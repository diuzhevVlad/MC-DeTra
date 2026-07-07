from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils

from detra_repro.config import DataConfig
from detra_repro.data.types import SceneSample


LABEL_TYPES = {
    v.number: v.name.replace("TYPE_", "")
    for v in label_pb2.Label.DESCRIPTOR.enum_types_by_name["Type"].values
}
LABEL_TYPE_IDS = {name: type_id for type_id, name in LABEL_TYPES.items()}

MAP_TOKEN_TYPES = (
    "lane",
    "road_line",
    "road_edge",
    "crosswalk",
    "speed_bump",
    "stop_sign",
    "driveway",
)
MAP_FEATURE_DIM = len(MAP_TOKEN_TYPES) + 5


def frame_pose_matrix(frame: open_dataset.Frame) -> np.ndarray:
    """Return the 4x4 vehicle-to-world pose matrix for a Waymo frame.

    Args:
        frame: Parsed Waymo frame.

    Returns:
        Float64 array with shape ``[4, 4]``. Waymo stores ``frame.pose.transform``
        as a flattened row-major rigid transform. Points produced by
        ``frame_utils.convert_range_image_to_point_cloud`` are in that frame's
        vehicle coordinates, so this matrix maps them into the global/world
        coordinate system.
    """

    return np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)


def transform_points_to_frame(
    xyz: np.ndarray,
    source_frame: open_dataset.Frame,
    target_frame: open_dataset.Frame,
) -> np.ndarray:
    """Transform lidar points from ``source_frame`` vehicle coords to target coords.

    Args:
        xyz: Source-frame points with shape ``[P, 3]``.
        source_frame: Frame whose vehicle coordinate system currently owns
            ``xyz``.
        target_frame: Frame whose vehicle coordinate system should own the
            returned points.

    Returns:
        Points with shape ``[P, 3]`` expressed in ``target_frame`` vehicle
        coordinates.

    Transform chain:
        ``target_xyz = inv(T_world_from_target) @ T_world_from_source @ xyz``.
    """

    if len(xyz) == 0:
        return xyz.astype(np.float32)

    source_to_world = frame_pose_matrix(source_frame)
    target_to_world = frame_pose_matrix(target_frame)
    source_to_target = np.linalg.inv(target_to_world) @ source_to_world

    homog = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=xyz.dtype)], axis=1)
    return (homog @ source_to_target.T)[:, :3].astype(np.float32)


def transform_world_points_to_frame(
    xyz_world: np.ndarray,
    target_frame: open_dataset.Frame,
) -> np.ndarray:
    """Transform world/map coordinates into a target vehicle frame.

    Args:
        xyz_world: World-frame coordinates with shape ``[P, 3]``.
        target_frame: Frame whose vehicle coordinate system should own output.

    Returns:
        Points with shape ``[P, 3]`` in ``target_frame`` vehicle coordinates.
    """

    if len(xyz_world) == 0:
        return xyz_world.astype(np.float32)

    world_to_target = np.linalg.inv(frame_pose_matrix(target_frame))
    homog = np.concatenate(
        [xyz_world, np.ones((len(xyz_world), 1), dtype=xyz_world.dtype)],
        axis=1,
    )
    return (homog @ world_to_target.T)[:, :3].astype(np.float32)


def _map_points_to_world(points, frame: open_dataset.Frame) -> np.ndarray:
    """Convert a repeated Waymo ``MapPoint`` field into world XYZ points.

    Waymo perception frames provide ``map_pose_offset`` together with
    ``map_features``. Adding that offset recovers the global/map coordinates
    used by the frame pose transform.
    """

    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    offset = frame.map_pose_offset
    return np.asarray(
        [[p.x + offset.x, p.y + offset.y, p.z + offset.z] for p in points],
        dtype=np.float32,
    )


def _sample_polyline_tokens(
    points_world: np.ndarray,
    target_frame: open_dataset.Frame,
    type_index: int,
    cfg: DataConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sample map tokens from a polyline or polygon boundary.

    Each output feature is ``[one_hot_type, dir_x, dir_y, z, length, closed]``.
    """

    if len(points_world) == 0:
        return [], []

    points_local = transform_world_points_to_frame(points_world, target_frame)
    x_min, y_min, x_max, y_max = cfg.roi_m
    one_hot = np.zeros(len(MAP_TOKEN_TYPES), dtype=np.float32)
    one_hot[type_index] = 1.0
    xy_tokens: list[np.ndarray] = []
    feature_tokens: list[np.ndarray] = []

    closed = float(type_index in {
        MAP_TOKEN_TYPES.index("crosswalk"),
        MAP_TOKEN_TYPES.index("speed_bump"),
        MAP_TOKEN_TYPES.index("driveway"),
    })
    if len(points_local) == 1:
        xy = points_local[0, :2]
        if x_min <= xy[0] < x_max and y_min <= xy[1] < y_max:
            xy_tokens.append(xy.astype(np.float32))
            feature_tokens.append(np.concatenate([one_hot, np.array([1.0, 0.0, points_local[0, 2], 0.0, 0.0], dtype=np.float32)]))
        return xy_tokens, feature_tokens

    for start, end in zip(points_local[:-1], points_local[1:]):
        delta = end[:2] - start[:2]
        length = float(np.linalg.norm(delta))
        if length < 1e-3:
            continue
        direction = delta / length
        samples = max(1, int(np.ceil(length / 3.0)))
        for sample_idx in range(samples):
            alpha = (sample_idx + 0.5) / samples
            point = start + alpha * (end - start)
            xy = point[:2]
            if not (x_min <= xy[0] < x_max and y_min <= xy[1] < y_max):
                continue
            scalars = np.array(
                [direction[0], direction[1], point[2], min(length, 50.0) / 50.0, closed],
                dtype=np.float32,
            )
            xy_tokens.append(xy.astype(np.float32))
            feature_tokens.append(np.concatenate([one_hot, scalars]))
    return xy_tokens, feature_tokens


def frame_map_tokens(
    frame: open_dataset.Frame,
    cfg: DataConfig,
    target_frame: open_dataset.Frame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract local vector-map tokens from Waymo perception ``map_features``.

    Args:
        frame: Waymo frame that owns ``map_features`` and ``map_pose_offset``.
            In perception TFRecords this is commonly only the first frame of a
            segment.
        cfg: Data config. Uses ``roi_m`` for token filtering.
        target_frame: Frame whose ego coordinate system should own output
            tokens. If omitted, tokens are produced in ``frame`` coordinates.

    Returns:
        ``(map_xy, map_features)``:
        - ``map_xy``: ``[M_map, 2]`` local-frame token coordinates.
        - ``map_features``: ``[M_map, MAP_FEATURE_DIM]`` features with map-type
          one-hot channels plus local direction/geometry scalars.
    """

    xy_tokens: list[np.ndarray] = []
    feature_tokens: list[np.ndarray] = []
    target_frame = target_frame or frame

    for feature in frame.map_features:
        if feature.HasField("lane"):
            type_name = "lane"
            points_world = _map_points_to_world(feature.lane.polyline, frame)
        elif feature.HasField("road_line"):
            type_name = "road_line"
            points_world = _map_points_to_world(feature.road_line.polyline, frame)
        elif feature.HasField("road_edge"):
            type_name = "road_edge"
            points_world = _map_points_to_world(feature.road_edge.polyline, frame)
        elif feature.HasField("crosswalk"):
            type_name = "crosswalk"
            points_world = _map_points_to_world(feature.crosswalk.polygon, frame)
        elif feature.HasField("speed_bump"):
            type_name = "speed_bump"
            points_world = _map_points_to_world(feature.speed_bump.polygon, frame)
        elif feature.HasField("stop_sign"):
            type_name = "stop_sign"
            position = feature.stop_sign.position
            points_world = _map_points_to_world([position], frame)
        elif feature.HasField("driveway"):
            type_name = "driveway"
            points_world = _map_points_to_world(feature.driveway.polygon, frame)
        else:
            continue

        new_xy, new_features = _sample_polyline_tokens(
            points_world,
            target_frame,
            MAP_TOKEN_TYPES.index(type_name),
            cfg,
        )
        xy_tokens.extend(new_xy)
        feature_tokens.extend(new_features)

    if not xy_tokens:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, MAP_FEATURE_DIM), dtype=np.float32),
        )
    return np.stack(xy_tokens).astype(np.float32), np.stack(feature_tokens).astype(np.float32)


def segment_map_frame(frames: list[open_dataset.Frame]) -> open_dataset.Frame | None:
    """Return the frame that carries segment map features, if any.

    Waymo perception TFRecords often store `map_features` only on the first
    frame. Reusing that map source and transforming it into every current ego
    frame keeps cached map tokens available across the segment.
    """

    return next((frame for frame in frames if frame.map_features), None)


def transform_box_to_frame(
    box: open_dataset.Label.Box,
    source_frame: open_dataset.Frame,
    target_frame: open_dataset.Frame,
) -> list[float]:
    """Transform one Waymo box center and yaw into target-frame coordinates.

    Args:
        box: Waymo laser label box from ``source_frame``.
        source_frame: Frame where ``box`` is expressed.
        target_frame: Desired local coordinate frame.

    Returns:
        List ``[x, y, z, length, width, height, heading]`` in target-frame
        coordinates.

    Heading handling:
        The center is transformed as a point. Heading is transformed by rotating
        a unit direction vector ``[cos(theta), sin(theta), 0]`` from source
        frame to target frame and taking ``atan2`` in the target XY plane. This
        is more robust than subtracting yaw angles from pose matrices manually.
    """

    center = np.array([[box.center_x, box.center_y, box.center_z]], dtype=np.float64)
    center_tgt = transform_points_to_frame(center, source_frame, target_frame)[0]

    source_to_world = frame_pose_matrix(source_frame)
    target_to_world = frame_pose_matrix(target_frame)
    rot_source_to_target = (np.linalg.inv(target_to_world) @ source_to_world)[:3, :3]
    heading_vec = np.array([np.cos(box.heading), np.sin(box.heading), 0.0])
    heading_tgt = rot_source_to_target @ heading_vec
    heading = float(np.arctan2(heading_tgt[1], heading_tgt[0]))

    return [
        float(center_tgt[0]),
        float(center_tgt[1]),
        float(center_tgt[2]),
        box.length,
        box.width,
        box.height,
        heading,
    ]


def read_waymo_frames(path: str | Path, max_frames: int | None = None) -> list[open_dataset.Frame]:
    """Read Waymo v1 TFRecord frames with the official Waymo API.

    Args:
        path: Path to a Waymo ``*.tfrecord`` segment.
        max_frames: Optional cap for debugging.

    Returns:
        List of parsed ``open_dataset.Frame`` protos. Each frame contains lidar
        range images, camera data, context metadata, and current-frame labels.

    Notes:
        Keep this function thin. Any expensive conversion should be cached by a
        preprocessing script so the training loop does not repeatedly parse
        protobufs.
    """

    if not tf.executing_eagerly():
        tf.enable_eager_execution()

    dataset = tf.data.TFRecordDataset(str(path), compression_type="")
    if max_frames is not None:
        dataset = dataset.take(max_frames)

    frames = []
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())
        frames.append(frame)
    return frames


def frame_lidar_points(frame: open_dataset.Frame) -> np.ndarray:
    """Convert one Waymo frame into XYZ lidar points using official helpers.

    Args:
        frame: One parsed Waymo frame.

    Returns:
        ``points`` as float array with shape ``[P, 3]`` and columns ``x, y, z``
        in the frame's vehicle coordinate system. ``P`` depends on sensor
        returns and valid range-image pixels.

    Scaling path:
        Use :func:`multi_sweep_lidar_points` to stack H sweeps transformed into
        the current ego frame, matching the paper more closely.
    """

    range_images, camera_projections, _, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    per_laser_points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose
    )
    return np.concatenate([np.asarray(p) for p in per_laser_points], axis=0)


def multi_sweep_lidar_points(
    frames: list[open_dataset.Frame],
    current_index: int,
    cfg: DataConfig,
) -> np.ndarray:
    """Stack H lidar sweeps into the current frame's local coordinate system.

    Args:
        frames: Parsed Waymo segment frames.
        current_index: Target/current frame index.
        cfg: Data config. Uses ``history_sweeps``, ``roi_m``, and ``z_range_m``.

    Returns:
        Float32 array with shape ``[P_total, 4]`` and columns
        ``x, y, z, dt`` in the current frame's vehicle coordinates.

        - ``dt`` is seconds relative to the current frame timestamp.
        - current sweep has ``dt=0``.
        - past sweeps have negative ``dt``.

    Paper mapping:
        DeTra uses H=5 past lidar sweeps over roughly 0.5 s, transformed into
        the coordinate frame of the most recent sweep. This function implements
        that geometric alignment.
    """

    current = frames[current_index]
    x_min, y_min, x_max, y_max = cfg.roi_m
    z_min, z_max = cfg.z_range_m
    sweeps = []

    first_index = max(0, current_index - cfg.history_sweeps + 1)
    for sweep_index in range(first_index, current_index + 1):
        sweep = frames[sweep_index]
        xyz = frame_lidar_points(sweep)
        xyz = transform_points_to_frame(xyz, sweep, current)
        dt = (sweep.timestamp_micros - current.timestamp_micros) * 1e-6

        keep = (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] < x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] < y_max)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] < z_max)
        )
        xyz = xyz[keep]
        dt_col = np.full((len(xyz), 1), dt, dtype=np.float32)
        sweeps.append(np.concatenate([xyz.astype(np.float32), dt_col], axis=1))

    if not sweeps:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(sweeps, axis=0)


def frame_detections(
    frame: open_dataset.Frame,
    target_frame: open_dataset.Frame | None = None,
    class_names: tuple[str, ...] = ("VEHICLE", "PEDESTRIAN", "CYCLIST"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract current-frame boxes, track ids, and class ids from laser labels.

    Args:
        frame: One parsed Waymo frame.
        target_frame: If provided, boxes are transformed from ``frame`` vehicle
            coordinates into ``target_frame`` vehicle coordinates.
        class_names: Class names to keep after stripping Waymo's ``TYPE_``
            prefix. Defaults to Waymo's standard moving-agent detection classes:
            vehicle, pedestrian, and cyclist.

    Returns:
        Tuple ``(boxes, track_ids, types)``:

        - ``boxes``: float32 array ``[M, 7]`` with columns
          ``x, y, z, length, width, height, heading``.
        - ``track_ids``: object id array ``[M]``.
        - ``types``: int64 class ids ``[M]`` using Waymo label enum values.

    Class filtering:
        Waymo labels also include signs and unknown objects. They are filtered
        here so detection/forecasting targets contain only vehicle, pedestrian,
        and cyclist by default.
    """

    keep_types = {LABEL_TYPE_IDS[name] for name in class_names}
    boxes, track_ids, types = [], [], []
    for label in frame.laser_labels:
        if label.type not in keep_types:
            continue
        box = label.box
        if target_frame is None or target_frame is frame:
            boxes.append([
                box.center_x,
                box.center_y,
                box.center_z,
                box.length,
                box.width,
                box.height,
                box.heading,
            ])
        else:
            boxes.append(transform_box_to_frame(box, frame, target_frame))
        track_ids.append(label.id)
        types.append(label.type)

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(track_ids, dtype=object),
        np.asarray(types, dtype=np.int64),
    )


def frame_detection_metadata(
    frame: open_dataset.Frame,
    target_frame: open_dataset.Frame | None = None,
    class_names: tuple[str, ...] = ("VEHICLE", "PEDESTRIAN", "CYCLIST"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract boxes plus Waymo evaluation metadata in one aligned pass."""

    keep_types = {LABEL_TYPE_IDS[name] for name in class_names}
    boxes, track_ids, types = [], [], []
    point_counts, detection_difficulty, tracking_difficulty = [], [], []
    for label in frame.laser_labels:
        if label.type not in keep_types:
            continue
        box = label.box
        if target_frame is None or target_frame is frame:
            boxes.append(
                [
                    box.center_x,
                    box.center_y,
                    box.center_z,
                    box.length,
                    box.width,
                    box.height,
                    box.heading,
                ]
            )
        else:
            boxes.append(transform_box_to_frame(box, frame, target_frame))
        track_ids.append(label.id)
        types.append(label.type)
        point_counts.append(label.num_lidar_points_in_box)
        detection_difficulty.append(label.detection_difficulty_level)
        tracking_difficulty.append(label.tracking_difficulty_level)
    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(track_ids, dtype=object),
        np.asarray(types, dtype=np.int64),
        np.asarray(point_counts, dtype=np.int32),
        np.asarray(detection_difficulty, dtype=np.int8),
        np.asarray(tracking_difficulty, dtype=np.int8),
    )


def frame_no_label_zones(frame: open_dataset.Frame) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Waymo no-label-zone polygons into vertices plus ragged offsets."""

    vertices: list[np.ndarray] = []
    offsets = [0]
    for polygon in frame.no_label_zones:
        count = min(len(polygon.x), len(polygon.y))
        if count:
            vertices.append(
                np.stack(
                    [
                        np.asarray(polygon.x[:count], dtype=np.float32),
                        np.asarray(polygon.y[:count], dtype=np.float32),
                    ],
                    axis=1,
                )
            )
        offsets.append(offsets[-1] + count)
    xy = np.concatenate(vertices, axis=0) if vertices else np.zeros((0, 2), dtype=np.float32)
    return xy, np.asarray(offsets, dtype=np.int32)


def _object_roi_mask(boxes: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Return current-frame objects whose centers are inside the configured ROI.

    Args:
        boxes: ``[M,7]`` local-frame boxes.
        cfg: Data configuration containing ``roi_m``.

    Returns:
        Boolean mask ``[M]``. This keeps the cached targets aligned with the
        LiDAR/map crop. It is especially important for constrained-range
        experiments, where objects outside the crop otherwise have no sensor
        evidence but still contribute detection and forecasting losses.
    """

    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)
    x_min, y_min, x_max, y_max = cfg.roi_m
    return (
        (boxes[:, 0] >= x_min)
        & (boxes[:, 0] < x_max)
        & (boxes[:, 1] >= y_min)
        & (boxes[:, 1] < y_max)
    )


def build_future_targets(
    frames: list[open_dataset.Frame],
    current_index: int,
    track_ids: np.ndarray,
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build future center trajectories by matching Waymo ``label.id``.

    Args:
        frames: Parsed segment frames in chronological order.
        current_index: Index of the frame used as ``t=0``.
        track_ids: Current-frame object ids with shape ``[M]``.
        cfg: Data settings. ``future_steps`` and ``frame_stride`` determine
            target timestamps.

    Returns:
        ``future_xy``: float32 array ``[M, T_future, 2]``.
        ``future_mask``: bool array ``[M, T_future]``.

    Coordinate convention:
        Future boxes are transformed into the current frame's vehicle
        coordinates with Waymo ego poses, matching the local frame used by the
        current lidar sweeps and current GT boxes.

    Missing-label behavior:
        Each target timestamp is looked up independently. A missing ``label.id``
        leaves that waypoint at padding with ``future_mask=False``; it is not
        interpolated and does not invalidate later timestamps if the ID
        reappears. Actors are selected from the current frame only.
    """

    target_frame = frames[current_index]
    by_frame = []
    for frame in frames:
        by_id = {label.id: label for label in frame.laser_labels}
        by_frame.append(by_id)

    m = len(track_ids)
    future_xy = np.zeros((m, cfg.future_steps, 2), dtype=np.float32)
    future_mask = np.zeros((m, cfg.future_steps), dtype=bool)

    for obj_idx, track_id in enumerate(track_ids):
        for t in range(cfg.future_steps):
            frame_idx = current_index + (t + 1) * cfg.frame_stride
            if frame_idx >= len(frames):
                continue
            label = by_frame[frame_idx].get(track_id)
            if label is None:
                continue
            future_box = transform_box_to_frame(label.box, frames[frame_idx], target_frame)
            future_xy[obj_idx, t] = future_box[:2]
            future_mask[obj_idx, t] = True

    return future_xy, future_mask


def build_history_targets(
    frames: list[open_dataset.Frame],
    current_index: int,
    track_ids: np.ndarray,
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build past/current center histories by matching Waymo ``label.id``.

    Args:
        frames: Parsed segment frames in chronological order.
        current_index: Index of the frame used as ``t=0``.
        track_ids: Current-frame object ids with shape ``[M]``.
        cfg: Data settings. ``history_sweeps`` determines the number of
            history points and ``frame_stride`` determines temporal spacing.

    Returns:
        ``history_xy``: float32 array ``[M, H_history, 2]``.
        ``history_mask``: bool array ``[M, H_history]``.

    Notes:
        This is a diagnostic actor-history signal. The DeTra paper uses past
        LiDAR sweeps with point timestamps, not explicit tracker histories, but
        this target is useful for isolating whether our simplified model can use
        motion cues when object association is provided.
    """

    target_frame = frames[current_index]
    by_frame = []
    for frame in frames:
        by_id = {label.id: label for label in frame.laser_labels}
        by_frame.append(by_id)

    m = len(track_ids)
    h = cfg.history_sweeps
    history_xy = np.zeros((m, h, 2), dtype=np.float32)
    history_mask = np.zeros((m, h), dtype=bool)

    for obj_idx, track_id in enumerate(track_ids):
        for hist_idx in range(h):
            # Oldest to newest; newest slot is current frame.
            steps_back = h - 1 - hist_idx
            frame_idx = current_index - steps_back * cfg.frame_stride
            if frame_idx < 0:
                continue
            label = by_frame[frame_idx].get(track_id)
            if label is None:
                continue
            history_box = transform_box_to_frame(label.box, frames[frame_idx], target_frame)
            history_xy[obj_idx, hist_idx] = history_box[:2]
            history_mask[obj_idx, hist_idx] = True

    return history_xy, history_mask


def make_scene_sample(
    frames: list[open_dataset.Frame],
    current_index: int,
    cfg: DataConfig,
) -> SceneSample:
    """Create one DeTra training sample from parsed Waymo frames.

    Args:
        frames: Parsed segment frames.
        current_index: Target/current frame index.
        cfg: Data configuration.

    Returns:
        ``SceneSample`` with H-sweep lidar points transformed into the current
        local frame and matched future targets transformed into that same frame.

    Expected shapes:
        - ``points``: ``[P, 4]`` with ``x,y,z,dt``.
        - ``gt_boxes``: ``[M, 7]``.
        - ``gt_future``: ``[M, T_future, 2]``.
        - ``map_xy``: ``[M_map, 2]``.
        - ``map_features``: ``[M_map, C_map]``.

    Coordinate convention:
        All points, current boxes, and future centers are expressed in the
        current frame's vehicle coordinate system. This is the local frame used
        by the model.
    """

    frame = frames[current_index]
    points = multi_sweep_lidar_points(frames, current_index, cfg)
    (
        boxes,
        track_ids,
        types,
        point_counts,
        detection_difficulty,
        tracking_difficulty,
    ) = frame_detection_metadata(
        frame,
        target_frame=frame,
        class_names=cfg.class_names,
    )
    future, future_mask = build_future_targets(frames, current_index, track_ids, cfg)
    history, history_mask = build_history_targets(frames, current_index, track_ids, cfg)
    keep_objects = _object_roi_mask(boxes, cfg)
    boxes = boxes[keep_objects]
    track_ids = track_ids[keep_objects]
    types = types[keep_objects]
    point_counts = point_counts[keep_objects]
    detection_difficulty = detection_difficulty[keep_objects]
    tracking_difficulty = tracking_difficulty[keep_objects]
    future = future[keep_objects]
    future_mask = future_mask[keep_objects]
    history = history[keep_objects]
    history_mask = history_mask[keep_objects]
    map_source = frame if frame.map_features else segment_map_frame(frames)
    if map_source is None:
        map_xy, map_features = (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, MAP_FEATURE_DIM), dtype=np.float32),
        )
    else:
        map_xy, map_features = frame_map_tokens(map_source, cfg, target_frame=frame)
    nlz_xy, nlz_offsets = frame_no_label_zones(frame)
    paper_vehicle_eligible = (types == LABEL_TYPE_IDS["VEHICLE"]) & (point_counts >= 1)
    return SceneSample(
        points=points,
        gt_boxes=boxes,
        gt_track_ids=track_ids,
        gt_types=types,
        gt_future=future,
        gt_future_mask=future_mask,
        gt_history=history,
        gt_history_mask=history_mask,
        map_xy=map_xy,
        map_features=map_features,
        gt_num_lidar_points=point_counts,
        gt_detection_difficulty=detection_difficulty,
        gt_tracking_difficulty=tracking_difficulty,
        paper_vehicle_eligible=paper_vehicle_eligible,
        context_name=str(frame.context.name),
        frame_index=int(current_index),
        frame_timestamp_micros=int(frame.timestamp_micros),
        ego_pose=np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4),
        nlz_xy=nlz_xy,
        nlz_offsets=nlz_offsets,
    )

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneSample:
    """One model training/evaluation sample.

    This is the canonical data object produced by preprocessing. Keep this
    object small and serializable; heavy raw Waymo protos should not flow into
    the training loop.

    Attributes:
        points: Lidar points in the current ego frame.
            Shape: ``[P, 4]`` with columns ``x, y, z, dt``.
            ``P`` is variable. ``dt`` is seconds relative to current frame,
            e.g. current sweep has dt=0 and past sweeps have negative dt.
        gt_boxes: Current-frame ground-truth boxes.
            Shape: ``[M, 7]`` with columns
            ``x, y, z, length, width, height, heading``.
        gt_track_ids: Stable Waymo laser-label ids for current objects.
            Shape: ``[M]``. These are used to link future labels into
            trajectories.
        gt_types: Integer semantic classes for current objects.
            Shape: ``[M]``. The default preprocessing keeps Waymo vehicle,
            pedestrian, and cyclist labels.
        gt_future: Future object centers.
            Shape: ``[M, T_future, 2]`` containing future ``x, y`` centers in
            the current ego frame. Missing future waypoints should be paired
            with ``gt_future_mask=False``.
        gt_future_mask: Future validity mask.
            Shape: ``[M, T_future]``. ``True`` where the object exists and has
            an associated future label.
        gt_history: Past object centers.
            Shape: ``[M, H_history, 2]`` in chronological order, oldest to
            newest. The newest slot corresponds to the current frame center.
            These are diagnostic actor-history features, not part of the
            paper's official DeTra input.
        gt_history_mask: History validity mask.
            Shape: ``[M, H_history]``. ``True`` where the object exists and has
            an associated past/current label.
        map_xy: Vector-map token coordinates.
            Shape: ``[M_map, 2]`` in the current ego frame.
        map_features: Per-token vector-map features.
            Shape: ``[M_map, C_map]``. The first channels are map-type one-hot
            features, followed by local direction and scalar geometry cues.
    """

    points: np.ndarray
    gt_boxes: np.ndarray
    gt_track_ids: np.ndarray
    gt_types: np.ndarray
    gt_future: np.ndarray
    gt_future_mask: np.ndarray
    gt_history: np.ndarray
    gt_history_mask: np.ndarray
    map_xy: np.ndarray
    map_features: np.ndarray
    gt_num_lidar_points: np.ndarray
    gt_detection_difficulty: np.ndarray
    gt_tracking_difficulty: np.ndarray
    paper_vehicle_eligible: np.ndarray
    context_name: str
    frame_index: int
    frame_timestamp_micros: int
    ego_pose: np.ndarray
    nlz_xy: np.ndarray
    nlz_offsets: np.ndarray


@dataclass
class Batch:
    """Torch-ready mini-batch after collation.

    Suggested tensor shapes after converting numpy arrays to torch tensors:

    - ``bev``: ``[B, C_in, H, W]`` float32 BEV raster.
    - ``gt_boxes``: ``[B, M_max, 7]`` float32 padded boxes.
    - ``gt_valid``: ``[B, M_max]`` bool, valid object slots.
    - ``gt_future``: ``[B, M_max, T_future, 2]`` float32 trajectories.
    - ``gt_future_mask``: ``[B, M_max, T_future]`` bool.
    - ``gt_types``: ``[B, M_max]`` int64 class ids.
    - ``map_xy``: ``[B, K_max, 2]`` float32 vector-map token coordinates.
    - ``map_features``: ``[B, K_max, C_map]`` float32 vector-map token features.
    - ``map_mask``: ``[B, K_max]`` bool valid map-token mask.

    The minimal training loop can start with only ``bev``, ``gt_boxes``, and
    ``gt_future``. Detection matching and forecasting masks can be added without
    changing model outputs.
    """

    bev: object
    gt_boxes: object
    gt_valid: object
    gt_future: object
    gt_future_mask: object

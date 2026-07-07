import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from detra_repro.data.cache_v2 import CACHE_SCHEMA_VERSION, MANIFEST_NAME, REQUIRED_V2_KEYS

DEFAULT_VEHICLE_TYPE_ID = 1


class CachedWaymoDataset(Dataset):
    """Dataset over cached ``sample_*.npz`` files.

    Expected arrays per file:
        - ``points``: ``[P,4]`` float32, columns ``x,y,z,dt``.
        - ``gt_boxes``: ``[M,7]`` float32.
        - ``gt_future``: ``[M,T,2]`` float32.
        - ``gt_future_mask``: ``[M,T]`` bool.
        - all cache-v2 metadata fields documented in ``data/cache_v2.py``.

    The collate function handles variable ``P`` and ``M`` by sampling/padding
    points and padding GT objects.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        require_schema_version: int | None = CACHE_SCHEMA_VERSION,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.require_schema_version = (
            CACHE_SCHEMA_VERSION if require_schema_version is None else require_schema_version
        )
        manifest_path = self.cache_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} is required; cache-v1 and manifest-less caches are not supported"
            )
        manifest_version = int(
            json.loads(manifest_path.read_text(encoding="utf-8")).get("cache_schema_version", 0)
        )
        if manifest_version != self.require_schema_version:
            raise ValueError(
                f"{manifest_path}: cache schema {manifest_version}, "
                f"required {self.require_schema_version}"
            )
        self.files = sorted(self.cache_dir.glob("**/sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files found under {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample_path = self.files[index]
        with np.load(sample_path, allow_pickle=True) as data:
            missing = REQUIRED_V2_KEYS.difference(data.files)
            if missing:
                raise ValueError(f"{sample_path}: missing cache-v2 keys {sorted(missing)}")
            schema_version = int(data["cache_schema_version"])
            if schema_version != self.require_schema_version:
                raise ValueError(
                    f"{sample_path}: cache schema {schema_version}, "
                    f"required {self.require_schema_version}"
                )
            # Empty Waymo frames may be serialized from an empty Python list as
            # shape ``(0,)``. Normalize them to the documented box shape so the
            # padded collator can handle legitimate zero-object frames.
            gt_boxes = data["gt_boxes"].astype(np.float32).reshape(-1, 7)
            return {
                "sample_index": np.asarray(index, dtype=np.int64),
                "sample_path": str(sample_path),
                "points": data["points"].astype(np.float32),
                "gt_boxes": gt_boxes,
                "gt_types": data["gt_types"].astype(np.int64),
                "gt_future": data["gt_future"].astype(np.float32),
                "gt_future_mask": data["gt_future_mask"].astype(bool),
                "gt_history": data["gt_history"].astype(np.float32),
                "gt_history_mask": data["gt_history_mask"].astype(bool),
                "map_xy": data["map_xy"].astype(np.float32),
                "map_features": data["map_features"].astype(np.float32),
                "gt_num_lidar_points": data["gt_num_lidar_points"].astype(np.int32),
                "gt_detection_difficulty": data["gt_detection_difficulty"].astype(np.int8),
                "gt_tracking_difficulty": data["gt_tracking_difficulty"].astype(np.int8),
                "paper_vehicle_eligible": data["paper_vehicle_eligible"].astype(bool),
                "cache_schema_version": np.asarray(schema_version, dtype=np.int16),
                "context_name": str(data["context_name"]),
                "frame_index": np.asarray(data["frame_index"], dtype=np.int32),
                "frame_timestamp_micros": np.asarray(data["frame_timestamp_micros"], dtype=np.int64),
                "source_tfrecord": str(data["source_tfrecord"]),
                "ego_pose": data["ego_pose"].astype(np.float64),
                "nlz_xy": data["nlz_xy"].astype(np.float32),
                "nlz_offsets": data["nlz_offsets"].astype(np.int32),
            }


def collate_cached_waymo(
    samples: list[dict[str, np.ndarray]],
    max_points: int | None = 32768,
    max_objects: int = 128,
    max_map_tokens: int = 2048,
    deterministic_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collate cached samples into fixed-shape tensors.

    Args:
        samples: Dataset samples.
        max_points: Number of lidar points to sample/pad per scene.
        max_objects: Maximum GT objects retained per scene.
        max_map_tokens: Maximum vector-map tokens retained per scene.
        deterministic_seed: Optional base seed for deterministic point
            subsampling. When set, each sample uses ``seed + sample_index`` so
            overfit/debug runs see identical lidar subsets on every pass.

    Returns:
        Dict with:
        - ``points``: ``[B,max_points,4]``.
        - ``point_mask``: ``[B,max_points]``.
        - ``gt_boxes``: ``[B,max_objects,7]``.
        - ``gt_types``: ``[B,max_objects]``.
        - ``gt_valid``: ``[B,max_objects]``.
        - ``gt_future``: ``[B,max_objects,T,2]``.
        - ``gt_future_mask``: ``[B,max_objects,T]``.
        - ``gt_history``: ``[B,max_objects,H,2]``.
        - ``gt_history_mask``: ``[B,max_objects,H]``.
        - ``map_xy``: ``[B,max_map_tokens,2]``.
        - ``map_features``: ``[B,max_map_tokens,C_map]``.
        - ``map_mask``: ``[B,max_map_tokens]``.
    """

    batch_size = len(samples)
    future_steps = samples[0]["gt_future"].shape[1]
    history_steps = samples[0]["gt_history"].shape[1]
    map_feature_dim = samples[0]["map_features"].shape[1]
    point_capacity = (
        max(len(sample["points"]) for sample in samples) if max_points is None else max_points
    )
    points = torch.zeros(batch_size, point_capacity, 4, dtype=torch.float32)
    point_mask = torch.zeros(batch_size, point_capacity, dtype=torch.bool)
    gt_boxes = torch.zeros(batch_size, max_objects, 7, dtype=torch.float32)
    gt_types = torch.full(
        (batch_size, max_objects),
        DEFAULT_VEHICLE_TYPE_ID,
        dtype=torch.long,
    )
    gt_valid = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    gt_future = torch.zeros(batch_size, max_objects, future_steps, 2, dtype=torch.float32)
    gt_future_mask = torch.zeros(batch_size, max_objects, future_steps, dtype=torch.bool)
    gt_history = torch.zeros(batch_size, max_objects, history_steps, 2, dtype=torch.float32)
    gt_history_mask = torch.zeros(batch_size, max_objects, history_steps, dtype=torch.bool)
    map_xy = torch.zeros(batch_size, max_map_tokens, 2, dtype=torch.float32)
    map_features = torch.zeros(batch_size, max_map_tokens, map_feature_dim, dtype=torch.float32)
    map_mask = torch.zeros(batch_size, max_map_tokens, dtype=torch.bool)
    sample_indices = torch.zeros(batch_size, dtype=torch.long)
    sample_paths: list[str] = []
    gt_num_lidar_points = torch.full((batch_size, max_objects), -1, dtype=torch.int32)
    gt_detection_difficulty = torch.zeros(batch_size, max_objects, dtype=torch.int8)
    gt_tracking_difficulty = torch.zeros(batch_size, max_objects, dtype=torch.int8)
    paper_vehicle_eligible = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    context_names: list[str] = []
    source_tfrecords: list[str] = []
    frame_indices = torch.full((batch_size,), -1, dtype=torch.int32)
    frame_timestamps = torch.full((batch_size,), -1, dtype=torch.int64)
    ego_poses = torch.eye(4, dtype=torch.float64).repeat(batch_size, 1, 1)
    nlz_xy: list[np.ndarray] = []
    nlz_offsets: list[np.ndarray] = []
    cache_schema_versions = torch.ones(batch_size, dtype=torch.int16)

    for batch_idx, sample in enumerate(samples):
        sample_indices[batch_idx] = int(sample.get("sample_index", np.asarray(batch_idx)))
        sample_paths.append(str(sample.get("sample_path", "")))
        sample_points = sample["points"]
        num_points = min(len(sample_points), point_capacity)
        if len(sample_points) > point_capacity:
            if deterministic_seed is None:
                rng = np.random.default_rng()
            else:
                sample_index = int(sample.get("sample_index", np.asarray(batch_idx)))
                rng = np.random.default_rng(deterministic_seed + sample_index)
            selected = rng.choice(len(sample_points), point_capacity, replace=False)
            sample_points = sample_points[selected]
        points[batch_idx, :num_points] = torch.from_numpy(sample_points[:num_points])
        point_mask[batch_idx, :num_points] = True

        num_objects = min(len(sample["gt_boxes"]), max_objects)
        gt_boxes[batch_idx, :num_objects] = torch.from_numpy(sample["gt_boxes"][:num_objects])
        gt_types[batch_idx, :num_objects] = torch.from_numpy(sample["gt_types"][:num_objects])
        gt_valid[batch_idx, :num_objects] = True
        gt_num_lidar_points[batch_idx, :num_objects] = torch.from_numpy(
            sample["gt_num_lidar_points"][:num_objects]
        )
        gt_detection_difficulty[batch_idx, :num_objects] = torch.from_numpy(
            sample["gt_detection_difficulty"][:num_objects]
        )
        gt_tracking_difficulty[batch_idx, :num_objects] = torch.from_numpy(
            sample["gt_tracking_difficulty"][:num_objects]
        )
        paper_vehicle_eligible[batch_idx, :num_objects] = torch.from_numpy(
            sample["paper_vehicle_eligible"][:num_objects]
        )
        gt_future[batch_idx, :num_objects] = torch.from_numpy(sample["gt_future"][:num_objects])
        gt_future_mask[batch_idx, :num_objects] = torch.from_numpy(
            sample["gt_future_mask"][:num_objects]
        )
        gt_history[batch_idx, :num_objects] = torch.from_numpy(sample["gt_history"][:num_objects])
        gt_history_mask[batch_idx, :num_objects] = torch.from_numpy(
            sample["gt_history_mask"][:num_objects]
        )
        num_map = min(len(sample["map_xy"]), max_map_tokens)
        if num_map:
            if len(sample["map_xy"]) > max_map_tokens:
                if deterministic_seed is None:
                    rng = np.random.default_rng()
                else:
                    sample_index = int(sample.get("sample_index", np.asarray(batch_idx)))
                    rng = np.random.default_rng(deterministic_seed + 104729 + sample_index)
                selected = rng.choice(len(sample["map_xy"]), max_map_tokens, replace=False)
                sample_map_xy = sample["map_xy"][selected]
                sample_map_features = sample["map_features"][selected]
            else:
                sample_map_xy = sample["map_xy"]
                sample_map_features = sample["map_features"]
            map_xy[batch_idx, :num_map] = torch.from_numpy(sample_map_xy[:num_map])
            map_features[batch_idx, :num_map] = torch.from_numpy(sample_map_features[:num_map])
            map_mask[batch_idx, :num_map] = True
        context_names.append(sample["context_name"])
        source_tfrecords.append(sample["source_tfrecord"])
        frame_indices[batch_idx] = int(sample["frame_index"])
        frame_timestamps[batch_idx] = int(sample["frame_timestamp_micros"])
        ego_poses[batch_idx] = torch.from_numpy(sample["ego_pose"])
        nlz_xy.append(sample["nlz_xy"])
        nlz_offsets.append(sample["nlz_offsets"])
        cache_schema_versions[batch_idx] = int(sample["cache_schema_version"])

    return {
        "points": points,
        "point_mask": point_mask,
        "gt_boxes": gt_boxes,
        "gt_types": gt_types,
        "gt_valid": gt_valid,
        "gt_future": gt_future,
        "gt_future_mask": gt_future_mask,
        "gt_history": gt_history,
        "gt_history_mask": gt_history_mask,
        "map_xy": map_xy,
        "map_features": map_features,
        "map_mask": map_mask,
        "sample_indices": sample_indices,
        "sample_paths": sample_paths,
        "gt_num_lidar_points": gt_num_lidar_points,
        "gt_detection_difficulty": gt_detection_difficulty,
        "gt_tracking_difficulty": gt_tracking_difficulty,
        "paper_vehicle_eligible": paper_vehicle_eligible,
        "context_names": context_names,
        "source_tfrecords": source_tfrecords,
        "frame_indices": frame_indices,
        "frame_timestamp_micros": frame_timestamps,
        "ego_poses": ego_poses,
        "nlz_xy": nlz_xy,
        "nlz_offsets": nlz_offsets,
        "cache_schema_versions": cache_schema_versions,
    }

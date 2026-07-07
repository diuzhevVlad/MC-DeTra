from pathlib import Path

import numpy as np

from detra_repro.config import DataConfig
from detra_repro.data.waymo import make_scene_sample, read_waymo_frames


def cache_segment_samples(
    tfrecord_path: str | Path,
    output_dir: str | Path,
    cfg: DataConfig,
    max_frames: int | None = None,
) -> None:
    """Preprocess a Waymo segment into small ``.npz`` training samples.

    Args:
        tfrecord_path: Input Waymo v1 segment.
        output_dir: Directory where ``sample_000000.npz`` files are written.
        cfg: Data configuration.
        max_frames: Optional cap for debugging.

    Saved arrays per sample:
        - ``points``: ``[P, 4]`` float32, ``x,y,z,dt``.
        - ``gt_boxes``: ``[M, 7]`` float32.
        - ``gt_track_ids``: ``[M]`` object strings.
        - ``gt_types``: ``[M]`` int64.
        - ``gt_future``: ``[M, T_future, 2]`` float32.
        - ``gt_future_mask``: ``[M, T_future]`` bool.
        - ``gt_history``: ``[M, H_history, 2]`` float32.
        - ``gt_history_mask``: ``[M, H_history]`` bool.
        - ``map_xy``: ``[M_map, 2]`` float32.
        - ``map_features``: ``[M_map, C_map]`` float32.

    Design reason:
        Raw Waymo TFRecords are expensive to parse. Caching this normalized
        format lets model experiments iterate quickly and keeps the data/model
        boundary stable.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_waymo_frames(tfrecord_path, max_frames=max_frames)
    last_usable = len(frames) - cfg.future_steps * cfg.frame_stride
    for current_index in range(max(0, last_usable)):
        sample = make_scene_sample(frames, current_index, cfg)
        np.savez_compressed(
            output_dir / f"sample_{current_index:06d}.npz",
            points=sample.points,
            gt_boxes=sample.gt_boxes,
            gt_track_ids=sample.gt_track_ids,
            gt_types=sample.gt_types,
            gt_future=sample.gt_future,
            gt_future_mask=sample.gt_future_mask,
            gt_history=sample.gt_history,
            gt_history_mask=sample.gt_history_mask,
            map_xy=sample.map_xy,
            map_features=sample.map_features,
        )

"""Validate a completed cache-v2 and spot-check labels against raw Waymo frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from detra_repro.data.cache_v2 import CACHE_SCHEMA_VERSION, MANIFEST_NAME, validate_v2_sample
from detra_repro.data.waymo import frame_detection_metadata, read_waymo_frames


def _inside_roi(boxes: np.ndarray, roi: list[float]) -> np.ndarray:
    x_min, y_min, x_max, y_max = roi
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        # frame_detection_metadata returns a flat array when the frame has 0 or 1
        # boxes; normalize to (M, D) so per-column indexing below works.
        boxes = boxes.reshape(0, 7) if boxes.size == 0 else boxes.reshape(1, -1)
    return (
        (boxes[:, 0] >= x_min)
        & (boxes[:, 0] < x_max)
        & (boxes[:, 1] >= y_min)
        & (boxes[:, 1] < y_max)
    )


def validate(cache: str | Path, random_samples: int, seed: int) -> dict:
    cache = Path(cache)
    manifest = json.loads((cache / MANIFEST_NAME).read_text(encoding="utf-8"))
    if int(manifest.get("cache_schema_version", 0)) != CACHE_SCHEMA_VERSION:
        raise ValueError("expected cache schema v2")
    files = sorted(cache.glob("*/sample_*.npz"))
    if len(files) != int(manifest["stats"]["samples"]):
        raise ValueError(
            f"manifest has {manifest['stats']['samples']} samples but filesystem has {len(files)}"
        )
    rng = np.random.default_rng(seed)
    selected = (
        list(rng.choice(files, size=min(random_samples, len(files)), replace=False))
        if files
        else []
    )
    source_map = {Path(item["path"]).name: item["path"] for item in manifest["sources"]}
    frame_cache: dict[str, list] = {}
    checked = []
    for path in selected:
        validate_v2_sample(path, deep=True)
        with np.load(path, allow_pickle=True) as data:
            source_name = str(data["source_tfrecord"])
            source_path = source_map[source_name]
            if source_path not in frame_cache:
                frame_cache[source_path] = read_waymo_frames(source_path)
            frames = frame_cache[source_path]
            frame_index = int(data["frame_index"])
            if frame_index < 4 or frame_index + 50 >= len(frames):
                raise ValueError(f"{path}: frame does not have complete history/future coverage")
            frame = frames[frame_index]
            boxes, _, types, counts, detection_difficulty, tracking_difficulty = (
                frame_detection_metadata(frame, target_frame=frame)
            )
            # For frames with 0 or 1 boxes the metadata boxes come back flat;
            # normalize to (M, 7) so it matches the cached gt_boxes layout.
            boxes = np.asarray(boxes)
            if boxes.ndim == 1:
                boxes = boxes.reshape(0, 7) if boxes.size == 0 else boxes.reshape(1, -1)
            keep = _inside_roi(boxes, manifest["config"]["roi_m"])
            np.testing.assert_allclose(data["gt_boxes"], boxes[keep], rtol=0, atol=1e-5)
            np.testing.assert_array_equal(data["gt_types"], types[keep])
            np.testing.assert_array_equal(data["gt_num_lidar_points"], counts[keep])
            np.testing.assert_array_equal(
                data["gt_detection_difficulty"], detection_difficulty[keep]
            )
            np.testing.assert_array_equal(
                data["gt_tracking_difficulty"], tracking_difficulty[keep]
            )
            if str(data["context_name"]) != frame.context.name:
                raise ValueError(f"{path}: context name mismatch")
            if int(data["frame_timestamp_micros"]) != frame.timestamp_micros:
                raise ValueError(f"{path}: timestamp mismatch")
            checked.append(str(path))
    return {
        "cache": str(cache),
        "num_samples": len(files),
        "num_segments": int(manifest["num_segments"]),
        "deep_random_samples_checked": len(checked),
        "checked_samples": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--random-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    result = validate(args.cache, args.random_samples, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

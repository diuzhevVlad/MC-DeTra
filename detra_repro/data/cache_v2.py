"""Versioned, metadata-complete cache helpers for paper-facing Waymo evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from detra_repro.config import DataConfig
from detra_repro.data.types import SceneSample


CACHE_SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"
SEGMENT_MANIFEST_NAME = "_segment_manifest.json"

REQUIRED_V2_KEYS = {
    "points",
    "gt_boxes",
    "gt_track_ids",
    "gt_types",
    "gt_future",
    "gt_future_mask",
    "gt_history",
    "gt_history_mask",
    "map_xy",
    "map_features",
    "gt_num_lidar_points",
    "gt_detection_difficulty",
    "gt_tracking_difficulty",
    "paper_vehicle_eligible",
    "context_name",
    "frame_index",
    "frame_timestamp_micros",
    "source_tfrecord",
    "ego_pose",
    "nlz_xy",
    "nlz_offsets",
    "cache_schema_version",
}


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_arrays(sample: SceneSample, source_tfrecord: str) -> dict[str, Any]:
    return {
        "points": sample.points.astype(np.float32, copy=False),
        "gt_boxes": sample.gt_boxes.astype(np.float32, copy=False).reshape(-1, 7),
        "gt_track_ids": sample.gt_track_ids,
        "gt_types": sample.gt_types.astype(np.int64, copy=False),
        "gt_future": sample.gt_future.astype(np.float32, copy=False),
        "gt_future_mask": sample.gt_future_mask.astype(bool, copy=False),
        "gt_history": sample.gt_history.astype(np.float32, copy=False),
        "gt_history_mask": sample.gt_history_mask.astype(bool, copy=False),
        "map_xy": sample.map_xy.astype(np.float32, copy=False),
        "map_features": sample.map_features.astype(np.float32, copy=False),
        "gt_num_lidar_points": sample.gt_num_lidar_points.astype(np.int32, copy=False),
        "gt_detection_difficulty": sample.gt_detection_difficulty.astype(np.int8, copy=False),
        "gt_tracking_difficulty": sample.gt_tracking_difficulty.astype(np.int8, copy=False),
        "paper_vehicle_eligible": sample.paper_vehicle_eligible.astype(bool, copy=False),
        "context_name": np.asarray(sample.context_name),
        "frame_index": np.asarray(sample.frame_index, dtype=np.int32),
        "frame_timestamp_micros": np.asarray(sample.frame_timestamp_micros, dtype=np.int64),
        "source_tfrecord": np.asarray(source_tfrecord),
        "ego_pose": sample.ego_pose.astype(np.float64, copy=False),
        "nlz_xy": sample.nlz_xy.astype(np.float32, copy=False),
        "nlz_offsets": sample.nlz_offsets.astype(np.int32, copy=False),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
    }


def atomic_save_sample(path: str | Path, sample: SceneSample, source_tfrecord: str) -> None:
    """Write one NPZ completely before atomically publishing its final path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **_sample_arrays(sample, source_tfrecord))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_v2_sample(path: str | Path, *, deep: bool = False) -> dict[str, int]:
    """Validate schema and aligned dimensions, returning compact sample statistics."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        missing = REQUIRED_V2_KEYS.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing cache-v2 keys {sorted(missing)}")
        version = int(data["cache_schema_version"])
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(f"{path}: cache schema {version}, expected {CACHE_SCHEMA_VERSION}")
        num_gt = len(data["gt_boxes"])
        if data["gt_boxes"].shape != (num_gt, 7):
            raise ValueError(f"{path}: gt_boxes must have shape ({num_gt},7)")
        aligned = (
            "gt_track_ids",
            "gt_types",
            "gt_future",
            "gt_future_mask",
            "gt_history",
            "gt_history_mask",
            "gt_num_lidar_points",
            "gt_detection_difficulty",
            "gt_tracking_difficulty",
            "paper_vehicle_eligible",
        )
        for key in aligned:
            if len(data[key]) != num_gt:
                raise ValueError(f"{path}: {key} has {len(data[key])} rows, expected {num_gt}")
        offsets = data["nlz_offsets"]
        if offsets.ndim != 1 or len(offsets) == 0 or int(offsets[0]) != 0:
            raise ValueError(f"{path}: malformed nlz_offsets")
        if int(offsets[-1]) != len(data["nlz_xy"]) or np.any(np.diff(offsets) < 0):
            raise ValueError(f"{path}: nlz offsets do not align with nlz_xy")
        if data["ego_pose"].shape != (4, 4):
            raise ValueError(f"{path}: ego_pose must have shape (4,4)")
        if deep:
            expected = (data["gt_types"] == 1) & (data["gt_num_lidar_points"] >= 1)
            if not np.array_equal(data["paper_vehicle_eligible"], expected):
                raise ValueError(f"{path}: paper_vehicle_eligible is inconsistent")
            for key in ("points", "gt_boxes", "gt_future", "gt_history"):
                if not np.isfinite(data[key]).all():
                    raise ValueError(f"{path}: non-finite values in {key}")
        return {
            "num_points": int(len(data["points"])),
            "num_gt": int(num_gt),
            "num_paper_vehicle_gt": int(data["paper_vehicle_eligible"].sum()),
        }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_segment_manifest(
    segment_dir: str | Path,
    *,
    source_tfrecord: str,
    source_sha256: str | None,
    cfg: DataConfig,
    sample_stride: int,
    stats: dict[str, int],
) -> None:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_tfrecord": source_tfrecord,
        "source_sha256": source_sha256,
        "config": asdict(cfg),
        "sample_stride": int(sample_stride),
        "stats": stats,
    }
    atomic_write_json(Path(segment_dir) / SEGMENT_MANIFEST_NAME, payload)


def finalize_cache_manifest(
    cache_root: str | Path,
    *,
    split: str,
    split_json: str | Path,
    cfg: DataConfig,
    sample_stride: int,
) -> dict[str, Any]:
    """Merge per-segment reports into the authoritative root manifest."""

    cache_root = Path(cache_root)
    segment_reports = []
    for path in sorted(cache_root.glob(f"*/{SEGMENT_MANIFEST_NAME}")):
        segment_reports.append(json.loads(path.read_text(encoding="utf-8")))
    expected_config = json.loads(json.dumps(asdict(cfg)))
    inconsistent = [
        report["source_tfrecord"]
        for report in segment_reports
        if int(report.get("cache_schema_version", 0)) != CACHE_SCHEMA_VERSION
        or int(report.get("sample_stride", -1)) != sample_stride
        or report.get("config") != expected_config
    ]
    if inconsistent:
        raise ValueError(f"inconsistent cache-v2 segment manifests: {inconsistent[:5]}")
    totals = {
        key: sum(int(report["stats"].get(key, 0)) for report in segment_reports)
        for key in (
            "samples",
            "expected_samples",
            "raw_frames",
            "points",
            "gt",
            "paper_vehicle_gt",
            "bytes",
        )
    }
    split_payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    expected_sources = split_payload[split]
    expected_names = {
        Path(path).stem.replace("_with_camera_labels", "") for path in expected_sources
    }
    reported_names = {
        Path(report["source_tfrecord"]).stem.replace("_with_camera_labels", "")
        for report in segment_reports
    }
    missing_segments = sorted(expected_names.difference(reported_names))
    unexpected_segments = sorted(reported_names.difference(expected_names))
    incomplete_segments = [
        report["source_tfrecord"]
        for report in segment_reports
        if int(report["stats"].get("samples", 0))
        != int(report["stats"].get("expected_samples", -1))
    ]
    if missing_segments or unexpected_segments or incomplete_segments:
        raise ValueError(
            "cache-v2 coverage validation failed: "
            f"missing={missing_segments[:5]} unexpected={unexpected_segments[:5]} "
            f"incomplete={incomplete_segments[:5]}"
        )
    point_min = min(
        (int(report["stats"]["point_min"]) for report in segment_reports if report["stats"]["samples"]),
        default=0,
    )
    point_max = max(
        (int(report["stats"]["point_max"]) for report in segment_reports if report["stats"]["samples"]),
        default=0,
    )
    sources = [
        {
            "path": report["source_tfrecord"],
            "sha256": report.get("source_sha256"),
        }
        for report in segment_reports
    ]
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "split_json": str(split_json),
        "config": asdict(cfg),
        "sample_stride": int(sample_stride),
        "num_segments": len(segment_reports),
        "expected_num_segments": len(expected_sources),
        "sources": sources,
        "stats": {
            **totals,
            "point_min": point_min,
            "point_max": point_max,
            "point_mean": totals["points"] / max(totals["samples"], 1),
        },
    }
    atomic_write_json(cache_root / MANIFEST_NAME, payload)
    return payload

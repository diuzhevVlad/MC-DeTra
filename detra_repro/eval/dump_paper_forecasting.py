"""Dump strict cache-v2 vehicle forecasts for detection-conditioned metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from detra_repro.data.cache_v2 import CACHE_SCHEMA_VERSION, MANIFEST_NAME
from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
from detra_repro.eval.dump_predictions import _load_experiment_cfg
from detra_repro.eval.paper_detection_ap import vehicle_confidence
from detra_repro.evaluate_detection_pr import (
    _build_configs,
    _inside_roi,
    _load_checkpoint,
    _load_models,
    _predict_batch,
)
from detra_repro.losses import current_boxes_from_pred


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_strict_cache_manifest(cache: Path) -> dict:
    manifest_path = cache / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"strict forecasting requires {manifest_path}")
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(cache_manifest.get("cache_schema_version", 0)) != CACHE_SCHEMA_VERSION:
        raise ValueError("strict forecasting requires cache schema v2")
    if tuple(cache_manifest["config"]["roi_m"]) != (-75.0, -75.0, 75.0, 75.0):
        raise ValueError("strict forecasting requires the paper 150x150 m ROI")
    if int(cache_manifest.get("sample_stride", -1)) != 1:
        raise ValueError("strict forecasting requires all eligible frames (sample_stride=1)")
    if any(not source.get("sha256") for source in cache_manifest.get("sources", [])):
        raise ValueError("strict forecasting requires SHA-256 fingerprints for all source TFRecords")
    return cache_manifest


@torch.no_grad()
def dump(args: argparse.Namespace) -> dict:
    cache = Path(args.cache)
    cache_manifest = _load_strict_cache_manifest(cache)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    ckpt = _load_checkpoint(checkpoint_path, device)
    exp_cfg = _load_experiment_cfg(args.experiment_json)
    dataset = CachedWaymoDataset(cache, require_schema_version=CACHE_SCHEMA_VERSION)
    expected_samples = int(cache_manifest["stats"]["samples"])
    if len(dataset) != expected_samples:
        raise ValueError(
            f"cache manifest lists {expected_samples} samples, filesystem has {len(dataset)}"
        )
    data_cfg, model_cfg = _build_configs(ckpt, exp_cfg, dataset)
    if model_cfg.num_queries != 600:
        raise ValueError(f"paper WOD forecasting requires 600 queries, got {model_cfg.num_queries}")
    norm_type = args.norm_type or str(exp_cfg.get("norm_type", "group"))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda samples: collate_cached_waymo(
            samples,
            max_points=None,
            max_objects=model_cfg.num_queries,
            max_map_tokens=args.max_map_tokens,
            deterministic_seed=args.deterministic_seed,
        ),
    )
    lidar_encoder, proposal_head, map_encoder, model = _load_models(
        ckpt, data_cfg, model_cfg, norm_type, device
    )

    pred_boxes_all: list[np.ndarray] = []
    pred_scores_all: list[np.ndarray] = []
    pred_future_all: list[np.ndarray] = []
    pred_modelogit_all: list[np.ndarray] = []
    pred_offsets = [0]

    gt_boxes_all: list[np.ndarray] = []
    gt_future_all: list[np.ndarray] = []
    gt_future_mask_all: list[np.ndarray] = []
    gt_offsets = [0]

    frame_ids: list[int] = []
    context_names: list[str] = []
    timestamps: list[int] = []
    future_steps = None
    num_modes = model_cfg.num_modes
    started_at = time.monotonic()

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        pred = _predict_batch(
            batch,
            lidar_encoder,
            proposal_head,
            map_encoder,
            model,
            data_cfg,
            model_cfg,
            device,
            proposal_nms_radius_m=0.1,
            proposal_nms_mode="rotated",
            proposal_center_mode="regression",
            proposal_peak_filter=True,
        )
        boxes = current_boxes_from_pred(pred)[0].detach().cpu().numpy()
        object_logits = pred["object_logits"][0].detach().cpu().numpy()
        class_logits = pred["class_logits"][0].detach().cpu().numpy()
        scores = vehicle_confidence(object_logits, class_logits)
        future_xy = pred["future_xy"][0].detach().cpu().numpy()
        mode_logits = pred["mode_logits"][0].detach().cpu().numpy()
        inside = _inside_roi(torch.from_numpy(boxes), (-75.0, -75.0, 75.0, 75.0)).numpy()
        boxes, scores = boxes[inside], scores[inside]
        future_xy, mode_logits = future_xy[inside], mode_logits[inside]

        eligible = batch["paper_vehicle_eligible"][0].numpy()
        gt_boxes = batch["gt_boxes"][0].numpy()[eligible]
        gt_future = batch["gt_future"][0].numpy()[eligible]
        gt_future_mask = batch["gt_future_mask"][0].numpy()[eligible].astype(bool)
        if future_steps is None:
            future_steps = int(batch["gt_future"].shape[2])

        frame_ids.append(batch_idx)
        context_names.append(batch["context_names"][0])
        timestamps.append(int(batch["frame_timestamp_micros"][0]))
        pred_boxes_all.append(boxes.astype(np.float32))
        pred_scores_all.append(scores.astype(np.float32))
        pred_future_all.append(future_xy.astype(np.float32))
        pred_modelogit_all.append(mode_logits.astype(np.float32))
        pred_offsets.append(pred_offsets[-1] + len(boxes))
        gt_boxes_all.append(gt_boxes.astype(np.float32))
        gt_future_all.append(gt_future.astype(np.float32))
        gt_future_mask_all.append(gt_future_mask)
        gt_offsets.append(gt_offsets[-1] + len(gt_boxes))

        completed = batch_idx + 1
        if args.progress_every and completed % args.progress_every == 0:
            elapsed = time.monotonic() - started_at
            rate = completed / max(elapsed, 1e-9)
            remaining = (expected_samples - completed) / max(rate, 1e-9)
            print(
                f"progress frames={completed}/{expected_samples} "
                f"rate={rate:.2f} frame/s dump_eta_s={remaining:.0f}",
                flush=True,
            )

    def cat(arrays: list[np.ndarray], shape_tail: tuple[int, ...]) -> np.ndarray:
        if arrays:
            return np.concatenate(arrays, axis=0)
        return np.zeros((0, *shape_tail), dtype=np.float32)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tf = int(future_steps or 0)
    np.savez_compressed(
        output / "predictions.npz",
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        context_names=np.asarray(context_names),
        frame_timestamp_micros=np.asarray(timestamps, dtype=np.int64),
        pred_frame_offsets=np.asarray(pred_offsets, dtype=np.int64),
        gt_frame_offsets=np.asarray(gt_offsets, dtype=np.int64),
        pred_boxes=cat(pred_boxes_all, (7,)),
        pred_scores=(
            np.concatenate(pred_scores_all) if pred_scores_all else np.zeros(0, np.float32)
        ),
        pred_classes=np.zeros(pred_offsets[-1], dtype=np.int64),
        pred_future_xy=cat(pred_future_all, (num_modes, tf, 2)),
        pred_mode_logits=cat(pred_modelogit_all, (num_modes,)),
        gt_boxes=cat(gt_boxes_all, (7,)),
        gt_classes=np.zeros(gt_offsets[-1], dtype=np.int64),
        gt_future_xy=cat(gt_future_all, (tf, 2)),
        gt_future_mask=(
            np.concatenate(gt_future_mask_all)
            if gt_future_mask_all
            else np.zeros((0, tf), dtype=bool)
        ),
    )
    meta = {
        "paper_strict_forecasting": True,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache": str(cache),
        "cache_manifest": cache_manifest,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "experiment_json": args.experiment_json,
        "num_frames": len(frame_ids),
        "num_modes": int(num_modes),
        "future_steps": tf,
        "num_pred_total": pred_offsets[-1],
        "num_gt_total": gt_offsets[-1],
        "complete_validation": args.max_batches is None and len(frame_ids) == expected_samples,
        "roi_m": [-75.0, -75.0, 75.0, 75.0],
        "max_points": None,
        "vehicle_score": "sigmoid(object_logits)*softmax(class_logits)[vehicle]",
        "proposal_nms": {"mode": "rotated", "iou_threshold": 0.1},
        "gt_population": "paper_vehicle_eligible",
        "class_names": ["vehicle"],
    }
    (output / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: meta[key]
                for key in (
                    "num_frames",
                    "num_modes",
                    "future_steps",
                    "num_pred_total",
                    "num_gt_total",
                    "complete_validation",
                )
            },
            indent=2,
        )
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-json", default=None)
    parser.add_argument("--max-map-tokens", type=int, default=4096)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--deterministic-seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--norm-type", default=None)
    dump(parser.parse_args())


if __name__ == "__main__":
    main()

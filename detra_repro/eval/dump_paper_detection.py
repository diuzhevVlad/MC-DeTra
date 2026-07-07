"""Dump strict cache-v2 vehicle detections for paper-facing BEV AP."""

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
from detra_repro.eval.paper_detection_ap import boxes_overlap_nlz, vehicle_confidence
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


@torch.no_grad()
def dump(args: argparse.Namespace) -> dict:
    cache = Path(args.cache)
    manifest_path = cache / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"strict evaluation requires {manifest_path}")
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(cache_manifest.get("cache_schema_version", 0)) != CACHE_SCHEMA_VERSION:
        raise ValueError("strict evaluation requires cache schema v2")
    if tuple(cache_manifest["config"]["roi_m"]) != (-75.0, -75.0, 75.0, 75.0):
        raise ValueError("strict evaluation requires the paper 150x150 m ROI")
    if int(cache_manifest.get("sample_stride", -1)) != 1:
        raise ValueError("strict evaluation requires all eligible frames (sample_stride=1)")
    if any(not source.get("sha256") for source in cache_manifest.get("sources", [])):
        raise ValueError("strict evaluation requires SHA-256 fingerprints for all source TFRecords")

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
        raise ValueError(f"paper WOD evaluation requires 600 queries, got {model_cfg.num_queries}")
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

    pred_boxes_all, pred_scores_all, pred_nlz_all = [], [], []
    gt_boxes_all, gt_difficulty_all = [], []
    pred_offsets, gt_offsets = [0], [0]
    frame_ids, context_names, timestamps = [], [], []
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
        inside = _inside_roi(torch.from_numpy(boxes), (-75.0, -75.0, 75.0, 75.0)).numpy()
        boxes, scores = boxes[inside], scores[inside]
        overlap_nlz = boxes_overlap_nlz(boxes, batch["nlz_xy"][0], batch["nlz_offsets"][0])

        eligible = batch["paper_vehicle_eligible"][0].numpy()
        gt_boxes = batch["gt_boxes"][0].numpy()[eligible]
        gt_difficulty = batch["gt_detection_difficulty"][0].numpy()[eligible]

        frame_ids.append(batch_idx)
        context_names.append(batch["context_names"][0])
        timestamps.append(int(batch["frame_timestamp_micros"][0]))
        pred_boxes_all.append(boxes.astype(np.float32))
        pred_scores_all.append(scores.astype(np.float32))
        pred_nlz_all.append(overlap_nlz)
        gt_boxes_all.append(gt_boxes.astype(np.float32))
        gt_difficulty_all.append(gt_difficulty.astype(np.uint8))
        pred_offsets.append(pred_offsets[-1] + len(boxes))
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

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cat_boxes = lambda arrays: np.concatenate(arrays) if arrays else np.zeros((0, 7), np.float32)
    np.savez_compressed(
        output / "predictions.npz",
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        context_names=np.asarray(context_names),
        frame_timestamp_micros=np.asarray(timestamps, dtype=np.int64),
        pred_frame_offsets=np.asarray(pred_offsets, dtype=np.int64),
        gt_frame_offsets=np.asarray(gt_offsets, dtype=np.int64),
        pred_boxes=cat_boxes(pred_boxes_all),
        pred_scores=np.concatenate(pred_scores_all) if pred_scores_all else np.zeros(0, np.float32),
        pred_classes=np.zeros(pred_offsets[-1], dtype=np.int64),
        pred_overlap_nlz=(
            np.concatenate(pred_nlz_all) if pred_nlz_all else np.zeros(0, dtype=bool)
        ),
        gt_boxes=cat_boxes(gt_boxes_all),
        gt_classes=np.zeros(gt_offsets[-1], dtype=np.int64),
        gt_difficulty=(
            np.concatenate(gt_difficulty_all)
            if gt_difficulty_all
            else np.zeros(0, dtype=np.uint8)
        ),
    )
    meta = {
        "paper_strict": True,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache": str(cache),
        "cache_manifest": cache_manifest,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "experiment_json": args.experiment_json,
        "num_frames": len(frame_ids),
        "num_pred_total": pred_offsets[-1],
        "num_gt_total": gt_offsets[-1],
        "complete_validation": args.max_batches is None and len(frame_ids) == expected_samples,
        "roi_m": [-75.0, -75.0, 75.0, 75.0],
        "max_points": None,
        "vehicle_score": "sigmoid(object_logits)*softmax(class_logits)[vehicle]",
        "proposal_nms": {"mode": "rotated", "iou_threshold": 0.1},
    }
    (output / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: meta[k] for k in ("num_frames", "num_pred_total", "num_gt_total")}, indent=2))
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

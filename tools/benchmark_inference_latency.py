#!/usr/bin/env python3
"""Benchmark cached Waymo inference latency for one checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
from detra_repro.evaluate_detection_pr import _build_configs, _load_checkpoint, _load_models, _predict_batch
from detra_repro.losses import current_boxes_from_pred


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p90_ms": _percentile(values, 0.90) * 1000.0,
        "min_ms": min(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def _selected_indices(dataset: CachedWaymoDataset, cache_root: Path, selected_json: Path | None, limit: int | None) -> list[int]:
    if selected_json is None:
        indices = list(range(len(dataset)))
        return indices[:limit] if limit else indices
    payload = json.loads(selected_json.read_text(encoding="utf-8"))
    index_by_rel = {
        path.relative_to(cache_root).as_posix(): idx
        for idx, path in enumerate(dataset.files)
    }
    indices = [index_by_rel[item["rel_path"]] for item in payload if item["rel_path"] in index_by_rel]
    return indices[:limit] if limit else indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selected-json", type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-queries", type=int, default=600)
    parser.add_argument("--max-points", type=int, default=750000)
    parser.add_argument("--norm-type", choices=["batch", "group"], default="group")
    parser.add_argument("--proposal-center-mode", choices=["cell", "anchor", "regression"], default="regression")
    parser.add_argument("--proposal-nms-mode", choices=["center", "rotated"], default="center")
    parser.add_argument("--proposal-nms-radius-m", type=float, default=1.0)
    parser.add_argument("--proposal-peak-filter", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.30)
    args = parser.parse_args()

    dataset = CachedWaymoDataset(args.cache)
    indices = _selected_indices(dataset, args.cache, args.selected_json, args.num_samples)
    if not indices:
        raise SystemExit("No samples selected")

    ckpt = _load_checkpoint(args.checkpoint, args.device)
    data_cfg, model_cfg = _build_configs(ckpt, {}, dataset)
    model_cfg = replace(model_cfg, num_queries=args.num_queries)
    lidar_encoder, proposal_head, map_encoder, model = _load_models(
        ckpt, data_cfg, model_cfg, args.norm_type, args.device
    )

    timings: dict[str, list[float]] = {"load": [], "collate": [], "infer": [], "post": [], "total": []}
    kept_counts: list[int] = []
    point_counts: list[int] = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def run_once(sample_idx: int, record: bool) -> None:
        t0 = time.perf_counter()
        sample = dataset[sample_idx]
        t1 = time.perf_counter()
        batch = collate_cached_waymo([sample], max_points=args.max_points)
        t2 = time.perf_counter()
        _sync(args.device)
        t3 = time.perf_counter()
        with torch.inference_mode():
            pred = _predict_batch(
                batch,
                lidar_encoder,
                proposal_head,
                map_encoder,
                model,
                data_cfg,
                model_cfg,
                args.device,
                args.proposal_nms_radius_m,
                args.proposal_nms_mode,
                args.proposal_center_mode,
                args.proposal_peak_filter,
            )
        _sync(args.device)
        t4 = time.perf_counter()
        scores = pred["object_logits"].sigmoid()
        boxes = current_boxes_from_pred(pred)
        keep = scores[0] >= args.score_threshold
        _ = boxes[0, keep]
        _sync(args.device)
        t5 = time.perf_counter()
        if record:
            timings["load"].append(t1 - t0)
            timings["collate"].append(t2 - t1)
            timings["infer"].append(t4 - t3)
            timings["post"].append(t5 - t4)
            timings["total"].append(t5 - t0)
            kept_counts.append(int(keep.sum().item()))
            point_counts.append(int(batch["points"].shape[1]))

    for i in range(args.warmup):
        run_once(indices[i % len(indices)], record=False)

    for _ in range(args.repeats):
        for idx in indices:
            run_once(idx, record=True)

    print(f"device={args.device}")
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name} total_vram_gb={props.total_memory / 1024**3:.2f}")
        print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"samples={len(indices)} repeats={args.repeats} measured_runs={len(timings['total'])}")
    print(f"max_points={args.max_points} num_queries={args.num_queries} nms={args.proposal_nms_mode}:{args.proposal_nms_radius_m}")
    print(f"points_mean={statistics.fmean(point_counts):.1f} points_max={max(point_counts)} kept_mean={statistics.fmean(kept_counts):.1f}")
    for name in ["load", "collate", "infer", "post", "total"]:
        stats = _summary(timings[name])
        print(
            f"{name}: mean={stats['mean_ms']:.2f}ms median={stats['median_ms']:.2f}ms "
            f"p90={stats['p90_ms']:.2f}ms min={stats['min_ms']:.2f}ms max={stats['max_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()

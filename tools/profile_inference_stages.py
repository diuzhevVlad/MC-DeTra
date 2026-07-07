#!/usr/bin/env python3
"""Profile stage-level latency for cached Waymo inference."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
from detra_repro.evaluate_detection_pr import _build_configs, _load_checkpoint, _load_models
from detra_repro.losses import current_boxes_from_pred


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _mark(device: str) -> float:
    _sync(device)
    return time.perf_counter()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def _summary(values: list[float]) -> str:
    return (
        f"mean={statistics.fmean(values) * 1000:.2f}ms "
        f"median={statistics.median(values) * 1000:.2f}ms "
        f"p90={_percentile(values, 0.90) * 1000:.2f}ms "
        f"min={min(values) * 1000:.2f}ms "
        f"max={max(values) * 1000:.2f}ms"
    )


def _selected_indices(dataset: CachedWaymoDataset, cache_root: Path, selected_json: Path | None, limit: int) -> list[int]:
    if selected_json is None:
        return list(range(min(limit, len(dataset))))
    payload = json.loads(selected_json.read_text(encoding="utf-8"))
    index_by_rel = {
        path.relative_to(cache_root).as_posix(): idx
        for idx, path in enumerate(dataset.files)
    }
    return [index_by_rel[item["rel_path"]] for item in payload[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selected-json", type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
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
    ckpt = _load_checkpoint(args.checkpoint, args.device)
    data_cfg, model_cfg = _build_configs(ckpt, {}, dataset)
    model_cfg = replace(model_cfg, num_queries=args.num_queries)
    lidar_encoder, proposal_head, map_encoder, model = _load_models(
        ckpt, data_cfg, model_cfg, args.norm_type, args.device
    )

    timings = {
        "load_npz": [],
        "collate_cpu": [],
        "to_device": [],
        "lidar_encoder": [],
        "proposal_head": [],
        "proposal_decode_nms": [],
        "map_to_device": [],
        "map_encoder": [],
        "refinement": [],
        "postprocess": [],
        "inference_total": [],
        "end_to_end": [],
    }
    kept_counts: list[int] = []
    map_token_counts: list[int] = []
    point_counts: list[int] = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def run_one(sample_idx: int, record: bool) -> None:
        t0 = time.perf_counter()
        sample = dataset[sample_idx]
        t1 = time.perf_counter()
        batch = collate_cached_waymo([sample], max_points=args.max_points)
        t2 = time.perf_counter()

        with torch.inference_mode():
            ti0 = _mark(args.device)
            points = batch["points"].to(args.device, non_blocking=True)
            point_mask = batch["point_mask"].to(args.device, non_blocking=True)
            ti1 = _mark(args.device)
            encoder_out = lidar_encoder(points, point_mask=point_mask)
            ti2 = _mark(args.device)
            proposal_outputs = proposal_head(encoder_out["feature_map"])
            ti3 = _mark(args.device)
            proposals = proposal_head.decode(
                proposal_outputs,
                data_cfg,
                num_proposals=model_cfg.num_queries,
                output_stride=model_cfg.proposal_output_stride,
                nms_radius_m=args.proposal_nms_radius_m,
                center_mode=args.proposal_center_mode,
                peak_filter=args.proposal_peak_filter,
                nms_mode=args.proposal_nms_mode,
            )
            ti4 = _mark(args.device)

            map_tokens = map_xy = map_mask = None
            tm0 = tm1 = tm2 = ti4
            if map_encoder is not None:
                tm0 = _mark(args.device)
                map_xy = batch["map_xy"].to(args.device, non_blocking=True)
                map_features = batch["map_features"].to(args.device, non_blocking=True)
                map_mask = batch["map_mask"].to(args.device, non_blocking=True)
                tm1 = _mark(args.device)
                if map_mask.any():
                    map_tokens = map_encoder(map_features, map_xy, map_mask)
                tm2 = _mark(args.device)

            pred = model(
                encoder_out["multi_scale_features"],
                proposals["poses"],
                initial_boxes=proposals["boxes"],
                map_tokens=map_tokens,
                map_xy=map_xy if map_tokens is not None else None,
                map_mask=map_mask if map_tokens is not None else None,
            )
            ti5 = _mark(args.device)
            scores = pred["object_logits"].sigmoid()
            boxes = current_boxes_from_pred(pred)
            keep = scores[0] >= args.score_threshold
            _ = boxes[0, keep]
            ti6 = _mark(args.device)

        if record:
            timings["load_npz"].append(t1 - t0)
            timings["collate_cpu"].append(t2 - t1)
            timings["to_device"].append(ti1 - ti0)
            timings["lidar_encoder"].append(ti2 - ti1)
            timings["proposal_head"].append(ti3 - ti2)
            timings["proposal_decode_nms"].append(ti4 - ti3)
            timings["map_to_device"].append(tm1 - tm0)
            timings["map_encoder"].append(tm2 - tm1)
            timings["refinement"].append(ti5 - tm2)
            timings["postprocess"].append(ti6 - ti5)
            timings["inference_total"].append(ti6 - ti0)
            timings["end_to_end"].append(ti6 - t0)
            kept_counts.append(int(keep.sum().item()))
            point_counts.append(int(batch["points"].shape[1]))
            map_token_counts.append(int(batch["map_mask"].sum().item()) if "map_mask" in batch else 0)

    for i in range(args.warmup):
        run_one(indices[i % len(indices)], record=False)
    for _ in range(args.repeats):
        for idx in indices:
            run_one(idx, record=True)

    print(f"device={args.device}")
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name} total_vram_gb={props.total_memory / 1024**3:.2f}")
        print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"samples={len(indices)} repeats={args.repeats} measured_runs={len(timings['end_to_end'])}")
    print(f"points_mean={statistics.fmean(point_counts):.1f} points_max={max(point_counts)}")
    print(f"map_tokens_mean={statistics.fmean(map_token_counts):.1f} kept_mean={statistics.fmean(kept_counts):.1f}")
    for name, values in timings.items():
        print(f"{name}: {_summary(values)}")


if __name__ == "__main__":
    main()

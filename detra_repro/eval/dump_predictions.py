"""Dump model predictions + ground truth for offline evaluation (trajpred env).

Runs the DeTra model once over a cached split and writes a single compressed
``predictions.npz`` plus ``meta.json``. The model forward path is reused from
``detra_repro.evaluate_detection_pr`` so this stays in lockstep with the PR
diagnostics. The dump is then consumed by:
- ``detra_repro.eval.eval_forecasting`` (forecasting metrics + operating point),
- ``detra_repro.eval.waymo_ap`` (official Waymo AP/APH, in the ``mtr`` env).

Storage format (ragged tensors stored flat + per-frame offsets to avoid
pickled object arrays):
- predictions are concatenated across frames along axis 0;
- ``pred_frame_offsets[i]:pred_frame_offsets[i+1]`` slices frame ``i``;
- ground truth uses the same offset scheme with ``gt_frame_offsets``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
from detra_repro.evaluate_detection_pr import (
    _build_configs,
    _inside_roi,
    _load_checkpoint,
    _load_models,
    _predict_batch,
)
from detra_repro.losses import current_boxes_from_pred, waymo_types_to_class_indices


def _resolve_proposal_settings(args: argparse.Namespace, exp_cfg: dict) -> dict:
    """Resolve proposal decode settings from CLI args then experiment json."""

    nms_radius = (
        args.proposal_nms_radius_m
        if args.proposal_nms_radius_m is not None
        else float(exp_cfg.get("proposal_nms_radius_m", 1.0))
    )
    return {
        "proposal_nms_radius_m": nms_radius,
        "proposal_nms_mode": args.proposal_nms_mode or str(exp_cfg.get("proposal_nms_mode", "center")),
        "proposal_center_mode": args.proposal_center_mode
        or str(exp_cfg.get("proposal_center_mode", "regression")),
        "proposal_peak_filter": bool(exp_cfg.get("proposal_peak_filter", True))
        and not args.no_proposal_peak_filter,
    }


def _load_experiment_cfg(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f).get("config", {})


@torch.no_grad()
def dump(args: argparse.Namespace) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    exp_cfg = _load_experiment_cfg(args.experiment_json)
    dataset = CachedWaymoDataset(args.cache)
    data_cfg, model_cfg = _build_configs(ckpt, exp_cfg, dataset)
    norm_type = args.norm_type or str(exp_cfg.get("norm_type", "group"))
    proposal_settings = _resolve_proposal_settings(args, exp_cfg)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda samples: collate_cached_waymo(
            samples,
            max_points=args.max_points,
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
    pred_classes_all: list[np.ndarray] = []
    pred_future_all: list[np.ndarray] = []
    pred_modelogit_all: list[np.ndarray] = []
    pred_past_all: list[np.ndarray] = []  # M1 past head, [H,2] per query (optional)
    has_past = False
    past_horizon = 0
    pred_offsets = [0]

    gt_boxes_all: list[np.ndarray] = []
    gt_classes_all: list[np.ndarray] = []
    gt_future_all: list[np.ndarray] = []
    gt_future_mask_all: list[np.ndarray] = []
    gt_offsets = [0]
    gt_history_all: list[np.ndarray] = []
    gt_history_mask_all: list[np.ndarray] = []
    history_steps = 0

    frame_ids: list[int] = []
    num_modes = model_cfg.num_modes
    future_steps = None

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
            **proposal_settings,
        )
        boxes = current_boxes_from_pred(pred).detach().cpu().numpy()
        scores = pred["object_logits"].sigmoid().detach().cpu().numpy()
        future_xy = pred["future_xy"].detach().cpu().numpy()  # [B,N,F,Tf,2]
        mode_logits = pred["mode_logits"].detach().cpu().numpy()  # [B,N,F]
        past_t = pred.get("past_xy")  # [B,N,H,2] (M1), newest->oldest; optional
        if past_t is not None:
            has_past = True
            past_xy = past_t.detach().cpu().numpy()
            past_horizon = past_xy.shape[2]
        else:
            past_xy = None
        class_logits = pred.get("class_logits")
        if class_logits is not None:
            pred_classes = class_logits.detach().cpu().softmax(dim=-1).argmax(dim=-1).numpy()
        else:
            pred_classes = np.zeros(scores.shape, dtype=np.int64)

        gt_boxes = batch["gt_boxes"].numpy()
        gt_valid = batch["gt_valid"].numpy().astype(bool)
        gt_classes = waymo_types_to_class_indices(batch["gt_types"]).numpy()
        gt_future = batch["gt_future"].numpy()
        gt_future_mask = batch["gt_future_mask"].numpy().astype(bool)
        gt_history = batch["gt_history"].numpy()  # [B,M,H,2]
        gt_history_mask = batch["gt_history_mask"].numpy().astype(bool)
        if future_steps is None:
            future_steps = gt_future.shape[2]
            history_steps = gt_history.shape[2]

        sample_indices = batch["sample_indices"].numpy()
        b = boxes.shape[0]
        for i in range(b):
            frame_id = int(sample_indices[i])
            frame_ids.append(frame_id)

            # Predictions: all queries (no score threshold here; thresholding is
            # an evaluation-time decision).
            p_boxes = boxes[i]
            p_scores = scores[i]
            p_classes = pred_classes[i]
            p_future = future_xy[i]
            p_modelogit = mode_logits[i]
            p_past = past_xy[i] if past_xy is not None else None
            if args.filter_roi:
                keep = _inside_roi(torch.from_numpy(p_boxes), data_cfg.roi_m).numpy()
                p_boxes = p_boxes[keep]
                p_scores = p_scores[keep]
                p_classes = p_classes[keep]
                p_future = p_future[keep]
                p_modelogit = p_modelogit[keep]
                if p_past is not None:
                    p_past = p_past[keep]
            if p_past is not None:
                pred_past_all.append(p_past.astype(np.float32))
            pred_boxes_all.append(p_boxes.astype(np.float32))
            pred_scores_all.append(p_scores.astype(np.float32))
            pred_classes_all.append(p_classes.astype(np.int64))
            pred_future_all.append(p_future.astype(np.float32))
            pred_modelogit_all.append(p_modelogit.astype(np.float32))
            pred_offsets.append(pred_offsets[-1] + len(p_boxes))

            # Ground truth: valid objects only.
            valid = gt_valid[i].copy()
            if args.filter_roi and valid.any():
                roi = _inside_roi(torch.from_numpy(gt_boxes[i]), data_cfg.roi_m).numpy()
                valid = valid & roi
            g_boxes = gt_boxes[i][valid]
            g_classes = gt_classes[i][valid]
            g_future = gt_future[i][valid]
            g_future_mask = gt_future_mask[i][valid]
            g_history = gt_history[i][valid]
            g_history_mask = gt_history_mask[i][valid]
            gt_boxes_all.append(g_boxes.astype(np.float32))
            gt_classes_all.append(g_classes.astype(np.int64))
            gt_future_all.append(g_future.astype(np.float32))
            gt_future_mask_all.append(g_future_mask.astype(bool))
            gt_history_all.append(g_history.astype(np.float32))
            gt_history_mask_all.append(g_history_mask.astype(bool))
            gt_offsets.append(gt_offsets[-1] + len(g_boxes))

    def cat(arrays: list[np.ndarray], shape_tail: tuple) -> np.ndarray:
        if arrays:
            return np.concatenate(arrays, axis=0)
        return np.zeros((0, *shape_tail), dtype=np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tf = future_steps or 0
    np.savez_compressed(
        output_dir / "predictions.npz",
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        pred_frame_offsets=np.asarray(pred_offsets, dtype=np.int64),
        gt_frame_offsets=np.asarray(gt_offsets, dtype=np.int64),
        pred_boxes=cat(pred_boxes_all, (7,)),
        pred_scores=np.concatenate(pred_scores_all) if pred_scores_all else np.zeros((0,), np.float32),
        pred_classes=np.concatenate(pred_classes_all) if pred_classes_all else np.zeros((0,), np.int64),
        pred_future_xy=cat(pred_future_all, (num_modes, tf, 2)),
        pred_mode_logits=cat(pred_modelogit_all, (num_modes,)),
        pred_past_xy=(
            cat(pred_past_all, (past_horizon, 2))
            if has_past
            else np.zeros((0, 0, 2), np.float32)
        ),
        gt_boxes=cat(gt_boxes_all, (7,)),
        gt_classes=np.concatenate(gt_classes_all) if gt_classes_all else np.zeros((0,), np.int64),
        gt_future_xy=cat(gt_future_all, (tf, 2)),
        gt_future_mask=(
            np.concatenate(gt_future_mask_all) if gt_future_mask_all else np.zeros((0, tf), bool)
        ),
        gt_history_xy=cat(gt_history_all, (history_steps, 2)),
        gt_history_mask=(
            np.concatenate(gt_history_mask_all)
            if gt_history_mask_all
            else np.zeros((0, history_steps), bool)
        ),
    )
    meta = {
        "cache": str(args.cache),
        "checkpoint": str(args.checkpoint),
        "experiment_json": args.experiment_json,
        "device": device,
        "num_frames": len(frame_ids),
        "num_modes": int(num_modes),
        "future_steps": int(tf),
        "has_past": bool(has_past),
        "past_horizon": int(past_horizon),
        "roi_m": list(data_cfg.roi_m),
        "voxel_size_m": float(data_cfg.voxel_size_m),
        "filter_roi": bool(args.filter_roi),
        "norm_type": norm_type,
        "proposal_settings": proposal_settings,
        "num_pred_total": int(pred_offsets[-1]),
        "num_gt_total": int(gt_offsets[-1]),
        "class_names": ["vehicle", "pedestrian", "cyclist"],
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump DeTra predictions + GT for offline eval.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-json", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=750000)
    parser.add_argument("--max-map-tokens", type=int, default=4096)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic-seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--norm-type", default=None)
    parser.add_argument("--no-filter-roi", dest="filter_roi", action="store_false")
    parser.set_defaults(filter_roi=True)
    parser.add_argument("--proposal-nms-radius-m", type=float, default=None)
    parser.add_argument("--proposal-nms-mode", choices=("rotated", "center", "none"), default=None)
    parser.add_argument("--proposal-center-mode", choices=("regression", "cell"), default=None)
    parser.add_argument("--no-proposal-peak-filter", action="store_true")
    args = parser.parse_args()
    dump(args)


if __name__ == "__main__":
    main()

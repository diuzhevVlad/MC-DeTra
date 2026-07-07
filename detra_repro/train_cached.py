import json
import math
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from detra_repro.checkpoints import (
    capture_rng_state,
    load_checkpoint,
    load_compatible,
    restore_rng_state,
    save_checkpoint_atomic,
)
from detra_repro.config import DataConfig, ModelConfig
from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
from detra_repro.experiment import ExperimentLogger, load_experiment_config, to_jsonable
from detra_repro.losses import (
    current_boxes_from_pred,
    detra_loss,
    proposal_loss,
    sigmoid_focal_loss,
    waymo_types_to_class_indices,
)
from detra_repro.models.bev import PaperLikeLidarEncoder
from detra_repro.models.proposals import PaperProposalHead, ProposalHead
from detra_repro.models.refinement import DeTraMini, MapTokenEncoder, build_map_encoder
from detra_repro.training.metrics import (
    _add_forecast_bin_ratios,
    _normalized_histogram,
    _recall_at_radii,
    _topk_recall_at_radii,
    compute_metrics,
)
from detra_repro.training.consistency import (
    rasterize_gt_motion_targets,
    trajectory_occupancy_consistency_loss,
)
from detra_repro.training.oracle import (
    dense_history_motion_loss,
    gt_history_step_velocity,
    gt_history_step_velocity_valid,
    gt_point_support,
    heatmap_topk_xy,
    make_oracle_proposals,
    motion_balanced_sample_weights,
    query_velocity_from_gt_match,
    rasterize_gt_history_motion_targets,
    rasterize_gt_history_occupancy,
)


def _load_compatible(module: torch.nn.Module, state_dict: dict) -> None:
    """Load only shape-matching parameters from ``state_dict`` into ``module``.

    Keys absent from the module or with a mismatched tensor shape (e.g. a box
    head that grew from 6 to 7/8 channels for z/height regression, or the
    motion-aux branch present only in some checkpoints) are skipped so older
    checkpoints resume cleanly; the new parameters keep their initialization.
    """

    skipped = load_compatible(module, state_dict)
    if skipped:
        print(f"_load_compatible: skipped {len(skipped)} mismatched/extra keys: {skipped[:6]}", flush=True)


def _ewta_top_k(step: int, anneal_steps: int, num_modes: int) -> int:
    """Return the evolving-WTA top-k width for a training step.

    With ``anneal_steps <= 0`` this is always 1 (plain winner-takes-all, the
    default that leaves behavior unchanged). Otherwise ``k`` decays linearly
    from ``num_modes`` at step 1 to 1 at ``anneal_steps``, then stays at 1.
    """

    if anneal_steps <= 0 or num_modes <= 1:
        return 1
    if step >= anneal_steps:
        return 1
    frac = (anneal_steps - step) / anneal_steps  # 1 -> 0 over the schedule
    return max(1, min(num_modes, int(round(1 + frac * (num_modes - 1)))))


def build_loader(
    cache_dir: str | Path,
    batch_size: int,
    max_points: int,
    max_objects: int,
    shuffle: bool,
    max_map_tokens: int = 2048,
    deterministic_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    sample_weights: torch.Tensor | None = None,
) -> tuple[CachedWaymoDataset, DataLoader]:
    """Build cached Waymo dataset and dataloader."""

    dataset = CachedWaymoDataset(cache_dir)
    sampler = None
    if sample_weights is not None:
        generator = None
        if deterministic_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(deterministic_seed)
        sampler = WeightedRandomSampler(
            sample_weights.double(),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": partial(
            collate_cached_waymo,
            max_points=max_points,
            max_objects=max_objects,
            max_map_tokens=max_map_tokens,
            deterministic_seed=deterministic_seed,
        ),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    return dataset, loader


# Curated per-step train metrics mirrored to Comet. The local JSONL keeps the
# full payload (all loss components, grad diagnostics); Comet gets only this
# headline set to keep the experiment view readable (<10 series).
COMET_TRAIN_METRIC_KEYS = (
    "loss",
    "final_loss_detection",
    "final_loss_forecast",
    "final_loss_mode",
    "final_loss_past",
    "final_loss_yaw_consistency",
    "loss_motion_aux",
    "loss_traj_occ",
    "grad_norm",
)


def forward_batch(
    batch: dict[str, torch.Tensor],
    lidar_encoder: PaperLikeLidarEncoder,
    proposal_head: ProposalHead,
    map_encoder: MapTokenEncoder | None,
    model: DeTraMini,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    device: str,
    objective: str = "joint",
    proposal_mode: str = "decoded",
    proposal_noise_std_m: float = 0.0,
    forecast_weight: float = 0.1,
    proposal_heatmap_weight: float = 1.0,
    proposal_box_weight: float = 1.0,
    proposal_giou_weight: float = 0.1,
    detection_box_l1_weight: float = 0.01,
    detection_giou_weight: float = 0.1,
    detection_class_weight: float = 1.0,
    detection_z_weight: float = 0.0,
    detection_height_weight: float = 0.0,
    forecast_gate_radius_m: float | None = None,
    forecast_loss_type: str = "laplace",
    forecast_top_k: int = 1,
    velocity_weight: float = 0.0,
    yaw_consistency_weight: float = 0.0,
    yaw_consistency_vthresh_m: float = 1.0,
    past_weight: float = 0.0,
    init_weight: float = 1.0,
    proposal_nms_radius_m: float = 0.1,
    proposal_nms_mode: str = "rotated",
    proposal_center_mode: str = "regression",
    proposal_peak_filter: bool = True,
    support_filter_radius_m: float | None = None,
    query_velocity_source: str = "none",
    query_velocity_match_radius_m: float = 2.0,
    forecast_base_mode: str = "current",
    gt_history_occupancy: bool = False,
    gt_history_occupancy_radius_m: float = 0.5,
    dense_history_motion_weight: float = 0.0,
    dense_history_motion_flow_weight: float = 1.0,
    occ_aux_future: bool = False,
    traj_occ_consistency_weight: float = 0.0,
    traj_occ_mode_weighting: str = "prob",
    compute_diagnostics: bool = True,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run encoder, proposals, refinement, and loss for one batch.

    When ``compute_diagnostics`` is False, the metrics-only proposal decodes
    (``raw_decoded_proposals`` with no NMS, and the heatmap top-k) are skipped to
    save per-step CPU work; the corresponding ``targets`` keys are omitted and
    ``compute_metrics`` simply does not report those recall breakdowns. The
    decoded proposals that actually feed the model are always computed.
    """

    non_blocking = device.startswith("cuda")
    points = batch["points"].to(device, non_blocking=non_blocking)
    point_mask = batch["point_mask"].to(device, non_blocking=non_blocking)
    gt_boxes = batch["gt_boxes"].to(device, non_blocking=non_blocking)
    gt_valid = batch["gt_valid"].to(device, non_blocking=non_blocking)
    gt_types = batch["gt_types"].to(device, non_blocking=non_blocking)
    gt_future = batch["gt_future"].to(device, non_blocking=non_blocking)
    gt_future_mask = batch["gt_future_mask"].to(device, non_blocking=non_blocking)
    gt_history = batch["gt_history"].to(device, non_blocking=non_blocking)
    gt_history_mask = batch["gt_history_mask"].to(device, non_blocking=non_blocking)
    gt_velocity = gt_history_step_velocity(gt_history, gt_history_mask, gt_valid)
    gt_velocity_valid = gt_history_step_velocity_valid(gt_history_mask, gt_valid)
    map_xy = batch["map_xy"].to(device, non_blocking=non_blocking)
    map_features = batch["map_features"].to(device, non_blocking=non_blocking)
    map_mask = batch["map_mask"].to(device, non_blocking=non_blocking)
    support_gt_valid_2m = gt_point_support(points, point_mask, gt_boxes, gt_valid, radius_m=2.0)
    train_gt_valid = gt_valid
    if support_filter_radius_m is not None:
        train_gt_valid = gt_valid & gt_point_support(
            points,
            point_mask,
            gt_boxes,
            gt_valid,
            radius_m=support_filter_radius_m,
        )

    history_occupancy = None
    if gt_history_occupancy:
        history_occupancy = rasterize_gt_history_occupancy(
            gt_history,
            gt_history_mask,
            gt_valid,
            data_cfg,
            radius_m=gt_history_occupancy_radius_m,
        )
    history_motion_targets = None
    if dense_history_motion_weight > 0:
        if occ_aux_future:
            history_motion_targets = rasterize_gt_motion_targets(
                gt_history,
                gt_history_mask,
                gt_future,
                gt_future_mask,
                gt_valid,
                data_cfg,
                output_stride=model_cfg.proposal_output_stride,
                radius_m=gt_history_occupancy_radius_m,
            )
        else:
            history_motion_targets = rasterize_gt_history_motion_targets(
                gt_history,
                gt_history_mask,
                gt_valid,
                data_cfg,
                output_stride=model_cfg.proposal_output_stride,
                radius_m=gt_history_occupancy_radius_m,
            )

    encoder_out = lidar_encoder(points, point_mask=point_mask, history_occupancy=history_occupancy)
    motion_aux_losses = dense_history_motion_loss(
        encoder_out.get("motion_aux") if isinstance(encoder_out, dict) else None,
        history_motion_targets,
        flow_weight=dense_history_motion_flow_weight,
    )
    proposal_outputs = proposal_head(encoder_out["feature_map"])
    decoded_proposals = proposal_head.decode(
        proposal_outputs,
        data_cfg,
        num_proposals=model_cfg.num_queries,
        output_stride=model_cfg.proposal_output_stride,
        nms_radius_m=proposal_nms_radius_m,
        center_mode=proposal_center_mode,
        peak_filter=proposal_peak_filter,
        nms_mode=proposal_nms_mode,
    )
    # Metrics-only decodes: skipped on non-diagnostic steps to cut CPU work.
    raw_decoded_proposals = None
    heatmap_xy = None
    if compute_diagnostics:
        raw_decoded_proposals = proposal_head.decode(
            proposal_outputs,
            data_cfg,
            num_proposals=model_cfg.num_queries,
            output_stride=model_cfg.proposal_output_stride,
            nms_radius_m=0.0,
            center_mode=proposal_center_mode,
            peak_filter=proposal_peak_filter,
            nms_mode="none",
        )
        heatmap_xy = heatmap_topk_xy(
            proposal_outputs,
            data_cfg,
            num_proposals=model_cfg.num_queries,
            output_stride=model_cfg.proposal_output_stride,
        )

    if objective == "proposal_only":
        losses = proposal_loss(
            proposal_outputs,
            gt_boxes,
            train_gt_valid,
            data_cfg,
            output_stride=model_cfg.proposal_output_stride,
            heatmap_weight=proposal_heatmap_weight,
            box_weight=proposal_box_weight,
            giou_weight=proposal_giou_weight,
        )
        losses.update(motion_aux_losses)
        losses["loss"] = losses["loss_init"] + dense_history_motion_weight * losses["loss_motion_aux"]
        targets = {
            "gt_boxes": gt_boxes,
            "gt_valid": gt_valid,
            "gt_types": gt_types,
            "train_gt_valid": train_gt_valid,
            "support_gt_valid_2m": support_gt_valid_2m,
            "gt_future": gt_future,
            "gt_future_mask": gt_future_mask,
            "proposal_boxes": decoded_proposals["boxes"],
            "proposal_scores": decoded_proposals["scores"],
            "map_mask": map_mask,
        }
        if raw_decoded_proposals is not None:
            targets["raw_proposal_boxes"] = raw_decoded_proposals["boxes"]
        if heatmap_xy is not None:
            targets["heatmap_xy"] = heatmap_xy
        if history_occupancy is not None:
            targets["history_occupancy_count"] = history_occupancy.bool().float().sum()
        if history_motion_targets is not None:
            targets["dense_history_motion_cells"] = history_motion_targets["flow_mask"].float().sum()
        return None, losses, targets

    if proposal_mode == "decoded":
        proposals = decoded_proposals
    elif proposal_mode == "oracle":
        proposals = make_oracle_proposals(gt_boxes, train_gt_valid, model_cfg.num_queries)
    elif proposal_mode == "noisy_gt":
        proposals = make_oracle_proposals(
            gt_boxes,
            train_gt_valid,
            model_cfg.num_queries,
            noise_std_m=proposal_noise_std_m,
        )
    else:
        raise ValueError("proposal_mode must be one of decoded, oracle, noisy_gt")

    if query_velocity_source == "none":
        query_velocity = None
    elif query_velocity_source == "gt_match":
        query_velocity = query_velocity_from_gt_match(
            proposals["boxes"],
            gt_boxes,
            train_gt_valid,
            gt_history,
            gt_history_mask,
            max_distance_m=query_velocity_match_radius_m,
        )
    else:
        raise ValueError("query_velocity_source must be one of none, gt_match")

    pred = model(
        encoder_out["multi_scale_features"],
        proposals["poses"],
        initial_boxes=proposals["boxes"],
        map_tokens=map_encoder(map_features, map_xy, map_mask) if map_encoder is not None and map_mask.any() else None,
        map_xy=map_xy if map_encoder is not None and map_mask.any() else None,
        map_mask=map_mask if map_encoder is not None and map_mask.any() else None,
        query_velocity=query_velocity,
    )
    try:
        losses = detra_loss(
            pred,
            gt_boxes,
            train_gt_valid,
            gt_types,
            gt_future,
            gt_future_mask,
            proposal_outputs=proposal_outputs,
            data_cfg=data_cfg,
            proposal_output_stride=model_cfg.proposal_output_stride,
            proposal_heatmap_weight=proposal_heatmap_weight,
            proposal_box_weight=proposal_box_weight,
            proposal_giou_weight=proposal_giou_weight,
            detection_box_l1_weight=detection_box_l1_weight,
            detection_giou_weight=detection_giou_weight,
            detection_class_weight=detection_class_weight,
            detection_z_weight=detection_z_weight,
            detection_height_weight=detection_height_weight,
            forecast_weight=forecast_weight,
            forecast_gate_radius_m=forecast_gate_radius_m,
            forecast_loss_type=forecast_loss_type,
            forecast_top_k=forecast_top_k,
            velocity_weight=velocity_weight,
            gt_velocity=gt_velocity,
            gt_velocity_valid=gt_velocity_valid,
            yaw_consistency_weight=yaw_consistency_weight,
            yaw_consistency_vthresh_m=yaw_consistency_vthresh_m,
            yaw_use_velocity_dir=forecast_base_mode == "learned_velocity",
            past_weight=past_weight,
            gt_history=gt_history,
            gt_history_mask=gt_history_mask,
            init_weight=init_weight,
        )
        losses.update(motion_aux_losses)
        losses["loss"] = losses["loss"] + dense_history_motion_weight * losses["loss_motion_aux"]
        if traj_occ_consistency_weight > 0:
            consistency = trajectory_occupancy_consistency_loss(
                pred,
                encoder_out.get("motion_aux") if isinstance(encoder_out, dict) else None,
                losses["aligned"]["query_object_targets"],
                data_cfg,
                output_stride=model_cfg.proposal_output_stride,
                mode_weighting=traj_occ_mode_weighting,
            )
            losses["loss_traj_occ"] = consistency["loss_traj_occ"]
            losses["traj_occ_cells"] = consistency["traj_occ_cells"]
            losses["loss"] = losses["loss"] + traj_occ_consistency_weight * consistency["loss_traj_occ"]
    except ValueError as exc:
        raise ValueError(_batch_debug_report(batch, points, point_mask, gt_boxes, train_gt_valid, pred, exc)) from exc
    targets = {
        "gt_boxes": gt_boxes,
        "gt_valid": gt_valid,
        "gt_types": gt_types,
        "train_gt_valid": train_gt_valid,
        "support_gt_valid_2m": support_gt_valid_2m,
        "gt_future": gt_future,
        "gt_future_mask": gt_future_mask,
        "proposal_boxes": decoded_proposals["boxes"],
        "proposal_scores": decoded_proposals["scores"],
        "map_mask": map_mask,
    }
    if raw_decoded_proposals is not None:
        targets["raw_proposal_boxes"] = raw_decoded_proposals["boxes"]
    if heatmap_xy is not None:
        targets["heatmap_xy"] = heatmap_xy
    if history_occupancy is not None:
        targets["history_occupancy_count"] = history_occupancy.bool().float().sum()
    if history_motion_targets is not None:
        targets["dense_history_motion_cells"] = history_motion_targets["flow_mask"].float().sum()
    return pred, losses, targets


def _finite_stats(name: str, tensor: torch.Tensor) -> str:
    """Return compact finite-value diagnostics for logs/errors."""

    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    total = tensor.numel()
    if finite_count == 0:
        return f"{name}: shape={tuple(tensor.shape)} finite=0/{total}"
    vals = tensor[finite]
    return (
        f"{name}: shape={tuple(tensor.shape)} finite={finite_count}/{total} "
        f"min={float(vals.min().item()):.6g} max={float(vals.max().item()):.6g} "
        f"mean={float(vals.mean().item()):.6g}"
    )


def _batch_debug_report(
    batch: dict[str, torch.Tensor],
    points: torch.Tensor,
    point_mask: torch.Tensor,
    gt_boxes: torch.Tensor,
    train_gt_valid: torch.Tensor,
    pred: dict[str, torch.Tensor],
    exc: Exception,
) -> str:
    """Attach sample identity and tensor stats to a lower-level training error."""

    sample_paths = batch.get("sample_paths", [])
    sample_indices = batch.get("sample_indices")
    if isinstance(sample_indices, torch.Tensor):
        sample_indices_text = [int(x) for x in sample_indices.detach().cpu().tolist()]
    else:
        sample_indices_text = sample_indices
    pred_boxes = current_boxes_from_pred(pred)
    lines = [
        "forward_batch failed during loss/matching",
        f"sample_indices={sample_indices_text}",
        f"sample_paths={sample_paths}",
        f"gt_counts={train_gt_valid.sum(dim=1).detach().cpu().tolist()}",
        f"point_counts={point_mask.sum(dim=1).detach().cpu().tolist()}",
        _finite_stats("points", points),
        _finite_stats("gt_boxes", gt_boxes),
        _finite_stats("pred_boxes", pred_boxes),
        _finite_stats("object_logits", pred["object_logits"]),
        _finite_stats("class_logits", pred.get("class_logits", pred["object_logits"][..., None])),
        _finite_stats("mode_logits", pred["mode_logits"]),
        _finite_stats("poses", pred["poses"]),
        f"cause={exc}",
    ]
    future_xy = pred.get("future_xy")
    if future_xy is not None:
        lines.insert(-1, _finite_stats("future_xy", future_xy))
    future_log_scale = pred.get("future_log_scale")
    if future_log_scale is not None:
        lines.insert(-1, _finite_stats("future_log_scale", future_log_scale))
    return "\n".join(lines)


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    lidar_encoder: PaperLikeLidarEncoder,
    proposal_head: ProposalHead,
    map_encoder: MapTokenEncoder | None,
    model: DeTraMini,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    device: str,
    max_batches: int | None = None,
    objective: str = "joint",
    proposal_mode: str = "decoded",
    proposal_noise_std_m: float = 0.0,
    forecast_weight: float = 0.1,
    proposal_heatmap_weight: float = 1.0,
    proposal_box_weight: float = 1.0,
    proposal_giou_weight: float = 0.1,
    detection_box_l1_weight: float = 0.01,
    detection_giou_weight: float = 0.1,
    detection_class_weight: float = 1.0,
    detection_z_weight: float = 0.0,
    detection_height_weight: float = 0.0,
    forecast_gate_radius_m: float | None = None,
    forecast_loss_type: str = "laplace",
    forecast_top_k: int = 1,
    velocity_weight: float = 0.0,
    init_weight: float = 1.0,
    train_mode_eval: bool = False,
    proposal_nms_radius_m: float = 0.1,
    proposal_nms_mode: str = "rotated",
    proposal_center_mode: str = "regression",
    proposal_peak_filter: bool = True,
    support_filter_radius_m: float | None = None,
    query_velocity_source: str = "none",
    query_velocity_match_radius_m: float = 2.0,
    gt_history_occupancy: bool = False,
    gt_history_occupancy_radius_m: float = 0.5,
    dense_history_motion_weight: float = 0.0,
    dense_history_motion_flow_weight: float = 1.0,
    occ_aux_future: bool = False,
    traj_occ_consistency_weight: float = 0.0,
    traj_occ_mode_weighting: str = "prob",
) -> dict[str, float]:
    """Evaluate validation loss and simple ADE/FDE/recall metrics."""

    if train_mode_eval:
        lidar_encoder.train()
        proposal_head.train()
        if map_encoder is not None:
            map_encoder.train()
        model.train()
    else:
        lidar_encoder.eval()
        proposal_head.eval()
        if map_encoder is not None:
            map_encoder.eval()
        model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        pred, losses, targets = forward_batch(
            batch,
            lidar_encoder,
            proposal_head,
            map_encoder,
            model,
            data_cfg,
            model_cfg,
            device,
            objective=objective,
            proposal_mode=proposal_mode,
            proposal_noise_std_m=proposal_noise_std_m,
            forecast_weight=forecast_weight,
            proposal_heatmap_weight=proposal_heatmap_weight,
            proposal_box_weight=proposal_box_weight,
            proposal_giou_weight=proposal_giou_weight,
            detection_z_weight=detection_z_weight,
            detection_height_weight=detection_height_weight,
            forecast_gate_radius_m=forecast_gate_radius_m,
            forecast_loss_type=forecast_loss_type,
            forecast_top_k=forecast_top_k,
            velocity_weight=velocity_weight,
            init_weight=init_weight,
            proposal_nms_radius_m=proposal_nms_radius_m,
            proposal_nms_mode=proposal_nms_mode,
            proposal_center_mode=proposal_center_mode,
            proposal_peak_filter=proposal_peak_filter,
            support_filter_radius_m=support_filter_radius_m,
            query_velocity_source=query_velocity_source,
            query_velocity_match_radius_m=query_velocity_match_radius_m,
            gt_history_occupancy=gt_history_occupancy,
            gt_history_occupancy_radius_m=gt_history_occupancy_radius_m,
            dense_history_motion_weight=dense_history_motion_weight,
            dense_history_motion_flow_weight=dense_history_motion_flow_weight,
            occ_aux_future=occ_aux_future,
            traj_occ_consistency_weight=traj_occ_consistency_weight,
            traj_occ_mode_weighting=traj_occ_mode_weighting,
        )
        metrics = compute_metrics(pred, losses, targets)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
    averaged = {key: value / max(count, 1) for key, value in sums.items()}
    return _add_forecast_bin_ratios(averaged)


def _format_metrics(metrics: dict[str, float], keys: tuple[str, ...]) -> str:
    """Format selected metric keys when present."""

    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


def _scalar_loss_metrics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return scalar loss values with one device-to-host synchronization.

    The full ``compute_metrics`` path intentionally computes rich proposal,
    detection, trajectory, and mode diagnostics. That is useful but expensive
    when done every step, because it performs extra tensor work and forces
    synchronization. This helper is the lightweight per-step path.
    """

    keys: list[str] = []
    values: list[torch.Tensor] = []
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            keys.append(key)
            values.append(value.detach().float())
    if not values:
        return {}
    target_device = next((value.device for value in values if value.is_cuda), values[0].device)
    stacked = torch.stack([value.to(target_device) for value in values]).cpu().tolist()
    return {key: float(value) for key, value in zip(keys, stacked)}


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""

    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _sync_for_timing(enabled: bool, device: str) -> None:
    """Synchronize CUDA work when accurate phase timing is requested."""

    if enabled and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_timing_window(sums: dict[str, float], count: int) -> str:
    """Format average phase timings accumulated over a profile window."""

    if count <= 0:
        return ""
    keys = (
        "data",
        "forward",
        "backward",
        "optimizer",
        "finite_check",
        "metrics",
        "total",
    )
    parts = [f"window_steps={count}"]
    for key in keys:
        if key in sums:
            parts.append(f"{key}_s={sums[key] / count:.4f}")
    return " ".join(parts)


def _write_experiment_log(
    path: str | Path,
    config: dict[str, object],
    final_train_metrics: dict[str, float] | None,
    final_val_metrics: dict[str, float] | None,
    best_train_metrics: dict[str, float] | None,
    best_val_metrics: dict[str, float] | None,
    val_metrics_by_step: dict[int, dict[str, float]] | None = None,
) -> None:
    """Write one structured experiment record as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "config": config,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        # Per-eval history (keyed by step) so training curves are reconstructable
        # from the log alone, not only from parsed console output.
        "val_metrics_by_step": (
            {str(k): v for k, v in sorted(val_metrics_by_step.items())}
            if val_metrics_by_step
            else {}
        ),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)


def _parameters_are_finite(params: list[torch.nn.Parameter]) -> bool:
    """Return true if every trainable parameter is finite."""

    return all(torch.isfinite(param).all().item() for param in params if param.requires_grad)


def _module_grad_norm(module: torch.nn.Module) -> float:
    """Total L2 norm of the current ``.grad`` over a module's parameters.

    Call after ``backward()`` and before ``optimizer.zero_grad()``. Used to
    diagnose which subsystem (BEV encoder, proposal head, refinement, map
    encoder) is driving or starving the gradient, since loss magnitude alone does
    not reveal gradient contribution under the shared-query detection/forecast
    tension.
    """

    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().norm().item()) ** 2
    return total ** 0.5


def _term_grad_norms(
    weighted_terms: dict[str, tuple[torch.Tensor | None, float]],
    shared_params: list[torch.nn.Parameter],
) -> dict[str, float]:
    """Per-term gradient-norm contribution at a shared parameter set.

    For loss calibration: returns ``weight_i * ||grad(term_i)||`` at the shared
    BEV trunk that every loss flows through (linearity makes this the gradient
    norm of the weighted contribution ``weight_i * term_i``). Comparing these
    reveals which term actually dominates the shared features, which the loss
    magnitude alone hides. Must be called before the real ``backward`` (uses
    ``retain_graph``); expensive (one backward per term), so gate on cadence.
    """

    out: dict[str, float] = {}
    for name, (term, weight) in weighted_terms.items():
        if term is None or weight == 0.0 or not (
            isinstance(term, torch.Tensor) and term.requires_grad
        ):
            out[name] = 0.0
            continue
        if not torch.isfinite(term).all():
            out[name] = float("nan")
            continue
        grads = torch.autograd.grad(
            term, shared_params, retain_graph=True, allow_unused=True
        )
        sq = 0.0
        for grad in grads:
            if grad is not None:
                sq += float(grad.detach().norm().item()) ** 2
        out[name] = weight * (sq ** 0.5)
    return out


def _is_better_metric(
    metrics: dict[str, float],
    metric_name: str,
    best_value: float | None,
    mode: str,
) -> tuple[bool, float | None]:
    """Return whether ``metrics[metric_name]`` improves the previous best."""

    if metric_name not in metrics:
        return False, best_value
    value = float(metrics[metric_name])
    if not torch.isfinite(torch.tensor(value)).item():
        return False, best_value
    if best_value is None:
        return True, value
    if mode == "min":
        return value < best_value, value if value < best_value else best_value
    if mode == "max":
        return value > best_value, value if value > best_value else best_value
    raise ValueError("mode must be 'min' or 'max'")


def _lr_for_step(
    step: int,
    *,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    decay: str,
    min_scale: float,
) -> float:
    """Return the LR for a global optimizer-attempt step.

    Keeping this schedule as a pure function makes a resumed run use exactly
    the same LR as an uninterrupted run when ``total_steps`` is unchanged.
    """

    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * min(1.0, step / float(warmup_steps))
    if decay == "cosine":
        decay_total = max(1, total_steps - warmup_steps)
        decay_progress = min(1.0, (step - warmup_steps) / float(decay_total))
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return base_lr * (min_scale + (1.0 - min_scale) * cosine)
    return base_lr


def _capture_rng_state() -> dict[str, object]:
    """Capture process RNG state needed for the closest practical resume."""

    return capture_rng_state()


def _restore_rng_state(state: dict[str, object] | None) -> None:
    """Restore RNG state from a full-state checkpoint when available."""

    restore_rng_state(state)


def _save_checkpoint(
    checkpoint_path: str | Path,
    *,
    step: int,
    model: DeTraMini,
    lidar_encoder: PaperLikeLidarEncoder,
    proposal_head: ProposalHead,
    map_encoder: MapTokenEncoder,
    metrics_key: str,
    metrics: dict[str, float],
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    save_best_by: str,
    save_best_metric: str,
    save_best_mode: str,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler_state: dict[str, object] | None = None,
    training_state: dict[str, object] | None = None,
) -> None:
    """Atomically save model and, when supplied, full training state."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 2,
        "step": step,
        "model": model.state_dict(),
        "lidar_encoder": lidar_encoder.state_dict(),
        "proposal_head": proposal_head.state_dict(),
        "map_encoder": map_encoder.state_dict(),
        metrics_key: metrics,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
        "save_best_by": save_best_by,
        "save_best_metric": save_best_metric,
        "save_best_mode": save_best_mode,
        "rng_state": _capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if lr_scheduler_state is not None:
        payload["lr_scheduler"] = lr_scheduler_state
    if training_state is not None:
        payload["training_state"] = training_state

    save_checkpoint_atomic(checkpoint_path, payload)


def train_cached(
    train_cache: str | Path,
    val_cache: str | Path | None = None,
    steps: int = 10,
    batch_size: int = 1,
    max_points: int = 32768,
    max_map_tokens: int = 2048,
    lr: float = 1e-4,
    roi: tuple[float, float, float, float] = (-80.0, -80.0, 80.0, 80.0),
    voxel_size: float = 0.1,
    eval_every: int = 10,
    val_batches: int | None = None,
    initial_eval: bool = True,
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_weights_only: bool = False,
    save_best_by: str = "val",
    save_best_metric: str = "loss",
    save_best_mode: str = "min",
    save_every_eval: bool = False,
    save_every_steps: int = 0,
    keep_last_n: int = 3,
    save_best_forecast_metric: str | None = None,
    device: str | None = None,
    objective: str = "joint",
    proposal_mode: str = "decoded",
    proposal_noise_std_m: float = 0.0,
    forecast_weight: float = 0.1,
    lr_warmup_steps: int = 0,
    forecast_weight_warmup_steps: int = 0,
    grad_clip_max_norm: float = 5.0,
    lr_decay: str = "none",
    lr_min_scale: float = 0.0,
    lr_decay_total_steps: int = 0,
    grad_accum_steps: int = 1,
    proposal_heatmap_weight: float = 1.0,
    proposal_box_weight: float = 1.0,
    proposal_giou_weight: float = 0.1,
    detection_box_l1_weight: float = 0.01,
    detection_giou_weight: float = 0.1,
    detection_class_weight: float = 1.0,
    detection_z_weight: float = 0.0,
    detection_height_weight: float = 0.0,
    forecast_gate_radius_m: float | None = None,
    forecast_loss_type: str = "laplace",
    forecast_top_k: int = 1,
    ewta_anneal_steps: int = 0,
    forecast_gate_warmup_steps: int = 0,
    forecast_gate_warmup_radius_m: float = 2.0,
    velocity_weight: float = 0.0,
    yaw_consistency_weight: float = 0.0,
    yaw_consistency_vthresh_m: float = 1.0,
    past_weight: float = 0.0,
    init_weight: float = 1.0,
    proposal_nms_radius_m: float = 0.1,
    proposal_nms_mode: str = "rotated",
    proposal_center_mode: str = "regression",
    proposal_peak_filter: bool = True,
    support_filter_radius_m: float | None = None,
    deterministic_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    eval_train_mode: bool = False,
    norm_type: str = "batch",
    proposal_head_type: str = "shallow",
    num_queries: int = 128,
    num_refinement_blocks: int = 3,
    deformable_radius_m: float = 2.0,
    attention_type: str = "deformable",
    lidar_attention_anchor: str = "current",
    detach_pose_refinement: bool = True,
    motion_balanced_sampling: bool = False,
    motion_static_weight: float = 1.0,
    motion_moving_weight: float = 3.0,
    motion_fast_weight: float = 6.0,
    query_velocity_source: str = "none",
    query_velocity_match_radius_m: float = 2.0,
    forecast_base_mode: str = "current",
    forecast_velocity_frame: str = "ego",
    predict_past: bool = False,
    map_encoder_type: str = "graph",
    map_graph_k: int = 8,
    map_graph_layers: int = 2,
    gt_history_occupancy: bool = False,
    gt_history_occupancy_radius_m: float = 0.5,
    dense_history_motion_weight: float = 0.0,
    dense_history_motion_flow_weight: float = 1.0,
    occ_aux_future: bool = False,
    traj_occ_consistency_weight: float = 0.0,
    traj_occ_mode_weighting: str = "prob",
    grad_calibration: bool = False,
    aux_weight_controller: bool = False,
    aux_controller_interval: int = 250,
    aux_controller_warmup: int = 500,
    aux_controller_alpha: float = 0.5,
    aux_controller_beta: float = 0.5,
    aux_weight_min: float = 0.005,
    aux_weight_max: float = 1.5,
    past_target_frac: float = 0.0,
    motion_target_frac: float = 0.0,
    yaw_target_frac: float = 0.0,
    run_dir: str | Path | None = None,
    comet: bool = False,
    comet_project: str | None = None,
    comet_workspace: str | None = None,
    comet_run_name: str | None = None,
    comet_tags: list[str] | None = None,
    log_every: int = 1,
    metrics_every: int = 1,
    finite_check_every: int = 1,
    profile_timing: bool = False,
    profile_timing_sync: bool = False,
    experiment_log: str | Path | None = None,
) -> None:
    """Run training over cached Waymo samples with optional validation."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        # Autotune conv algorithms for the fixed BEV grid size. This only selects
        # the fastest kernel; it does not change numerics or training stability.
        torch.backends.cudnn.benchmark = True
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    if metrics_every < 1:
        raise ValueError("metrics_every must be >= 1")
    if finite_check_every < 1:
        raise ValueError("finite_check_every must be >= 1")
    if save_every_steps < 0:
        raise ValueError("save_every_steps must be >= 0")
    if lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be >= 0")
    if forecast_weight_warmup_steps < 0:
        raise ValueError("forecast_weight_warmup_steps must be >= 0")
    if grad_clip_max_norm <= 0:
        raise ValueError("grad_clip_max_norm must be > 0")
    if save_best_mode not in {"min", "max"}:
        raise ValueError("save_best_mode must be 'min' or 'max'")
    if save_best_by == "train" and val_cache is not None:
        print(
            f"save_best_by=train save_best_metric={save_best_metric} save_best_mode={save_best_mode}",
            flush=True,
        )
    elif save_best_by == "val":
        print(
            f"save_best_by=val save_best_metric={save_best_metric} save_best_mode={save_best_mode}",
            flush=True,
        )
    use_pin_memory = pin_memory and device.startswith("cuda")
    if run_dir is None:
        run_name = (
            Path(checkpoint_path).stem
            if checkpoint_path is not None
            else f"train_cached_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir = Path("runs") / run_name
    run_config = {
        key: to_jsonable(value)
        for key, value in locals().items()
        if key
        not in {
            "run_config",
            "experiment_logger",
        }
    }
    experiment_logger = ExperimentLogger(
        run_dir,
        config=run_config,
        comet_enabled=comet,
        comet_project=comet_project,
        comet_workspace=comet_workspace,
        comet_run_name=comet_run_name,
        comet_tags=comet_tags or [],
    )
    data_cfg = DataConfig(roi_m=roi, voxel_size_m=voxel_size)

    # Need future + past horizons before constructing ModelConfig.
    train_dataset_probe = CachedWaymoDataset(train_cache)
    future_steps = train_dataset_probe[0]["gt_future"].shape[1]
    # past_horizon = strictly-past slots (history length minus the current frame).
    history_steps = train_dataset_probe[0]["gt_history"].shape[1]
    past_horizon = max(history_steps - 1, 0) if predict_past else 0
    # M2 occupancy head width: history-only (H) by default, or H + T_future when
    # the future occupancy channels are enabled, so the motion-aux head matches
    # the combined past+future target raster.
    occupancy_channels = history_steps + (future_steps if occ_aux_future else 0)
    model_cfg = ModelConfig(
        num_queries=num_queries,
        num_refinement_blocks=num_refinement_blocks,
        num_times=future_steps + 1,
        deformable_radius_m=deformable_radius_m,
        attention_type=attention_type,
        lidar_attention_anchor=lidar_attention_anchor,
        detach_pose_refinement=detach_pose_refinement,
        use_query_velocity=query_velocity_source != "none",
        forecast_base_mode=forecast_base_mode,
        forecast_velocity_frame=forecast_velocity_frame,
        use_history_occupancy=gt_history_occupancy,
        history_occupancy_channels=occupancy_channels,
        map_encoder_type=map_encoder_type,
        map_graph_k=map_graph_k,
        map_graph_layers=map_graph_layers,
        predict_past=predict_past,
        past_horizon=past_horizon,
    )

    train_sample_weights = None
    if motion_balanced_sampling:
        train_sample_weights = motion_balanced_sample_weights(
            train_dataset_probe,
            static_weight=motion_static_weight,
            moving_weight=motion_moving_weight,
            fast_weight=motion_fast_weight,
        )
        print(
            "motion_balanced_sampling "
            f"static_weight={motion_static_weight} moving_weight={motion_moving_weight} "
            f"fast_weight={motion_fast_weight} "
            f"mean_weight={float(train_sample_weights.mean().item()):.4f} "
            f"max_weight={float(train_sample_weights.max().item()):.4f}",
            flush=True,
        )

    _, train_loader = build_loader(
        train_cache,
        batch_size=batch_size,
        max_points=max_points,
        max_objects=model_cfg.num_queries,
        shuffle=True,
        max_map_tokens=max_map_tokens,
        deterministic_seed=deterministic_seed,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        sample_weights=train_sample_weights,
    )
    val_loader = None
    if val_cache is not None:
        _, val_loader = build_loader(
            val_cache,
            batch_size=batch_size,
            max_points=max_points,
            max_objects=model_cfg.num_queries,
            shuffle=False,
            max_map_tokens=max_map_tokens,
            deterministic_seed=deterministic_seed,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

    lidar_encoder = PaperLikeLidarEncoder(
        data_cfg,
        point_feature_dim=model_cfg.point_feature_dim,
        hidden_dim=model_cfg.hidden_dim,
        dynamic_conv_experts=model_cfg.dynamic_conv_experts,
        norm_type=norm_type,
        history_occupancy_channels=(
            model_cfg.history_occupancy_channels
            if gt_history_occupancy or dense_history_motion_weight > 0
            else 0
        ),
    ).to(device)
    if proposal_head_type == "paper":
        # Paper A.3: FPN-fused stride-4 map -> separate 4-conv BN/ReLU score and
        # box branches. The conv norm follows the encoder norm_type so a
        # GroupNorm launcher stays consistent end-to-end.
        proposal_head = PaperProposalHead(
            model_cfg.hidden_dim,
            num_convs=4,
            norm_type=norm_type if norm_type in {"batch", "group"} else "group",
        ).to(device)
    elif proposal_head_type == "shallow":
        proposal_head = ProposalHead(model_cfg.hidden_dim).to(device)
    else:
        raise ValueError("proposal_head_type must be 'shallow' or 'paper'")
    map_encoder = build_map_encoder(model_cfg).to(device)
    model = DeTraMini(model_cfg, data_cfg).to(device)
    resume_ckpt: dict[str, object] | None = None
    if resume_checkpoint is not None:
        resume_ckpt = load_checkpoint(resume_checkpoint, device)
        _load_compatible(lidar_encoder, resume_ckpt["lidar_encoder"])
        _load_compatible(proposal_head, resume_ckpt["proposal_head"])
        if "map_encoder" in resume_ckpt:
            _load_compatible(map_encoder, resume_ckpt["map_encoder"])
        _load_compatible(model, resume_ckpt["model"])
    params = list(lidar_encoder.parameters()) + list(proposal_head.parameters())
    if objective != "proposal_only":
        params += list(map_encoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best_val: float | None = None
    best_train: float | None = None
    best_forecast: float | None = None
    final_train_metrics = None
    final_val_metrics = None
    best_train_metrics = None
    best_val_metrics = None
    val_metrics_by_step: dict[int, dict[str, float]] = {}
    saved_eval_ckpts: deque[Path] = deque()
    start_time = time.monotonic()
    timing_sums: dict[str, float] = {}
    timing_count = 0
    loss_ema: float | None = None
    loss_ema_beta = 0.98
    clip_events = 0  # running count of steps whose pre-clip grad norm exceeded the cap
    applied_steps = 0  # steps that actually took an optimizer step (not skipped)
    clip_window: deque[int] = deque(maxlen=200)
    clip_scale_window: deque[float] = deque(maxlen=200)
    start_step = 0
    if resume_ckpt is not None:
        checkpoint_step = int(resume_ckpt.get("step", 0))
        if resume_weights_only:
            print(
                f"resumed_checkpoint={resume_checkpoint} mode=weights_only "
                f"checkpoint_step={checkpoint_step} start_step=0",
                flush=True,
            )
        else:
            start_step = checkpoint_step
            if "optimizer" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                optimizer_restored = True
            else:
                optimizer_restored = False
            saved_training_state = resume_ckpt.get("training_state", {})
            if isinstance(saved_training_state, dict) and saved_training_state:
                applied_steps = int(saved_training_state.get("applied_steps", 0))
                clip_events = int(saved_training_state.get("clip_events", 0))
                loss_ema = saved_training_state.get("loss_ema")
                best_val = saved_training_state.get("best_val")
                best_train = saved_training_state.get("best_train")
                best_forecast = saved_training_state.get("best_forecast")
                best_train_metrics = saved_training_state.get("best_train_metrics")
                best_val_metrics = saved_training_state.get("best_val_metrics")
                restored_history = saved_training_state.get("val_metrics_by_step", {})
                if isinstance(restored_history, dict):
                    val_metrics_by_step = {
                        int(key): value for key, value in restored_history.items()
                    }
                clip_window.extend(saved_training_state.get("clip_window", []))
                clip_scale_window.extend(
                    saved_training_state.get("clip_scale_window", [])
                )
            elif save_best_by == "val" and "val_metrics" in resume_ckpt:
                best_val = resume_ckpt["val_metrics"].get(save_best_metric)
            elif save_best_by == "train" and "train_metrics" in resume_ckpt:
                best_train = resume_ckpt["train_metrics"].get(save_best_metric)
            _restore_rng_state(resume_ckpt.get("rng_state"))
            saved_scheduler = resume_ckpt.get("lr_scheduler")
            if isinstance(saved_scheduler, dict):
                expected_scheduler = {
                    "decay": lr_decay,
                    "base_lr": lr,
                    "warmup_steps": lr_warmup_steps,
                    "min_scale": lr_min_scale,
                    "total_steps": lr_decay_total_steps if lr_decay_total_steps else steps,
                }
                mismatches = {
                    key: (saved_scheduler.get(key), value)
                    for key, value in expected_scheduler.items()
                    if saved_scheduler.get(key) != value
                }
                if mismatches:
                    print(
                        f"resume_scheduler_config_mismatch={mismatches}",
                        flush=True,
                    )
            print(
                f"resumed_checkpoint={resume_checkpoint} mode=full "
                f"checkpoint_step={checkpoint_step} start_step={start_step} "
                f"optimizer_restored={int(optimizer_restored)}",
                flush=True,
            )
            if not optimizer_restored:
                print(
                    "resume_warning=legacy_checkpoint_has_no_optimizer_state; "
                    "global schedule resumes but AdamW moments restart",
                    flush=True,
                )
    if start_step >= steps:
        raise ValueError(
            f"resume checkpoint step {start_step} is not below target --steps {steps}"
        )

    base_checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if base_checkpoint_path is not None:
        step_prefix = f"{base_checkpoint_path.stem}_step"
        existing_step_ckpts = sorted(
            (
                path
                for path in base_checkpoint_path.parent.glob(
                    f"{step_prefix}*{base_checkpoint_path.suffix}"
                )
                if path.stem[len(step_prefix):].isdigit()
            ),
            key=lambda path: int(path.stem[len(step_prefix):]),
        )
        saved_eval_ckpts.extend(existing_step_ckpts)

    termination_signal: int | None = None

    def _request_termination(signum: int, _frame) -> None:
        nonlocal termination_signal
        termination_signal = signum
        print(
            f"signal_received={signal.Signals(signum).name} "
            f"will_checkpoint_after_current_step=1",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _request_termination)
    signal.signal(signal.SIGINT, _request_termination)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _request_termination)

    def _scheduler_state(step: int) -> dict[str, object]:
        return {
            "kind": "manual_global_step",
            "last_step": step,
            "decay": lr_decay,
            "base_lr": lr,
            "warmup_steps": lr_warmup_steps,
            "min_scale": lr_min_scale,
            "total_steps": lr_decay_total_steps if lr_decay_total_steps else steps,
        }

    def _training_state(step: int) -> dict[str, object]:
        return {
            "global_step": step,
            "applied_steps": applied_steps,
            "clip_events": clip_events,
            "loss_ema": loss_ema,
            "clip_window": list(clip_window),
            "clip_scale_window": list(clip_scale_window),
            "best_val": best_val,
            "best_train": best_train,
            "best_forecast": best_forecast,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "val_metrics_by_step": val_metrics_by_step,
            "termination_signal": termination_signal,
        }

    def _save_training_checkpoint(
        path: Path,
        *,
        step: int,
        metrics_key: str,
        metrics: dict[str, float],
        selected_by: str = save_best_by,
        selected_metric: str = save_best_metric,
        selected_mode: str = save_best_mode,
    ) -> None:
        _save_checkpoint(
            path,
            step=step,
            model=model,
            lidar_encoder=lidar_encoder,
            proposal_head=proposal_head,
            map_encoder=map_encoder,
            metrics_key=metrics_key,
            metrics=metrics,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            save_best_by=selected_by,
            save_best_metric=selected_metric,
            save_best_mode=selected_mode,
            optimizer=optimizer,
            lr_scheduler_state=_scheduler_state(step),
            training_state=_training_state(step),
        )

    def _save_rolling_checkpoint(
        *,
        step: int,
        metrics_key: str,
        metrics: dict[str, float],
    ) -> None:
        if base_checkpoint_path is None:
            return
        last_path = base_checkpoint_path.with_name(
            f"{base_checkpoint_path.stem}_last{base_checkpoint_path.suffix}"
        )
        step_path = base_checkpoint_path.with_name(
            f"{base_checkpoint_path.stem}_step{step}{base_checkpoint_path.suffix}"
        )
        _save_training_checkpoint(
            last_path,
            step=step,
            metrics_key=metrics_key,
            metrics=metrics,
        )
        _save_training_checkpoint(
            step_path,
            step=step,
            metrics_key=metrics_key,
            metrics=metrics,
        )
        experiment_logger.log_event(
            "checkpoint_saved",
            {
                "step": step,
                "reason": "rolling",
                "last_path": str(last_path),
                "step_path": str(step_path),
                "metrics_key": metrics_key,
            },
        )
        if step_path not in saved_eval_ckpts:
            saved_eval_ckpts.append(step_path)
        while keep_last_n > 0 and len(saved_eval_ckpts) > keep_last_n:
            old = saved_eval_ckpts.popleft()
            try:
                old.unlink()
            except OSError:
                pass

    iterator = iter(train_loader)

    # Named modules for per-subsystem grad-norm diagnostics on metrics steps.
    grad_norm_modules = {
        "bev": lidar_encoder,
        "proposal": proposal_head,
        "refine": model,
        "map": map_encoder,
    }
    # The aux-weight controller (GradNorm-style) needs the per-term grad norms, so
    # it implies grad_calibration. Targets are per-term fractions of the forecast
    # term's grad norm at the shared trunk; only terms with a positive target are
    # actively managed. Weights are the plain ``train()`` float locals, rebound each
    # controller tick so the next step's forward_batch picks up the new value.
    if aux_weight_controller:
        grad_calibration = True
    aux_controller_targets = {
        "past": past_target_frac,
        "motion_aux": motion_target_frac,
        "yaw_consistency": yaw_target_frac,
    }
    # Shared bottleneck every loss flows through; used to compare per-term gradient
    # contributions when --grad-calibration is set.
    calibration_shared_params = (
        [p for p in lidar_encoder.parameters() if p.requires_grad]
        if grad_calibration
        else []
    )
    for step in range(start_step + 1, steps + 1):
        current_lr = _lr_for_step(
            step,
            total_steps=lr_decay_total_steps if lr_decay_total_steps else steps,
            base_lr=lr,
            warmup_steps=lr_warmup_steps,
            decay=lr_decay,
            min_scale=lr_min_scale,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if forecast_weight_warmup_steps > 0:
            step_forecast_weight = forecast_weight * min(
                1.0,
                step / float(forecast_weight_warmup_steps),
            )
        else:
            step_forecast_weight = forecast_weight
        step_start = time.monotonic()
        timing_step: dict[str, float] = {}
        if profile_timing:
            _sync_for_timing(profile_timing_sync, device)
            timing_last = time.monotonic()
        lidar_encoder.train()
        proposal_head.train()
        map_encoder.train()
        model.train()

        # Evolving-WTA: anneal the number of supervised modes from F down to 1
        # over ``ewta_anneal_steps`` so non-winner modes survive early training.
        # With annealing disabled, the static ``forecast_top_k`` is used as-is.
        if ewta_anneal_steps > 0:
            step_top_k = _ewta_top_k(step, ewta_anneal_steps, model_cfg.num_modes)
        else:
            step_top_k = forecast_top_k
        # Forecast TP-gate warmup: relax the IoU>0.5 gate for the first
        # ``forecast_gate_warmup_steps`` steps (when no explicit radius gate is
        # configured) so forecasting gets gradient before detection localizes.
        step_forecast_gate_radius_m = forecast_gate_radius_m
        if (
            forecast_gate_radius_m is None
            and forecast_gate_warmup_steps > 0
            and step <= forecast_gate_warmup_steps
        ):
            step_forecast_gate_radius_m = forecast_gate_warmup_radius_m

        # Full diagnostics (and the metrics-only proposal decodes) only on metric
        # steps; this gates the expensive raw-decode/heatmap-topk in forward_batch.
        should_compute_full_train_metrics = (
            step == 1
            or step % metrics_every == 0
            or step == steps
            or (checkpoint_path is not None and save_best_by == "train")
        )

        # Gradient accumulation: run ``grad_accum_steps`` micro-batches, scaling
        # each loss by 1/grad_accum_steps, and take a single optimizer step at
        # the end. Effective batch = batch_size * grad_accum_steps (paper: 16).
        # The last micro-batch's pred/losses/targets/batch feed the downstream
        # metrics/diagnostics, mirroring the previous single-batch behavior.
        optimizer.zero_grad(set_to_none=True)
        accum_loss_value = 0.0
        nonfinite_microbatch = False
        calibration_grad_norms: dict[str, float] = {}
        for micro_idx in range(grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            if profile_timing and micro_idx == 0:
                _sync_for_timing(profile_timing_sync, device)
                timing_now = time.monotonic()
                timing_step["data"] = timing_now - timing_last
                timing_last = timing_now

            pred, losses, targets = forward_batch(
                batch,
                lidar_encoder,
                proposal_head,
                map_encoder if objective != "proposal_only" else None,
                model,
                data_cfg,
                model_cfg,
                device,
                objective=objective,
                proposal_mode=proposal_mode,
                proposal_noise_std_m=proposal_noise_std_m,
                forecast_weight=step_forecast_weight,
                proposal_heatmap_weight=proposal_heatmap_weight,
                proposal_box_weight=proposal_box_weight,
                proposal_giou_weight=proposal_giou_weight,
                detection_box_l1_weight=detection_box_l1_weight,
                detection_giou_weight=detection_giou_weight,
                detection_class_weight=detection_class_weight,
                detection_z_weight=detection_z_weight,
                detection_height_weight=detection_height_weight,
                forecast_gate_radius_m=step_forecast_gate_radius_m,
                forecast_loss_type=forecast_loss_type,
                forecast_top_k=step_top_k,
                velocity_weight=velocity_weight,
                yaw_consistency_weight=yaw_consistency_weight,
                yaw_consistency_vthresh_m=yaw_consistency_vthresh_m,
                past_weight=past_weight,
                init_weight=init_weight,
                proposal_nms_radius_m=proposal_nms_radius_m,
                proposal_nms_mode=proposal_nms_mode,
                proposal_center_mode=proposal_center_mode,
                proposal_peak_filter=proposal_peak_filter,
                support_filter_radius_m=support_filter_radius_m,
                query_velocity_source=query_velocity_source,
                query_velocity_match_radius_m=query_velocity_match_radius_m,
                gt_history_occupancy=gt_history_occupancy,
                gt_history_occupancy_radius_m=gt_history_occupancy_radius_m,
                dense_history_motion_weight=dense_history_motion_weight,
                dense_history_motion_flow_weight=dense_history_motion_flow_weight,
            occ_aux_future=occ_aux_future,
            traj_occ_consistency_weight=traj_occ_consistency_weight,
            traj_occ_mode_weighting=traj_occ_mode_weighting,
                compute_diagnostics=should_compute_full_train_metrics,
            )
            if profile_timing and micro_idx == 0:
                _sync_for_timing(profile_timing_sync, device)
                timing_now = time.monotonic()
                timing_step["forward"] = timing_now - timing_last
                timing_last = timing_now

            # Per-term gradient-norm calibration: must run before backward frees
            # the graph. On full-metric steps (diagnostic) and on controller ticks
            # (so the controller has fresh per-term grad norms to balance against).
            controller_tick = (
                aux_weight_controller
                and step >= aux_controller_warmup
                and step % aux_controller_interval == 0
            )
            if (
                grad_calibration
                and (should_compute_full_train_metrics or controller_tick)
                and micro_idx == 0
                and torch.isfinite(losses["loss"]).all()
            ):
                calibration_grad_norms = _term_grad_norms(
                    {
                        "detection": (losses.get("final_loss_detection"), 1.0),
                        "forecast": (losses.get("final_loss_forecast"), step_forecast_weight),
                        "past": (losses.get("final_loss_past"), past_weight),
                        "yaw_consistency": (
                            losses.get("final_loss_yaw_consistency"),
                            yaw_consistency_weight,
                        ),
                        "motion_aux": (losses.get("loss_motion_aux"), dense_history_motion_weight),
                        "traj_occ": (losses.get("loss_traj_occ"), traj_occ_consistency_weight),
                    },
                    calibration_shared_params,
                )

            if not torch.isfinite(losses["loss"]).item():
                # Surface which scalar loss components are non-finite to localize the cause.
                bad = {
                    key: float(value.detach().cpu())
                    for key, value in losses.items()
                    if isinstance(value, torch.Tensor) and value.ndim == 0
                    and not torch.isfinite(value).all()
                }
                sample_paths = batch.get("sample_paths", [])
                print(
                    f"train step={step}/{steps} skipped_nonfinite_loss "
                    f"loss={float(losses['loss'].detach().cpu())} nonfinite_terms={bad} "
                    f"sample_paths={sample_paths}",
                    flush=True,
                )
                nonfinite_microbatch = True
                break
            # Scale so accumulated gradients match a single batch of the
            # effective size; backward accumulates into the shared .grad buffers.
            (losses["loss"] / grad_accum_steps).backward()
            accum_loss_value += float(losses["loss"].detach().cpu()) / grad_accum_steps

        if nonfinite_microbatch:
            optimizer.zero_grad(set_to_none=True)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_max_norm)
        grad_norm_value = float(grad_norm.detach().cpu())
        if not torch.isfinite(grad_norm).item():
            optimizer.zero_grad(set_to_none=True)
            print(
                f"train step={step}/{steps} skipped_nonfinite_grad "
                f"grad_norm={grad_norm_value}",
                flush=True,
            )
            continue
        applied_steps += 1
        clip_active = grad_norm_value > grad_clip_max_norm
        if clip_active:
            clip_events += 1
        clip_fraction = clip_events / max(applied_steps, 1)
        clip_scale = min(1.0, grad_clip_max_norm / max(grad_norm_value, 1e-12))
        clip_window.append(1 if clip_active else 0)
        clip_scale_window.append(clip_scale)
        clip_frac_window = sum(clip_window) / max(len(clip_window), 1)
        clip_scale_window_mean = sum(clip_scale_window) / max(len(clip_scale_window), 1)
        loss_value = accum_loss_value
        loss_ema = loss_value if loss_ema is None else loss_ema_beta * loss_ema + (1 - loss_ema_beta) * loss_value
        # Per-subsystem grad norms (cheap; only on full-metric steps).
        module_grad_norms = (
            {name: _module_grad_norm(mod) for name, mod in grad_norm_modules.items()}
            if should_compute_full_train_metrics
            else {}
        )
        if profile_timing:
            _sync_for_timing(profile_timing_sync, device)
            timing_now = time.monotonic()
            timing_step["backward"] = timing_now - timing_last
            timing_last = timing_now
        optimizer.step()
        # Aux-weight controller (GradNorm-style): on controller ticks, rebind each
        # managed aux weight toward its target fraction of the forecast term's grad
        # norm at the shared trunk. w <- w * (target*g_forecast/g_k)^alpha, EMA-mixed
        # with the old weight (beta) and clamped. Takes effect next step.
        if (
            aux_weight_controller
            and step >= aux_controller_warmup
            and step % aux_controller_interval == 0
            and calibration_grad_norms
        ):
            g_forecast = calibration_grad_norms.get("forecast", 0.0)
            if math.isfinite(g_forecast) and g_forecast > 1e-12:
                _wmap = {
                    "past": past_weight,
                    "motion_aux": dense_history_motion_weight,
                    "yaw_consistency": yaw_consistency_weight,
                }
                for _term, _frac in aux_controller_targets.items():
                    if _frac <= 0.0:
                        continue
                    g_k = calibration_grad_norms.get(_term, 0.0)
                    w_old = _wmap[_term]
                    if not math.isfinite(g_k) or g_k <= 1e-12 or w_old <= 0.0:
                        continue
                    w_target = w_old * (_frac * g_forecast / g_k) ** aux_controller_alpha
                    w_new = (1.0 - aux_controller_beta) * w_old + aux_controller_beta * w_target
                    _wmap[_term] = float(min(max(w_new, aux_weight_min), aux_weight_max))
                past_weight = _wmap["past"]
                dense_history_motion_weight = _wmap["motion_aux"]
                yaw_consistency_weight = _wmap["yaw_consistency"]
        if profile_timing:
            _sync_for_timing(profile_timing_sync, device)
            timing_now = time.monotonic()
            timing_step["optimizer"] = timing_now - timing_last
            timing_last = timing_now
        should_check_params = step == 1 or step % finite_check_every == 0 or step == steps
        if should_check_params and not _parameters_are_finite(params):
            raise FloatingPointError(f"model parameters became non-finite after optimizer step={step}")
        if profile_timing:
            _sync_for_timing(profile_timing_sync, device)
            timing_now = time.monotonic()
            timing_step["finite_check"] = timing_now - timing_last
            timing_last = timing_now
        step_elapsed = time.monotonic() - step_start
        elapsed = time.monotonic() - start_time
        run_steps_completed = step - start_step
        eta = elapsed / max(run_steps_completed, 1) * (steps - step)
        # (should_compute_full_train_metrics computed before forward_batch above.)
        should_log_train = (
            step == 1
            or step % log_every == 0
            or step == steps
            or should_compute_full_train_metrics
        )
        if should_compute_full_train_metrics:
            train_metrics = compute_metrics(pred, losses, targets)
            train_metric_mode = "full"
            train_line_keys = (
                "loss",
                "loss_init_heatmap",
                "loss_init_box",
                "loss_init_giou",
                "loss_init_heatmap_weighted",
                "loss_init_box_weighted",
                "loss_init_giou_weighted",
                "final_loss_class_ce",
                "heatmap_recall_2m",
                "support_heatmap_recall_2m",
                "raw_proposal_recall_2m",
                "final_loss_detection",
                "final_loss_forecast",
                "final_loss_mode",
                "final_loss_velocity",
                "final_velocity_l2",
                "final_velocity_supervised_queries",
                "final_forecast_candidate_queries",
                "final_forecast_supervised_queries",
                "final_forecast_gate_removed_queries",
                "loss_motion_aux",
                "loss_motion_occ",
                "loss_motion_flow",
                "motion_flow_l2",
                "motion_flow_cells",
                "proposal_recall_2m",
                "support_proposal_recall_2m",
                "refined_recall_2m",
                "top32_refined_recall_2m",
                "det_recall_2m",
                "matched_center_dist",
                "matched_z_err",
                "matched_height_err",
                "matched_score_rank",
                "score_max",
                "ade",
                "fde",
                "gt_final_disp",
                "pred_final_disp",
                "moving_ade",
                "moving_fde",
                "moving_gt_final_disp",
                "moving_pred_final_disp",
                "ade_t1",
                "ade_t3",
                "ade_t5",
                "ade_t10",
                "gt_disp_t10",
                "pred_disp_t10",
                "mode_entropy",
                "mode_entropy_normalized",
                "mode_effective_count",
                "mode_top_prob",
                "mode_wta_is_top_rate",
                "mode_wta_prob",
                "mode_top_hist_0",
                "mode_top_hist_1",
                "mode_top_hist_2",
                "mode_top_hist_3",
                "mode_top_hist_4",
                "mode_top_hist_5",
                "mode_wta_hist_0",
                "mode_wta_hist_1",
                "mode_wta_hist_2",
                "mode_wta_hist_3",
                "mode_wta_hist_4",
                "mode_wta_hist_5",
                "forecast_scale",
                "map_token_count",
                "history_occupancy_count",
                "dense_history_motion_cells",
            )
        else:
            train_metrics = _scalar_loss_metrics(losses)
            train_metric_mode = "loss"
            train_line_keys = (
                "loss",
                "loss_init_heatmap",
                "loss_init_box",
                "loss_init_giou",
                "loss_init_heatmap_weighted",
                "loss_init_box_weighted",
                "loss_init_giou_weighted",
                "final_loss_detection",
                "final_loss_forecast",
                "final_loss_mode",
                "final_loss_velocity",
                "loss_motion_aux",
                "loss_motion_occ",
                "loss_motion_flow",
            )
        if profile_timing:
            _sync_for_timing(profile_timing_sync, device)
            timing_now = time.monotonic()
            timing_step["metrics"] = timing_now - timing_last
            timing_step["total"] = timing_now - step_start
            timing_last = timing_now
            for key, value in timing_step.items():
                timing_sums[key] = timing_sums.get(key, 0.0) + value
            timing_count += 1
        if should_log_train:
            train_line = _format_metrics(train_metrics, train_line_keys)
            opt_line = (
                f"loss_ema={loss_ema:.4f} grad_norm={grad_norm_value:.4f} "
                f"clip_frac={clip_fraction:.3f} clip_frac_window_200={clip_frac_window:.3f} "
                f"clip_active={int(clip_active)} clip_scale={clip_scale:.4f} "
                f"clip_scale_window_200={clip_scale_window_mean:.4f} "
                f"lr={current_lr:.2e} forecast_weight={step_forecast_weight:.4f}"
            )
            if module_grad_norms:
                opt_line += " " + " ".join(
                    f"gnorm_{name}={value:.4f}" for name, value in module_grad_norms.items()
                )
            if calibration_grad_norms:
                opt_line += " " + " ".join(
                    f"cgn_{name}={value:.4f}" for name, value in calibration_grad_norms.items()
                )
            if aux_weight_controller:
                opt_line += (
                    f" auxw_past={past_weight:.4f}"
                    f" auxw_motion={dense_history_motion_weight:.4f}"
                    f" auxw_yaw={yaw_consistency_weight:.4f}"
                )
            print(
                f"train step={step}/{steps} step_time={_format_duration(step_elapsed)} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
                f"metrics={train_metric_mode} {opt_line} {train_line}",
                flush=True,
            )
            if profile_timing:
                print(
                    f"timing step={step}/{steps} sync={int(profile_timing_sync)} "
                    f"{_format_timing_window(timing_sums, timing_count)}",
                    flush=True,
                )
                timing_sums = {}
                timing_count = 0
            experiment_logger.log_metrics(
                "train",
                step,
                {
                    **train_metrics,
                    "loss_ema": loss_ema,
                    "grad_norm": grad_norm_value,
                    "clip_fraction": clip_fraction,
                    "clip_fraction_window_200": clip_frac_window,
                    "clip_scale": clip_scale,
                    "clip_scale_window_200": clip_scale_window_mean,
                    "lr": current_lr,
                    "forecast_weight_active": step_forecast_weight,
                    **{f"grad_norm_{name}": value for name, value in module_grad_norms.items()},
                    **{f"gradnorm_{name}": value for name, value in calibration_grad_norms.items()},
                    **(
                        {
                            "aux_weight_past": past_weight,
                            "aux_weight_motion": dense_history_motion_weight,
                            "aux_weight_yaw": yaw_consistency_weight,
                        }
                        if aux_weight_controller
                        else {}
                    ),
                },
                comet_keys=COMET_TRAIN_METRIC_KEYS,
            )
        final_train_metrics = train_metrics

        should_eval = (
            val_loader is not None
            and ((initial_eval and step == 1) or step % eval_every == 0 or step == steps)
        )
        if checkpoint_path is not None and save_best_by == "train":
            improved, best_train = _is_better_metric(
                train_metrics,
                save_best_metric,
                best_train,
                save_best_mode,
            )
            if improved:
                best_train_metrics = train_metrics
                _save_training_checkpoint(
                    Path(checkpoint_path),
                    step=step,
                    metrics_key="train_metrics",
                    metrics=train_metrics,
                )
                experiment_logger.log_event(
                    "checkpoint_saved",
                    {
                        "step": step,
                        "reason": "best_train",
                        "path": str(checkpoint_path),
                        "metric": save_best_metric,
                        "mode": save_best_mode,
                    },
                )

        if (
            save_every_steps > 0
            and step % save_every_steps == 0
            and not (save_every_eval and should_eval)
        ):
            _save_rolling_checkpoint(
                step=step,
                metrics_key="train_metrics",
                metrics=train_metrics,
            )
            print(f"checkpoint_saved step={step} reason=periodic", flush=True)

        if termination_signal is not None:
            _save_rolling_checkpoint(
                step=step,
                metrics_key="train_metrics",
                metrics=train_metrics,
            )
            if base_checkpoint_path is not None:
                interrupted_path = base_checkpoint_path.with_name(
                    f"{base_checkpoint_path.stem}_interrupted"
                    f"{base_checkpoint_path.suffix}"
                )
                _save_training_checkpoint(
                    interrupted_path,
                    step=step,
                    metrics_key="train_metrics",
                    metrics=train_metrics,
                )
            signal_name = signal.Signals(termination_signal).name
            print(
                f"checkpoint_saved step={step} reason=signal signal={signal_name}",
                flush=True,
            )
            raise SystemExit(128 + termination_signal)

        if should_eval:
            metrics = evaluate(
                val_loader,
                lidar_encoder,
                proposal_head,
                map_encoder if objective != "proposal_only" else None,
                model,
                data_cfg,
                model_cfg,
                device,
                max_batches=val_batches,
                objective=objective,
                proposal_mode=proposal_mode,
                proposal_noise_std_m=proposal_noise_std_m,
                forecast_weight=forecast_weight,
                proposal_heatmap_weight=proposal_heatmap_weight,
                proposal_box_weight=proposal_box_weight,
                proposal_giou_weight=proposal_giou_weight,
                detection_box_l1_weight=detection_box_l1_weight,
                detection_giou_weight=detection_giou_weight,
                detection_class_weight=detection_class_weight,
                detection_z_weight=detection_z_weight,
                detection_height_weight=detection_height_weight,
                forecast_gate_radius_m=forecast_gate_radius_m,
                forecast_loss_type=forecast_loss_type,
                forecast_top_k=forecast_top_k,
                velocity_weight=velocity_weight,
                    init_weight=init_weight,
                proposal_nms_radius_m=proposal_nms_radius_m,
                proposal_nms_mode=proposal_nms_mode,
                proposal_center_mode=proposal_center_mode,
                proposal_peak_filter=proposal_peak_filter,
                support_filter_radius_m=support_filter_radius_m,
                query_velocity_source=query_velocity_source,
                query_velocity_match_radius_m=query_velocity_match_radius_m,
                gt_history_occupancy=gt_history_occupancy,
                gt_history_occupancy_radius_m=gt_history_occupancy_radius_m,
                dense_history_motion_weight=dense_history_motion_weight,
                dense_history_motion_flow_weight=dense_history_motion_flow_weight,
            occ_aux_future=occ_aux_future,
            traj_occ_consistency_weight=traj_occ_consistency_weight,
            traj_occ_mode_weighting=traj_occ_mode_weighting,
            )
            final_val_metrics = metrics
            val_line = _format_metrics(
                metrics,
                (
                    "loss",
                    "loss_init_heatmap",
                    "loss_init_box",
                    "loss_init_giou",
                    "loss_init_heatmap_weighted",
                    "loss_init_box_weighted",
                    "loss_init_giou_weighted",
                    "final_loss_class_ce",
                    "heatmap_recall_1m",
                    "heatmap_recall_2m",
                    "heatmap_recall_4m",
                    "support_heatmap_recall_1m",
                    "support_heatmap_recall_2m",
                    "support_heatmap_recall_4m",
                    "raw_proposal_recall_1m",
                    "raw_proposal_recall_2m",
                    "raw_proposal_recall_4m",
                    "final_loss_detection",
                    "final_loss_forecast",
                    "final_loss_mode",
                    "final_loss_velocity",
                    "final_velocity_l2",
                    "final_velocity_supervised_queries",
                    "final_forecast_candidate_queries",
                    "final_forecast_supervised_queries",
                    "final_forecast_gate_removed_queries",
                    "loss_motion_aux",
                    "loss_motion_occ",
                    "loss_motion_flow",
                    "motion_flow_l2",
                    "motion_flow_cells",
                    "proposal_recall_1m",
                    "proposal_recall_2m",
                    "proposal_recall_4m",
                    "support_proposal_recall_1m",
                    "support_proposal_recall_2m",
                    "support_proposal_recall_4m",
                    "refined_recall_1m",
                    "refined_recall_2m",
                    "refined_recall_4m",
                    "top32_refined_recall_1m",
                    "top32_refined_recall_2m",
                    "top32_refined_recall_4m",
                    "det_recall_2m",
                    "matched_center_dist",
                    "matched_z_err",
                    "matched_height_err",
                    "matched_score_rank",
                    "score_max",
                    "ade",
                    "fde",
                    "gt_final_disp",
                    "pred_final_disp",
                    "moving_track_count",
                    "moving_ade",
                    "moving_fde",
                    "moving_gt_final_disp",
                    "moving_pred_final_disp",
                    "minade_k1",
                    "minfde_k1",
                    "minade_k6",
                    "minfde_k6",
                    "mr_k6",
                    "minade_k6_veh",
                    "minfde_k6_veh",
                    "minade_k6_ped",
                    "minfde_k6_ped",
                    "minade_k6_cyc",
                    "minfde_k6_cyc",
                    "det_recall_2m_veh",
                    "det_recall_2m_ped",
                    "det_recall_2m_cyc",
                    "track_count_veh",
                    "track_count_ped",
                    "track_count_cyc",
                    "ade_t1",
                    "ade_t3",
                    "ade_t5",
                    "ade_t10",
                    "gt_disp_t10",
                    "pred_disp_t10",
                    "mode_entropy",
                    "mode_entropy_normalized",
                    "mode_effective_count",
                    "mode_top_prob",
                    "mode_wta_is_top_rate",
                    "mode_wta_prob",
                    "mode_top_hist_0",
                    "mode_top_hist_1",
                    "mode_top_hist_2",
                    "mode_top_hist_3",
                    "mode_top_hist_4",
                    "mode_top_hist_5",
                    "mode_wta_hist_0",
                    "mode_wta_hist_1",
                    "mode_wta_hist_2",
                    "mode_wta_hist_3",
                    "mode_wta_hist_4",
                    "mode_wta_hist_5",
                    "forecast_scale",
                    "gt_count",
                    "train_gt_count",
                    "support_gt_count",
                    "future_track_count",
                    "map_token_count",
                    "history_occupancy_count",
                    "dense_history_motion_cells",
                ),
            )
            print(
                f"val step={step}/{steps} elapsed={_format_duration(time.monotonic() - start_time)} "
                f"eta={_format_duration(eta)} {val_line}",
                flush=True,
            )
            experiment_logger.log_metrics("val", step, metrics)
            if eval_train_mode:
                train_mode_metrics = evaluate(
                    val_loader,
                    lidar_encoder,
                    proposal_head,
                    map_encoder if objective != "proposal_only" else None,
                    model,
                    data_cfg,
                    model_cfg,
                    device,
                    max_batches=val_batches,
                    objective=objective,
                    proposal_mode=proposal_mode,
                    proposal_noise_std_m=proposal_noise_std_m,
                    forecast_weight=forecast_weight,
                    proposal_heatmap_weight=proposal_heatmap_weight,
                    proposal_box_weight=proposal_box_weight,
                    proposal_giou_weight=proposal_giou_weight,
                    detection_box_l1_weight=detection_box_l1_weight,
                    detection_giou_weight=detection_giou_weight,
                    detection_class_weight=detection_class_weight,
                    detection_z_weight=detection_z_weight,
                    detection_height_weight=detection_height_weight,
                    forecast_gate_radius_m=forecast_gate_radius_m,
                    forecast_loss_type=forecast_loss_type,
                    forecast_top_k=forecast_top_k,
                    velocity_weight=velocity_weight,
                            init_weight=init_weight,
                    proposal_nms_radius_m=proposal_nms_radius_m,
                    proposal_nms_mode=proposal_nms_mode,
                    proposal_center_mode=proposal_center_mode,
                    proposal_peak_filter=proposal_peak_filter,
                    support_filter_radius_m=support_filter_radius_m,
                    query_velocity_source=query_velocity_source,
                    query_velocity_match_radius_m=query_velocity_match_radius_m,
                    gt_history_occupancy=gt_history_occupancy,
                    gt_history_occupancy_radius_m=gt_history_occupancy_radius_m,
                    dense_history_motion_weight=dense_history_motion_weight,
                    dense_history_motion_flow_weight=dense_history_motion_flow_weight,
            occ_aux_future=occ_aux_future,
            traj_occ_consistency_weight=traj_occ_consistency_weight,
            traj_occ_mode_weighting=traj_occ_mode_weighting,
                    train_mode_eval=True,
                )
                train_mode_line = _format_metrics(
                    train_mode_metrics,
                    (
                        "loss",
                        "proposal_recall_2m",
                        "refined_recall_2m",
                        "matched_center_dist",
                        "matched_z_err",
                        "matched_height_err",
                        "ade",
                        "fde",
                    ),
                )
                print(f"val_train_mode step={step} {train_mode_line}", flush=True)
                experiment_logger.log_metrics("val_train_mode", step, train_mode_metrics)

            # Record per-eval metric history before any checkpoint is written so
            # a resumed run retains the complete curve through this evaluation.
            val_metrics_by_step[int(step)] = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            if checkpoint_path is not None and save_best_by == "val":
                improved, best_val = _is_better_metric(
                    metrics,
                    save_best_metric,
                    best_val,
                    save_best_mode,
                )
            else:
                improved = False
            if improved:
                best_val_metrics = metrics
                _save_training_checkpoint(
                    Path(checkpoint_path),
                    step=step,
                    metrics_key="val_metrics",
                    metrics=metrics,
                )
                experiment_logger.log_event(
                    "checkpoint_saved",
                    {
                        "step": step,
                        "reason": "best_val",
                        "path": str(checkpoint_path),
                        "metric": save_best_metric,
                        "mode": save_best_mode,
                    },
                )

            if checkpoint_path is not None:
                base = Path(checkpoint_path)
                # Best-by-forecast pointer (e.g. moving_fde, min): a separate path
                # so a joint model keeps its best-forecasting weights even after
                # detection (the primary save metric) plateaus.
                if save_best_forecast_metric is not None and save_best_forecast_metric in metrics:
                    f_improved, best_forecast = _is_better_metric(
                        metrics, save_best_forecast_metric, best_forecast, "min"
                    )
                    if f_improved:
                        fc_path = base.with_name(f"{base.stem}_best_{save_best_forecast_metric}{base.suffix}")
                        _save_training_checkpoint(
                            fc_path,
                            step=step,
                            metrics_key="val_metrics",
                            metrics=metrics,
                            selected_by="val",
                            selected_metric=save_best_forecast_metric,
                            selected_mode="min",
                        )
                        experiment_logger.log_event(
                            "checkpoint_saved",
                            {
                                "step": step,
                                "reason": "best_forecast",
                                "path": str(fc_path),
                                "metric": save_best_forecast_metric,
                                "mode": "min",
                            },
                        )
                # Save a checkpoint at EVERY eval (rolling last + step-tagged ring),
                # independent of any metric, so improving forecasting is never lost
                # just because the best-metric (detection) has plateaued.
                if save_every_eval:
                    _save_rolling_checkpoint(
                        step=step,
                        metrics_key="val_metrics",
                        metrics=metrics,
                    )

        if termination_signal is not None:
            checkpoint_metrics = final_val_metrics or train_metrics
            checkpoint_metrics_key = (
                "val_metrics" if final_val_metrics is not None else "train_metrics"
            )
            _save_rolling_checkpoint(
                step=step,
                metrics_key=checkpoint_metrics_key,
                metrics=checkpoint_metrics,
            )
            if base_checkpoint_path is not None:
                interrupted_path = base_checkpoint_path.with_name(
                    f"{base_checkpoint_path.stem}_interrupted"
                    f"{base_checkpoint_path.suffix}"
                )
                _save_training_checkpoint(
                    interrupted_path,
                    step=step,
                    metrics_key=checkpoint_metrics_key,
                    metrics=checkpoint_metrics,
                )
            signal_name = signal.Signals(termination_signal).name
            print(
                f"checkpoint_saved step={step} reason=signal signal={signal_name}",
                flush=True,
            )
            raise SystemExit(128 + termination_signal)

    run_summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "val_metrics_by_step": {str(k): v for k, v in sorted(val_metrics_by_step.items())},
    }
    experiment_logger.write_summary(run_summary)
    experiment_logger.close()

    if experiment_log is not None:
        _write_experiment_log(
            experiment_log,
            config={
                "train_cache": str(train_cache),
                "val_cache": str(val_cache) if val_cache is not None else None,
                "steps": steps,
                "batch_size": batch_size,
                "max_points": max_points,
                "max_map_tokens": max_map_tokens,
                "lr": lr,
                "roi": list(roi),
                "voxel_size": voxel_size,
                "eval_every": eval_every,
                "val_batches": val_batches,
                "initial_eval": initial_eval,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
                "resume_weights_only": resume_weights_only,
                "save_best_by": save_best_by,
                "save_best_metric": save_best_metric,
                "save_best_mode": save_best_mode,
                "save_every_steps": save_every_steps,
                "device": device,
                "objective": objective,
                "proposal_mode": proposal_mode,
                "proposal_noise_std_m": proposal_noise_std_m,
                "forecast_weight": forecast_weight,
                "lr_warmup_steps": lr_warmup_steps,
                "forecast_weight_warmup_steps": forecast_weight_warmup_steps,
                "grad_clip_max_norm": grad_clip_max_norm,
                "lr_decay": lr_decay,
                "lr_min_scale": lr_min_scale,
                "grad_accum_steps": grad_accum_steps,
                "proposal_heatmap_weight": proposal_heatmap_weight,
                "proposal_box_weight": proposal_box_weight,
                "proposal_giou_weight": proposal_giou_weight,
                "detection_box_l1_weight": detection_box_l1_weight,
                "detection_giou_weight": detection_giou_weight,
                "detection_class_weight": detection_class_weight,
                "detection_z_weight": detection_z_weight,
                "detection_height_weight": detection_height_weight,
                "forecast_gate_radius_m": forecast_gate_radius_m,
                "forecast_loss_type": forecast_loss_type,
                "forecast_top_k": forecast_top_k,
                "ewta_anneal_steps": ewta_anneal_steps,
                "forecast_gate_warmup_steps": forecast_gate_warmup_steps,
                "forecast_gate_warmup_radius_m": forecast_gate_warmup_radius_m,
                "velocity_weight": velocity_weight,
                "proposal_nms_radius_m": proposal_nms_radius_m,
                "proposal_nms_mode": proposal_nms_mode,
                "proposal_center_mode": proposal_center_mode,
                "proposal_peak_filter": proposal_peak_filter,
                "support_filter_radius_m": support_filter_radius_m,
                "deterministic_seed": deterministic_seed,
                "num_workers": num_workers,
                "pin_memory": use_pin_memory,
                "eval_train_mode": eval_train_mode,
                "norm_type": norm_type,
                "num_queries": num_queries,
                "num_refinement_blocks": num_refinement_blocks,
                "deformable_radius_m": deformable_radius_m,
                "attention_type": attention_type,
                "lidar_attention_anchor": lidar_attention_anchor,
                "detach_pose_refinement": detach_pose_refinement,
                "motion_balanced_sampling": motion_balanced_sampling,
                "motion_static_weight": motion_static_weight,
                "motion_moving_weight": motion_moving_weight,
                "motion_fast_weight": motion_fast_weight,
                "query_velocity_source": query_velocity_source,
                "query_velocity_match_radius_m": query_velocity_match_radius_m,
                "forecast_base_mode": forecast_base_mode,
                "forecast_velocity_frame": forecast_velocity_frame,
                "map_encoder_type": map_encoder_type,
                "map_graph_k": map_graph_k,
                "map_graph_layers": map_graph_layers,
                "gt_history_occupancy": gt_history_occupancy,
                "gt_history_occupancy_radius_m": gt_history_occupancy_radius_m,
                "dense_history_motion_weight": dense_history_motion_weight,
                "dense_history_motion_flow_weight": dense_history_motion_flow_weight,
                "log_every": log_every,
                "metrics_every": metrics_every,
                "finite_check_every": finite_check_every,
                "profile_timing": profile_timing,
                "profile_timing_sync": profile_timing_sync,
            },
            final_train_metrics=final_train_metrics,
            final_val_metrics=final_val_metrics,
            best_train_metrics=best_train_metrics,
            best_val_metrics=best_val_metrics,
            val_metrics_by_step=val_metrics_by_step,
        )


def _merge_experiment_args(args, parser) -> None:
    cfg = load_experiment_config(args.experiment_json)
    if not cfg:
        return
    defaults = {
        action.dest: parser.get_default(action.dest)
        for action in parser._actions
        if action.dest != "help"
    }
    special = {
        "initial_eval": ("no_initial_eval", lambda value: not bool(value)),
        "pin_memory": ("no_pin_memory", lambda value: not bool(value)),
        "proposal_peak_filter": ("no_proposal_peak_filter", lambda value: not bool(value)),
    }
    for raw_key, value in cfg.items():
        key = raw_key.replace("-", "_")
        if key in special:
            dest, transform = special[key]
            if getattr(args, dest) == defaults.get(dest):
                setattr(args, dest, transform(value))
            continue
        if not hasattr(args, key):
            print(f"experiment_json_ignored_unknown_key={raw_key}", flush=True)
            continue
        current = getattr(args, key)
        if current == defaults.get(key) or current is None:
            setattr(args, key, value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-json", default=None)
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=32768)
    parser.add_argument("--max-map-tokens", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--roi", type=float, nargs=4, default=(-80.0, -80.0, 80.0, 80.0))
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument(
        "--no-initial-eval",
        action="store_true",
        help="Skip validation at step 1 and evaluate only on eval_every/final steps.",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help=(
            "Load model weights but reset optimizer, scheduler/global step, and "
            "training statistics. By default a checkpoint resumes from its next "
            "global step and restores full state when available."
        ),
    )
    parser.add_argument("--save-best-by", choices=("val", "train"), default="val")
    parser.add_argument(
        "--save-best-metric",
        default="loss",
        help="Metric name used for best-checkpoint selection, e.g. loss or moving_fde.",
    )
    parser.add_argument("--save-best-mode", choices=("min", "max"), default="min")
    parser.add_argument(
        "--save-every-eval",
        action="store_true",
        help="Save a checkpoint at every eval (rolling *_last.pt + *_step{N}.pt ring), "
        "independent of the best metric, so improving forecasting is never lost.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help=(
            "Save rolling *_last.pt and step-tagged full-state checkpoints every "
            "N global steps, independently of validation (0 disables)."
        ),
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=3,
        help=(
            "With rolling eval/periodic saves, keep only the last N step-tagged "
            "checkpoints (0=keep all)."
        ),
    )
    parser.add_argument(
        "--save-best-forecast-metric",
        default=None,
        help="Optional second best-checkpoint pointer on a forecasting metric "
        "(min), e.g. moving_fde. Saved to *_best_<metric>.pt.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--objective", choices=("joint", "proposal_only"), default="joint")
    parser.add_argument(
        "--proposal-mode",
        choices=("decoded", "oracle", "noisy_gt"),
        default="decoded",
        help="Use decoded proposals, GT oracle proposals, or noisy GT proposals for refinement.",
    )
    parser.add_argument("--proposal-noise-std-m", type=float, default=0.0)
    parser.add_argument("--forecast-weight", type=float, default=0.1)
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Linearly warm LR from 0 to --lr over this many optimizer steps.",
    )
    parser.add_argument(
        "--forecast-weight-warmup-steps",
        type=int,
        default=0,
        help="Linearly warm forecast loss weight from 0 to --forecast-weight over this many steps.",
    )
    parser.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=5.0,
        help="Global gradient norm clipping threshold.",
    )
    parser.add_argument(
        "--lr-decay",
        choices=("none", "cosine"),
        default="none",
        help="Post-warmup LR schedule. Paper uses cosine decay from --lr to 0.",
    )
    parser.add_argument(
        "--lr-min-scale",
        type=float,
        default=0.0,
        help="Cosine-decay floor as a fraction of --lr (paper decays to 0).",
    )
    parser.add_argument(
        "--lr-decay-total-steps",
        type=int,
        default=0,
        help=(
            "Horizon (in optimizer steps) over which the cosine decays to its "
            "floor, decoupled from --steps. 0 (default) ties the horizon to "
            "--steps (train-to-budget: anneal fully to 0 at the stop step). Set "
            "this to the paper's 240000 while stopping early via --steps to get "
            "a schedule-faithful trajectory prefix that is cleanly resumable "
            "toward 240k (the run then stops 'hot' at a high LR, not annealed)."
        ),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients over this many micro-batches before an "
            "optimizer step. Effective batch = --batch-size * this. Paper uses "
            "an effective batch of 16."
        ),
    )
    parser.add_argument("--proposal-heatmap-weight", type=float, default=1.0)
    parser.add_argument("--proposal-box-weight", type=float, default=1.0)
    parser.add_argument("--proposal-giou-weight", type=float, default=0.1)
    parser.add_argument(
        "--detection-box-l1-weight",
        type=float,
        default=0.01,
        help="Paper beta: weight on the matched smooth-L1 box term (paper 0.01).",
    )
    parser.add_argument(
        "--detection-giou-weight",
        type=float,
        default=0.1,
        help="Paper gamma: weight on the matched BEV gIoU term (paper 0.1).",
    )
    parser.add_argument(
        "--detection-class-weight",
        type=float,
        default=1.0,
        help="Weight on the per-class cross-entropy detection term.",
    )
    parser.add_argument("--detection-z-weight", type=float, default=0.0)
    parser.add_argument("--detection-height-weight", type=float, default=0.0)
    parser.add_argument("--forecast-gate-radius-m", type=float, default=None)
    parser.add_argument("--forecast-loss-type", choices=("laplace", "smooth_l1"), default="laplace")
    parser.add_argument(
        "--forecast-top-k",
        type=int,
        default=1,
        help="Static evolving-WTA width. 1 = plain winner-takes-all. Overridden per-step when --ewta-anneal-steps > 0.",
    )
    parser.add_argument(
        "--ewta-anneal-steps",
        type=int,
        default=0,
        help="Anneal the number of supervised forecast modes from F down to 1 over this many steps (0 = disabled).",
    )
    parser.add_argument(
        "--forecast-gate-warmup-steps",
        type=int,
        default=0,
        help="Relax the IoU>0.5 forecast TP gate to a center-radius gate for the first N steps (0 = pure IoU gate).",
    )
    parser.add_argument(
        "--forecast-gate-warmup-radius-m",
        type=float,
        default=2.0,
        help="Center-distance radius used by the forecast gate during warmup steps.",
    )
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument(
        "--yaw-consistency-weight",
        type=float,
        default=0.0,
        help="Weight on the yaw<->motion-direction consistency loss (M3). 0 disables.",
    )
    parser.add_argument(
        "--yaw-consistency-vthresh-m",
        type=float,
        default=1.0,
        help="Min GT next-step displacement (m) for an actor to count as moving in M3.",
    )
    parser.add_argument(
        "--predict-past",
        action="store_true",
        help="Enable the M1 past-trajectory head (decodes the observed past track).",
    )
    parser.add_argument(
        "--past-weight",
        type=float,
        default=0.0,
        help="Weight on the M1 past-trajectory SmoothL1 loss. 0 disables.",
    )
    parser.add_argument("--init-weight", type=float, default=1.0)
    parser.add_argument("--proposal-nms-radius-m", type=float, default=0.1)
    parser.add_argument("--proposal-nms-mode", choices=("rotated", "center", "none"), default="rotated")
    parser.add_argument("--proposal-center-mode", choices=("regression", "cell"), default="regression")
    parser.add_argument("--no-proposal-peak-filter", action="store_true")
    parser.add_argument("--support-filter-radius-m", type=float, default=None)
    parser.add_argument("--deterministic-seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--eval-train-mode", action="store_true")
    parser.add_argument("--norm-type", choices=("batch", "group", "instance"), default="batch")
    parser.add_argument(
        "--proposal-head",
        dest="proposal_head_type",
        choices=("shallow", "paper"),
        default="shallow",
        help=(
            "Initial pose head. 'shallow' = 1x 3x3 + 1x1 per branch (legacy). "
            "'paper' = suppl A.3 four-conv BN/ReLU score and box branches. Pair "
            "'paper' with --proposal-nms-mode rotated --proposal-nms-radius-m 0.1."
        ),
    )
    parser.add_argument("--num-queries", type=int, default=600)
    parser.add_argument("--num-refinement-blocks", type=int, default=3)
    parser.add_argument("--deformable-radius-m", type=float, default=2.0)
    parser.add_argument("--attention-type", choices=("deformable", "center"), default="deformable")
    parser.add_argument("--lidar-attention-anchor", choices=("current", "time"), default="current")
    parser.add_argument(
        "--detach-pose-refinement",
        action="store_true",
        help=(
            "Deprecated/no-op: pose-gradient detach is now on by default "
            "(paper-faithful). Kept for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--no-detach-pose-refinement",
        action="store_true",
        help=(
            "Ablation: let pose gradients flow between refinement blocks. The "
            "paper stops them, so the default is detach=on."
        ),
    )
    parser.add_argument("--motion-balanced-sampling", action="store_true")
    parser.add_argument("--motion-static-weight", type=float, default=1.0)
    parser.add_argument("--motion-moving-weight", type=float, default=3.0)
    parser.add_argument("--motion-fast-weight", type=float, default=6.0)
    parser.add_argument("--query-velocity-source", choices=("none", "gt_match"), default="none")
    parser.add_argument("--query-velocity-match-radius-m", type=float, default=2.0)
    parser.add_argument(
        "--forecast-base-mode",
        choices=("current", "constant_velocity", "learned_velocity"),
        default="current",
    )
    parser.add_argument("--forecast-velocity-frame", choices=("ego", "object"), default="ego")
    parser.add_argument(
        "--map-encoder-type",
        choices=("graph", "mlp"),
        default="graph",
        help="Map token encoder: 'graph' = runtime kNN-graph message passing, 'mlp' = legacy per-token MLP.",
    )
    parser.add_argument("--map-graph-k", type=int, default=8, help="kNN neighbors for the graph map encoder.")
    parser.add_argument("--map-graph-layers", type=int, default=2, help="Message-passing layers for the graph map encoder.")
    parser.add_argument("--gt-history-occupancy", action="store_true")
    parser.add_argument("--gt-history-occupancy-radius-m", type=float, default=0.5)
    parser.add_argument("--dense-history-motion-weight", type=float, default=0.0)
    parser.add_argument("--dense-history-motion-flow-weight", type=float, default=1.0)
    parser.add_argument(
        "--occ-aux-future",
        action="store_true",
        help="M2: extend the dense occupancy/flow aux head to future channels "
        "(target = concatenated past+future track). Requires "
        "--dense-history-motion-weight > 0.",
    )
    parser.add_argument(
        "--traj-occ-consistency-weight",
        type=float,
        default=0.0,
        help="M2: weight on the trajectory<->occupancy coverage consistency loss. "
        "0 disables.",
    )
    parser.add_argument(
        "--traj-occ-mode-weighting",
        choices=("prob", "uniform"),
        default="prob",
        help="M2: how future modes are combined into the soft trajectory union for "
        "the occupancy consistency loss. 'prob' weights each mode by its softmax "
        "probability (marginal); 'uniform' gives every mode equal mass.",
    )
    parser.add_argument(
        "--grad-calibration",
        action="store_true",
        help="Diagnostic: on full-metric steps, log per-term gradient-norm "
        "contributions (weight * ||grad(term)|| at the shared BEV trunk) to "
        "calibrate auxiliary loss weights. Local JSONL/console only, not Comet.",
    )
    parser.add_argument(
        "--aux-weight-controller",
        action="store_true",
        help="Online GradNorm-style controller: every --aux-controller-interval "
        "steps (after warmup), rebind each aux weight toward its --*-target-frac "
        "fraction of the forecast term's grad norm at the shared trunk. Implies "
        "--grad-calibration.",
    )
    parser.add_argument("--aux-controller-interval", type=int, default=250)
    parser.add_argument("--aux-controller-warmup", type=int, default=500)
    parser.add_argument("--aux-controller-alpha", type=float, default=0.5)
    parser.add_argument("--aux-controller-beta", type=float, default=0.5)
    parser.add_argument("--aux-weight-min", type=float, default=0.005)
    parser.add_argument("--aux-weight-max", type=float, default=1.5)
    parser.add_argument("--past-target-frac", type=float, default=0.0)
    parser.add_argument("--motion-target-frac", type=float, default=0.0)
    parser.add_argument("--yaw-target-frac", type=float, default=0.0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print train progress every N steps. Full diagnostics still print on metrics steps.",
    )
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="Compute full train diagnostics every N steps; other logged steps use scalar losses only.",
    )
    parser.add_argument(
        "--finite-check-every",
        type=int,
        default=1,
        help="Scan all trainable parameters for finite values every N steps.",
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Print train-loop phase timing windows on train log steps.",
    )
    parser.add_argument(
        "--profile-timing-sync",
        action="store_true",
        help="Synchronize CUDA before each timing mark for accurate but slower GPU phase timing.",
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--comet", action="store_true", help="Mirror local JSONL metrics to Comet.")
    parser.add_argument("--comet-project", default=None)
    parser.add_argument("--comet-workspace", default=None)
    parser.add_argument("--comet-run-name", default=None)
    parser.add_argument(
        "--comet-tags",
        nargs="*",
        default=None,
        help="Optional Comet tags. In JSON this may be a list of strings.",
    )
    parser.add_argument("--experiment-log", default=None)
    args = parser.parse_args()
    _merge_experiment_args(args, parser)
    if args.train_cache is None:
        raise SystemExit("--train-cache is required unless provided by --experiment-json")
    train_cached(
        args.train_cache,
        val_cache=args.val_cache,
        steps=args.steps,
        batch_size=args.batch_size,
        max_points=args.max_points,
        max_map_tokens=args.max_map_tokens,
        lr=args.lr,
        roi=tuple(args.roi),
        voxel_size=args.voxel_size,
        eval_every=args.eval_every,
        val_batches=args.val_batches,
        initial_eval=not args.no_initial_eval,
        checkpoint_path=args.checkpoint_path,
        resume_checkpoint=args.resume_checkpoint,
        resume_weights_only=args.resume_weights_only,
        save_best_by=args.save_best_by,
        save_best_metric=args.save_best_metric,
        save_best_mode=args.save_best_mode,
        save_every_eval=args.save_every_eval,
        save_every_steps=args.save_every_steps,
        keep_last_n=args.keep_last_n,
        save_best_forecast_metric=args.save_best_forecast_metric,
        device=args.device,
        objective=args.objective,
        proposal_mode=args.proposal_mode,
        proposal_noise_std_m=args.proposal_noise_std_m,
        forecast_weight=args.forecast_weight,
        lr_warmup_steps=args.lr_warmup_steps,
        forecast_weight_warmup_steps=args.forecast_weight_warmup_steps,
        grad_clip_max_norm=args.grad_clip_max_norm,
        lr_decay=args.lr_decay,
        lr_min_scale=args.lr_min_scale,
        lr_decay_total_steps=args.lr_decay_total_steps,
        grad_accum_steps=args.grad_accum_steps,
        proposal_heatmap_weight=args.proposal_heatmap_weight,
        proposal_box_weight=args.proposal_box_weight,
        proposal_giou_weight=args.proposal_giou_weight,
        detection_box_l1_weight=args.detection_box_l1_weight,
        detection_giou_weight=args.detection_giou_weight,
        detection_class_weight=args.detection_class_weight,
        detection_z_weight=args.detection_z_weight,
        detection_height_weight=args.detection_height_weight,
        forecast_gate_radius_m=args.forecast_gate_radius_m,
        forecast_loss_type=args.forecast_loss_type,
        forecast_top_k=args.forecast_top_k,
        ewta_anneal_steps=args.ewta_anneal_steps,
        forecast_gate_warmup_steps=args.forecast_gate_warmup_steps,
        forecast_gate_warmup_radius_m=args.forecast_gate_warmup_radius_m,
        velocity_weight=args.velocity_weight,
        yaw_consistency_weight=args.yaw_consistency_weight,
        yaw_consistency_vthresh_m=args.yaw_consistency_vthresh_m,
        past_weight=args.past_weight,
        predict_past=args.predict_past,
        init_weight=args.init_weight,
        proposal_nms_radius_m=args.proposal_nms_radius_m,
        proposal_nms_mode=args.proposal_nms_mode,
        proposal_center_mode=args.proposal_center_mode,
        proposal_peak_filter=not args.no_proposal_peak_filter,
        support_filter_radius_m=args.support_filter_radius_m,
        deterministic_seed=args.deterministic_seed,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        eval_train_mode=args.eval_train_mode,
        norm_type=args.norm_type,
        proposal_head_type=args.proposal_head_type,
        num_queries=args.num_queries,
        num_refinement_blocks=args.num_refinement_blocks,
        deformable_radius_m=args.deformable_radius_m,
        attention_type=args.attention_type,
        lidar_attention_anchor=args.lidar_attention_anchor,
        detach_pose_refinement=not args.no_detach_pose_refinement,
        motion_balanced_sampling=args.motion_balanced_sampling,
        motion_static_weight=args.motion_static_weight,
        motion_moving_weight=args.motion_moving_weight,
        motion_fast_weight=args.motion_fast_weight,
        query_velocity_source=args.query_velocity_source,
        query_velocity_match_radius_m=args.query_velocity_match_radius_m,
        forecast_base_mode=args.forecast_base_mode,
        forecast_velocity_frame=args.forecast_velocity_frame,
        map_encoder_type=args.map_encoder_type,
        map_graph_k=args.map_graph_k,
        map_graph_layers=args.map_graph_layers,
        gt_history_occupancy=args.gt_history_occupancy,
        gt_history_occupancy_radius_m=args.gt_history_occupancy_radius_m,
        dense_history_motion_weight=args.dense_history_motion_weight,
        dense_history_motion_flow_weight=args.dense_history_motion_flow_weight,
        occ_aux_future=args.occ_aux_future,
        traj_occ_consistency_weight=args.traj_occ_consistency_weight,
        traj_occ_mode_weighting=args.traj_occ_mode_weighting,
        grad_calibration=args.grad_calibration,
        aux_weight_controller=args.aux_weight_controller,
        aux_controller_interval=args.aux_controller_interval,
        aux_controller_warmup=args.aux_controller_warmup,
        aux_controller_alpha=args.aux_controller_alpha,
        aux_controller_beta=args.aux_controller_beta,
        aux_weight_min=args.aux_weight_min,
        aux_weight_max=args.aux_weight_max,
        past_target_frac=args.past_target_frac,
        motion_target_frac=args.motion_target_frac,
        yaw_target_frac=args.yaw_target_frac,
        run_dir=args.run_dir,
        comet=args.comet,
        comet_project=args.comet_project,
        comet_workspace=args.comet_workspace,
        comet_run_name=args.comet_run_name,
        comet_tags=args.comet_tags,
        log_every=args.log_every,
        metrics_every=args.metrics_every,
        finite_check_every=args.finite_check_every,
        profile_timing=args.profile_timing,
        profile_timing_sync=args.profile_timing_sync,
        experiment_log=args.experiment_log,
    )

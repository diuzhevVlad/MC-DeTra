"""Detection/forecasting metric computation for cached-Waymo training.

Pure metric helpers and ``compute_metrics``/``_add_forecast_bin_ratios``, moved
verbatim out of ``detra_repro.train_cached`` during the 2026 refactor and
re-exported there for backward compatibility. No training state; inputs are
prediction/target tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from detra_repro.losses import current_boxes_from_pred, waymo_types_to_class_indices


def _recall_at_radii(
    pred_xy: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    radii: tuple[float, ...] = (1.0, 2.0, 4.0),
    prefix: str = "proposal",
) -> dict[str, torch.Tensor]:
    """Measure GT recall if any prediction center is within each radius."""

    out: dict[str, torch.Tensor] = {}
    device = gt_boxes.device
    total_hits = {r: gt_boxes.new_tensor(0.0) for r in radii}
    total_gt = gt_boxes.new_tensor(0.0)

    for batch_idx in range(gt_boxes.shape[0]):
        valid = gt_valid[batch_idx]
        if not valid.any():
            continue
        gt_xy = gt_boxes[batch_idx, valid, :2]
        dist = torch.cdist(gt_xy, pred_xy[batch_idx])
        min_dist = dist.min(dim=1).values
        total_gt = total_gt + valid.float().sum()
        for radius in radii:
            total_hits[radius] = total_hits[radius] + (min_dist < radius).float().sum()

    denom = total_gt.clamp_min(1.0)
    for radius in radii:
        out[f"{prefix}_recall_{int(radius)}m"] = total_hits[radius] / denom
    out[f"{prefix}_gt_count"] = total_gt.to(device)
    return out


def _topk_recall_at_radii(
    pred_xy: torch.Tensor,
    scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    topk: int,
    radii: tuple[float, ...] = (1.0, 2.0, 4.0),
    prefix: str = "topk_refined",
) -> dict[str, torch.Tensor]:
    """Measure recall using only the top-scored prediction centers."""

    k = min(topk, pred_xy.shape[1])
    order = scores.argsort(dim=1, descending=True)[:, :k]
    gather_idx = order[..., None].expand(-1, -1, 2)
    top_xy = pred_xy.gather(dim=1, index=gather_idx)
    return _recall_at_radii(top_xy, gt_boxes, gt_valid, radii=radii, prefix=prefix)


def _normalized_histogram(values: torch.Tensor, num_bins: int, prefix: str) -> dict[str, torch.Tensor]:
    """Return normalized per-bin histogram metrics for compact scalar logging."""

    hist = torch.bincount(values.reshape(-1), minlength=num_bins).float()
    hist = hist / hist.sum().clamp_min(1.0)
    return {f"{prefix}_{idx}": hist[idx] for idx in range(num_bins)}


@torch.no_grad()
def compute_metrics(
    pred: dict[str, torch.Tensor] | None,
    losses: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    recall_radius_m: float = 2.0,
) -> dict[str, float]:
    """Compute validation metrics from final matching.

    Metrics:
        - ``loss``: total validation loss.
        - ``ade``: matched winner-mode average displacement error.
        - ``fde``: matched winner-mode final displacement error.
        - ``det_recall_2m``: matched detections whose center is within 2m.

    These are not paper metrics yet; they are early training health metrics.
    """

    metrics: dict[str, torch.Tensor] = {"loss": losses["loss"].detach()}
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            metrics[key] = value.detach()

    gt_boxes = targets["gt_boxes"]
    gt_valid = targets["gt_valid"]
    gt_future_mask = targets["gt_future_mask"]
    metrics.update(
        _recall_at_radii(
            targets["proposal_boxes"][..., :2],
            gt_boxes,
            gt_valid,
            prefix="proposal",
        )
    )
    support_gt_valid = targets["support_gt_valid_2m"]
    metrics.update(
        _recall_at_radii(
            targets["proposal_boxes"][..., :2],
            gt_boxes,
            support_gt_valid,
            prefix="support_proposal",
        )
    )
    # raw_proposal / heatmap recalls are diagnostic-only; present only when
    # forward_batch ran with compute_diagnostics=True.
    if "raw_proposal_boxes" in targets:
        metrics.update(
            _recall_at_radii(
                targets["raw_proposal_boxes"][..., :2],
                gt_boxes,
                gt_valid,
                prefix="raw_proposal",
            )
        )
    if "heatmap_xy" in targets:
        metrics.update(
            _recall_at_radii(
                targets["heatmap_xy"],
                gt_boxes,
                gt_valid,
                prefix="heatmap",
            )
        )
        metrics.update(
            _recall_at_radii(
                targets["heatmap_xy"],
                gt_boxes,
                support_gt_valid,
                prefix="support_heatmap",
            )
        )
    metrics["gt_count"] = gt_valid.float().sum()
    metrics["train_gt_count"] = targets["train_gt_valid"].float().sum()
    metrics["support_gt_count"] = support_gt_valid.float().sum()
    metrics["future_track_count"] = (gt_valid & gt_future_mask.any(dim=-1)).float().sum()
    if "map_mask" in targets:
        metrics["map_token_count"] = targets["map_mask"].float().sum()
    if "history_occupancy_count" in targets:
        metrics["history_occupancy_count"] = targets["history_occupancy_count"].detach()
    if "dense_history_motion_cells" in targets:
        metrics["dense_history_motion_cells"] = targets["dense_history_motion_cells"].detach()

    aligned = losses.get("aligned")
    if pred is None or aligned is None:
        metrics.update(
            {
                "ade": gt_boxes.new_tensor(0.0),
                "fde": gt_boxes.new_tensor(0.0),
                "det_recall_2m": gt_boxes.new_tensor(0.0),
                "matched_center_dist": gt_boxes.new_tensor(0.0),
            }
        )
        return {key: float(value.detach()) for key, value in metrics.items()}

    object_targets = aligned["query_object_targets"].bool()
    future_mask = aligned["query_gt_future_mask"]
    valid_query = object_targets & future_mask.any(dim=-1)
    if not valid_query.any():
        metrics.update(
            {
                "ade": gt_boxes.new_tensor(0.0),
                "fde": gt_boxes.new_tensor(0.0),
                "det_recall_2m": gt_boxes.new_tensor(0.0),
                "matched_center_dist": gt_boxes.new_tensor(0.0),
            }
        )
        return {key: float(value.detach()) for key, value in metrics.items()}

    pred_xy = pred.get("future_xy")
    if pred_xy is None:
        pred_xy = pred["poses"][:, :, :, 1:, :2]
    gt_future = aligned["query_gt_future"]
    mask = future_mask[:, :, None, :, None].float()
    errors = (pred_xy - gt_future[:, :, None]).norm(dim=-1) * mask.squeeze(-1)
    denom = future_mask[:, :, None].sum(dim=-1).clamp_min(1)
    ade_by_mode = errors.sum(dim=-1) / denom
    winner = ade_by_mode.argmin(dim=2)

    b, n, _, t, _ = pred_xy.shape
    gather_idx = winner[:, :, None, None, None].expand(b, n, 1, t, 2)
    best_xy = pred_xy.gather(2, gather_idx).squeeze(2)
    point_error = (best_xy - gt_future).norm(dim=-1)
    ade = (point_error[future_mask & valid_query[:, :, None]]).mean()

    current_gt_xy = aligned["query_gt_boxes"][..., :2]
    last_idx = future_mask.long().sum(dim=-1).clamp_min(1) - 1
    fde_map = torch.zeros_like(future_mask, dtype=point_error.dtype)[..., 0]
    gt_final_disp_map = torch.zeros_like(fde_map)
    pred_final_disp_map = torch.zeros_like(fde_map)
    fde_vals = []
    gt_disp_vals = []
    pred_disp_vals = []
    for batch_idx, query_idx in valid_query.nonzero(as_tuple=False):
        bi = int(batch_idx)
        qi = int(query_idx)
        li = int(last_idx[bi, qi])
        query_fde = point_error[bi, qi, li]
        query_gt_disp = (gt_future[bi, qi, li] - current_gt_xy[bi, qi]).norm()
        query_pred_disp = (best_xy[bi, qi, li] - current_gt_xy[bi, qi]).norm()
        fde_map[bi, qi] = query_fde
        gt_final_disp_map[bi, qi] = query_gt_disp
        pred_final_disp_map[bi, qi] = query_pred_disp
        fde_vals.append(query_fde)
        gt_disp_vals.append(query_gt_disp)
        pred_disp_vals.append(query_pred_disp)
    fde = torch.stack(fde_vals).mean() if fde_vals else ade.new_tensor(0.0)
    gt_final_disp = torch.stack(gt_disp_vals).mean() if gt_disp_vals else ade.new_tensor(0.0)
    pred_final_disp = torch.stack(pred_disp_vals).mean() if pred_disp_vals else ade.new_tensor(0.0)
    moving_query = valid_query & (gt_final_disp_map > 2.0)
    if moving_query.any():
        moving_point_mask = future_mask & moving_query[:, :, None]
        moving_ade = point_error[moving_point_mask].mean()
        moving_fde = fde_map[moving_query].mean()
        moving_gt_final_disp = gt_final_disp_map[moving_query].mean()
        moving_pred_final_disp = pred_final_disp_map[moving_query].mean()
    else:
        moving_ade = ade.new_tensor(0.0)
        moving_fde = ade.new_tensor(0.0)
        moving_gt_final_disp = ade.new_tensor(0.0)
        moving_pred_final_disp = ade.new_tensor(0.0)

    # --- K=6 (best-of-modes) metrics, the paper's reported forecasting numbers. ---
    # ade_by_mode is [B,N,F]; the K=6 minADE picks the best mode per query (vs the
    # K=1 path above, which uses the winner mode by predicted probability).
    last_step = (future_mask.long().sum(dim=-1).clamp_min(1) - 1)  # [B,N]
    # Per-mode FDE: error at each query's last valid step, for every mode.
    fde_by_mode = errors.gather(
        3, last_step[:, :, None, None].expand(b, n, pred_xy.shape[2], 1)
    ).squeeze(3)  # [B,N,F]
    if valid_query.any():
        vq = valid_query
        minade_k6 = ade_by_mode.min(dim=2).values[vq].mean()
        minfde_k6 = fde_by_mode.min(dim=2).values[vq].mean()
        # Miss rate K=6: missed if the best mode's FDE exceeds 2 m.
        mr_k6 = (fde_by_mode.min(dim=2).values[vq] > 2.0).float().mean()
        minade_k1 = ade_by_mode.gather(2, winner.unsqueeze(2)).squeeze(2)[vq].mean()
        minfde_k1 = fde_by_mode.gather(2, winner.unsqueeze(2)).squeeze(2)[vq].mean()
    else:
        minade_k6 = ade.new_tensor(0.0)
        minfde_k6 = ade.new_tensor(0.0)
        mr_k6 = ade.new_tensor(0.0)
        minade_k1 = ade.new_tensor(0.0)
        minfde_k1 = ade.new_tensor(0.0)

    # Displacement-bin sums are reduced globally in ``evaluate``. This avoids
    # treating batches with no actors in a bin as zero-error batches.
    bin_specs = (
        ("static_0_0p5", 0.0, 0.5),
        ("near_static_0p5_2", 0.5, 2.0),
        ("moving_2_10", 2.0, 10.0),
        ("fast_10p", 10.0, None),
    )
    bin_metrics: dict[str, torch.Tensor] = {}
    for name, lower, upper in bin_specs:
        query_mask = valid_query & (gt_final_disp_map >= lower)
        if upper is not None:
            query_mask = query_mask & (gt_final_disp_map < upper)
        point_mask = future_mask & query_mask[:, :, None]
        query_count = query_mask.float().sum()
        point_count = point_mask.float().sum()
        prefix = f"dispbin_{name}"
        bin_metrics[f"{prefix}_query_count"] = query_count.detach()
        bin_metrics[f"{prefix}_point_count"] = point_count.detach()
        if query_mask.any():
            bin_metrics[f"{prefix}_fde_sum"] = fde_map[query_mask].sum().detach()
            bin_metrics[f"{prefix}_gt_final_disp_sum"] = gt_final_disp_map[query_mask].sum().detach()
            bin_metrics[f"{prefix}_pred_final_disp_sum"] = pred_final_disp_map[query_mask].sum().detach()
        else:
            bin_metrics[f"{prefix}_fde_sum"] = ade.new_tensor(0.0)
            bin_metrics[f"{prefix}_gt_final_disp_sum"] = ade.new_tensor(0.0)
            bin_metrics[f"{prefix}_pred_final_disp_sum"] = ade.new_tensor(0.0)
        if point_mask.any():
            bin_metrics[f"{prefix}_ade_sum"] = point_error[point_mask].sum().detach()
        else:
            bin_metrics[f"{prefix}_ade_sum"] = ade.new_tensor(0.0)

    horizon_metrics: dict[str, torch.Tensor] = {}
    future_valid = future_mask & valid_query[:, :, None]
    gt_step_disp = (gt_future - current_gt_xy[:, :, None]).norm(dim=-1)
    pred_step_disp = (best_xy - current_gt_xy[:, :, None]).norm(dim=-1)
    for horizon_idx in range(point_error.shape[-1]):
        step_mask = future_valid[..., horizon_idx]
        if step_mask.any():
            suffix = horizon_idx + 1
            horizon_metrics[f"ade_t{suffix}"] = point_error[..., horizon_idx][step_mask].mean().detach()
            horizon_metrics[f"gt_disp_t{suffix}"] = gt_step_disp[..., horizon_idx][step_mask].mean().detach()
            horizon_metrics[f"pred_disp_t{suffix}"] = pred_step_disp[..., horizon_idx][step_mask].mean().detach()

    pred_boxes = current_boxes_from_pred(pred)
    gt_boxes = aligned["query_gt_boxes"]
    center_error = (pred_boxes[..., :2] - gt_boxes[..., :2]).norm(dim=-1)
    det_recall = (center_error[object_targets] < recall_radius_m).float().mean()

    # --- Per-class forecasting (K=6) + detection recall (vehicle/ped/cyclist). ---
    per_class_metrics: dict[str, torch.Tensor] = {}
    gt_types_q = aligned.get("query_gt_types")
    if gt_types_q is not None:
        class_ids = waymo_types_to_class_indices(gt_types_q)  # [B,N] -> 0/1/2
        minade_best = ade_by_mode.min(dim=2).values
        minfde_best = fde_by_mode.min(dim=2).values
        for cname, cid in (("veh", 0), ("ped", 1), ("cyc", 2)):
            cls_q = valid_query & (class_ids == cid)
            if bool(cls_q.any()):
                per_class_metrics[f"minade_k6_{cname}"] = minade_best[cls_q].mean().detach()
                per_class_metrics[f"minfde_k6_{cname}"] = minfde_best[cls_q].mean().detach()
            per_class_metrics[f"track_count_{cname}"] = cls_q.float().sum().detach()
            det_cls = object_targets & (class_ids == cid)
            if bool(det_cls.any()):
                per_class_metrics[f"det_recall_2m_{cname}"] = (
                    (center_error[det_cls] < recall_radius_m).float().mean().detach()
                )
    object_scores = pred["object_logits"].sigmoid()
    mode_prob = pred["mode_logits"].softmax(dim=-1)
    mode_entropy = -(mode_prob * mode_prob.clamp_min(1e-8).log()).sum(dim=-1)
    top_mode = mode_prob.argmax(dim=-1)
    top_prob = mode_prob.max(dim=-1).values
    wta_is_top = (winner == top_mode)[valid_query].float()
    winner_prob = mode_prob.gather(-1, winner[..., None]).squeeze(-1)
    future_log_scale = pred.get("future_log_scale")
    if future_log_scale is None:
        forecast_scale = ade.new_tensor(0.0)
    else:
        forecast_scale = future_log_scale.exp()[valid_query].mean()
    metrics.update(
        _recall_at_radii(
            pred_boxes[..., :2],
            targets["gt_boxes"],
            targets["gt_valid"],
            prefix="refined",
        )
    )
    metrics.update(
        _topk_recall_at_radii(
            pred_boxes[..., :2],
            object_scores,
            targets["gt_boxes"],
            targets["gt_valid"],
            topk=32,
            prefix="top32_refined",
        )
    )
    order = object_scores.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(order.shape[1], device=order.device)[None].expand_as(order))
    matched_ranks = ranks[object_targets]
    metrics.update(
        {
            "ade": ade.detach(),
            "fde": fde.detach(),
            "det_recall_2m": det_recall.detach(),
            "matched_center_dist": center_error[object_targets].mean().detach(),
            "matched_z_err": (pred_boxes[..., 2] - gt_boxes[..., 2]).abs()[object_targets].mean().detach(),
            "matched_height_err": (pred_boxes[..., 5] - gt_boxes[..., 5]).abs()[object_targets].mean().detach(),
            "matched_score_rank": matched_ranks.float().mean().detach(),
            "score_mean": object_scores.mean().detach(),
            "score_max": object_scores.max().detach(),
            "gt_final_disp": gt_final_disp.detach(),
            "pred_final_disp": pred_final_disp.detach(),
            "moving_track_count": moving_query.float().sum().detach(),
            "moving_ade": moving_ade.detach(),
            "moving_fde": moving_fde.detach(),
            "moving_gt_final_disp": moving_gt_final_disp.detach(),
            "moving_pred_final_disp": moving_pred_final_disp.detach(),
            "minade_k1": minade_k1.detach(),
            "minfde_k1": minfde_k1.detach(),
            "minade_k6": minade_k6.detach(),
            "minfde_k6": minfde_k6.detach(),
            "mr_k6": mr_k6.detach(),
            "mode_entropy": mode_entropy[valid_query].mean().detach(),
            "mode_entropy_normalized": (
                mode_entropy[valid_query].mean()
                / pred["mode_logits"].new_tensor(
                    float(pred["mode_logits"].shape[-1])
                ).log()
            ).detach(),
            "mode_effective_count": mode_entropy[valid_query].mean().exp().detach(),
            "mode_top_prob": top_prob[valid_query].mean().detach(),
            "mode_wta_is_top_rate": wta_is_top.mean().detach(),
            "mode_wta_prob": winner_prob[valid_query].mean().detach(),
            "forecast_scale": forecast_scale.detach(),
        }
    )
    metrics.update(
        _normalized_histogram(
            top_mode[valid_query],
            pred["mode_logits"].shape[-1],
            prefix="mode_top_hist",
        )
    )
    metrics.update(
        _normalized_histogram(
            winner[valid_query],
            pred["mode_logits"].shape[-1],
            prefix="mode_wta_hist",
        )
    )
    metrics.update(horizon_metrics)
    metrics.update(bin_metrics)
    metrics.update(per_class_metrics)
    return {key: float(value.detach()) for key, value in metrics.items()}


def _add_forecast_bin_ratios(metrics: dict[str, float]) -> dict[str, float]:
    """Add derived global ADE/FDE/disp ratios from displacement-bin sums."""

    for name in (
        "static_0_0p5",
        "near_static_0p5_2",
        "moving_2_10",
        "fast_10p",
    ):
        prefix = f"dispbin_{name}"
        point_count = metrics.get(f"{prefix}_point_count", 0.0)
        query_count = metrics.get(f"{prefix}_query_count", 0.0)
        if point_count > 0:
            metrics[f"{prefix}_ade"] = metrics.get(f"{prefix}_ade_sum", 0.0) / point_count
        if query_count > 0:
            metrics[f"{prefix}_fde"] = metrics.get(f"{prefix}_fde_sum", 0.0) / query_count
            metrics[f"{prefix}_gt_final_disp"] = (
                metrics.get(f"{prefix}_gt_final_disp_sum", 0.0) / query_count
            )
            metrics[f"{prefix}_pred_final_disp"] = (
                metrics.get(f"{prefix}_pred_final_disp_sum", 0.0) / query_count
            )
    return metrics

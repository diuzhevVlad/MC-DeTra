"""Trajectory<->occupancy consistency (M2).

Two train-only, inference-safe pieces:

- ``rasterize_gt_motion_targets`` builds a combined past+future dense BEV
  occupancy/flow target by concatenating the GT history and future tracks and
  reusing the existing history rasterizer. Paired with the encoder's
  ``motion_aux_head`` it supervises occupancy over both past and future (2A).

- ``trajectory_occupancy_consistency_loss`` requires the decoded per-object
  trajectories (current box + past + future waypoints) to "explain" the predicted
  occupancy field: every occupied cell should be covered by some trajectory. The
  occupancy field is the (detached) teacher, so the gradient flows only into the
  trajectory heads, pulling waypoints toward occupied regions (2B). The
  trajectory raster uses a differentiable bilinear splat so the loss is smooth in
  the predicted waypoint coordinates.
"""

from __future__ import annotations

import torch

from detra_repro.config import DataConfig
from detra_repro.training.oracle import rasterize_gt_history_motion_targets


def rasterize_gt_motion_targets(
    gt_history: torch.Tensor,
    gt_history_mask: torch.Tensor,
    gt_future: torch.Tensor,
    gt_future_mask: torch.Tensor,
    gt_valid: torch.Tensor,
    data_cfg: DataConfig,
    output_stride: int = 4,
    radius_m: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Combined past+future dense occupancy/flow targets (2A).

    Concatenates the history and future tracks along the time axis and reuses
    :func:`rasterize_gt_history_motion_targets`; the result has ``H + T_future``
    time channels (history oldest->newest, then future). The occupancy head width
    must equal ``H + T_future`` (set ``history_occupancy_channels`` accordingly).
    """

    track = torch.cat([gt_history, gt_future], dim=2)
    track_mask = torch.cat([gt_history_mask, gt_future_mask], dim=2)
    return rasterize_gt_history_motion_targets(
        track,
        track_mask,
        gt_valid,
        data_cfg,
        output_stride=output_stride,
        radius_m=radius_m,
    )


def _bilinear_soft_occupancy(
    centers_xy: torch.Tensor,
    valid: torch.Tensor,
    batch_size: int,
    grid_h: int,
    grid_w: int,
    x_min: float,
    y_min: float,
    cell: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable bilinear splat of points into a ``[B, grid_h, grid_w]`` map.

    Each valid point distributes a mass (``weights`` if given, else unit) to its
    four neighboring cells with bilinear weights (smooth in ``xy``). Mass
    accumulates across points, so splatting multiple trajectory modes into the
    same map is a differentiable soft union (soft-OR). The returned soft occupancy
    is ``1 - exp(-accumulated_mass)`` in ``[0, 1)``, differentiable w.r.t. the
    point coordinates through the bilinear weights (and w.r.t. ``weights``).
    """

    device = centers_xy.device
    acc = centers_xy.new_zeros((batch_size, grid_h, grid_w))
    flat = acc.view(-1)

    # batch index per point
    b_idx = torch.arange(batch_size, device=device).view(-1, *([1] * (centers_xy.dim() - 2)))
    b_idx = b_idx.expand(centers_xy.shape[:-1])

    sel = valid
    if not sel.any():
        return 1.0 - torch.exp(-acc)
    pts = centers_xy[sel]  # [E, 2]
    b_e = b_idx[sel]  # [E]
    w_e = weights[sel] if weights is not None else None  # [E]

    gx = (pts[:, 0] - x_min) / cell - 0.5
    gy = (pts[:, 1] - y_min) / cell - 0.5
    x0 = torch.floor(gx)
    y0 = torch.floor(gy)
    wx = gx - x0
    wy = gy - y0
    x0l = x0.long()
    y0l = y0.long()

    for dy, dx, weight in (
        (0, 0, (1 - wy) * (1 - wx)),
        (0, 1, (1 - wy) * wx),
        (1, 0, wy * (1 - wx)),
        (1, 1, wy * wx),
    ):
        mass = weight if w_e is None else weight * w_e
        xi = x0l + dx
        yi = y0l + dy
        in_bounds = (xi >= 0) & (xi < grid_w) & (yi >= 0) & (yi < grid_h)
        if not in_bounds.any():
            continue
        flat_idx = (b_e * grid_h + yi.clamp(0, grid_h - 1)) * grid_w + xi.clamp(0, grid_w - 1)
        flat.index_put_((flat_idx[in_bounds],), mass[in_bounds], accumulate=True)

    return 1.0 - torch.exp(-acc)


def trajectory_occupancy_consistency_loss(
    pred: dict[str, torch.Tensor],
    motion_aux: dict[str, torch.Tensor] | None,
    object_targets: torch.Tensor,
    data_cfg: DataConfig,
    output_stride: int = 4,
    mode_weighting: str = "prob",
) -> dict[str, torch.Tensor]:
    """Coverage loss: predicted trajectories must explain occupied cells (2B).

    Uses the detached, time-collapsed occupancy field as a teacher and penalizes
    occupied cells not covered by the union of decoded trajectory waypoints
    (current box center + past + every future mode) of true-positive queries.
    One-directional (occupancy is detached), so the gradient pulls waypoints
    toward occupied regions without the occupancy head chasing the trajectories.

    The occupancy field is unimodal: it is the marginal "where could this actor
    be" over the multimodal future. So all ``F`` future modes are splatted into a
    single soft union (not the mode-mean, which threads between modes through free
    space). With ``mode_weighting="prob"`` each mode's mass is weighted by its
    softmax probability (``mode_logits``), making the union a probabilistic
    marginal ``sum_k p_k * raster(mode_k)``; ``"uniform"`` gives every mode equal
    mass. WTA is intentionally not used here: it would collapse the union back to
    one mode and defeat the marginal match.
    """

    if motion_aux is None or "occupancy_logits" not in motion_aux:
        zero = pred["poses"].sum() * 0.0
        return {"loss_traj_occ": zero, "traj_occ_cells": zero}

    logits = motion_aux["occupancy_logits"]  # [B, C, Hs, Ws]
    b, _, grid_h, grid_w = logits.shape
    teacher = torch.sigmoid(logits).amax(dim=1).detach()  # [B, Hs, Ws]

    x_min, y_min, _, _ = data_cfg.roi_m
    cell = data_cfg.voxel_size_m * output_stride

    # Gather decoded waypoints per query with a per-point mass. Current center and
    # past are single-mode (mass 1); future keeps every mode (soft union),
    # optionally weighted by mode probability.
    current = pred["poses"][:, :, :, 0, :2].mean(dim=2)  # [B, N, 2]; modes coincide at t=0
    n = current.shape[1]
    point_list = [current[:, :, None, :]]
    weight_list = [current.new_ones((b, n, 1))]
    if "past_xy" in pred:
        past = pred["past_xy"]  # [B, N, H_past, 2]
        point_list.append(past)
        weight_list.append(past.new_ones(past.shape[:-1]))
    if "future_xy" in pred:
        fut = pred["future_xy"]  # [B, N, F, T-1, 2]
        bb, nn, f, t, _ = fut.shape
        point_list.append(fut.reshape(bb, nn, f * t, 2))
        if mode_weighting == "prob" and "mode_logits" in pred:
            probs = torch.softmax(pred["mode_logits"], dim=-1)  # [B, N, F]
            weight_list.append(probs[:, :, :, None].expand(bb, nn, f, t).reshape(bb, nn, f * t))
        else:
            weight_list.append(fut.new_ones((bb, nn, f * t)))
    points = torch.cat(point_list, dim=2)  # [B, N, K, 2]
    point_mass = torch.cat(weight_list, dim=2)  # [B, N, K]

    k = points.shape[2]
    valid = object_targets.bool()[:, :, None].expand(b, n, k)
    traj_occ = _bilinear_soft_occupancy(
        points, valid, b, grid_h, grid_w, x_min, y_min, cell, weights=point_mass
    )  # [B, Hs, Ws]

    denom = teacher.sum().clamp_min(1.0)
    loss = (teacher * (1.0 - traj_occ)).sum() / denom
    return {"loss_traj_occ": loss, "traj_occ_cells": (teacher > 0.5).float().sum()}

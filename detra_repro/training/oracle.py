"""GT-derived oracle and diagnostic helpers for training.

These functions use ground-truth labels (actor histories, GT-matched velocity,
GT occupancy rasters) to build *training-only* targets, sampling weights, and
diagnostics. None of them are paper-faithful inference inputs: the DeTra
reproduction must infer motion from multi-sweep LiDAR, not GT actor history.
They are kept apart from the training loop so reportable runs never accidentally
consume GT-oracle signal at inference time.

Moved verbatim out of ``detra_repro.train_cached`` during the 2026 refactor;
re-exported there for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from detra_repro.losses import sigmoid_focal_loss


def motion_balanced_sample_weights(
    dataset: CachedWaymoDataset,
    static_weight: float = 1.0,
    moving_weight: float = 3.0,
    fast_weight: float = 6.0,
    moving_threshold_m: float = 2.0,
    fast_threshold_m: float = 10.0,
) -> torch.Tensor:
    """Return per-sample weights from cached GT future displacement.

    A sample receives ``fast_weight`` when any valid actor's final displacement
    is at least ``fast_threshold_m``, ``moving_weight`` when any actor reaches
    ``moving_threshold_m``, and ``static_weight`` otherwise.
    """

    weights = []
    for path in dataset.files:
        with np.load(path, allow_pickle=True) as data:
            boxes = data["gt_boxes"].astype(np.float32)
            future = data["gt_future"].astype(np.float32)
            future_mask = data["gt_future_mask"].astype(bool)
        max_disp = 0.0
        for obj_idx in range(len(boxes)):
            valid = np.where(future_mask[obj_idx])[0]
            if len(valid) == 0:
                continue
            disp = float(np.linalg.norm(future[obj_idx, valid[-1]] - boxes[obj_idx, :2]))
            max_disp = max(max_disp, disp)
        if max_disp >= fast_threshold_m:
            weights.append(fast_weight)
        elif max_disp >= moving_threshold_m:
            weights.append(moving_weight)
        else:
            weights.append(static_weight)
    return torch.tensor(weights, dtype=torch.float32)


def make_oracle_proposals(
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    num_queries: int,
    noise_std_m: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Create proposal tensors by copying GT boxes into query slots.

    Args:
        gt_boxes: ``[B,M,7]`` ground-truth boxes.
        gt_valid: ``[B,M]`` valid-object mask.
        num_queries: Number of DeTra query slots.
        noise_std_m: Optional Gaussian noise applied to ``x,y`` centers.

    Returns:
        Dict compatible with ``ProposalHead.decode``:
        ``boxes [B,N,7]``, ``poses [B,N,3]``, ``scores [B,N]``.

    Use this only for diagnosis. If oracle proposals overfit but decoded
    proposals do not, the initial proposal stage or proposal-to-query matching
    is the bottleneck.
    """

    b = gt_boxes.shape[0]
    boxes = gt_boxes.new_zeros((b, num_queries, 7))
    scores = gt_boxes.new_zeros((b, num_queries))
    boxes[..., 3:6] = gt_boxes.new_tensor([1.0, 1.0, 1.7])

    for batch_idx in range(b):
        valid_idx = torch.where(gt_valid[batch_idx])[0][:num_queries]
        count = len(valid_idx)
        if count == 0:
            continue
        copied = gt_boxes[batch_idx, valid_idx].clone()
        if noise_std_m > 0:
            copied[:, :2] = copied[:, :2] + torch.randn_like(copied[:, :2]) * noise_std_m
        boxes[batch_idx, :count] = copied
        scores[batch_idx, :count] = 1.0

    poses = boxes[..., [0, 1, 6]]
    return {"boxes": boxes, "poses": poses, "scores": scores}


def gt_history_step_velocity(
    gt_history: torch.Tensor,
    gt_history_mask: torch.Tensor,
    gt_valid: torch.Tensor,
) -> torch.Tensor:
    """Estimate object displacement per cached history step from GT history.

    The newest history slot is the current center. For each object, this returns
    ``current_xy - previous_valid_xy`` when possible, otherwise zero. The units
    match the future target step spacing used by the cache, so this is a
    displacement-per-target-step diagnostic rather than SI velocity.
    """

    b, m, h, _ = gt_history.shape
    velocity = gt_history.new_zeros((b, m, 2))
    if h < 2:
        return velocity

    mask = gt_history_mask.bool()  # [B,M,H]
    steps = torch.arange(h, device=gt_history.device)
    valid_pos = torch.where(mask, steps, torch.full_like(steps, -1))  # [B,M,H]
    last_idx = valid_pos.max(dim=-1).values  # [B,M], -1 if no valid step
    # Second-to-last valid step: re-take the max after masking out the last.
    valid_pos_2 = torch.where(steps == last_idx[..., None], torch.full_like(steps, -1), valid_pos)
    prev_idx = valid_pos_2.max(dim=-1).values  # [B,M], -1 if <2 valid steps

    has_two = gt_valid.bool() & (prev_idx >= 0)  # matches len(valid_steps) >= 2
    last_safe = last_idx.clamp_min(0)
    prev_safe = prev_idx.clamp_min(0)
    current = torch.gather(
        gt_history, 2, last_safe[..., None, None].expand(b, m, 1, 2)
    ).squeeze(2)
    previous = torch.gather(
        gt_history, 2, prev_safe[..., None, None].expand(b, m, 1, 2)
    ).squeeze(2)
    velocity = torch.where(has_two[..., None], current - previous, velocity)
    return velocity


def gt_history_step_velocity_valid(
    gt_history_mask: torch.Tensor,
    gt_valid: torch.Tensor,
) -> torch.Tensor:
    """Return true for objects with at least two valid cached history steps."""

    return gt_valid & (gt_history_mask.bool().sum(dim=-1) >= 2)


def query_velocity_from_gt_match(
    proposal_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    gt_history: torch.Tensor,
    gt_history_mask: torch.Tensor,
    max_distance_m: float = 2.0,
) -> torch.Tensor:
    """Assign GT-history velocity to proposal queries by nearest GT center.

    This is a diagnostic oracle signal. It intentionally uses GT labels before
    the model forward pass to test whether forecasting can improve when motion
    is made explicit.
    """

    gt_velocity = gt_history_step_velocity(gt_history, gt_history_mask, gt_valid)
    query_velocity = proposal_boxes.new_zeros(proposal_boxes.shape[:2] + (2,))
    for batch_idx in range(proposal_boxes.shape[0]):
        valid_idx = torch.where(gt_valid[batch_idx])[0]
        if len(valid_idx) == 0:
            continue
        gt_xy = gt_boxes[batch_idx, valid_idx, :2]
        prop_xy = proposal_boxes[batch_idx, :, :2]
        dist = torch.cdist(prop_xy[None], gt_xy[None]).squeeze(0)
        min_dist, nearest = dist.min(dim=1)
        matched = min_dist < max_distance_m
        if matched.any():
            query_velocity[batch_idx, matched] = gt_velocity[
                batch_idx, valid_idx[nearest[matched]]
            ]
    return query_velocity


@torch.no_grad()
def heatmap_topk_xy(
    proposal_outputs: dict[str, torch.Tensor],
    data_cfg: DataConfig,
    num_proposals: int,
    output_stride: int,
) -> torch.Tensor:
    """Return top-k heatmap cell centers without regression or NMS.

    Args:
        proposal_outputs:
            Proposal head output with ``heatmap_logits [B,1,H,W]``.
        data_cfg:
            ROI and voxel-size config.
        num_proposals:
            Number of heatmap peaks to keep.
        output_stride:
            Proposal feature stride relative to the raw BEV grid.

    Returns:
        ``xy``: ``[B,N,2]`` metric cell centers. This isolates whether the
        heatmap ranks GT neighborhoods before box regression and NMS are used.
    """

    heatmap = proposal_outputs["heatmap_logits"].sigmoid()
    b, _, h, w = heatmap.shape
    k = min(num_proposals, h * w)
    _, flat_idx = heatmap.flatten(1).topk(k, dim=1)
    rows = torch.div(flat_idx, w, rounding_mode="floor")
    cols = flat_idx % w
    x_min, y_min, _, _ = data_cfg.roi_m
    cell = data_cfg.voxel_size_m * output_stride
    x = x_min + (cols.float() + 0.5) * cell
    y = y_min + (rows.float() + 0.5) * cell
    xy = torch.stack([x, y], dim=-1)
    if k == num_proposals:
        return xy
    pad = xy[:, -1:].expand(b, num_proposals - k, 2)
    return torch.cat([xy, pad], dim=1)


@torch.no_grad()
def gt_point_support(
    points: torch.Tensor,
    point_mask: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    radius_m: float,
) -> torch.Tensor:
    """Return GT objects with at least one input point near the center.

    Args:
        points: ``[B,P,4]`` sampled lidar input.
        point_mask: ``[B,P]`` valid point mask.
        gt_boxes: ``[B,M,7]`` boxes.
        gt_valid: ``[B,M]`` valid object mask.
        radius_m: Center-distance support radius in meters.

    Returns:
        Boolean mask ``[B,M]``. This is a diagnostic proxy for "detectable from
        the current sampled lidar input"; it should not be used for final paper
        metrics, but it is useful for overfit debugging.
    """

    support = torch.zeros_like(gt_valid)
    for batch_idx in range(gt_boxes.shape[0]):
        valid_gt = gt_valid[batch_idx]
        valid_points = point_mask[batch_idx]
        if not valid_gt.any() or not valid_points.any():
            continue
        gt_xy = gt_boxes[batch_idx, valid_gt, :2]
        pts_xy = points[batch_idx, valid_points, :2]
        min_dist = torch.cdist(gt_xy[None], pts_xy[None]).squeeze(0).min(dim=1).values
        support[batch_idx, valid_gt] = min_dist < radius_m
    return support


def rasterize_gt_history_occupancy(
    gt_history: torch.Tensor,
    gt_history_mask: torch.Tensor,
    gt_valid: torch.Tensor,
    data_cfg: DataConfig,
    radius_m: float = 0.5,
) -> torch.Tensor:
    """Rasterize GT actor histories into raw-grid BEV occupancy channels.

    Args:
        gt_history: ``[B,M,H_hist,2]`` centers in current ego frame.
        gt_history_mask: ``[B,M,H_hist]`` validity mask.
        gt_valid: ``[B,M]`` current-object validity mask.
        data_cfg: BEV ROI/voxel configuration.
        radius_m: Disk radius used to thicken each center.

    Returns:
        ``[B,H_hist,H,W]`` float occupancy grid, oldest history channel first.

    This is an oracle diagnostic input. It intentionally uses GT histories from
    the cache to test whether clean actor-history occupancy fixes forecasting.
    """

    b, m, h_hist, _ = gt_history.shape
    x_min, y_min, x_max, y_max = data_cfg.roi_m
    voxel = data_cfg.voxel_size_m
    width = int(round((x_max - x_min) / voxel))
    height = int(round((y_max - y_min) / voxel))
    occ = gt_history.new_zeros((b, h_hist, height, width))
    xy = gt_history
    ix = torch.floor((xy[..., 0] - x_min) / voxel).long()
    iy = torch.floor((xy[..., 1] - y_min) / voxel).long()
    valid = (
        gt_valid[:, :, None]
        & gt_history_mask
        & (ix >= 0)
        & (ix < width)
        & (iy >= 0)
        & (iy < height)
    )
    if valid.any():
        batch_ids = torch.arange(b, device=gt_history.device)[:, None, None].expand(b, m, h_hist)
        time_ids = torch.arange(h_hist, device=gt_history.device)[None, None].expand(b, m, h_hist)
        flat_index = (
            batch_ids * (h_hist * height * width)
            + time_ids * (height * width)
            + iy.clamp(0, height - 1) * width
            + ix.clamp(0, width - 1)
        )
        occ.view(-1).index_fill_(0, flat_index[valid], 1.0)
    radius_cells = max(0, int(round(radius_m / voxel)))
    if radius_cells > 0:
        kernel = radius_cells * 2 + 1
        occ = F.max_pool2d(
            occ.reshape(b * h_hist, 1, height, width),
            kernel_size=kernel,
            stride=1,
            padding=radius_cells,
        ).reshape(b, h_hist, height, width)
    return occ


def rasterize_gt_history_motion_targets(
    gt_history: torch.Tensor,
    gt_history_mask: torch.Tensor,
    gt_valid: torch.Tensor,
    data_cfg: DataConfig,
    output_stride: int = 4,
    radius_m: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Rasterize dense BEV history occupancy and flow targets.

    Args:
        gt_history: ``[B,M,H_hist,2]`` centers in current ego frame, oldest to
            current.
        gt_history_mask: ``[B,M,H_hist]`` validity mask.
        gt_valid: ``[B,M]`` current-object validity mask.
        data_cfg: BEV ROI/voxel configuration.
        output_stride: Target stride relative to raw BEV cells. The auxiliary
            head predicts from the encoder's stride-4 feature map by default.
        radius_m: Occupancy disk radius at raw-grid resolution.

    Returns:
        Dict with:
        - ``occupancy``: ``[B,H_hist,Hs,Ws]`` binary target.
        - ``flow``: ``[B,H_hist,2,Hs,Ws]`` displacement to the next valid
          history slot in meters per cached history step.
        - ``flow_mask``: ``[B,H_hist,Hs,Ws]`` valid flow cells.

    The current slot has no next history slot, so it uses current-minus-previous
    valid center. This gives all history channels a direction target while
    preserving the newest-step velocity definition used by query-level velocity
    supervision.
    """

    b, m, h_hist, _ = gt_history.shape
    device = gt_history.device
    x_min, y_min, x_max, y_max = data_cfg.roi_m
    voxel = data_cfg.voxel_size_m
    raw_w = int(round((x_max - x_min) / voxel))
    raw_h = int(round((y_max - y_min) / voxel))
    occ = gt_history.new_zeros((b, h_hist, raw_h, raw_w))
    flow_sum = gt_history.new_zeros((b, h_hist, 2, raw_h, raw_w))
    flow_count = gt_history.new_zeros((b, h_hist, 1, raw_h, raw_w))

    radius_cells = max(0, int(round(radius_m / voxel)))
    if radius_cells > 0:
        offsets = [
            (dy, dx)
            for dy in range(-radius_cells, radius_cells + 1)
            for dx in range(-radius_cells, radius_cells + 1)
            if dx * dx + dy * dy <= radius_cells * radius_cells
        ]
    else:
        offsets = [(0, 0)]

    # --- Vectorized flow target (replaces the per-object/step/offset loops). ---
    # An entry is a valid history step of an object with >= 2 valid steps. Its
    # flow points to the next valid step (or, for the last valid step, comes from
    # the previous valid step), exactly matching the original loop semantics.
    mask = gt_history_mask.bool() & gt_valid[:, :, None]  # [B, M, H]
    obj_ok = mask.sum(dim=-1) >= 2  # [B, M]
    step_idx = torch.arange(h_hist, device=device)
    # next valid index strictly after each step (H if none), prev valid before it.
    after = mask[:, :, None, :] & (step_idx[None, None, None, :] > step_idx[None, None, :, None])
    before = mask[:, :, None, :] & (step_idx[None, None, None, :] < step_idx[None, None, :, None])
    next_idx = torch.where(after, step_idx[None, None, None, :], h_hist).min(dim=-1).values  # [B,M,H]
    prev_idx = torch.where(before, step_idx[None, None, None, :], -1).max(dim=-1).values  # [B,M,H]
    has_next = next_idx < h_hist

    gathered_next = torch.gather(
        gt_history, 2, next_idx.clamp(0, h_hist - 1)[..., None].expand(b, m, h_hist, 2)
    )
    gathered_prev = torch.gather(
        gt_history, 2, prev_idx.clamp(0, h_hist - 1)[..., None].expand(b, m, h_hist, 2)
    )
    flow = torch.where(
        has_next[..., None], gathered_next - gt_history, gt_history - gathered_prev
    )  # [B, M, H, 2]

    ix = torch.floor((gt_history[..., 0] - x_min) / voxel).long()  # [B,M,H]
    iy = torch.floor((gt_history[..., 1] - y_min) / voxel).long()
    entry = mask & obj_ok[:, :, None] & (has_next | (prev_idx >= 0))
    entry = entry & (ix >= 0) & (ix < raw_w) & (iy >= 0) & (iy < raw_h)

    if entry.any():
        oy = torch.tensor([o[0] for o in offsets], device=device)
        ox = torch.tensor([o[1] for o in offsets], device=device)
        num_off = len(offsets)
        batch_ids = torch.arange(b, device=device)[:, None, None].expand(b, m, h_hist)
        step_ch = step_idx[None, None, :].expand(b, m, h_hist)
        # Broadcast each entry over the disk stencil: [B, M, H, num_off].
        yy = iy[..., None] + oy[None, None, None, :]
        xx = ix[..., None] + ox[None, None, None, :]
        off_valid = (
            entry[..., None]
            & (yy >= 0)
            & (yy < raw_h)
            & (xx >= 0)
            & (xx < raw_w)
        )
        b_e = batch_ids[..., None].expand(b, m, h_hist, num_off)[off_valid]
        c_e = step_ch[..., None].expand(b, m, h_hist, num_off)[off_valid]
        y_e = yy[off_valid]
        x_e = xx[off_valid]
        flow_e = flow[:, :, :, None, :].expand(b, m, h_hist, num_off, 2)[off_valid]  # [E, 2]

        cell = ((b_e * h_hist + c_e) * raw_h + y_e) * raw_w + x_e  # flat (b,step,yy,xx)
        occ.view(-1).index_fill_(0, cell, 1.0)
        flow_count.view(-1).index_put_((cell,), torch.ones_like(cell, dtype=flow_count.dtype), accumulate=True)
        # Accumulate flow into a flat [N, 2] buffer (last-dim contiguous), then
        # reshape to [B, H, 2, raw_h, raw_w] to match the original layout.
        flow_sum_flat = gt_history.new_zeros((b * h_hist * raw_h * raw_w, 2))
        flow_sum_flat.index_put_((cell,), flow_e, accumulate=True)
        flow_sum = flow_sum_flat.view(b, h_hist, raw_h, raw_w, 2).permute(0, 1, 4, 2, 3).contiguous()

    if output_stride > 1:
        occ_ds = F.max_pool2d(occ, kernel_size=output_stride, stride=output_stride)
        count_ds = F.avg_pool2d(
            flow_count.reshape(b * h_hist, 1, raw_h, raw_w),
            kernel_size=output_stride,
            stride=output_stride,
        ).reshape(b, h_hist, 1, occ_ds.shape[-2], occ_ds.shape[-1])
        flow_sum_ds = F.avg_pool2d(
            flow_sum.reshape(b * h_hist, 2, raw_h, raw_w),
            kernel_size=output_stride,
            stride=output_stride,
        ).reshape(b, h_hist, 2, occ_ds.shape[-2], occ_ds.shape[-1])
        flow = flow_sum_ds / count_ds.clamp_min(1e-6)
        flow_mask = count_ds.squeeze(2) > 0
        return {
            "occupancy": occ_ds,
            "flow": flow,
            "flow_mask": flow_mask,
        }

    flow = flow_sum / flow_count.clamp_min(1.0)
    return {
        "occupancy": occ,
        "flow": flow,
        "flow_mask": flow_count.squeeze(2) > 0,
    }


def dense_history_motion_loss(
    motion_aux: dict[str, torch.Tensor] | None,
    targets: dict[str, torch.Tensor] | None,
    flow_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Auxiliary dense BEV occupancy/flow loss for history motion cues."""

    if motion_aux is None or targets is None:
        zero = torch.tensor(0.0)
        return {
            "loss_motion_aux": zero,
            "loss_motion_occ": zero,
            "loss_motion_flow": zero,
            "motion_flow_l2": zero,
            "motion_flow_cells": zero,
        }
    logits = motion_aux["occupancy_logits"]
    pred_flow = motion_aux["flow"]
    occupancy = targets["occupancy"].to(device=logits.device, dtype=logits.dtype)
    flow = targets["flow"].to(device=logits.device, dtype=logits.dtype)
    flow_mask = targets["flow_mask"].to(device=logits.device)
    if occupancy.shape[-2:] != logits.shape[-2:]:
        occupancy = F.interpolate(
            occupancy,
            size=logits.shape[-2:],
            mode="nearest",
        )
        flow = F.interpolate(
            flow.flatten(1, 2),
            size=logits.shape[-2:],
            mode="nearest",
        ).view(flow.shape[0], flow.shape[1], 2, logits.shape[-2], logits.shape[-1])
        flow_mask = F.interpolate(
            flow_mask.float(),
            size=logits.shape[-2:],
            mode="nearest",
        ).bool()

    loss_occ = sigmoid_focal_loss(logits, occupancy, alpha=0.75, gamma=2.0, reduction="mean")
    if flow_mask.any():
        loss_flow = F.smooth_l1_loss(
            pred_flow.permute(0, 1, 3, 4, 2)[flow_mask],
            flow.permute(0, 1, 3, 4, 2)[flow_mask],
        )
        flow_l2 = (
            pred_flow.permute(0, 1, 3, 4, 2)[flow_mask]
            - flow.permute(0, 1, 3, 4, 2)[flow_mask]
        ).norm(dim=-1).mean()
        flow_cells = flow_mask.float().sum()
    else:
        loss_flow = pred_flow.sum() * 0.0
        flow_l2 = loss_flow
        flow_cells = loss_flow
    return {
        "loss_motion_aux": loss_occ + flow_weight * loss_flow,
        "loss_motion_occ": loss_occ,
        "loss_motion_flow": loss_flow,
        "motion_flow_l2": flow_l2,
        "motion_flow_cells": flow_cells,
    }

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from detra_repro.config import DataConfig

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback keeps imports alive without scipy.
    linear_sum_assignment = None


@dataclass
class MatchResult:
    """Hungarian assignment between DeTra queries and ground-truth objects.

    Attributes:
        pred_indices: List of length ``B``. Each tensor contains matched query
            indices with shape ``[K_b]``.
        gt_indices: List of length ``B``. Each tensor contains matched GT
            indices with shape ``[K_b]``.

    Meaning:
        For batch item ``b``, prediction query ``pred_indices[b][i]`` is matched
        to ground-truth object ``gt_indices[b][i]``. Unmatched queries are
        treated as background for object classification and ignored for box and
        forecasting regression.
    """

    pred_indices: list[torch.Tensor]
    gt_indices: list[torch.Tensor]


WAYMO_TYPE_TO_CLASS = {
    1: 0,  # VEHICLE
    2: 1,  # PEDESTRIAN
    4: 2,  # CYCLIST
}


def current_boxes_from_pred(pred: dict[str, torch.Tensor]) -> torch.Tensor:
    """Decode current-frame boxes from DeTra outputs.

    Args:
        pred:
            Output dict from ``DeTraMini``.

            Required keys:
            - ``poses``: ``[B, N, F, T, 3]``.
            - ``box_params``: ``[B, N, 5]`` as
              ``dz,dlog_length,dlog_width,dlog_height,heading_delta``.
            - optional ``base_boxes``: ``[B,N,7]`` decoded proposals.

    Returns:
        ``boxes``: ``[B, N, 7]`` as
        ``x, y, z, length, width, height, heading``.

    Note:
        The paper predicts dimensions and detection pose at ``t=0``. This
        scaffold averages the current pose across modes, then applies a small
        residual heading from ``box_params``. ``z`` is set to zero because the
        current DeTra core is BEV-first.
    """

    current_pose = pred["poses"][:, :, :, 0].mean(dim=2)
    box_params = pred["box_params"]
    base_boxes = pred.get("base_boxes")
    if base_boxes is None:
        base_z = torch.zeros_like(current_pose[..., 0])
        base_size = torch.ones(*current_pose.shape[:2], 3, device=current_pose.device)
        base_heading = current_pose[..., 2]
    else:
        base_z = base_boxes[..., 2]
        base_size = base_boxes[..., 3:6].clamp_min(1e-2)
        base_heading = base_boxes[..., 6]

    z = base_z + box_params[..., 0]
    size = base_size * box_params[..., 1:4].clamp(-4.0, 4.0).exp()
    heading = base_heading + box_params[..., 4]
    return torch.stack(
        [
            current_pose[..., 0],
            current_pose[..., 1],
            z,
            size[..., 0],
            size[..., 1],
            size[..., 2],
            heading,
        ],
        dim=-1,
    )


def _angle_l1(pred_heading: torch.Tensor, gt_heading: torch.Tensor) -> torch.Tensor:
    """Circular heading distance with shape broadcast from inputs."""

    return torch.atan2(
        torch.sin(pred_heading - gt_heading),
        torch.cos(pred_heading - gt_heading),
    ).abs()


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary sigmoid focal loss.

    Args:
        logits: Arbitrary shape.
        targets: Same shape, values in ``{0,1}``.
        alpha: Positive/negative balancing factor.
        gamma: Modulating exponent.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def _axis_aligned_bev(boxes: torch.Tensor) -> torch.Tensor:
    """Approximate rotated boxes by axis-aligned BEV boxes for IoU losses."""

    half = boxes[..., 3:5].clamp_min(1e-2) * 0.5
    return torch.cat([boxes[..., :2] - half, boxes[..., :2] + half], dim=-1)


def waymo_types_to_class_indices(gt_types: torch.Tensor) -> torch.Tensor:
    """Map Waymo laser-label enum ids to contiguous class ids.

    Returns class ids in ``{0: vehicle, 1: pedestrian, 2: cyclist}``. Unknown
    or missing values default to vehicle for backwards-compatible old caches.
    """

    out = torch.zeros_like(gt_types, dtype=torch.long)
    out = torch.where(gt_types == 2, torch.ones_like(out), out)
    out = torch.where(gt_types == 4, torch.full_like(out, 2), out)
    return out


def rotated_box_corners_bev(boxes: torch.Tensor) -> torch.Tensor:
    """Return BEV corners for rotated boxes.

    Args:
        boxes: ``[..., 7]`` as ``x,y,z,length,width,height,heading``.

    Returns:
        Corners ``[..., 4, 2]`` in counter-clockwise order.
    """

    half_l = boxes[..., 3].clamp_min(1e-3) * 0.5
    half_w = boxes[..., 4].clamp_min(1e-3) * 0.5
    local = torch.stack(
        [
            torch.stack([half_l, half_w], dim=-1),
            torch.stack([-half_l, half_w], dim=-1),
            torch.stack([-half_l, -half_w], dim=-1),
            torch.stack([half_l, -half_w], dim=-1),
        ],
        dim=-2,
    )
    c = torch.cos(boxes[..., 6])
    s = torch.sin(boxes[..., 6])
    rot = torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )
    return local @ rot.transpose(-1, -2) + boxes[..., None, :2]


def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2D scalar cross product over the last axis: ``a.x*b.y - a.y*b.x``."""

    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _convex_poly_intersection_area(poly_a: torch.Tensor, poly_b: torch.Tensor) -> torch.Tensor:
    """Vectorized intersection area of two batches of convex quads.

    Args:
        poly_a: ``[N,4,2]`` CCW corners.
        poly_b: ``[N,4,2]`` CCW corners.

    Returns:
        ``[N]`` intersection areas. Fully batched (no Python/per-pair loop) and
        differentiable w.r.t. the corner coordinates. The intersection polygon
        of two convex shapes is the convex hull of (a) vertices of A inside B,
        (b) vertices of B inside A, and (c) edge-edge intersection points; its
        area is the centroid-relative shoelace sum with invalid candidate
        vertices collapsed onto the centroid so they contribute zero area
        regardless of sort order.
    """

    n = poly_a.shape[0]
    if n == 0:
        return poly_a.new_zeros((0,))
    eps = 1e-9

    a_next = poly_a.roll(-1, dims=1)  # [N,4,2] edge end for each A vertex
    b_next = poly_b.roll(-1, dims=1)

    def _inside(points: torch.Tensor, poly: torch.Tensor, poly_next: torch.Tensor) -> torch.Tensor:
        # points: [N,P,2]; poly/poly_next: [N,4,2]. Inside CCW poly => left of all edges.
        edge = (poly_next - poly)[:, None, :, :]  # [N,1,4,2]
        rel = points[:, :, None, :] - poly[:, None, :, :]  # [N,P,4,2]
        return (_cross2d(edge, rel) >= -1e-6).all(dim=-1)  # [N,P]

    a_in_b = _inside(poly_a, poly_b, b_next)  # [N,4]
    b_in_a = _inside(poly_b, poly_a, a_next)  # [N,4]

    # Edge-edge intersections: 4 A-edges x 4 B-edges -> 16 candidate points.
    a1 = poly_a[:, :, None, :]  # [N,4,1,2]
    r = (a_next - poly_a)[:, :, None, :]  # [N,4,1,2]
    b1 = poly_b[:, None, :, :]  # [N,1,4,2]
    s = (b_next - poly_b)[:, None, :, :]  # [N,1,4,2]
    denom = _cross2d(r, s)  # [N,4,4]
    qp = b1 - a1  # [N,4,4,2]
    safe_denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    t = _cross2d(qp, s) / safe_denom  # [N,4,4]
    u = _cross2d(qp, r) / safe_denom  # [N,4,4]
    inter_pts = a1 + t[..., None] * r  # [N,4,4,2]
    inter_valid = (denom.abs() >= eps) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

    pts = torch.cat(
        [poly_a, poly_b, inter_pts.reshape(n, 16, 2)], dim=1
    )  # [N,24,2]
    mask = torch.cat([a_in_b, b_in_a, inter_valid.reshape(n, 16)], dim=1)  # [N,24]

    maskf = mask.to(pts.dtype)
    count = maskf.sum(dim=1, keepdim=True)  # [N,1]
    centroid = (pts * maskf[..., None]).sum(dim=1) / count.clamp_min(1.0)  # [N,2]

    rel = pts - centroid[:, None, :]  # [N,24,2]
    angle = torch.atan2(rel[..., 1], rel[..., 0])  # [N,24]
    # Sort valid vertices CCW; invalid vertices get a large angle so they land
    # contiguously at the tail of the ordering.
    angle = torch.where(mask, angle, angle.new_full((), 1e4))
    order = torch.argsort(angle, dim=1)  # [N,24]
    rel_sorted = torch.gather(rel, 1, order[..., None].expand(-1, -1, 2))  # [N,24,2]
    mask_sorted = torch.gather(mask, 1, order)  # [N,24]
    # Replace tail (invalid) vertices with the first valid vertex so the ring
    # closes on the true V_{k-1}->V0 edge: the duplicated edges contribute zero
    # area, but the closing edge is preserved.
    first_valid = rel_sorted[:, 0:1, :]  # [N,1,2]
    rel_sorted = torch.where(mask_sorted[..., None], rel_sorted, first_valid)

    rel_next = rel_sorted.roll(-1, dims=1)
    area = 0.5 * _cross2d(rel_sorted, rel_next).sum(dim=1).abs()  # [N]
    return torch.where(count.squeeze(1) >= 3.0, area, area.new_zeros(()))


def rotated_bev_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Paired rotated BEV IoU.

    Args:
        pred_boxes: ``[..., 7]``.
        gt_boxes: ``[..., 7]``.

    Returns:
        IoU tensor with shape ``[...]``.
    """

    flat_pred = pred_boxes.reshape(-1, 7)
    flat_gt = gt_boxes.reshape(-1, 7)
    if flat_pred.shape[0] == 0:
        return pred_boxes.new_zeros(pred_boxes.shape[:-1])
    pred_corners = rotated_box_corners_bev(flat_pred)
    gt_corners = rotated_box_corners_bev(flat_gt)
    inter = _convex_poly_intersection_area(pred_corners, gt_corners)
    pred_area = flat_pred[:, 3].clamp_min(1e-3) * flat_pred[:, 4].clamp_min(1e-3)
    gt_area = flat_gt[:, 3].clamp_min(1e-3) * flat_gt[:, 4].clamp_min(1e-3)
    union = (pred_area + gt_area - inter).clamp_min(1e-6)
    return (inter / union).reshape(pred_boxes.shape[:-1])


def rotated_bev_iou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Rotated BEV IoU matrix ``[N,M]`` for proposal NMS/diagnostics."""

    n, m = pred_boxes.shape[0], gt_boxes.shape[0]
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return pred_boxes.new_zeros((n, m))
    pred_grid = pred_boxes[:, None, :].expand(n, m, 7).reshape(-1, 7)
    gt_grid = gt_boxes[None, :, :].expand(n, m, 7).reshape(-1, 7)
    return rotated_bev_iou(pred_grid, gt_grid).reshape(n, m)


def generalized_iou_loss_bev(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Rotated BEV generalized IoU-style loss for matched boxes.

    Args:
        pred_boxes: ``[K,7]``.
        gt_boxes: ``[K,7]``.

    Returns:
        Scalar ``1 - gIoU`` mean.

    Note:
        Intersection uses exact rotated polygon clipping. The enclosing term is
        the axis-aligned enclosure of the two rotated rectangles, which is a
        stable dependency-free approximation to rotated gIoU.
    """

    pred_corners = rotated_box_corners_bev(pred_boxes)
    gt_corners = rotated_box_corners_bev(gt_boxes)
    inter = _convex_poly_intersection_area(pred_corners, gt_corners)
    pred_area = pred_boxes[:, 3].clamp_min(1e-3) * pred_boxes[:, 4].clamp_min(1e-3)
    gt_area = gt_boxes[:, 3].clamp_min(1e-3) * gt_boxes[:, 4].clamp_min(1e-3)
    union = (pred_area + gt_area - inter).clamp_min(1e-6)
    iou = inter / union

    all_corners = torch.cat([pred_corners, gt_corners], dim=1)
    enc_lt = all_corners.min(dim=1).values
    enc_rb = all_corners.max(dim=1).values
    enc_wh = (enc_rb - enc_lt).clamp_min(0)
    enc_area = (enc_wh[:, 0] * enc_wh[:, 1]).clamp_min(1e-6)
    giou = iou - (enc_area - union) / enc_area
    return (1.0 - giou).mean()


def axis_aligned_bev_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Axis-aligned BEV IoU between paired boxes.

    Args:
        pred_boxes: ``[..., 7]``.
        gt_boxes: ``[..., 7]`` (same leading shape as ``pred_boxes``).

    Returns:
        IoU ``[...]``.

    Note:
        This ignores heading, matching the axis-aligned approximation used by
        :func:`generalized_iou_loss_bev`. It is used to define true-positive
        detections (IoU > 0.5) for forecast supervision, as in the paper. A
        faithful reproduction should replace it with rotated BEV IoU.
    """

    pred = _axis_aligned_bev(pred_boxes)
    gt = _axis_aligned_bev(gt_boxes)
    lt = torch.maximum(pred[..., :2], gt[..., :2])
    rb = torch.minimum(pred[..., 2:], gt[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    pred_area = (pred[..., 2] - pred[..., 0]).clamp_min(0) * (pred[..., 3] - pred[..., 1]).clamp_min(0)
    gt_area = (gt[..., 2] - gt[..., 0]).clamp_min(0) * (gt[..., 3] - gt[..., 1]).clamp_min(0)
    union = (pred_area + gt_area - inter).clamp_min(1e-6)
    return inter / union


def _greedy_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Small fallback assignment if SciPy is unavailable."""

    cost = cost.clone()
    pred_ids, gt_ids = [], []
    while cost.numel() and len(pred_ids) < min(cost.shape):
        flat_idx = cost.argmin()
        pred_idx = flat_idx // cost.shape[1]
        gt_idx = flat_idx % cost.shape[1]
        pred_ids.append(pred_idx)
        gt_ids.append(gt_idx)
        cost[pred_idx, :] = float("inf")
        cost[:, gt_idx] = float("inf")
    if not pred_ids:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    return torch.stack(pred_ids), torch.stack(gt_ids)


def _tensor_debug_stats(name: str, tensor: torch.Tensor) -> str:
    """Return compact finite-value diagnostics for error messages."""

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


@torch.no_grad()
def hungarian_match(
    pred: dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    class_weight: float = 2.0,
    center_weight: float = 1.0,
    size_weight: float = 0.25,
    heading_weight: float = 0.5,
) -> MatchResult:
    """Match query predictions to ground truth with Hungarian assignment.

    Args:
        pred:
            Model output dict.
            ``object_logits``: ``[B, N]``.
            decoded boxes from ``current_boxes_from_pred``: ``[B, N, 7]``.
        gt_boxes:
            Padded GT boxes ``[B, M, 7]``.
        gt_valid:
            Boolean mask ``[B, M]``.
        class_weight, center_weight, size_weight, heading_weight:
            Cost weights.

    Returns:
        ``MatchResult``.

    Cost:
        ``-object_probability + L1(center_xy) + L1(size_lwh) +
        circular_heading_distance``.
    """

    device = gt_boxes.device
    pred_boxes = current_boxes_from_pred(pred)
    probs = pred["object_logits"].sigmoid()
    b = gt_boxes.shape[0]

    # Build the full [B,N,M] cost tensor in one batched pass (single set of GPU
    # kernels, one finite-check sync, one host transfer) instead of per-batch
    # cdist calls each followed by their own GPU->CPU sync.
    cls_cost = -probs[:, :, None]  # [B,N,1] broadcast over M
    center_cost = torch.cdist(pred_boxes[:, :, :2], gt_boxes[:, :, :2], p=1)  # [B,N,M]
    size_cost = torch.cdist(pred_boxes[:, :, 3:6], gt_boxes[:, :, 3:6], p=1)
    heading_cost = _angle_l1(pred_boxes[:, :, None, 6], gt_boxes[:, None, :, 6])
    cost = (
        class_weight * cls_cost
        + center_weight * center_cost
        + size_weight * size_cost
        + heading_weight * heading_cost
    )  # [B,N,M]

    valid_mask = gt_valid[:, None, :].expand_as(cost)  # [B,N,M]
    if not torch.isfinite(cost[valid_mask]).all():
        details = ["Hungarian cost contains non-finite values"]
        details.append(_tensor_debug_stats("cost", cost))
        details.append(_tensor_debug_stats("pred_boxes", pred_boxes))
        details.append(_tensor_debug_stats("gt_boxes", gt_boxes))
        details.append(_tensor_debug_stats("object_prob", probs))
        details.append(_tensor_debug_stats("center_cost", center_cost))
        details.append(_tensor_debug_stats("size_cost", size_cost))
        details.append(_tensor_debug_stats("heading_cost", heading_cost))
        raise ValueError("\n".join(details))

    cost_cpu = cost.detach().cpu().numpy()
    valid_np = gt_valid.detach().cpu().numpy()
    pred_indices, gt_indices = [], []
    for batch_idx in range(b):
        valid_cols = valid_np[batch_idx].nonzero()[0]
        if valid_cols.size == 0:
            pred_indices.append(torch.empty(0, dtype=torch.long, device=device))
            gt_indices.append(torch.empty(0, dtype=torch.long, device=device))
            continue
        sub = cost_cpu[batch_idx][:, valid_cols]  # [N, num_valid]
        valid_gt = torch.as_tensor(valid_cols, dtype=torch.long, device=device)
        if linear_sum_assignment is not None:
            row, col = linear_sum_assignment(sub)
            pred_idx = torch.as_tensor(row, dtype=torch.long, device=device)
            gt_idx = valid_gt[torch.as_tensor(col, dtype=torch.long, device=device)]
        else:
            pred_idx, local_gt_idx = _greedy_assignment(torch.as_tensor(sub, device=device))
            gt_idx = valid_gt[local_gt_idx]
        pred_indices.append(pred_idx)
        gt_indices.append(gt_idx)

    return MatchResult(pred_indices, gt_indices)


def align_gt_to_queries(
    matches: MatchResult,
    gt_boxes: torch.Tensor,
    gt_types: torch.Tensor | None,
    gt_future: torch.Tensor,
    gt_future_mask: torch.Tensor,
    num_queries: int,
    gt_velocity: torch.Tensor | None = None,
    gt_velocity_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Place matched GT data into prediction query slots.

    Args:
        matches:
            Output of ``hungarian_match``.
        gt_boxes:
            ``[B, M, 7]``.
        gt_types:
            Optional Waymo type ids ``[B, M]``.
        gt_future:
            ``[B, M, T_future, 2]``.
        gt_future_mask:
            ``[B, M, T_future]``.
        num_queries:
            Number of prediction queries ``N``.
        gt_velocity:
            Optional ``[B,M,2]`` per-object displacement per forecast step.
        gt_velocity_valid:
            Optional ``[B,M]`` mask for valid velocity targets.

    Returns:
        Dict with:
        - ``query_gt_boxes``: ``[B, N, 7]``.
        - ``query_gt_future``: ``[B, N, T_future, 2]``.
        - ``query_gt_future_mask``: ``[B, N, T_future]``.
        - ``query_object_targets``: ``[B, N]`` with 1 for matched queries.

    Why this exists:
        DeTra predicts futures per query slot, not per GT object row. After
        Hungarian matching says "query 17 corresponds to GT object 3", the
        future trajectory of GT object 3 must be copied into slot 17 so the WTA
        forecasting loss compares the right query to the right future.
    """

    b = gt_boxes.shape[0]
    t_future = gt_future.shape[2]
    query_gt_boxes = gt_boxes.new_zeros((b, num_queries, 7))
    query_gt_types = torch.zeros((b, num_queries), dtype=torch.long, device=gt_boxes.device)
    query_gt_future = gt_future.new_zeros((b, num_queries, t_future, 2))
    query_gt_future_mask = gt_future_mask.new_zeros((b, num_queries, t_future))
    query_gt_velocity = gt_boxes.new_zeros((b, num_queries, 2))
    query_gt_velocity_valid = torch.zeros((b, num_queries), dtype=torch.bool, device=gt_boxes.device)
    query_object_targets = gt_boxes.new_zeros((b, num_queries))

    for batch_idx, (pred_idx, gt_idx) in enumerate(zip(matches.pred_indices, matches.gt_indices)):
        if len(pred_idx) == 0:
            continue
        query_gt_boxes[batch_idx, pred_idx] = gt_boxes[batch_idx, gt_idx]
        if gt_types is not None:
            query_gt_types[batch_idx, pred_idx] = gt_types[batch_idx, gt_idx].long()
        query_gt_future[batch_idx, pred_idx] = gt_future[batch_idx, gt_idx]
        query_gt_future_mask[batch_idx, pred_idx] = gt_future_mask[batch_idx, gt_idx]
        if gt_velocity is not None:
            query_gt_velocity[batch_idx, pred_idx] = gt_velocity[batch_idx, gt_idx]
        if gt_velocity_valid is not None:
            query_gt_velocity_valid[batch_idx, pred_idx] = gt_velocity_valid[batch_idx, gt_idx]
        query_object_targets[batch_idx, pred_idx] = 1.0

    return {
        "query_gt_boxes": query_gt_boxes,
        "query_gt_types": query_gt_types,
        "query_gt_future": query_gt_future,
        "query_gt_future_mask": query_gt_future_mask,
        "query_gt_velocity": query_gt_velocity,
        "query_gt_velocity_valid": query_gt_velocity_valid,
        "query_object_targets": query_object_targets,
    }


def align_gt_history_to_queries(
    matches: MatchResult,
    gt_history: torch.Tensor,
    gt_history_mask: torch.Tensor,
    num_queries: int,
) -> dict[str, torch.Tensor]:
    """Place matched GT past tracks into prediction query slots (M1).

    ``gt_history`` is ``[B, M, H, 2]`` ordered oldest->newest with the newest
    slot equal to the current center (already supervised by detection). We drop
    that current slot and reverse to newest->oldest so target slot 0 is ``t=-1``,
    matching ``pred["past_xy"]`` from :class:`PoseUpdate`.

    Returns ``query_gt_past [B,N,H-1,2]`` and ``query_gt_past_mask [B,N,H-1]``.
    """

    b, m, h, _ = gt_history.shape
    past_steps = max(h - 1, 0)
    query_gt_past = gt_history.new_zeros((b, num_queries, past_steps, 2))
    query_gt_past_mask = torch.zeros(
        (b, num_queries, past_steps), dtype=torch.bool, device=gt_history.device
    )
    if past_steps == 0:
        return {"query_gt_past": query_gt_past, "query_gt_past_mask": query_gt_past_mask}

    # strictly-past slots (drop newest=current), reversed to newest->oldest
    past = gt_history[:, :, :-1, :].flip(dims=[2])
    past_mask = gt_history_mask[:, :, :-1].flip(dims=[2])
    for batch_idx, (pred_idx, gt_idx) in enumerate(zip(matches.pred_indices, matches.gt_indices)):
        if len(pred_idx) == 0:
            continue
        query_gt_past[batch_idx, pred_idx] = past[batch_idx, gt_idx]
        query_gt_past_mask[batch_idx, pred_idx] = past_mask[batch_idx, gt_idx]
    return {"query_gt_past": query_gt_past, "query_gt_past_mask": query_gt_past_mask}


def past_trajectory_loss(
    pred: dict[str, torch.Tensor],
    query_gt_past: torch.Tensor,
    query_gt_past_mask: torch.Tensor,
    object_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Masked SmoothL1 between predicted and observed past tracks (M1).

    Single-mode (no WTA): the past is observed, so there is one target per query.
    Supervised only on gated (true-positive) queries with valid past slots.
    """

    past_xy = pred.get("past_xy")
    if past_xy is None:
        zero = pred["poses"].sum() * 0.0
        return {"loss_past": zero, "past_supervised_queries": zero}
    steps = min(past_xy.shape[2], query_gt_past.shape[2])
    valid = object_targets[:, :, None].bool() & query_gt_past_mask[..., :steps]
    if not valid.any():
        zero = past_xy.sum() * 0.0
        return {"loss_past": zero, "past_supervised_queries": zero}
    pred_sel = past_xy[:, :, :steps][valid]
    target_sel = query_gt_past[:, :, :steps][valid]
    loss = F.smooth_l1_loss(pred_sel, target_sel)
    return {"loss_past": loss, "past_supervised_queries": valid.float().sum()}


def detection_loss(
    pred: dict[str, torch.Tensor],
    aligned: dict[str, torch.Tensor],
    box_l1_weight: float = 1.0,
    giou_weight: float = 0.1,
    class_weight: float = 1.0,
    z_weight: float = 0.0,
    height_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Detection classification and matched box regression loss.

    Args:
        pred:
            Model output dict.
        aligned:
            Output of ``align_gt_to_queries``.
        box_l1_weight:
            Regression weight on the full 7-dim smooth-L1.
        giou_weight:
            Weight on the BEV gIoU term (ignores z/height by construction).
        class_weight:
            Weight on the per-class cross-entropy.
        z_weight, height_weight:
            Extra decoupled smooth-L1 weight on the vertical dimensions (z
            center, box height). BEV IoU ignores z/height, so without these the
            only z/height signal is 2 of 7 terms in ``box_l1`` and they stay
            loose enough to destroy 3D IoU (see CHANGELOG diagnosis). Default 0
            preserves prior behavior; the launch sets them > 0.

    Returns:
        Dict with scalar ``loss_cls``, ``loss_box_l1``, ``loss_giou``,
        ``loss_z_l1``, ``loss_height_l1``, ``z_l1`` / ``height_l1`` (unweighted
        diagnostics), and ``loss_detection``.
    """

    targets = aligned["query_object_targets"]
    num_objects = targets.sum().clamp_min(1.0)
    loss_cls = sigmoid_focal_loss(pred["object_logits"], targets, reduction="sum") / num_objects

    matched = targets.bool()
    pred_boxes = current_boxes_from_pred(pred)
    if matched.any():
        gt_boxes = aligned["query_gt_boxes"]
        pm = pred_boxes[matched]
        gm = gt_boxes[matched]
        box_l1 = F.smooth_l1_loss(pm, gm)
        giou = generalized_iou_loss_bev(pm, gm)
        # Decoupled vertical terms (column 2 = z center, column 5 = height).
        z_l1 = F.smooth_l1_loss(pm[:, 2], gm[:, 2])
        height_l1 = F.smooth_l1_loss(pm[:, 5], gm[:, 5])
        if "class_logits" in pred:
            class_targets = waymo_types_to_class_indices(aligned["query_gt_types"][matched])
            class_ce = F.cross_entropy(pred["class_logits"][matched], class_targets)
        else:
            class_ce = pred["object_logits"].sum() * 0.0
    else:
        box_l1 = pred["object_logits"].sum() * 0.0
        giou = box_l1
        z_l1 = box_l1
        height_l1 = box_l1
        class_ce = box_l1

    loss_detection = (
        loss_cls
        + class_weight * class_ce
        + box_l1_weight * box_l1
        + giou_weight * giou
        + z_weight * z_l1
        + height_weight * height_l1
    )
    return {
        "loss_cls": loss_cls,
        "loss_class_ce": class_ce,
        "loss_box_l1": box_l1,
        "loss_giou": giou,
        "loss_z_l1": z_l1,
        "loss_height_l1": height_l1,
        "loss_detection": loss_detection,
    }


def build_proposal_targets(
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    data_cfg: DataConfig,
    map_shape: tuple[int, int],
    output_stride: int,
    gaussian_radius: int = 1,
) -> dict[str, torch.Tensor]:
    """Build dense targets for the initial proposal head.

    Args:
        gt_boxes: ``[B,M,7]``.
        gt_valid: ``[B,M]``.
        data_cfg: ROI and voxel-size config.
        map_shape: ``(H,W)`` of proposal head output.
        output_stride: Feature-map stride relative to raw BEV grid.
        gaussian_radius: Radius in output cells for positive heatmap blobs.

    Returns:
        Dict with:
        - ``heatmap``: ``[B,1,H,W]``.
        - ``box_target``: ``[B,8,H,W]`` as
          ``dx,dy,log_length,log_width,sin_heading,cos_heading,z,log_height``.
        - ``box_mask``: ``[B,1,H,W]`` positive center cells.
        - ``box7_target``: ``[B,7,H,W]`` full boxes at positive center cells.
    """

    b, m, _ = gt_boxes.shape
    height, width = map_shape
    heatmap = gt_boxes.new_zeros((b, 1, height, width))
    box_target = gt_boxes.new_zeros((b, 8, height, width))
    box7_target = gt_boxes.new_zeros((b, 7, height, width))
    box_mask = gt_boxes.new_zeros((b, 1, height, width))
    x_min, y_min, _, _ = data_cfg.roi_m
    cell = data_cfg.voxel_size_m * output_stride
    device = gt_boxes.device

    # Flatten valid (batch, object) entries; preserve row-major (batch then
    # object) order so same-cell collisions resolve last-write-wins exactly as
    # the original double Python loop did.
    flat_valid = gt_valid.reshape(b * m)
    batch_of = torch.arange(b, device=device).repeat_interleave(m)  # [b*m]
    boxes_flat = gt_boxes.reshape(b * m, 7)
    col = torch.round((boxes_flat[:, 0] - x_min) / cell - 0.5).long()
    row = torch.round((boxes_flat[:, 1] - y_min) / cell - 0.5).long()
    in_range = flat_valid & (row >= 0) & (row < height) & (col >= 0) & (col < width)
    sel = torch.where(in_range)[0]
    if sel.numel() == 0:
        return {
            "heatmap": heatmap,
            "box_target": box_target,
            "box7_target": box7_target,
            "box_mask": box_mask,
        }

    bsel = batch_of[sel]
    rsel = row[sel]
    csel = col[sel]
    box = boxes_flat[sel]  # [K,7]

    cell_x = x_min + (csel.to(box.dtype) + 0.5) * cell
    cell_y = y_min + (rsel.to(box.dtype) + 0.5) * cell
    box8 = torch.stack(
        [
            box[:, 0] - cell_x,
            box[:, 1] - cell_y,
            box[:, 3].clamp_min(1e-2).log(),
            box[:, 4].clamp_min(1e-2).log(),
            torch.sin(box[:, 6]),
            torch.cos(box[:, 6]),
            box[:, 2],
            box[:, 5].clamp_min(1e-2).log(),
        ],
        dim=-1,
    )  # [K,8]
    # Center-cell scatters (last-write-wins along K).
    box_target[bsel, :, rsel, csel] = box8
    box7_target[bsel, :, rsel, csel] = box
    box_mask[bsel, 0, rsel, csel] = 1.0

    # Gaussian heatmap: isotropic blob over a (2r+1)^2 stencil, scattered with
    # amax across overlapping blobs (equivalent to the per-center max-draw).
    diameter = 2 * gaussian_radius + 1
    offs = torch.arange(diameter, device=device) - gaussian_radius
    dyy, dxx = torch.meshgrid(offs, offs, indexing="ij")
    dyy = dyy.reshape(-1)  # [S]
    dxx = dxx.reshape(-1)
    sigma = max(diameter / 6.0, 1e-6)
    gvals = torch.exp(
        -(dxx.to(box.dtype) ** 2 + dyy.to(box.dtype) ** 2) / (2 * sigma * sigma)
    )  # [S]

    tr = rsel[:, None] + dyy[None, :]  # [K,S]
    tc = csel[:, None] + dxx[None, :]
    valid_off = (tr >= 0) & (tr < height) & (tc >= 0) & (tc < width)
    flat_idx = bsel[:, None] * (height * width) + tr * width + tc  # [K,S]
    g_broadcast = gvals[None, :].expand_as(flat_idx)

    flat_idx = flat_idx[valid_off]
    g_sel = g_broadcast[valid_off]
    heatmap.reshape(-1).scatter_reduce_(0, flat_idx, g_sel, reduce="amax", include_self=True)

    return {
        "heatmap": heatmap,
        "box_target": box_target,
        "box7_target": box7_target,
        "box_mask": box_mask,
    }


def dense_proposal_boxes(
    box_regression: torch.Tensor,
    data_cfg: DataConfig,
    output_stride: int,
    default_z: float = 0.0,
    default_height: float = 1.7,
) -> torch.Tensor:
    """Decode dense proposal regression map into BEV boxes.

    Args:
        box_regression: ``[B,8,H,W]`` with
            ``dx,dy,log_length,log_width,sin_heading,cos_heading,z,log_height``.
            Legacy 6-channel maps are also accepted (z/height fall back to the
            defaults).
        data_cfg: ROI and voxel-size config.
        output_stride: Feature-map stride relative to raw BEV grid.
        default_z: z center used when the head emits no z channel.
        default_height: height used when the head emits no log-height channel.

    Returns:
        ``boxes``: ``[B,7,H,W]`` as
        ``x,y,z,length,width,height,heading``.
    """

    b, channels, height, width = box_regression.shape
    x_min, y_min, _, _ = data_cfg.roi_m
    cell = data_cfg.voxel_size_m * output_stride
    ys = torch.arange(height, device=box_regression.device, dtype=box_regression.dtype)
    xs = torch.arange(width, device=box_regression.device, dtype=box_regression.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cell_x = x_min + (xx + 0.5) * cell
    cell_y = y_min + (yy + 0.5) * cell

    dx, dy, log_l, log_w, sin_h, cos_h = box_regression[:, :6].unbind(dim=1)
    center_x = cell_x[None] + dx
    center_y = cell_y[None] + dy
    if channels >= 8:
        z = box_regression[:, 6]
        height_t = box_regression[:, 7].exp().clamp_min(0.1)
    else:
        z = torch.full_like(center_x, default_z)
        height_t = torch.full_like(center_x, default_height)
    length = log_l.exp().clamp_min(0.1)
    width_t = log_w.exp().clamp_min(0.1)
    heading = torch.atan2(sin_h, cos_h)
    return torch.stack([center_x, center_y, z, length, width_t, height_t, heading], dim=1)


def proposal_loss(
    proposal_outputs: dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    data_cfg: DataConfig,
    output_stride: int = 4,
    heatmap_weight: float = 1.0,
    box_weight: float = 1.0,
    giou_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Initial dense proposal loss for high-recall pose initialization.

    Args:
        proposal_outputs:
            ``heatmap_logits``: ``[B,1,H,W]``.
            ``box_regression``: ``[B,6,H,W]``.
        gt_boxes: ``[B,M,7]``.
        gt_valid: ``[B,M]``.
        data_cfg: Data config.
        output_stride: Proposal output stride.
        heatmap_weight: Dense heatmap focal loss weight.
        box_weight: Regression weight.
        giou_weight: Rotated BEV gIoU loss weight.

    Returns:
        Dict with ``loss_init_heatmap``, ``loss_init_box``,
        ``loss_init_giou``, and ``loss_init``.
    """

    _, _, height, width = proposal_outputs["heatmap_logits"].shape
    targets = build_proposal_targets(
        gt_boxes,
        gt_valid,
        data_cfg,
        map_shape=(height, width),
        output_stride=output_stride,
    )
    heatmap_target = targets["heatmap"]
    # CenterNet-style focal heatmap. Gaussian cells are soft positives.
    pred = proposal_outputs["heatmap_logits"].sigmoid().clamp(1e-4, 1.0 - 1e-4)
    pos = heatmap_target.eq(1.0)
    neg = heatmap_target.lt(1.0)
    neg_weights = (1.0 - heatmap_target).pow(4)
    pos_loss = -(pred.log()) * (1.0 - pred).pow(2) * pos
    neg_loss = -((1.0 - pred).log()) * pred.pow(2) * neg_weights * neg
    num_pos = pos.float().sum().clamp_min(1.0)
    loss_heatmap = (pos_loss.sum() + neg_loss.sum()) / num_pos

    box_mask = targets["box_mask"].expand_as(proposal_outputs["box_regression"]).bool()
    if box_mask.any():
        loss_box = F.smooth_l1_loss(
            proposal_outputs["box_regression"][box_mask],
            targets["box_target"][box_mask],
        )
        dense_boxes = dense_proposal_boxes(
            proposal_outputs["box_regression"],
            data_cfg,
            output_stride=output_stride,
        )
        loss_giou = generalized_iou_loss_bev(
            dense_boxes.permute(0, 2, 3, 1)[targets["box_mask"].squeeze(1).bool()],
            targets["box7_target"].permute(0, 2, 3, 1)[targets["box_mask"].squeeze(1).bool()],
        )
    else:
        loss_box = proposal_outputs["box_regression"].sum() * 0.0
        loss_giou = loss_box

    loss_heatmap_weighted = heatmap_weight * loss_heatmap
    loss_box_weighted = box_weight * loss_box
    loss_giou_weighted = giou_weight * loss_giou
    return {
        "loss_init_heatmap": loss_heatmap,
        "loss_init_box": loss_box,
        "loss_init_giou": loss_giou,
        "loss_init_heatmap_weighted": loss_heatmap_weighted,
        "loss_init_box_weighted": loss_box_weighted,
        "loss_init_giou_weighted": loss_giou_weighted,
        "loss_init": loss_heatmap_weighted + loss_box_weighted + loss_giou_weighted,
    }


def wta_forecasting_loss(
    pred: dict[str, torch.Tensor],
    matched_gt_future: torch.Tensor,
    matched_future_mask: torch.Tensor,
    object_targets: torch.Tensor,
    mode_ce_weight: float = 1.0,
    loss_type: str = "laplace",
    top_k: int = 1,
) -> dict[str, torch.Tensor]:
    """Winner-takes-all multi-modal forecasting loss with optional EWTA.

    Args:
        pred:
            Preferred key ``future_xy``: ``[B,N,F,T_future,2]``. If absent,
            falls back to ``poses[:, :, :, 1:, :2]`` for older tests.
            ``mode_logits``: ``[B, N, F]``.
        matched_gt_future:
            Query-aligned future centers ``[B, N, T_future, 2]``.
        matched_future_mask:
            Query-aligned future validity ``[B, N, T_future]``.
        object_targets:
            Query object targets ``[B, N]``; only matched queries are supervised.
        mode_ce_weight:
            Cross-entropy weight for winner mode.
        loss_type:
            ``laplace`` uses predicted log scales when available, matching the
            paper-style NLL. ``smooth_l1`` ignores predicted scales and is a
            fixed-scale diagnostic loss for overfit experiments.
        top_k:
            Evolving-WTA width. With ``top_k=1`` (default) only the single best
            mode is regressed -- identical to plain winner-takes-all. With
            ``top_k>1`` the ``k`` lowest-loss modes are all regressed (mean of
            their per-mode losses), which keeps non-winner modes alive early in
            training and prevents mode collapse. Anneal ``k`` from ``F`` down to
            ``1`` over training. The mode cross-entropy always targets the single
            best winner regardless of ``k``.

    Returns:
        Dict with ``loss_forecast_reg``, ``loss_mode_ce``, and
        ``loss_forecast``.
    """

    pred_xy = pred.get("future_xy")
    if pred_xy is None:
        pred_xy = pred["poses"][:, :, :, 1:, :2]
    mask = matched_future_mask[:, :, None, :, None].float()
    valid_query = object_targets.bool() & matched_future_mask.any(dim=-1)

    if not valid_query.any():
        zero = pred["poses"].sum() * 0.0
        return {
            "loss_forecast_reg": zero,
            "loss_mode_ce": zero,
            "loss_forecast": zero,
        }

    if loss_type not in {"laplace", "smooth_l1"}:
        raise ValueError("loss_type must be laplace or smooth_l1")
    future_log_scale = pred.get("future_log_scale") if loss_type == "laplace" else None
    abs_error = (pred_xy - matched_gt_future[:, :, None]).abs()
    denom = mask.sum(dim=(3, 4)).clamp_min(1.0)

    # Winner SELECTION is always by L1 distance on the predicted centroids vs GT
    # (paper Sec. 3.3 / suppl A.5: "the winner mode is the one with trajectory
    # waypoints closest to the ground truth, measured with an L1 loss on the
    # Laplacian centroids"). This is detached so assignment is independent of the
    # supervision loss and of the predicted scale -- previously the laplace path
    # selected by NLL, letting a mode win by inflating/among-collapsing sigma.
    with torch.no_grad():
        select_metric = (abs_error * mask).sum(dim=(3, 4)) / denom  # [B, N, F]

    # Supervision LOSS per mode: NLL for laplace, masked L1 for smooth_l1.
    if future_log_scale is None:
        mode_loss = select_metric
    else:
        scale = future_log_scale.exp().clamp_min(1e-3)
        # Bound the per-point NLL ratio. With future_log_scale floored at -2 in
        # the model (sigma_min ~= 0.135 m) a normal waypoint stays well under
        # this cap; the clamp only defuses rare large-error/small-sigma winners
        # whose d/d(log_sigma) = -abs_error/sigma would otherwise spike the
        # refinement-head gradient (the 2026-06-20 NaN/clip source). clamp(max)
        # passes gradient through below the cap and zeroes it above, so the
        # pathological tail contributes no gradient.
        ratio = (abs_error / scale).clamp(max=50.0)
        nll = ratio + future_log_scale
        nll = nll * mask
        mode_loss = nll.sum(dim=(3, 4)) / denom

    winner = select_metric.argmin(dim=2)

    b, n, t = pred_xy.shape[0], pred_xy.shape[1], pred_xy.shape[3]
    f = pred_xy.shape[2]
    k = max(1, min(int(top_k), f))
    # The k modes with the smallest L1-centroid selection metric (k=1 -> plain
    # winner-takes-all). Selection uses L1, supervision uses mode_loss.
    topk_modes = (-select_metric).topk(k, dim=2).indices  # [B, N, k]

    if future_log_scale is None:
        # Mean smooth-L1 over the k selected modes, only on valid future points.
        point_mask = (matched_future_mask & valid_query[:, :, None])[:, :, None, :]
        gather_xy = topk_modes[:, :, :, None, None].expand(b, n, k, t, 2)
        sel_xy = pred_xy.gather(dim=2, index=gather_xy)  # [B, N, k, T, 2]
        target_xy = matched_gt_future[:, :, None].expand(b, n, k, t, 2)
        sel_point_mask = point_mask.expand(b, n, k, t)
        loss_reg = F.smooth_l1_loss(sel_xy[sel_point_mask], target_xy[sel_point_mask])
    else:
        # Mean per-mode NLL over the k selected modes for supervised queries.
        sel_loss = mode_loss.gather(dim=2, index=topk_modes)  # [B, N, k]
        loss_reg = sel_loss[valid_query].mean()
    loss_ce = F.cross_entropy(pred["mode_logits"][valid_query], winner[valid_query])
    return {
        "loss_forecast_reg": loss_reg,
        "loss_mode_ce": loss_ce,
        "loss_forecast": loss_reg + mode_ce_weight * loss_ce,
    }


def velocity_loss(
    pred: dict[str, torch.Tensor],
    aligned: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """SmoothL1 auxiliary loss for learned query velocity.

    Args:
        pred:
            Prediction dict. Uses optional ``pred_velocity [B,N,2]``.
        aligned:
            Query-aligned GT dict from :func:`align_gt_to_queries`.

    Returns:
        Dict with ``loss_velocity`` and ``velocity_supervised_queries``.
    """

    pred_velocity = pred.get("pred_velocity")
    if pred_velocity is None:
        zero = pred["poses"].sum() * 0.0
        return {
            "loss_velocity": zero,
            "velocity_supervised_queries": zero,
            "velocity_l2": zero,
        }
    valid = aligned["query_object_targets"].bool() & aligned["query_gt_velocity_valid"].bool()
    if not valid.any():
        zero = pred_velocity.sum() * 0.0
        return {
            "loss_velocity": zero,
            "velocity_supervised_queries": zero,
            "velocity_l2": zero,
        }
    target = aligned["query_gt_velocity"]
    loss = F.smooth_l1_loss(pred_velocity[valid], target[valid])
    l2 = (pred_velocity[valid] - target[valid]).norm(dim=-1).mean()
    return {
        "loss_velocity": loss,
        "velocity_supervised_queries": valid.float().sum(),
        "velocity_l2": l2,
    }


def yaw_motion_consistency_loss(
    pred: dict[str, torch.Tensor],
    aligned: dict[str, torch.Tensor],
    object_targets: torch.Tensor,
    *,
    v_thresh_m: float = 1.0,
    use_velocity_dir: bool = False,
) -> dict[str, torch.Tensor]:
    """Penalize disagreement between predicted yaw and GT motion direction.

    Moving actors should face the way they move. For every matched, moving query
    we add ``1 - cos(yaw - dir)`` where ``dir`` is the GT next-step displacement
    direction. The term is computed entirely in the ego frame (the detection yaw
    and ``pred_velocity`` are both ego-frame), and is masked to actors whose GT
    next-step displacement exceeds ``v_thresh_m`` so stationary actors (whose
    heading is unconstrained by motion) are excluded.

    Args:
        pred:
            Prediction dict from a refinement block.
        aligned:
            Query-aligned GT dict from :func:`align_gt_to_queries`.
        object_targets:
            ``[B,N]`` gate (typically the forecast true-positive gate) selecting
            which queries are supervised.
        v_thresh_m:
            Minimum GT next-step displacement (meters) to count as moving.
        use_velocity_dir:
            When true and ``pred_velocity`` is present, also align the predicted
            velocity direction with the GT motion direction. Intended for
            ``forecast_base_mode=learned_velocity`` (the finite-difference future
            headings are degenerate w.r.t. their own waypoints, so they are not
            constrained here).

    Returns:
        Dict with ``loss_yaw_consistency`` and ``yaw_supervised_queries``.
    """

    boxes = current_boxes_from_pred(pred)
    yaw = boxes[..., 6]  # [B, N]
    gt_future = aligned["query_gt_future"]  # [B, N, T_future, 2]
    gt_future_mask = aligned["query_gt_future_mask"]  # [B, N, T_future]
    gt_boxes = aligned["query_gt_boxes"]  # [B, N, 7]

    direction = gt_future[:, :, 0, :] - gt_boxes[:, :, :2]  # [B, N, 2]
    speed = direction.norm(dim=-1)  # [B, N]
    moving = (
        (speed > v_thresh_m)
        & gt_future_mask[:, :, 0].bool()
        & object_targets.bool()
    )
    if not moving.any():
        zero = yaw.sum() * 0.0
        return {"loss_yaw_consistency": zero, "yaw_supervised_queries": zero}

    # Work with unit direction vectors and dot products rather than atan2 angle
    # differences: atan2 has an undefined (NaN) gradient at the origin, and
    # ``pred_velocity`` is zero-initialized, which would inject NaNs early.
    # ``eps`` floors every denominator so a (near-)zero vector yields a finite,
    # bounded gradient instead. ``1 - u.v`` equals ``1 - cos(angle diff)`` for
    # unit vectors.
    eps = 1e-6
    target_dir = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    yaw_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # [B, N, 2]
    loss = (1.0 - (yaw_dir * target_dir).sum(dim=-1))[moving].mean()
    num_terms = 1
    pred_velocity = pred.get("pred_velocity") if use_velocity_dir else None
    if pred_velocity is not None:
        pred_dir = pred_velocity / pred_velocity.norm(dim=-1, keepdim=True).clamp_min(eps)
        loss = loss + (1.0 - (pred_dir * target_dir).sum(dim=-1))[moving].mean()
        num_terms = 2
    return {
        "loss_yaw_consistency": loss / num_terms,
        "yaw_supervised_queries": moving.float().sum(),
    }


def detra_loss(
    pred: dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    gt_types: torch.Tensor | None,
    gt_future: torch.Tensor,
    gt_future_mask: torch.Tensor,
    proposal_outputs: dict[str, torch.Tensor] | None = None,
    data_cfg: DataConfig | None = None,
    proposal_output_stride: int = 4,
    proposal_heatmap_weight: float = 1.0,
    proposal_box_weight: float = 1.0,
    proposal_giou_weight: float = 0.1,
    forecast_weight: float = 0.1,
    aux_weight: float = 1.0,
    init_weight: float = 1.0,
    detection_box_l1_weight: float = 0.01,
    detection_giou_weight: float = 0.1,
    detection_class_weight: float = 1.0,
    detection_z_weight: float = 0.0,
    detection_height_weight: float = 0.0,
    forecast_gate_radius_m: float | None = None,
    forecast_tp_iou: float | None = 0.5,
    forecast_loss_type: str = "laplace",
    forecast_top_k: int = 1,
    velocity_weight: float = 0.0,
    gt_velocity: torch.Tensor | None = None,
    gt_velocity_valid: torch.Tensor | None = None,
    yaw_consistency_weight: float = 0.0,
    yaw_consistency_vthresh_m: float = 1.0,
    yaw_use_velocity_dir: bool = False,
    past_weight: float = 0.0,
    gt_history: torch.Tensor | None = None,
    gt_history_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute final + intermediate DeTra losses.

    Args:
        pred:
            Output from ``DeTraMini``. May contain ``aux_outputs``.
        gt_boxes:
            ``[B, M, 7]``.
        gt_valid:
            ``[B, M]``.
        gt_types:
            Optional Waymo type ids ``[B, M]``.
        gt_future:
            ``[B, M, T_future, 2]``.
        gt_future_mask:
            ``[B, M, T_future]``.
        proposal_outputs:
            Optional dense proposal head outputs for initial proposal loss.
        data_cfg:
            Required when ``proposal_outputs`` is provided.
        proposal_output_stride:
            Dense proposal map stride.
        proposal_heatmap_weight:
            Weight for the initial proposal heatmap loss.
        proposal_box_weight:
            Weight for the initial proposal box regression loss.
        proposal_giou_weight:
            Weight for the initial proposal BEV gIoU loss.
        detection_box_l1_weight:
            Weight (paper beta) on the matched 7-dim smooth-L1 box term. The
            paper uses beta=0.01; the previous default left this at the
            ``detection_loss`` default of 1.0 (100x too strong), which fights
            the gIoU localization term. Default here is the paper value.
        detection_giou_weight:
            Weight (paper gamma) on the BEV gIoU term. Paper uses gamma=0.1.
        detection_class_weight:
            Weight on the per-class cross-entropy term.
        forecast_weight:
            DeTra paper uses alpha=0.1 for forecasting loss.
        aux_weight:
            Weight for each intermediate block loss.
        forecast_gate_radius_m:
            Optional true-positive proxy for forecast supervision. When set,
            it takes precedence over ``forecast_tp_iou``: only matched queries
            whose current center is within this radius of the matched GT center
            receive forecast loss. Useful as a diagnostic.
        forecast_tp_iou:
            Paper-style true-positive gate. When ``forecast_gate_radius_m`` is
            ``None``, forecast loss is applied only to matched queries whose
            detection box has BEV IoU above this threshold with the matched GT
            box (the paper uses 0.5). Set to ``None`` to supervise all matched
            queries.
        forecast_loss_type:
            ``laplace`` for paper-style learned-scale WTA NLL, or ``smooth_l1``
            for fixed-scale overfit diagnostics.
        velocity_weight:
            Weight for auxiliary learned-velocity SmoothL1 loss.
        gt_velocity:
            Optional GT velocity ``[B,M,2]``.
        gt_velocity_valid:
            Optional GT velocity-valid mask ``[B,M]``.

    Returns:
        Dict containing scalar losses and the final ``matches``/``aligned``
        objects for debugging.
    """

    outputs = [pred] + list(pred.get("aux_outputs", [])[:-1])
    total = gt_boxes.new_tensor(0.0)
    logs: dict[str, torch.Tensor | MatchResult | dict[str, torch.Tensor]] = {}

    final_matches = None
    final_aligned = None
    for idx, output in enumerate(outputs):
        matches = hungarian_match(output, gt_boxes, gt_valid)
        aligned = align_gt_to_queries(
            matches,
            gt_boxes,
            gt_types,
            gt_future,
            gt_future_mask,
            num_queries=output["object_logits"].shape[1],
            gt_velocity=gt_velocity,
            gt_velocity_valid=gt_velocity_valid,
        )
        det = detection_loss(
            output,
            aligned,
            box_l1_weight=detection_box_l1_weight,
            giou_weight=detection_giou_weight,
            class_weight=detection_class_weight,
            z_weight=detection_z_weight,
            height_weight=detection_height_weight,
        )
        forecast_targets = aligned["query_object_targets"]
        forecast_candidate_targets = forecast_targets.clone()
        if forecast_gate_radius_m is not None:
            pred_boxes = current_boxes_from_pred(output)
            center_error = (pred_boxes[..., :2] - aligned["query_gt_boxes"][..., :2]).norm(dim=-1)
            forecast_targets = forecast_targets * (center_error < forecast_gate_radius_m).float()
        elif forecast_tp_iou is not None:
            pred_boxes = current_boxes_from_pred(output)
            iou = rotated_bev_iou(pred_boxes, aligned["query_gt_boxes"])
            forecast_targets = forecast_targets * (iou > forecast_tp_iou).float()
        forecast = wta_forecasting_loss(
            output,
            aligned["query_gt_future"],
            aligned["query_gt_future_mask"],
            forecast_targets,
            loss_type=forecast_loss_type,
            top_k=forecast_top_k,
        )
        vel = velocity_loss(output, aligned)
        yaw = yaw_motion_consistency_loss(
            output,
            aligned,
            forecast_targets,
            v_thresh_m=yaw_consistency_vthresh_m,
            use_velocity_dir=yaw_use_velocity_dir,
        )
        if "past_xy" in output and gt_history is not None:
            past_aligned = align_gt_history_to_queries(
                matches, gt_history, gt_history_mask, num_queries=output["object_logits"].shape[1]
            )
            past = past_trajectory_loss(
                output,
                past_aligned["query_gt_past"],
                past_aligned["query_gt_past_mask"],
                forecast_targets,
            )
        else:
            zero = total.new_tensor(0.0)
            past = {"loss_past": zero, "past_supervised_queries": zero}
        loss = (
            det["loss_detection"]
            + forecast_weight * forecast["loss_forecast"]
            + velocity_weight * vel["loss_velocity"]
            + yaw_consistency_weight * yaw["loss_yaw_consistency"]
            + past_weight * past["loss_past"]
        )
        weight = 1.0 if idx == 0 else aux_weight
        total = total + weight * loss

        prefix = "final" if idx == 0 else f"aux_{idx}"
        logs[f"{prefix}_loss_cls"] = det["loss_cls"]
        logs[f"{prefix}_loss_class_ce"] = det["loss_class_ce"]
        logs[f"{prefix}_loss_box_l1"] = det["loss_box_l1"]
        logs[f"{prefix}_loss_giou"] = det["loss_giou"]
        logs[f"{prefix}_loss_z_l1"] = det["loss_z_l1"]
        logs[f"{prefix}_loss_height_l1"] = det["loss_height_l1"]
        logs[f"{prefix}_loss_detection"] = det["loss_detection"]
        logs[f"{prefix}_loss_forecast_reg"] = forecast["loss_forecast_reg"]
        logs[f"{prefix}_loss_mode_ce"] = forecast["loss_mode_ce"]
        logs[f"{prefix}_loss_mode"] = forecast["loss_mode_ce"]
        logs[f"{prefix}_loss_forecast"] = forecast["loss_forecast"]
        logs[f"{prefix}_loss_velocity"] = vel["loss_velocity"]
        logs[f"{prefix}_velocity_supervised_queries"] = vel["velocity_supervised_queries"]
        logs[f"{prefix}_velocity_l2"] = vel["velocity_l2"]
        logs[f"{prefix}_loss_yaw_consistency"] = yaw["loss_yaw_consistency"]
        logs[f"{prefix}_yaw_supervised_queries"] = yaw["yaw_supervised_queries"]
        logs[f"{prefix}_loss_past"] = past["loss_past"]
        logs[f"{prefix}_past_supervised_queries"] = past["past_supervised_queries"]
        logs[f"{prefix}_forecast_candidate_queries"] = forecast_candidate_targets.bool().float().sum()
        logs[f"{prefix}_forecast_supervised_queries"] = forecast_targets.bool().float().sum()
        logs[f"{prefix}_forecast_gate_removed_queries"] = (
            forecast_candidate_targets.bool() & ~forecast_targets.bool()
        ).float().sum()
        if idx == 0:
            final_matches = matches
            final_aligned = aligned

    logs["loss"] = total
    if proposal_outputs is not None:
        if data_cfg is None:
            raise ValueError("data_cfg is required when proposal_outputs is provided")
        init = proposal_loss(
            proposal_outputs,
            gt_boxes,
            gt_valid,
            data_cfg,
            output_stride=proposal_output_stride,
            heatmap_weight=proposal_heatmap_weight,
            box_weight=proposal_box_weight,
            giou_weight=proposal_giou_weight,
        )
        total = total + init_weight * init["loss_init"]
        logs.update(init)
        logs["loss"] = total
    logs["matches"] = final_matches
    logs["aligned"] = final_aligned
    return logs

"""Detection true-positive matching and the fixed-recall operating point.

DeTra reports forecasting metrics at a common detection operating point: the
score threshold that yields a fixed detection recall (80%) at IoU 0.5. This
module provides:

- numpy BEV IoU (axis-aligned, fully vectorized; rotated, center-gated) used to
  gate true positives,
- a per-frame greedy, score-ordered, one-to-one IoU matcher,
- a single-pass PR-curve construction over all frames to find the score
  threshold achieving a target recall.

Everything is pure numpy so it runs in any env. The matcher mirrors the greedy
``criterion="iou"`` logic in ``detra_repro/evaluate_detection_pr._match_predictions``
but returns the matched ``(pred_idx, gt_idx)`` pairs so the forecasting code can
score trajectories only on true positives.
"""

from __future__ import annotations

import numpy as np

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_TARGET_RECALL = 0.80


def axis_aligned_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Vectorized axis-aligned BEV IoU ``[P, G]`` ignoring heading.

    Boxes are ``[*, 7]`` as ``x,y,z,l,w,h,heading``; only ``x,y,l,w`` are used.
    """

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    p_half = np.clip(pred_boxes[:, 3:5], 1e-2, None) * 0.5
    g_half = np.clip(gt_boxes[:, 3:5], 1e-2, None) * 0.5
    p_lt = pred_boxes[:, :2] - p_half
    p_rb = pred_boxes[:, :2] + p_half
    g_lt = gt_boxes[:, :2] - g_half
    g_rb = gt_boxes[:, :2] + g_half

    lt = np.maximum(p_lt[:, None, :], g_lt[None, :, :])
    rb = np.minimum(p_rb[:, None, :], g_rb[None, :, :])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    p_area = (p_rb[:, 0] - p_lt[:, 0]) * (p_rb[:, 1] - p_lt[:, 1])
    g_area = (g_rb[:, 0] - g_lt[:, 0]) * (g_rb[:, 1] - g_lt[:, 1])
    union = p_area[:, None] + g_area[None, :] - inter
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


def _box_corners(box: np.ndarray) -> np.ndarray:
    """Return CCW BEV corners ``[4, 2]`` for one rotated box ``[7]``."""

    half_l = max(float(box[3]), 1e-3) * 0.5
    half_w = max(float(box[4]), 1e-3) * 0.5
    local = np.array(
        [[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]],
        dtype=np.float64,
    )
    c, s = np.cos(box[6]), np.sin(box[6])
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + box[:2].astype(np.float64)


def _poly_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman convex polygon clipping."""

    output = list(subject)
    for i in range(len(clipper)):
        if not output:
            break
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        edge = b - a

        def inside(p: np.ndarray) -> bool:
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-9

        def intersect(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            r = q - p
            denom = edge[0] * r[1] - edge[1] * r[0]
            if abs(denom) < 1e-12:
                return q
            t = (edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])) / -denom
            return p + t * r

        new_output: list[np.ndarray] = []
        prev = output[-1]
        prev_in = inside(prev)
        for curr in output:
            curr_in = inside(curr)
            if curr_in:
                if not prev_in:
                    new_output.append(intersect(prev, curr))
                new_output.append(curr)
            elif prev_in:
                new_output.append(intersect(prev, curr))
            prev, prev_in = curr, curr_in
        output = new_output
    return np.array(output, dtype=np.float64) if output else np.zeros((0, 2))


def rotated_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Rotated BEV IoU ``[P, G]``, center-gated for speed.

    Pairs whose centers are farther than the sum of half-diagonals cannot
    overlap and are skipped (IoU 0). Remaining pairs use exact polygon
    clipping. Pure numpy; no shapely dependency.
    """

    p, g = len(pred_boxes), len(gt_boxes)
    out = np.zeros((p, g), dtype=np.float32)
    if p == 0 or g == 0:
        return out
    p_rad = 0.5 * np.linalg.norm(np.clip(pred_boxes[:, 3:5], 1e-3, None), axis=1)
    g_rad = 0.5 * np.linalg.norm(np.clip(gt_boxes[:, 3:5], 1e-3, None), axis=1)
    centers_d = np.linalg.norm(pred_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=-1)
    gate = centers_d <= (p_rad[:, None] + g_rad[None, :])
    p_corners = [_box_corners(b) for b in pred_boxes]
    g_corners = [_box_corners(b) for b in gt_boxes]
    p_areas = np.clip(pred_boxes[:, 3], 1e-3, None) * np.clip(pred_boxes[:, 4], 1e-3, None)
    g_areas = np.clip(gt_boxes[:, 3], 1e-3, None) * np.clip(gt_boxes[:, 4], 1e-3, None)
    for i in range(p):
        for j in range(g):
            if not gate[i, j]:
                continue
            inter = _poly_area(_clip(p_corners[i], g_corners[j]))
            union = float(p_areas[i] + g_areas[j] - inter)
            if union > 1e-6:
                out[i, j] = inter / union
    return out


def iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray, mode: str) -> np.ndarray:
    """Dispatch to ``axis_aligned`` or ``rotated`` IoU."""

    if mode == "axis_aligned":
        return axis_aligned_iou_matrix(pred_boxes, gt_boxes)
    if mode == "rotated":
        return rotated_iou_matrix(pred_boxes, gt_boxes)
    raise ValueError("mode must be 'axis_aligned' or 'rotated'")


def match_frame(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    *,
    iou_thr: float = DEFAULT_IOU_THRESHOLD,
    iou_mode: str = "rotated",
    class_aware: bool = True,
) -> dict[str, object]:
    """Greedy score-ordered one-to-one IoU match for one frame.

    Returns a dict with:
        - ``pairs``: list of ``(pred_idx, gt_idx)`` true-positive matches
          (indices into the original arrays).
        - ``labels``: list of ``(score, is_tp)`` for every prediction.
        - ``num_gt``: number of ground-truth objects.
    Predictions are matched in descending score order; each GT is used once.
    """

    p = len(pred_boxes)
    g = len(gt_boxes)
    order = np.argsort(-pred_scores) if p else np.array([], dtype=int)
    iou = iou_matrix(pred_boxes, gt_boxes, iou_mode) if (p and g) else np.zeros((p, g))

    used = np.zeros(g, dtype=bool)
    pairs: list[tuple[int, int]] = []
    labels: list[tuple[float, int]] = []
    for pred_idx in order:
        best_iou = iou_thr
        best_gt = -1
        for gt_idx in range(g):
            if used[gt_idx]:
                continue
            if class_aware and int(pred_classes[pred_idx]) != int(gt_classes[gt_idx]):
                continue
            if iou[pred_idx, gt_idx] >= best_iou:
                best_iou = iou[pred_idx, gt_idx]
                best_gt = gt_idx
        if best_gt >= 0:
            used[best_gt] = True
            pairs.append((int(pred_idx), int(best_gt)))
            labels.append((float(pred_scores[pred_idx]), 1))
        else:
            labels.append((float(pred_scores[pred_idx]), 0))
    return {"pairs": pairs, "labels": labels, "num_gt": int(g)}


def find_operating_point(
    frame_matches: list[dict[str, object]],
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> dict[str, float]:
    """Find the highest score threshold with detection recall >= target.

    Builds a single global PR curve from all frames' ``labels`` and ``num_gt``,
    sorted by descending score. Returns the threshold and the precision/recall
    achieved there. If the target recall is never reached, returns the lowest
    score seen (maximum achievable recall).
    """

    all_labels: list[tuple[float, int]] = []
    total_gt = 0
    for fm in frame_matches:
        all_labels.extend(fm["labels"])  # type: ignore[arg-type]
        total_gt += int(fm["num_gt"])  # type: ignore[index]
    if total_gt == 0 or not all_labels:
        return {
            "threshold": 1.0,
            "achieved_recall": 0.0,
            "precision_at_op": 0.0,
            "target_recall": float(target_recall),
            "total_gt": float(total_gt),
        }

    all_labels.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    chosen = None
    last = all_labels[-1]
    for score, is_tp in all_labels:
        tp += is_tp
        fp += 1 - is_tp
        recall = tp / total_gt
        if recall >= target_recall:
            chosen = (score, recall, tp / max(tp + fp, 1))
            break
    if chosen is None:
        # Target never reached; report the maximum-recall endpoint.
        recall = tp / total_gt
        chosen = (last[0], recall, tp / max(tp + fp, 1))
    return {
        "threshold": float(chosen[0]),
        "achieved_recall": float(chosen[1]),
        "precision_at_op": float(chosen[2]),
        "target_recall": float(target_recall),
        "total_gt": float(total_gt),
    }

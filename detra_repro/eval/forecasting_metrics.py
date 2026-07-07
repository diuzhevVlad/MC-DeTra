"""Multimodal trajectory metrics via the official Argoverse 2 implementation.

DeTra follows the Argoverse 2 Motion forecasting metrics: minADE, minFDE, miss
rate, and the Brier variant, reported at ``K=1`` (most-likely mode) and ``K=6``
(best of K). The per-object math is delegated to ``av2.datasets.
motion_forecasting.eval.metrics`` (``compute_ade``, ``compute_fde``,
``compute_brier_fde``, ``compute_is_missed_prediction``) so the numbers match the
official benchmark exactly. This module only adds the masking, K-selection, and
stationary/dynamic + class macro-averaging that the cached-Waymo setting needs.

Conventions:
- A prediction for one object is ``[K, T, 2]`` absolute local-frame waypoints.
- Ground truth is ``[T, 2]`` with a per-step validity ``mask [T]``. The av2
  functions take contiguous ``(K, N, 2)``/``(N, 2)`` with no mask, so we slice
  to the valid steps before calling them (ADE then averages over those steps,
  FDE uses the last one).
- ``minADE_K``/``minFDE_K``: min over K of av2's per-mode ADE/FDE.
- ``MR_K``: av2 ``compute_is_missed_prediction`` (FDE > 2 m); a miss at K means
  every mode missed.
- ``brier-minFDE_K``: av2 ``compute_brier_fde`` (FDE + ``(1-p)^2``), reported at
  the FDE-best mode.
- ``K=1`` variants use the single most-likely mode (argmax probability).
"""

from __future__ import annotations

import numpy as np

try:
    from av2.datasets.motion_forecasting.eval.metrics import (
        compute_ade,
        compute_brier_fde,
        compute_fde,
        compute_is_missed_prediction,
    )
except ModuleNotFoundError:
    def compute_ade(forecasted_trajectories: np.ndarray, gt_trajectory: np.ndarray) -> np.ndarray:
        """Fallback equivalent to AV2 ADE for one actor and K trajectories."""

        return np.linalg.norm(forecasted_trajectories - gt_trajectory, axis=2).mean(axis=1)

    def compute_fde(forecasted_trajectories: np.ndarray, gt_trajectory: np.ndarray) -> np.ndarray:
        """Fallback equivalent to AV2 FDE for one actor and K trajectories."""

        return np.linalg.norm(forecasted_trajectories[:, -1] - gt_trajectory[-1], axis=-1)

    def compute_is_missed_prediction(
        forecasted_trajectories: np.ndarray,
        gt_trajectory: np.ndarray,
        miss_threshold_m: float = 2.0,
    ) -> np.ndarray:
        """Fallback equivalent to AV2 miss-rate decision."""

        return compute_fde(forecasted_trajectories, gt_trajectory) > miss_threshold_m

    def compute_brier_fde(
        forecasted_trajectories: np.ndarray,
        gt_trajectory: np.ndarray,
        forecast_probabilities: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Fallback equivalent to AV2 Brier-FDE."""

        probs = forecast_probabilities.astype(np.float64)
        if normalize:
            probs = probs / np.sum(probs)
        if np.logical_or(probs < 0.0, probs > 1.0).any():
            raise ValueError("At least one forecast probability falls outside [0, 1].")
        return compute_fde(forecasted_trajectories, gt_trajectory) + np.square(1.0 - probs)

MISS_RATE_RADIUS_M = 2.0
MOVING_DISP_THRESHOLD_M = 2.0
CLASS_NAMES = ("vehicle", "pedestrian", "cyclist")


def _last_valid_index(mask: np.ndarray) -> int:
    """Return the last index where ``mask`` is True, or -1 if none."""

    valid = np.nonzero(mask)[0]
    if len(valid) == 0:
        return -1
    return int(valid[-1])


def trajectory_metrics(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    mask: np.ndarray,
    mode_probs: np.ndarray,
) -> dict[str, float] | None:
    """Compute per-object forecasting metrics for one matched true positive.

    Delegates the displacement math to the official av2 metric functions.

    Args:
        pred_xy: ``[K, T, 2]`` predicted waypoints (all modes).
        gt_xy: ``[T, 2]`` ground-truth future waypoints.
        mask: ``[T]`` boolean validity of each GT step.
        mode_probs: ``[K]`` mode probabilities (already softmaxed, sum to 1).

    Returns:
        Dict of scalar metrics, or ``None`` if there is no valid future step.
        Keys: ``minade_k1, minfde_k1, mr_k1, minade_k6, minfde_k6, mr_k6,
        brier_minfde_k6, gt_final_disp``.
    """

    mask = mask.astype(bool)
    last = _last_valid_index(mask)
    if last < 0:
        return None
    k = pred_xy.shape[0]

    # av2 expects contiguous trajectories with no gaps; restrict to valid steps.
    valid_idx = np.nonzero(mask)[0]
    pred_valid = pred_xy[:, valid_idx, :].astype(np.float64)  # [K, V, 2]
    gt_valid = gt_xy[valid_idx, :].astype(np.float64)  # [V, 2]
    probs = np.asarray(mode_probs, dtype=np.float64).clip(0.0, 1.0)

    ade_per_mode = compute_ade(pred_valid, gt_valid)  # [K]
    fde_per_mode = compute_fde(pred_valid, gt_valid)  # [K]
    missed = compute_is_missed_prediction(pred_valid, gt_valid, MISS_RATE_RADIUS_M)
    brier_fde_per_mode = compute_brier_fde(pred_valid, gt_valid, probs, normalize=False)

    top_mode = int(np.argmax(probs)) if k > 1 else 0

    # K=1: most-likely mode.
    minade_k1 = float(ade_per_mode[top_mode])
    minfde_k1 = float(fde_per_mode[top_mode])
    mr_k1 = float(missed[top_mode])

    # K=K: best of all modes (FDE-best selects the winner, as in av2).
    fde_best_mode = int(np.argmin(fde_per_mode))
    minade_k6 = float(ade_per_mode.min())
    minfde_k6 = float(fde_per_mode[fde_best_mode])
    mr_k6 = float(missed.all())
    brier_minfde_k6 = float(brier_fde_per_mode[fde_best_mode])

    gt_final_disp = float(np.linalg.norm(gt_valid[-1] - gt_valid[0])) if last > 0 else 0.0
    # gt_valid[0] is the first future step, not the current pose; the caller
    # overrides bucketing with ``current_disp`` from the matched-GT center.

    return {
        "minade_k1": minade_k1,
        "minfde_k1": minfde_k1,
        "mr_k1": mr_k1,
        "minade_k6": minade_k6,
        "minfde_k6": minfde_k6,
        "mr_k6": mr_k6,
        "brier_minfde_k6": brier_minfde_k6,
        "gt_final_disp": gt_final_disp,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""

    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


_METRIC_KEYS = (
    "minade_k1",
    "minfde_k1",
    "mr_k1",
    "minade_k6",
    "minfde_k6",
    "mr_k6",
    "brier_minfde_k6",
)


def aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    """Mean-aggregate a list of per-object metric dicts.

    Returns zeros with a ``count`` of 0 when ``records`` is empty.
    """

    out: dict[str, float] = {"count": float(len(records))}
    if not records:
        for key in _METRIC_KEYS:
            out[key] = 0.0
        return out
    for key in _METRIC_KEYS:
        out[key] = float(np.mean([r[key] for r in records]))
    return out


def aggregate_bucketed(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate overall and split into stationary / dynamic buckets.

    A record is "dynamic" when its ``current_disp`` (GT final displacement from
    the current center) exceeds :data:`MOVING_DISP_THRESHOLD_M`. The
    ``macro`` block is the unweighted mean of the stationary and dynamic blocks,
    matching DeTra's stationary/dynamic macro-averaging. Records missing
    ``current_disp`` fall back to ``gt_final_disp``.
    """

    stationary: list[dict[str, float]] = []
    dynamic: list[dict[str, float]] = []
    for r in records:
        disp = r.get("current_disp", r.get("gt_final_disp", 0.0))
        (dynamic if disp > MOVING_DISP_THRESHOLD_M else stationary).append(r)

    overall = aggregate(records)
    stat = aggregate(stationary)
    dyn = aggregate(dynamic)
    macro: dict[str, float] = {"count": overall["count"]}
    for key in _METRIC_KEYS:
        vals = []
        if stationary:
            vals.append(stat[key])
        if dynamic:
            vals.append(dyn[key])
        macro[key] = float(np.mean(vals)) if vals else 0.0
    return {
        "overall": overall,
        "stationary": stat,
        "dynamic": dyn,
        "macro": macro,
    }


def aggregate_by_class(
    records: list[dict[str, float]],
    class_key: str = "class_id",
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-class bucketed aggregation plus a class-macro average.

    Returns ``{class_name: bucketed, ..., "class_macro": bucketed}`` where the
    class-macro is the unweighted mean over classes that have records.
    """

    out: dict[str, dict[str, dict[str, float]]] = {}
    per_class_overall: list[dict[str, float]] = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        cls_records = [r for r in records if int(r.get(class_key, 0)) == cls_idx]
        out[name] = aggregate_bucketed(cls_records)
        if cls_records:
            per_class_overall.append(out[name]["overall"])

    class_macro: dict[str, float] = {"count": float(len(per_class_overall))}
    for key in _METRIC_KEYS:
        vals = [block[key] for block in per_class_overall]
        class_macro[key] = float(np.mean(vals)) if vals else 0.0
    out["class_macro"] = {"overall": class_macro}
    return out

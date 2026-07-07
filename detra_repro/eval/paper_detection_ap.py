"""Strict vehicle-only BEV AP for the DeTra Waymo task population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from detra_repro.eval.operating_point import rotated_iou_matrix


IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def vehicle_confidence(object_logits: np.ndarray, class_logits: np.ndarray) -> np.ndarray:
    object_score = 1.0 / (
        1.0 + np.exp(-np.clip(object_logits.astype(np.float64), -80.0, 80.0))
    )
    shifted = class_logits.astype(np.float64) - class_logits.max(axis=-1, keepdims=True)
    class_prob = np.exp(shifted)
    class_prob /= class_prob.sum(axis=-1, keepdims=True)
    return (object_score * class_prob[..., 0]).astype(np.float32)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-8
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    return (
        (abs(o1) <= eps and on_segment(a, c, b))
        or (abs(o2) <= eps and on_segment(a, d, b))
        or (abs(o3) <= eps and on_segment(c, a, d))
        or (abs(o4) <= eps and on_segment(c, b, d))
    )


def _box_corners(box: np.ndarray) -> np.ndarray:
    half_l, half_w = max(float(box[3]), 1e-3) * 0.5, max(float(box[4]), 1e-3) * 0.5
    local = np.asarray(
        [[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]],
        dtype=np.float64,
    )
    c, s = np.cos(float(box[6])), np.sin(float(box[6]))
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + box[None, :2]


def box_overlaps_polygon(box: np.ndarray, polygon: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    corners = _box_corners(box)
    if any(_point_in_polygon(corner, polygon) for corner in corners):
        return True
    if any(_point_in_polygon(vertex, corners) for vertex in polygon):
        return True
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for j in range(len(polygon)):
            if _segments_intersect(a, b, polygon[j], polygon[(j + 1) % len(polygon)]):
                return True
    return False


def boxes_overlap_nlz(
    boxes: np.ndarray, nlz_xy: np.ndarray, nlz_offsets: np.ndarray
) -> np.ndarray:
    polygons = [
        nlz_xy[int(nlz_offsets[i]) : int(nlz_offsets[i + 1])]
        for i in range(len(nlz_offsets) - 1)
    ]
    return np.asarray(
        [any(box_overlaps_polygon(box, polygon) for polygon in polygons) for box in boxes],
        dtype=bool,
    )


def score_first_ap(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_frame: np.ndarray,
    gt_boxes: np.ndarray,
    gt_frame: np.ndarray,
    *,
    iou_threshold: float,
    pred_overlap_nlz: np.ndarray | None = None,
) -> dict:
    return score_first_ap_multi(
        pred_boxes,
        pred_scores,
        pred_frame,
        gt_boxes,
        gt_frame,
        iou_thresholds=(iou_threshold,),
        pred_overlap_nlz=pred_overlap_nlz,
    )[float(iou_threshold)]


def score_first_ap_multi(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_frame: np.ndarray,
    gt_boxes: np.ndarray,
    gt_frame: np.ndarray,
    *,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
    pred_overlap_nlz: np.ndarray | None = None,
) -> dict[float, dict]:
    """Compute multiple AP thresholds while evaluating each rotated IoU only once."""

    states = {
        float(threshold): {
            "labels": np.full(len(pred_boxes), -1, dtype=np.int8),
            "ignored_nlz": 0,
        }
        for threshold in iou_thresholds
    }
    frame_ids = np.union1d(np.unique(pred_frame), np.unique(gt_frame))
    for frame_id in frame_ids:
        pred_indices = np.flatnonzero(pred_frame == frame_id)
        gt_indices = np.flatnonzero(gt_frame == frame_id)
        local_order = np.argsort(-pred_scores[pred_indices], kind="stable")
        pred_indices = pred_indices[local_order]
        frame_iou = rotated_iou_matrix(pred_boxes[pred_indices], gt_boxes[gt_indices])
        for threshold, state in states.items():
            matched_local = np.zeros(len(gt_indices), dtype=bool)
            for local_pred, pred_idx in enumerate(pred_indices):
                candidates = np.flatnonzero(
                    (~matched_local) & (frame_iou[local_pred] >= threshold)
                )
                if len(candidates):
                    best_local = candidates[np.argmax(frame_iou[local_pred, candidates])]
                    matched_local[best_local] = True
                    state["labels"][pred_idx] = 1
                elif pred_overlap_nlz is not None and bool(pred_overlap_nlz[pred_idx]):
                    state["ignored_nlz"] += 1
                else:
                    state["labels"][pred_idx] = 0

    results: dict[float, dict] = {}
    global_order = np.argsort(-pred_scores, kind="stable")
    for threshold, state in states.items():
        valid_order = global_order[state["labels"][global_order] >= 0]
        scores = pred_scores[valid_order].astype(np.float32, copy=False)
        tp = state["labels"][valid_order].astype(np.int64, copy=False)
        fp = 1 - tp
        tp_cum = np.cumsum(tp, dtype=np.int64)
        fp_cum = np.cumsum(fp, dtype=np.int64)
        recall = tp_cum / max(len(gt_boxes), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
        recall_prev = np.concatenate([np.zeros((1,), dtype=np.float64), recall[:-1]])
        ap = float(np.sum((recall - recall_prev) * envelope)) if len(recall) else 0.0
        tp_total = int(tp_cum[-1]) if len(tp_cum) else 0
        results[threshold] = {
            "ap": ap,
            "max_recall": float(recall[-1]) if len(recall) else 0.0,
            "num_gt": int(len(gt_boxes)),
            "num_predictions_scored": int(len(valid_order)),
            "num_predictions_ignored_nlz": int(state["ignored_nlz"]),
            "tp": tp_total,
            "fp": int(fp_cum[-1]) if len(fp_cum) else 0,
            "fn": int(len(gt_boxes) - tp_total),
            "pr_curve": {
                "score": scores,
                "precision": precision.astype(np.float32),
                "recall": recall.astype(np.float32),
                "interpolated_precision": envelope.astype(np.float32),
            },
        }
    return results


def evaluate_dump(dump_dir: str | Path, *, allow_partial: bool = False) -> dict:
    dump_dir = Path(dump_dir)
    meta = json.loads((dump_dir / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("paper_strict", False):
        raise ValueError("strict paper detection AP requires a paper_strict prediction dump")
    if not allow_partial and not meta.get("complete_validation", False):
        raise ValueError("strict paper detection AP requires the complete cache-v2 validation split")
    data = dict(np.load(dump_dir / "predictions.npz", allow_pickle=False))
    pred_frame = np.repeat(data["frame_ids"], np.diff(data["pred_frame_offsets"]))
    gt_frame = np.repeat(data["frame_ids"], np.diff(data["gt_frame_offsets"]))
    result = {
        "metric": "score_first_all_score_interpolated_bev_ap",
        "meta": meta,
        "num_frames": int(len(data["frame_ids"])),
        "iou_thresholds": {},
    }
    threshold_results = score_first_ap_multi(
        data["pred_boxes"],
        data["pred_scores"],
        pred_frame,
        data["gt_boxes"],
        gt_frame,
        pred_overlap_nlz=data.get("pred_overlap_nlz"),
    )
    for threshold, values in threshold_results.items():
        result["iou_thresholds"][f"{threshold:.2f}"] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-partial", action="store_true", help="Allow diagnostic partial dumps.")
    args = parser.parse_args()
    payload = evaluate_dump(args.dump_dir, allow_partial=args.allow_partial)
    summary = {
        threshold: {
            key: values[key]
            for key in ("ap", "max_recall", "tp", "fp", "fn")
        }
        for threshold, values in payload["iou_thresholds"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        curve_path = path.with_name(f"{path.stem}_pr_curves.npz")
        curve_arrays = {}
        for threshold, values in payload["iou_thresholds"].items():
            curve = values.pop("pr_curve")
            keys = {}
            for name, array in curve.items():
                key = f"iou_{threshold.replace('.', '_')}_{name}"
                curve_arrays[key] = array
                keys[name] = key
            values["pr_curve"] = {"npz": curve_path.name, "keys": keys}
        np.savez_compressed(curve_path, **curve_arrays)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Forecasting evaluation over a prediction dump (env-agnostic, pure numpy).

Loads ``predictions.npz`` from :mod:`detra_repro.eval.dump_predictions`, finds
the fixed detection-recall operating point (default 80% recall at IoU 0.5), then
computes Argoverse-2-style minADE/minFDE/MR/brier forecasting metrics over the
true positives at that operating point. Metrics are reported at K=1 (most-likely
mode) and K=6 (best of K), bucketed into stationary/dynamic with macro-averaging,
and per class.

The forecasting true positives are the IoU-matched ``(pred, gt)`` pairs among
predictions whose score passes the operating-point threshold.

Paper-facing Waymo reporting is vehicle-only and uses the stationary/dynamic
macro average as the primary row. ``overall`` is still emitted, but it is
diagnostic because stationary vehicles dominate the object count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from detra_repro.eval.forecasting_metrics import (
    _softmax,
    aggregate_bucketed,
    aggregate_by_class,
    trajectory_metrics,
)
from detra_repro.eval.operating_point import find_operating_point, match_frame


PAPER_CLASS_FILTER = 0
PAPER_PRIMARY_BUCKET = "macro"


def _load_dump(dump_dir: Path) -> tuple[dict, dict]:
    with (dump_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    data = dict(np.load(dump_dir / "predictions.npz", allow_pickle=False))
    return meta, data


def _frame_slices(data: dict) -> list[tuple[slice, slice]]:
    """Return per-frame (pred_slice, gt_slice) from the offset arrays."""

    po = data["pred_frame_offsets"]
    go = data["gt_frame_offsets"]
    return [
        (slice(int(po[i]), int(po[i + 1])), slice(int(go[i]), int(go[i + 1])))
        for i in range(len(po) - 1)
    ]


def evaluate(args: argparse.Namespace) -> dict:
    dump_dir = Path(args.dump_dir)
    meta, data = _load_dump(dump_dir)
    slices = _frame_slices(data)

    pred_boxes = data["pred_boxes"]
    pred_scores = data["pred_scores"]
    pred_classes = data["pred_classes"]
    pred_future = data["pred_future_xy"]
    pred_mode_logits = data["pred_mode_logits"]
    gt_boxes = data["gt_boxes"]
    gt_classes = data["gt_classes"]
    gt_future = data["gt_future_xy"]
    gt_future_mask = data["gt_future_mask"]

    # Optional single-class restriction. The paper's WOD Table 2 is vehicle-only,
    # so the operating point and recall must be computed over vehicle GT and
    # vehicle predictions only -- pooling all classes selects a different (lower)
    # global threshold and inflates recall with easier pedestrian/stationary
    # matches. The default class_filter is vehicle; pass ``--all-classes`` for
    # the old pooled diagnostic. The per-frame boolean masks below are applied
    # before any matching so the offsets in ``slices`` still index the original
    # arrays correctly.
    class_filter = args.class_filter

    def _pred_keep(ps: slice) -> np.ndarray:
        if class_filter is None:
            return np.ones(pred_boxes[ps].shape[0], dtype=bool)
        return pred_classes[ps] == class_filter

    def _gt_keep(gs: slice) -> np.ndarray:
        if class_filter is None:
            return np.ones(gt_boxes[gs].shape[0], dtype=bool)
        return gt_classes[gs] == class_filter

    # Pass 1: per-frame matches over all predictions -> global operating point.
    full_matches = []
    for ps, gs in slices:
        pk = _pred_keep(ps)
        gk = _gt_keep(gs)
        full_matches.append(
            match_frame(
                pred_boxes[ps][pk],
                pred_scores[ps][pk],
                pred_classes[ps][pk],
                gt_boxes[gs][gk],
                gt_classes[gs][gk],
                iou_thr=args.iou_threshold,
                iou_mode=args.match_iou,
                class_aware=not args.class_agnostic,
            )
        )
    op = find_operating_point(full_matches, target_recall=args.target_recall)
    threshold = op["threshold"]

    # Pass 2: re-match using only predictions above the operating threshold;
    # score forecasting on those true positives.
    records: list[dict[str, float]] = []
    for ps, gs in slices:
        pk = _pred_keep(ps)
        gk = _gt_keep(gs)
        fb = pred_boxes[ps][pk]
        fs = pred_scores[ps][pk]
        fc = pred_classes[ps][pk]
        ff = pred_future[ps][pk]
        fm = pred_mode_logits[ps][pk]
        keep = fs >= threshold
        gb = gt_boxes[gs][gk]
        gc = gt_classes[gs][gk]
        gf = gt_future[gs][gk]
        gfm = gt_future_mask[gs][gk]
        match = match_frame(
            fb[keep],
            fs[keep],
            fc[keep],
            gb,
            gc,
            iou_thr=args.iou_threshold,
            iou_mode=args.match_iou,
            class_aware=not args.class_agnostic,
        )
        kept_idx = np.nonzero(keep)[0]
        for pred_local, gt_idx in match["pairs"]:  # type: ignore[index]
            pred_idx = int(kept_idx[pred_local])
            probs = _softmax(fm[pred_idx])
            metrics = trajectory_metrics(
                ff[pred_idx], gf[gt_idx], gfm[gt_idx], probs
            )
            if metrics is None:
                continue
            # Displacement from the *current* matched-GT center (for bucketing).
            valid = np.nonzero(gfm[gt_idx])[0]
            last = int(valid[-1])
            metrics["current_disp"] = float(
                np.linalg.norm(gf[gt_idx, last] - gb[gt_idx, :2])
            )
            metrics["class_id"] = int(gc[gt_idx])
            records.append(metrics)

    forecasting = aggregate_bucketed(records)
    paper_comparable = {
        "is_paper_comparable": bool(
            args.class_filter == PAPER_CLASS_FILTER and not args.class_agnostic
        ),
        "required_class_filter": PAPER_CLASS_FILTER,
        "primary_bucket": PAPER_PRIMARY_BUCKET,
        "metric_block": forecasting[PAPER_PRIMARY_BUCKET],
        "dynamic_block": forecasting["dynamic"],
        "stationary_block": forecasting["stationary"],
        "note": (
            "Use metric_block (stationary/dynamic macro) for DeTra Table-2-style "
            "forecasting comparison. Overall is diagnostic because stationary "
            "vehicles dominate the scored population."
        ),
    }
    payload = {
        "dump_dir": str(dump_dir),
        "meta": meta,
        "iou_threshold": args.iou_threshold,
        "match_iou": args.match_iou,
        "class_agnostic": args.class_agnostic,
        "class_filter": args.class_filter,
        "operating_point": op,
        "num_true_positives": len(records),
        "forecasting": forecasting,
        "forecasting_by_class": aggregate_by_class(records),
        "paper_comparable": paper_comparable,
    }
    return payload


def _fmt(block: dict[str, float]) -> str:
    return (
        f"n={int(block['count'])} "
        f"minADE_k1={block['minade_k1']:.4f} minFDE_k1={block['minfde_k1']:.4f} "
        f"MR_k1={block['mr_k1']:.4f} | "
        f"minADE_k6={block['minade_k6']:.4f} minFDE_k6={block['minfde_k6']:.4f} "
        f"MR_k6={block['mr_k6']:.4f} bFDE_k6={block['brier_minfde_k6']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecasting metrics over a prediction dump.")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--match-iou", choices=("rotated", "axis_aligned"), default="rotated")
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument(
        "--class-filter",
        type=int,
        default=PAPER_CLASS_FILTER,
        help=(
            "Restrict operating point + forecasting to a single class id "
            "(0=vehicle, 1=pedestrian, 2=cyclist). The paper's WOD Table 2 is "
            "vehicle-only and the default is 0. Use --all-classes for the old "
            "pooled diagnostic."
        ),
    )
    parser.add_argument(
        "--vehicle-only",
        action="store_const",
        const=0,
        dest="class_filter",
        help="Shortcut for --class-filter 0 (paper-comparable vehicle forecasting).",
    )
    parser.add_argument(
        "--all-classes",
        action="store_const",
        const=None,
        dest="class_filter",
        help="Pool all classes. Diagnostic only; not paper-comparable.",
    )
    args = parser.parse_args()

    payload = evaluate(args)
    op = payload["operating_point"]
    cf = payload["class_filter"]
    print(f"class_filter={cf if cf is not None else 'all'} (0=vehicle,1=ped,2=cyc)")
    print(
        f"operating_point: threshold={op['threshold']:.4f} "
        f"achieved_recall={op['achieved_recall']:.4f} "
        f"(target={op['target_recall']:.2f}) "
        f"precision={op['precision_at_op']:.4f} total_gt={int(op['total_gt'])}"
    )
    print(f"true_positives_scored={payload['num_true_positives']}")
    fc = payload["forecasting"]
    pc = payload["paper_comparable"]
    print(
        "paper_primary",
        _fmt(pc["metric_block"]),
        f"(bucket={pc['primary_bucket']} vehicle-only={pc['is_paper_comparable']})",
    )
    print("paper_dynamic", _fmt(pc["dynamic_block"]))
    print("overall    ", _fmt(fc["overall"]))
    print("stationary ", _fmt(fc["stationary"]))
    print("dynamic    ", _fmt(fc["dynamic"]))
    print("macro      ", _fmt(fc["macro"]))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

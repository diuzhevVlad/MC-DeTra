"""Official Waymo detection AP/APH over a prediction dump (MTR env only).

This script must run in a conda env where the Waymo metrics op loads (verified:
``mtr``, TF 2.6, ``waymo-open-dataset-tf-2-6-0``). It is broken in ``trajpred``
(``metrics_ops.so`` absl ABI mismatch), which is exactly why predictions are
dumped to disk first.

It loads ``predictions.npz`` from :mod:`detra_repro.eval.dump_predictions` and
runs the official ``WODDetectionEvaluator`` for:
- BEV boxes (``TYPE_2D``: cx,cy,l,w,heading) with class-specific IoU
  (vehicle 0.7, pedestrian 0.5, cyclist 0.5),
- 3D boxes (``TYPE_3D``: full 7-dim) with the same IoU thresholds,
- DeTra-style BEV AP at a single shared IoU in {0.3, 0.5, 0.7}.

Class mapping: dump class ids 0/1/2 -> Waymo type ids 1/2/4
(vehicle/pedestrian/cyclist). Ground-truth difficulty is constant ``LEVEL_1``
(the cache has no difficulty field), so the ``LEVEL_2`` breakdown spans all
boxes. ``prediction_overlap_nlz`` is all False (no no-label-zones in the cache).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.python import config_util_py
from waymo_open_dataset.metrics.python.wod_detection_evaluator import WODDetectionEvaluator
from waymo_open_dataset.protos import breakdown_pb2, metrics_pb2

# Dump class id (0/1/2) -> Waymo laser-label type id.
CLASS_TO_WAYMO_TYPE = np.array([1, 2, 4], dtype=np.uint8)
CLASS_NAMES = ("vehicle", "pedestrian", "cyclist")


def _build_config(
    box_type: int,
    iou_override: float | None = None,
    num_score_cutoffs: int = 11,
) -> metrics_pb2.Config:
    """Build a WOD metrics config.

    Args:
        box_type: ``label_pb2.Label.Box.TYPE_2D`` or ``TYPE_3D``.
        iou_override: If set, use this IoU for every class (DeTra AP@IoU sweep).
            Otherwise use class-specific thresholds (veh 0.7, ped/cyc 0.5).
    """

    config = metrics_pb2.Config()
    config.box_type = box_type
    # iou_thresholds indexed by type id: 0 unknown,1 veh,2 ped,3 sign,4 cyclist.
    thresholds = [0.0, 0.7, 0.5, 0.5, 0.5]
    if iou_override is not None:
        thresholds = [0.0, iou_override, iou_override, iou_override, iou_override]
    for value in thresholds:
        config.iou_thresholds.append(value)
    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.OBJECT_TYPE)
    config.difficulties.add()  # default => LEVEL_2 (includes LEVEL_1)
    config.matcher_type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN
    config.num_desired_score_cutoffs = num_score_cutoffs
    config.min_precision = 0.0
    config.min_heading_accuracy = 0.0
    return config


def _frame_repeated(offsets: np.ndarray, frame_ids: np.ndarray) -> np.ndarray:
    """Expand per-frame counts into a flat per-box frame-id vector."""

    counts = np.diff(offsets)
    return np.repeat(frame_ids, counts).astype(np.int64)


def _to_box_tensor(boxes7: np.ndarray, box_type: int) -> np.ndarray:
    """Convert ``[M,7]`` (x,y,z,l,w,h,heading) to the WOD box layout."""

    if box_type == label_pb2.Label.Box.TYPE_3D:
        return boxes7.astype(np.float32)  # cx,cy,cz,l,w,h,heading
    # TYPE_2D: cx,cy,l,w,heading
    return np.stack(
        [boxes7[:, 0], boxes7[:, 1], boxes7[:, 3], boxes7[:, 4], boxes7[:, 6]],
        axis=1,
    ).astype(np.float32)


def _run_evaluator(
    data: dict,
    box_type: int,
    iou_override: float | None,
    *,
    num_score_cutoffs: int = 11,
) -> dict:
    frame_ids = data["frame_ids"]
    pred_frame = _frame_repeated(data["pred_frame_offsets"], frame_ids)
    gt_frame = _frame_repeated(data["gt_frame_offsets"], frame_ids)

    pred_boxes = _to_box_tensor(data["pred_boxes"], box_type)
    gt_boxes = _to_box_tensor(data["gt_boxes"], box_type)
    pred_types = CLASS_TO_WAYMO_TYPE[data["pred_classes"].astype(int)]
    gt_types = CLASS_TO_WAYMO_TYPE[data["gt_classes"].astype(int)]

    evaluator = WODDetectionEvaluator(
        config=_build_config(box_type, iou_override, num_score_cutoffs)
    )
    groundtruths = {
        "ground_truth_frame_id": tf.constant(gt_frame, tf.int64),
        "ground_truth_bbox": tf.constant(gt_boxes, tf.float32),
        "ground_truth_type": tf.constant(gt_types, tf.uint8),
        "ground_truth_difficulty": tf.constant(
            data.get(
                "gt_difficulty",
                np.full(len(gt_boxes), label_pb2.Label.LEVEL_1, dtype=np.uint8),
            ).astype(np.uint8),
            tf.uint8,
        ),
    }
    predictions = {
        "prediction_frame_id": tf.constant(pred_frame, tf.int64),
        "prediction_bbox": tf.constant(pred_boxes, tf.float32),
        "prediction_type": tf.constant(pred_types, tf.uint8),
        "prediction_score": tf.constant(data["pred_scores"].astype(np.float32), tf.float32),
        "prediction_overlap_nlz": tf.constant(
            data.get("pred_overlap_nlz", np.zeros(len(pred_boxes), dtype=bool)).astype(bool),
            tf.bool,
        ),
    }
    evaluator.update_state(groundtruths, predictions)
    result = evaluator.evaluate()

    names = config_util_py.get_breakdown_names_from_config(
        _build_config(box_type, iou_override, num_score_cutoffs)
    )
    ap = result.average_precision.numpy()
    aph = result.average_precision_ha_weighted.numpy()
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        out[name] = {"ap": float(ap[idx]), "aph": float(aph[idx])}
    return out


def evaluate(args: argparse.Namespace) -> dict:
    dump_dir = Path(args.dump_dir)
    with (dump_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    data = dict(np.load(dump_dir / "predictions.npz", allow_pickle=False))
    paper_task = bool(getattr(args, "paper_task", False))
    if paper_task and not meta.get("paper_strict", False):
        raise ValueError("--paper-task requires a paper_strict cache-v2 prediction dump")
    if paper_task and not meta.get("complete_validation", False):
        raise ValueError("--paper-task requires the complete cache-v2 validation split")
    score_cutoffs = 101 if paper_task else 11

    payload: dict[str, object] = {
        "dump_dir": str(dump_dir),
        "meta": meta,
        "evaluation_variant": (
            "waymo_hungarian_same_paper_population" if paper_task else "legacy"
        ),
        "num_desired_score_cutoffs": score_cutoffs,
        "bev_class_specific": _run_evaluator(
            data,
            label_pb2.Label.Box.TYPE_2D,
            None,
            num_score_cutoffs=score_cutoffs,
        ),
        "bev_ap_iou_sweep": {
            f"iou_{iou:.2f}": _run_evaluator(
                data,
                label_pb2.Label.Box.TYPE_2D,
                iou,
                num_score_cutoffs=score_cutoffs,
            )
            for iou in (0.3, 0.5, 0.7)
        },
    }
    # The paper detects in BEV only (box D=(x,y,l,w,theta,c)) and reports BEV
    # AP@{0.3,0.5,0.7}. 3D AP is NOT a paper metric, so it is opt-in via
    # --with-3d (the repo's z/height channels are an extension beyond the paper).
    if args.with_3d:
        payload["d3_class_specific"] = _run_evaluator(
            data,
            label_pb2.Label.Box.TYPE_3D,
            None,
            num_score_cutoffs=score_cutoffs,
        )
    return payload


# Paper DeTra Waymo vehicle BEV AP@IoU (arxiv 2406.04426v1, Table 2), for the
# primary-metric comparison printed at the end of an eval.
_PAPER_WOD_VEHICLE_BEV_AP = {"iou_0.30": 0.898, "iou_0.50": 0.865, "iou_0.70": 0.749}


def _print_paper_comparison(payload: dict) -> None:
    """Print the primary BEV AP@0.7 comparison against the paper's vehicle AP."""

    sweep = payload.get("bev_ap_iou_sweep", {})
    print("== Paper comparison: vehicle BEV AP@IoU (primary metric) ==")
    for iou_name, paper_ap in _PAPER_WOD_VEHICLE_BEV_AP.items():
        block = sweep.get(iou_name, {})
        veh = next(
            (vals["ap"] for name, vals in block.items() if "VEHICLE" in name.upper()),
            None,
        )
        ours = f"{veh:.4f}" if veh is not None else "n/a"
        print(f"  {iou_name}: ours={ours}  paper={paper_ap:.3f}")


def _print_block(title: str, block: dict) -> None:
    print(f"== {title} ==")
    for name, vals in block.items():
        print(f"  {name:40s} AP={vals['ap']:.4f} APH={vals['aph']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Official Waymo detection AP/APH over a dump.")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--with-3d",
        action="store_true",
        help=(
            "Also run the (non-paper) 3D AP evaluation. The paper detects in BEV "
            "only and reports BEV AP@{0.3,0.5,0.7}; 3D AP is off by default."
        ),
    )
    parser.add_argument(
        "--bev-only",
        action="store_true",
        help=argparse.SUPPRESS,  # Deprecated no-op: BEV-only is now the default.
    )
    parser.add_argument(
        "--paper-task",
        action="store_true",
        help=(
            "Require a strict cache-v2 vehicle dump and use 101 Waymo score "
            "cutoffs over that exact task population."
        ),
    )
    args = parser.parse_args()

    payload = evaluate(args)
    # Primary (paper) metric first: BEV AP sweep + vehicle comparison.
    for iou_name, block in payload["bev_ap_iou_sweep"].items():
        _print_block(f"BEV AP @ {iou_name}", block)
    _print_paper_comparison(payload)
    _print_block("BEV (TYPE_2D) class-specific IoU", payload["bev_class_specific"])
    if "d3_class_specific" in payload:
        _print_block("3D (TYPE_3D) class-specific IoU (non-paper)", payload["d3_class_specific"])

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detra_repro.checkpoints import load_checkpoint, load_compatible
from detra_repro.config import DataConfig, ModelConfig
from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
from detra_repro.losses import (
    current_boxes_from_pred,
    rotated_bev_iou_matrix,
    waymo_types_to_class_indices,
)
from detra_repro.models.bev import PaperLikeLidarEncoder
from detra_repro.models.proposals import PaperProposalHead, ProposalHead
from detra_repro.models.refinement import DeTraMini, MapTokenEncoder, build_map_encoder


CLASS_NAMES = ("vehicle", "pedestrian", "cyclist")


def _load_json(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f).get("config", {})


def _load_checkpoint(path: Path, device: str) -> dict:
    return load_checkpoint(path, device)


def _cfg_to_jsonable(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg) if isinstance(cfg, dict) else {}


def _build_configs(ckpt: dict, exp_cfg: dict, dataset: CachedWaymoDataset) -> tuple[DataConfig, ModelConfig]:
    data_cfg = ckpt.get("data_cfg")
    if not isinstance(data_cfg, DataConfig):
        data_cfg = DataConfig(
            roi_m=tuple(exp_cfg.get("roi", DataConfig().roi_m)),
            voxel_size_m=float(exp_cfg.get("voxel_size", DataConfig().voxel_size_m)),
        )
    model_cfg = ckpt.get("model_cfg")
    if not isinstance(model_cfg, ModelConfig):
        first = dataset[0]
        model_cfg = ModelConfig(
            num_queries=int(exp_cfg.get("num_queries", ModelConfig().num_queries)),
            num_times=int(first["gt_future"].shape[1]) + 1,
            num_refinement_blocks=int(
                exp_cfg.get("num_refinement_blocks", ModelConfig().num_refinement_blocks)
            ),
            deformable_radius_m=float(
                exp_cfg.get("deformable_radius_m", ModelConfig().deformable_radius_m)
            ),
            attention_type=str(exp_cfg.get("attention_type", ModelConfig().attention_type)),
            lidar_attention_anchor=str(
                exp_cfg.get("lidar_attention_anchor", ModelConfig().lidar_attention_anchor)
            ),
            use_query_velocity=str(exp_cfg.get("query_velocity_source", "none")) != "none",
            forecast_base_mode=str(
                exp_cfg.get("forecast_base_mode", ModelConfig().forecast_base_mode)
            ),
            forecast_velocity_frame=str(
                exp_cfg.get("forecast_velocity_frame", ModelConfig().forecast_velocity_frame)
            ),
        )
    return data_cfg, model_cfg


def _has_history_motion_branch(ckpt: dict) -> bool:
    lidar_state = ckpt.get("lidar_encoder", {})
    return any(
        key.startswith(("history_occupancy_net.", "motion_aux_head.", "motion_context_proj."))
        for key in lidar_state
    )


def _load_compatible(module, state_dict: dict) -> None:
    """Load only the parameters whose shapes match the current module.

    Keys present in ``state_dict`` but absent from the module, or whose tensor
    shape differs (e.g. the box head grew from 6 to 7/8 channels for z/height
    regression), are skipped. Those parameters keep their freshly initialized
    values, which is the intended behavior when loading a pre-z checkpoint.
    """

    skipped = load_compatible(module, state_dict)
    if skipped:
        print(f"_load_compatible: skipped {len(skipped)} mismatched/extra keys: {skipped[:6]}")


def _load_models(
    ckpt: dict,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    norm_type: str,
    device: str,
) -> tuple[PaperLikeLidarEncoder, ProposalHead, MapTokenEncoder | None, DeTraMini]:
    history_channels = model_cfg.history_occupancy_channels if _has_history_motion_branch(ckpt) else 0
    lidar_encoder = PaperLikeLidarEncoder(
        data_cfg,
        point_feature_dim=model_cfg.point_feature_dim,
        hidden_dim=model_cfg.hidden_dim,
        dynamic_conv_experts=model_cfg.dynamic_conv_experts,
        norm_type=norm_type,
        history_occupancy_channels=history_channels,
    ).to(device)
    # Auto-detect the proposal head architecture from the checkpoint so eval
    # matches training without an extra flag. The paper A.3 head (PaperProposalHead)
    # stacks four conv blocks per branch, so its heatmap Sequential has indices
    # well past the shallow head's max index of 2 (conv,relu,conv).
    head_state = ckpt["proposal_head"]
    head_max_idx = max(
        (int(k.split(".")[1]) for k in head_state if k.startswith("heatmap.") and k.split(".")[1].isdigit()),
        default=0,
    )
    if head_max_idx > 2:
        proposal_head = PaperProposalHead(
            model_cfg.hidden_dim,
            num_convs=4,
            norm_type=norm_type if norm_type in {"batch", "group"} else "group",
        ).to(device)
    else:
        proposal_head = ProposalHead(model_cfg.hidden_dim).to(device)
    map_encoder = build_map_encoder(model_cfg).to(device) if "map_encoder" in ckpt else None
    model = DeTraMini(model_cfg, data_cfg).to(device)

    _load_compatible(lidar_encoder, ckpt["lidar_encoder"])
    _load_compatible(proposal_head, ckpt["proposal_head"])
    if map_encoder is not None:
        _load_compatible(map_encoder, ckpt["map_encoder"])
    _load_compatible(model, ckpt["model"])
    lidar_encoder.eval()
    proposal_head.eval()
    if map_encoder is not None:
        map_encoder.eval()
    model.eval()
    return lidar_encoder, proposal_head, map_encoder, model


@torch.no_grad()
def _predict_batch(
    batch: dict,
    lidar_encoder: PaperLikeLidarEncoder,
    proposal_head: ProposalHead,
    map_encoder: MapTokenEncoder | None,
    model: DeTraMini,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    device: str,
    proposal_nms_radius_m: float,
    proposal_nms_mode: str,
    proposal_center_mode: str,
    proposal_peak_filter: bool,
) -> dict:
    non_blocking = device.startswith("cuda")
    points = batch["points"].to(device, non_blocking=non_blocking)
    point_mask = batch["point_mask"].to(device, non_blocking=non_blocking)
    encoder_out = lidar_encoder(points, point_mask=point_mask)
    proposal_outputs = proposal_head(encoder_out["feature_map"])
    proposals = proposal_head.decode(
        proposal_outputs,
        data_cfg,
        num_proposals=model_cfg.num_queries,
        output_stride=model_cfg.proposal_output_stride,
        nms_radius_m=proposal_nms_radius_m,
        center_mode=proposal_center_mode,
        peak_filter=proposal_peak_filter,
        nms_mode=proposal_nms_mode,
    )

    map_tokens = map_xy = map_mask = None
    if map_encoder is not None:
        map_xy = batch["map_xy"].to(device, non_blocking=non_blocking)
        map_features = batch["map_features"].to(device, non_blocking=non_blocking)
        map_mask = batch["map_mask"].to(device, non_blocking=non_blocking)
        if map_mask.any():
            map_tokens = map_encoder(map_features, map_xy, map_mask)

    return model(
        encoder_out["multi_scale_features"],
        proposals["poses"],
        initial_boxes=proposals["boxes"],
        map_tokens=map_tokens,
        map_xy=map_xy if map_tokens is not None else None,
        map_mask=map_mask if map_tokens is not None else None,
    )


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    *,
    threshold: float,
    criterion: str,
    center_radius_m: float,
    iou_threshold: float,
    class_aware: bool,
) -> tuple[int, int, int, list[tuple[float, int]], dict[int, tuple[int, int, int]]]:
    keep = pred_scores >= threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_classes = pred_classes[keep]
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_classes = pred_classes[order]

    matched_gt: set[int] = set()
    scored_labels: list[tuple[float, int]] = []
    per_class = {idx: [0, 0, int((gt_classes == idx).sum().item())] for idx in range(len(CLASS_NAMES))}

    if len(gt_boxes) and len(pred_boxes):
        center_dist = torch.cdist(pred_boxes[:, :2], gt_boxes[:, :2])
        iou = rotated_bev_iou_matrix(pred_boxes, gt_boxes) if criterion == "iou" else None
    else:
        center_dist = pred_boxes.new_zeros((len(pred_boxes), len(gt_boxes)))
        iou = pred_boxes.new_zeros((len(pred_boxes), len(gt_boxes)))

    tp = 0
    fp = 0
    for pred_idx in range(len(pred_boxes)):
        pred_class = int(pred_classes[pred_idx].item())
        per_class[pred_class][1] += 1
        candidates = []
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            if class_aware and pred_class != int(gt_classes[gt_idx].item()):
                continue
            if criterion == "center":
                ok = bool(center_dist[pred_idx, gt_idx].item() <= center_radius_m)
                quality = -float(center_dist[pred_idx, gt_idx].item())
            elif criterion == "iou":
                ok = bool(iou[pred_idx, gt_idx].item() >= iou_threshold)
                quality = float(iou[pred_idx, gt_idx].item())
            else:
                raise ValueError("criterion must be 'center' or 'iou'")
            if ok:
                candidates.append((quality, gt_idx))
        if candidates:
            _, gt_idx = max(candidates, key=lambda item: item[0])
            matched_gt.add(gt_idx)
            tp += 1
            per_class[pred_class][0] += 1
            scored_labels.append((float(pred_scores[pred_idx].item()), 1))
        else:
            fp += 1
            scored_labels.append((float(pred_scores[pred_idx].item()), 0))

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, scored_labels, {k: tuple(v) for k, v in per_class.items()}


def _inside_roi(boxes: torch.Tensor, roi: tuple[float, float, float, float]) -> torch.Tensor:
    """Return mask for boxes whose centers are inside the BEV evaluation ROI."""

    x_min, y_min, x_max, y_max = roi
    return (
        (boxes[:, 0] >= x_min)
        & (boxes[:, 0] <= x_max)
        & (boxes[:, 1] >= y_min)
        & (boxes[:, 1] <= y_max)
    )


def _ap_from_scored_labels(scored_labels: list[tuple[float, int]], num_gt: int) -> float:
    if num_gt <= 0 or not scored_labels:
        return 0.0
    labels = [label for _, label in sorted(scored_labels, key=lambda item: item[0], reverse=True)]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for label in labels:
        if label:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / num_gt)
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * max(recall - prev_recall, 0.0)
        prev_recall = recall
    return ap


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(args: argparse.Namespace) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    exp_cfg = _load_json(args.experiment_json)
    dataset = CachedWaymoDataset(args.cache)
    data_cfg, model_cfg = _build_configs(ckpt, exp_cfg, dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda samples: collate_cached_waymo(
            samples,
            max_points=args.max_points,
            max_objects=model_cfg.num_queries,
            max_map_tokens=args.max_map_tokens,
            deterministic_seed=args.deterministic_seed,
        ),
    )
    norm_type = args.norm_type or str(exp_cfg.get("norm_type", "group"))
    proposal_nms_radius_m = (
        args.proposal_nms_radius_m
        if args.proposal_nms_radius_m is not None
        else float(exp_cfg.get("proposal_nms_radius_m", 1.0))
    )
    proposal_nms_mode = args.proposal_nms_mode or str(exp_cfg.get("proposal_nms_mode", "center"))
    proposal_center_mode = args.proposal_center_mode or str(
        exp_cfg.get("proposal_center_mode", "regression")
    )
    proposal_peak_filter = bool(exp_cfg.get("proposal_peak_filter", True)) and not args.no_proposal_peak_filter
    lidar_encoder, proposal_head, map_encoder, model = _load_models(
        ckpt, data_cfg, model_cfg, norm_type, device
    )

    thresholds = args.thresholds
    criteria = {
        "center2m": {"criterion": "center", "center_radius_m": args.center_radius_m, "iou_threshold": 0.0},
        "iou0p5": {"criterion": "iou", "center_radius_m": 0.0, "iou_threshold": args.iou_threshold},
    }
    totals = {
        name: {
            threshold: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "scored": [],
                "per_class": {idx: [0, 0, 0] for idx in range(len(CLASS_NAMES))},
            }
            for threshold in thresholds
        }
        for name in criteria
    }
    frames = 0
    gt_total = 0
    pred_score_sum = 0.0
    pred_count = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        pred = _predict_batch(
            batch,
            lidar_encoder,
            proposal_head,
            map_encoder,
            model,
            data_cfg,
            model_cfg,
            device,
            proposal_nms_radius_m=proposal_nms_radius_m,
            proposal_nms_mode=proposal_nms_mode,
            proposal_center_mode=proposal_center_mode,
            proposal_peak_filter=proposal_peak_filter,
        )
        boxes = current_boxes_from_pred(pred).detach().cpu()
        object_scores = pred["object_logits"].sigmoid().detach().cpu()
        class_logits = pred.get("class_logits")
        if class_logits is not None:
            pred_classes = class_logits.detach().cpu().softmax(dim=-1).argmax(dim=-1)
        else:
            pred_classes = torch.zeros_like(object_scores, dtype=torch.long)
        gt_boxes = batch["gt_boxes"]
        gt_valid = batch["gt_valid"]
        gt_classes = waymo_types_to_class_indices(batch["gt_types"])

        for b in range(boxes.shape[0]):
            frames += 1
            valid_gt = gt_valid[b]
            if args.filter_roi:
                valid_gt = valid_gt.clone()
                valid_gt[valid_gt.clone()] &= _inside_roi(gt_boxes[b, valid_gt], data_cfg.roi_m)
            sample_gt_boxes = gt_boxes[b, valid_gt]
            sample_gt_classes = gt_classes[b, valid_gt]
            sample_pred_boxes = boxes[b]
            sample_object_scores = object_scores[b]
            sample_pred_classes = pred_classes[b]
            if args.filter_roi:
                pred_roi = _inside_roi(sample_pred_boxes, data_cfg.roi_m)
                sample_pred_boxes = sample_pred_boxes[pred_roi]
                sample_object_scores = sample_object_scores[pred_roi]
                sample_pred_classes = sample_pred_classes[pred_roi]
            gt_total += int(valid_gt.sum().item())
            pred_score_sum += float(sample_object_scores.sum().item())
            pred_count += int(sample_object_scores.shape[0])
            for criterion_name, criterion_kwargs in criteria.items():
                for threshold in thresholds:
                    tp, fp, fn, scored, per_class = _match_predictions(
                        sample_pred_boxes,
                        sample_object_scores,
                        sample_pred_classes,
                        sample_gt_boxes,
                        sample_gt_classes,
                        threshold=threshold,
                        class_aware=not args.class_agnostic,
                        **criterion_kwargs,
                    )
                    bucket = totals[criterion_name][threshold]
                    bucket["tp"] += tp
                    bucket["fp"] += fp
                    bucket["fn"] += fn
                    bucket["scored"].extend(scored)
                    for cls_idx, values in per_class.items():
                        for idx, value in enumerate(values):
                            bucket["per_class"][cls_idx][idx] += value

    metrics = {}
    for criterion_name, threshold_buckets in totals.items():
        metrics[criterion_name] = {}
        for threshold, bucket in threshold_buckets.items():
            row = {
                "threshold": threshold,
                "tp": bucket["tp"],
                "fp": bucket["fp"],
                "fn": bucket["fn"],
                "predictions_per_frame": (bucket["tp"] + bucket["fp"]) / max(frames, 1),
                "ap_like": _ap_from_scored_labels(bucket["scored"], gt_total),
                **_prf(bucket["tp"], bucket["fp"], bucket["fn"]),
            }
            row["per_class"] = {}
            for cls_idx, (cls_tp, cls_pred_count, cls_gt) in bucket["per_class"].items():
                cls_fp = cls_pred_count - cls_tp
                cls_fn = cls_gt - cls_tp
                row["per_class"][CLASS_NAMES[cls_idx]] = {
                    "tp": cls_tp,
                    "fp": cls_fp,
                    "fn": cls_fn,
                    "gt": cls_gt,
                    **_prf(cls_tp, cls_fp, cls_fn),
                }
            metrics[criterion_name][str(threshold)] = row

    return {
        "cache": str(args.cache),
        "checkpoint": str(args.checkpoint),
        "experiment_json": args.experiment_json,
        "device": device,
        "frames": frames,
        "gt_total": gt_total,
        "mean_objectness": pred_score_sum / max(pred_count, 1),
        "thresholds": thresholds,
        "class_aware": not args.class_agnostic,
        "center_radius_m": args.center_radius_m,
        "iou_threshold": args.iou_threshold,
        "filter_roi": args.filter_roi,
        "data_cfg": _cfg_to_jsonable(data_cfg),
        "model_cfg": _cfg_to_jsonable(model_cfg),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection PR diagnostics for cached DeTra outputs.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=750000)
    parser.add_argument("--max-map-tokens", type=int, default=2048)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic-seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--norm-type", default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    parser.add_argument("--center-radius-m", type=float, default=2.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--no-filter-roi", dest="filter_roi", action="store_false")
    parser.set_defaults(filter_roi=True)
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--proposal-nms-radius-m", type=float, default=None)
    parser.add_argument("--proposal-nms-mode", choices=("rotated", "center", "none"), default=None)
    parser.add_argument("--proposal-center-mode", choices=("regression", "cell"), default=None)
    parser.add_argument("--no-proposal-peak-filter", action="store_true")
    args = parser.parse_args()

    payload = evaluate(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

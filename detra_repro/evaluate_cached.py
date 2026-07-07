from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from detra_repro.config import DataConfig, ModelConfig
from detra_repro.models.bev import PaperLikeLidarEncoder
from detra_repro.models.proposals import ProposalHead
from detra_repro.models.refinement import DeTraMini, MapTokenEncoder, build_map_encoder
from detra_repro.train_cached import _load_compatible, build_loader, evaluate


def _config_from_experiment(path: Path | None) -> dict:
    """Load command config from an experiment JSON when available."""

    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        record = json.load(f)
    return record.get("config", {})


def _load_checkpoint(path: Path, device: str) -> dict:
    """Load a training checkpoint with dataclass configs preserved."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _value_or_default(value, default):
    """Return ``default`` when an experiment JSON stored an unset value."""

    return default if value is None else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a cached DeTra checkpoint.")
    parser.add_argument("--cache", required=True, help="Cached validation/train directory.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train_cached.")
    parser.add_argument("--experiment-json", default=None, help="Optional experiment JSON for runtime flags.")
    parser.add_argument("--output-json", default=None, help="Optional path for metrics JSON.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--max-map-tokens", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--deterministic-seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--norm-type", default=None)
    parser.add_argument("--objective", default=None, choices=("joint", "proposal_only"))
    parser.add_argument("--proposal-mode", default=None, choices=("decoded", "oracle", "noisy_gt"))
    parser.add_argument("--proposal-noise-std-m", type=float, default=None)
    parser.add_argument("--forecast-weight", type=float, default=None)
    parser.add_argument("--forecast-gate-radius-m", type=float, default=None)
    parser.add_argument("--forecast-loss-type", default=None, choices=("laplace", "smooth_l1"))
    parser.add_argument("--velocity-weight", type=float, default=None)
    parser.add_argument("--init-weight", type=float, default=None)
    parser.add_argument("--proposal-nms-radius-m", type=float, default=None)
    parser.add_argument("--proposal-nms-mode", default=None, choices=("rotated", "center", "none"))
    parser.add_argument("--proposal-center-mode", default=None, choices=("regression", "cell"))
    parser.add_argument("--no-proposal-peak-filter", action="store_true")
    parser.add_argument("--support-filter-radius-m", type=float, default=None)
    parser.add_argument("--query-velocity-source", default=None, choices=("none", "gt_match"))
    parser.add_argument("--query-velocity-match-radius-m", type=float, default=None)
    parser.add_argument(
        "--forecast-base-mode",
        default=None,
        choices=("current", "constant_velocity", "learned_velocity"),
    )
    parser.add_argument("--forecast-velocity-frame", default=None, choices=("ego", "object"))
    parser.add_argument("--gt-history-occupancy", action="store_true")
    parser.add_argument("--gt-history-occupancy-radius-m", type=float, default=None)
    parser.add_argument("--dense-history-motion-weight", type=float, default=None)
    parser.add_argument("--dense-history-motion-flow-weight", type=float, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    exp_cfg = _config_from_experiment(Path(args.experiment_json) if args.experiment_json else None)

    data_cfg = ckpt.get("data_cfg")
    if not isinstance(data_cfg, DataConfig):
        data_cfg = DataConfig(
            roi_m=tuple(exp_cfg.get("roi", DataConfig().roi_m)),
            voxel_size_m=float(exp_cfg.get("voxel_size", DataConfig().voxel_size_m)),
        )
    model_cfg = ckpt.get("model_cfg")
    if not isinstance(model_cfg, ModelConfig):
        model_cfg = ModelConfig(
            num_queries=int(exp_cfg.get("num_queries", ModelConfig().num_queries)),
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
            use_history_occupancy=bool(exp_cfg.get("gt_history_occupancy", False)),
        )
    elif not hasattr(model_cfg, "forecast_base_mode") or not hasattr(model_cfg, "forecast_velocity_frame"):
        model_cfg = ModelConfig(
            num_queries=model_cfg.num_queries,
            num_modes=model_cfg.num_modes,
            num_times=model_cfg.num_times,
            hidden_dim=model_cfg.hidden_dim,
            num_refinement_blocks=model_cfg.num_refinement_blocks,
            point_feature_dim=model_cfg.point_feature_dim,
            num_classes=model_cfg.num_classes,
            map_feature_dim=model_cfg.map_feature_dim,
            bev_channels=model_cfg.bev_channels,
            proposal_output_stride=model_cfg.proposal_output_stride,
            deformable_points=model_cfg.deformable_points,
            deformable_radius_m=model_cfg.deformable_radius_m,
            attention_type=model_cfg.attention_type,
            lidar_attention_anchor=model_cfg.lidar_attention_anchor,
            detach_pose_refinement=model_cfg.detach_pose_refinement,
            map_k=model_cfg.map_k,
            use_query_velocity=model_cfg.use_query_velocity,
            forecast_base_mode=str(exp_cfg.get("forecast_base_mode", "current")),
            forecast_velocity_frame=str(exp_cfg.get("forecast_velocity_frame", "ego")),
            use_history_occupancy=model_cfg.use_history_occupancy,
            history_occupancy_channels=model_cfg.history_occupancy_channels,
            dynamic_conv_experts=model_cfg.dynamic_conv_experts,
        )
    current_forecast_base_mode = getattr(
        model_cfg,
        "forecast_base_mode",
        ModelConfig().forecast_base_mode,
    )
    current_forecast_velocity_frame = getattr(
        model_cfg,
        "forecast_velocity_frame",
        ModelConfig().forecast_velocity_frame,
    )
    target_forecast_velocity_frame = args.forecast_velocity_frame or current_forecast_velocity_frame
    if (
        args.forecast_base_mode is not None
        and current_forecast_base_mode != args.forecast_base_mode
    ) or current_forecast_velocity_frame != target_forecast_velocity_frame:
        model_cfg = ModelConfig(
            num_queries=model_cfg.num_queries,
            num_modes=model_cfg.num_modes,
            num_times=model_cfg.num_times,
            hidden_dim=model_cfg.hidden_dim,
            num_refinement_blocks=model_cfg.num_refinement_blocks,
            point_feature_dim=model_cfg.point_feature_dim,
            num_classes=model_cfg.num_classes,
            map_feature_dim=model_cfg.map_feature_dim,
            bev_channels=model_cfg.bev_channels,
            proposal_output_stride=model_cfg.proposal_output_stride,
            deformable_points=model_cfg.deformable_points,
            deformable_radius_m=model_cfg.deformable_radius_m,
            attention_type=model_cfg.attention_type,
            lidar_attention_anchor=model_cfg.lidar_attention_anchor,
            detach_pose_refinement=model_cfg.detach_pose_refinement,
            map_k=model_cfg.map_k,
            use_query_velocity=model_cfg.use_query_velocity,
            forecast_base_mode=args.forecast_base_mode or current_forecast_base_mode,
            forecast_velocity_frame=target_forecast_velocity_frame,
            use_history_occupancy=model_cfg.use_history_occupancy,
            history_occupancy_channels=model_cfg.history_occupancy_channels,
            dynamic_conv_experts=model_cfg.dynamic_conv_experts,
        )

    batch_size = args.batch_size or int(exp_cfg.get("batch_size", 1))
    max_points = args.max_points or int(exp_cfg.get("max_points", 32768))
    max_map_tokens = args.max_map_tokens or int(exp_cfg.get("max_map_tokens", 2048))
    norm_type = args.norm_type or str(exp_cfg.get("norm_type", "batch"))
    objective = args.objective or str(exp_cfg.get("objective", "joint"))
    proposal_mode = args.proposal_mode or str(exp_cfg.get("proposal_mode", "decoded"))
    proposal_noise_std_m = (
        args.proposal_noise_std_m
        if args.proposal_noise_std_m is not None
        else float(exp_cfg.get("proposal_noise_std_m", 0.0))
    )
    forecast_weight = (
        args.forecast_weight
        if args.forecast_weight is not None
        else float(exp_cfg.get("forecast_weight", 0.1))
    )
    forecast_loss_type = args.forecast_loss_type or str(exp_cfg.get("forecast_loss_type", "laplace"))
    velocity_weight = (
        args.velocity_weight
        if args.velocity_weight is not None
        else float(exp_cfg.get("velocity_weight", 0.0))
    )
    init_weight = (
        args.init_weight
        if args.init_weight is not None
        else float(_value_or_default(exp_cfg.get("init_weight"), 1.0))
    )
    proposal_nms_radius_m = (
        args.proposal_nms_radius_m
        if args.proposal_nms_radius_m is not None
        else float(exp_cfg.get("proposal_nms_radius_m", 0.1))
    )
    proposal_nms_mode = args.proposal_nms_mode or str(exp_cfg.get("proposal_nms_mode", "rotated"))
    proposal_center_mode = args.proposal_center_mode or str(
        exp_cfg.get("proposal_center_mode", "regression")
    )
    proposal_peak_filter = bool(exp_cfg.get("proposal_peak_filter", True)) and not args.no_proposal_peak_filter
    support_filter_radius_m = (
        args.support_filter_radius_m
        if args.support_filter_radius_m is not None
        else exp_cfg.get("support_filter_radius_m")
    )
    query_velocity_source = args.query_velocity_source or str(exp_cfg.get("query_velocity_source", "none"))
    forecast_base_mode = args.forecast_base_mode or str(
        exp_cfg.get("forecast_base_mode", getattr(model_cfg, "forecast_base_mode", "current"))
    )
    query_velocity_match_radius_m = (
        args.query_velocity_match_radius_m
        if args.query_velocity_match_radius_m is not None
        else float(exp_cfg.get("query_velocity_match_radius_m", 2.0))
    )
    gt_history_occupancy = args.gt_history_occupancy or bool(exp_cfg.get("gt_history_occupancy", False))
    gt_history_occupancy_radius_m = (
        args.gt_history_occupancy_radius_m
        if args.gt_history_occupancy_radius_m is not None
        else float(exp_cfg.get("gt_history_occupancy_radius_m", 0.5))
    )
    dense_history_motion_weight = (
        args.dense_history_motion_weight
        if args.dense_history_motion_weight is not None
        else float(exp_cfg.get("dense_history_motion_weight", 0.0))
    )
    dense_history_motion_flow_weight = (
        args.dense_history_motion_flow_weight
        if args.dense_history_motion_flow_weight is not None
        else float(exp_cfg.get("dense_history_motion_flow_weight", 1.0))
    )

    _, loader = build_loader(
        args.cache,
        batch_size=batch_size,
        max_points=max_points,
        max_objects=model_cfg.num_queries,
        shuffle=False,
        max_map_tokens=max_map_tokens,
        deterministic_seed=args.deterministic_seed,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    lidar_encoder = PaperLikeLidarEncoder(
        data_cfg,
        point_feature_dim=model_cfg.point_feature_dim,
        hidden_dim=model_cfg.hidden_dim,
        dynamic_conv_experts=model_cfg.dynamic_conv_experts,
        norm_type=norm_type,
        history_occupancy_channels=(
            model_cfg.history_occupancy_channels
            if gt_history_occupancy or dense_history_motion_weight > 0
            else 0
        ),
    ).to(device)
    proposal_head = ProposalHead(model_cfg.hidden_dim).to(device)
    map_encoder = build_map_encoder(model_cfg).to(device)
    model = DeTraMini(model_cfg, data_cfg).to(device)

    lidar_encoder.load_state_dict(
        ckpt["lidar_encoder"],
        strict=not (gt_history_occupancy or dense_history_motion_weight > 0),
    )
    proposal_head.load_state_dict(ckpt["proposal_head"], strict=False)
    if "map_encoder" in ckpt:
        _load_compatible(map_encoder, ckpt["map_encoder"])
    model.load_state_dict(ckpt["model"], strict=False)

    metrics = evaluate(
        loader,
        lidar_encoder,
        proposal_head,
        map_encoder if objective != "proposal_only" else None,
        model,
        data_cfg,
        model_cfg,
        device,
        max_batches=args.max_batches,
        objective=objective,
        proposal_mode=proposal_mode,
        proposal_noise_std_m=proposal_noise_std_m,
        forecast_weight=forecast_weight,
        forecast_gate_radius_m=args.forecast_gate_radius_m,
        forecast_loss_type=forecast_loss_type,
        velocity_weight=velocity_weight,
        init_weight=init_weight,
        proposal_nms_radius_m=proposal_nms_radius_m,
        proposal_nms_mode=proposal_nms_mode,
        proposal_center_mode=proposal_center_mode,
        proposal_peak_filter=proposal_peak_filter,
        support_filter_radius_m=support_filter_radius_m,
        query_velocity_source=query_velocity_source,
        query_velocity_match_radius_m=query_velocity_match_radius_m,
        gt_history_occupancy=gt_history_occupancy,
        gt_history_occupancy_radius_m=gt_history_occupancy_radius_m,
        dense_history_motion_weight=dense_history_motion_weight,
        dense_history_motion_flow_weight=dense_history_motion_flow_weight,
    )

    payload = {
        "cache": str(args.cache),
        "checkpoint": str(args.checkpoint),
        "experiment_json": str(args.experiment_json) if args.experiment_json else None,
        "device": device,
        "batch_size": batch_size,
        "max_points": max_points,
        "max_map_tokens": max_map_tokens,
        "max_batches": args.max_batches,
        "query_velocity_source": query_velocity_source,
        "query_velocity_match_radius_m": query_velocity_match_radius_m,
        "forecast_base_mode": forecast_base_mode,
        "forecast_velocity_frame": getattr(model_cfg, "forecast_velocity_frame", "ego"),
        "gt_history_occupancy": gt_history_occupancy,
        "gt_history_occupancy_radius_m": gt_history_occupancy_radius_m,
        "dense_history_motion_weight": dense_history_motion_weight,
        "dense_history_motion_flow_weight": dense_history_motion_flow_weight,
        "velocity_weight": velocity_weight,
        "metrics": metrics,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

import torch

from detra_repro.config import DataConfig, ModelConfig
from detra_repro.losses import detra_loss
from detra_repro.models.bev import PaperLikeLidarEncoder
from detra_repro.models.proposals import ProposalHead
from detra_repro.models.refinement import DeTraMini


def overfit_one_batch() -> None:
    """Smoke-test the model interfaces on synthetic tensors.

    This is not real training. It exists to validate that the model pieces agree
    on tensor shapes before Waymo preprocessing and matching are complete.

    Synthetic shapes:
        - ``points``: ``[B=2, P=4096, C_point=4]`` using ``x,y,z,dt``.
        - raw scatter grid: ``[B=2, D=128, H=100, W=100]`` for the small test
          ROI ``[-20,20]`` at ``0.4 m``.
        - ``lidar_tokens``: ``[B=2, L=625, D=128]`` after stride-4 fusion.
        - ``initial_poses``: ``[B=2, N=128, 3]``.
        - ``poses`` output: ``[2, 128, 6, 11, 3]``.
    """

    cfg = ModelConfig()
    data_cfg = DataConfig(roi_m=(-20.0, -20.0, 20.0, 20.0), voxel_size_m=0.4)
    lidar_encoder = PaperLikeLidarEncoder(
        data_cfg,
        point_feature_dim=cfg.point_feature_dim,
        hidden_dim=cfg.hidden_dim,
        dynamic_conv_experts=cfg.dynamic_conv_experts,
    )
    proposal_head = ProposalHead(cfg.hidden_dim)
    model = DeTraMini(cfg, data_cfg)

    points = torch.empty(2, 4096, cfg.point_feature_dim).uniform_(-20.0, 20.0)
    points[..., 2] = torch.empty(2, 4096).uniform_(-2.0, 4.0)
    points[..., 3] = torch.empty(2, 4096).uniform_(-0.5, 0.0)
    encoder_out = lidar_encoder(points)
    lidar_tokens = encoder_out["tokens"]
    feature_map = encoder_out["feature_map"]
    dense_props = proposal_head(feature_map)
    proposals = proposal_head.decode(
        dense_props,
        data_cfg,
        num_proposals=cfg.num_queries,
        output_stride=cfg.proposal_output_stride,
    )

    pred = model(encoder_out["multi_scale_features"], proposals["poses"], initial_boxes=proposals["boxes"])

    gt_boxes = torch.zeros(2, 8, 7)
    gt_boxes[..., 0] = torch.empty(2, 8).uniform_(-15.0, 15.0)
    gt_boxes[..., 1] = torch.empty(2, 8).uniform_(-15.0, 15.0)
    gt_boxes[..., 3:6] = torch.tensor([4.5, 2.0, 1.7])
    gt_types = torch.ones(2, 8, dtype=torch.long)
    gt_valid = torch.ones(2, 8, dtype=torch.bool)
    gt_future = gt_boxes[..., None, :2] + torch.randn(2, 8, cfg.num_times - 1, 2) * 0.25
    gt_future_mask = torch.ones(2, 8, cfg.num_times - 1, dtype=torch.bool)
    losses = detra_loss(
        pred,
        gt_boxes,
        gt_valid,
        gt_types,
        gt_future,
        gt_future_mask,
        proposal_outputs=dense_props,
        data_cfg=data_cfg,
        proposal_output_stride=cfg.proposal_output_stride,
    )

    print("heatmap", tuple(dense_props["heatmap_logits"].shape))
    print("proposal_boxes", tuple(proposals["boxes"].shape))
    print("proposal_poses", tuple(proposals["poses"].shape))
    print("lidar_tokens", tuple(lidar_tokens.shape))
    print("stride4", tuple(encoder_out["multi_scale_features"]["stride4"].shape))
    print("stride8", tuple(encoder_out["multi_scale_features"]["stride8"].shape))
    print("stride16", tuple(encoder_out["multi_scale_features"]["stride16"].shape))
    print("poses", tuple(pred["poses"].shape))
    print("object_logits", tuple(pred["object_logits"].shape))
    print("class_logits", tuple(pred["class_logits"].shape))
    print("mode_logits", tuple(pred["mode_logits"].shape))
    print("loss", float(losses["loss"].detach()))


if __name__ == "__main__":
    overfit_one_batch()

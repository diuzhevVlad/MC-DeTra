import torch
import torch.nn.functional as F
from torch import nn

from detra_repro.config import DataConfig
from detra_repro.losses import rotated_bev_iou, rotated_bev_iou_matrix


def _norm2d(norm_type: str, channels: int) -> nn.Module:
    """Build a 2D normalization layer (``batch`` per paper A.3, or ``group``)."""

    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        groups = 1
        for g in (32, 16, 8, 4, 2):
            if channels % g == 0:
                groups = g
                break
        return nn.GroupNorm(groups, channels)
    raise ValueError("norm_type must be 'batch' or 'group'")


def _conv_branch(hidden_dim: int, out_channels: int, num_convs: int, norm_type: str) -> nn.Sequential:
    """4-conv (paper A.3) BN/ReLU branch ending in a 1x1 output conv."""

    layers: list[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False))
        layers.append(_norm2d(norm_type, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
    return nn.Sequential(*layers)


class ProposalHead(nn.Module):
    """Anchorless proposal head for initial DeTra poses.

    Input:
        ``feature_map``: ``[B, D, H, W]`` BEV features from the lidar encoder.

    Raw outputs:
        ``heatmap``: ``[B, 1, H, W]`` objectness logits.
        ``box_params``: ``[B, 8, H, W]`` box regression values:
        ``dx, dy, log_length, log_width, sin_heading, cos_heading, z,
        log_height``. The last two give the proposal a real vertical extent so
        downstream 3D detection metrics are achievable.

    Decoded proposal target shape:
        ``proposals``: ``[B, N, 7]`` with
        ``x, y, z, length, width, height, heading``.

    Decoding:
        ``decode`` converts dense maps into top-N initial proposals for DeTra:
        ``boxes: [B,N,7]``, ``poses: [B,N,3]``, and ``scores: [B,N]``.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.heatmap = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.box = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 8, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict dense proposal maps.

        Args:
            feature_map: ``[B, D, H, W]`` float tensor.

        Returns:
            Dict with:
            - ``heatmap_logits``: ``[B, 1, H, W]``.
            - ``box_regression``: ``[B, 8, H, W]``.
        """

        return {
            "heatmap_logits": self.heatmap(feature_map),
            "box_regression": self.box(feature_map),
        }

    @staticmethod
    def _greedy_nms(
        boxes: torch.Tensor,
        candidate_count: int,
        num_proposals: int,
        nms_mode: str,
        nms_radius_m: float,
    ) -> list[int]:
        """Greedy NMS over score-sorted candidate boxes; returns kept indices.

        ``boxes`` is ``[C, 7]`` already sorted by descending score (as produced
        by the heatmap top-k). This reproduces the original per-candidate greedy
        acceptance exactly, but operates on a prebuilt tensor and a precomputed
        center-distance matrix instead of allocating a box per Python iteration:

        - ``center``: suppress a candidate if it is within ``nms_radius_m`` of any
          already-kept box.
        - ``rotated``: suppress if its rotated BEV IoU with a near kept box
          exceeds ``nms_radius_m`` (interpreted as an IoU threshold).
        """

        center_dist = torch.cdist(boxes[:, :2], boxes[:, :2])  # [C, C]
        if nms_mode == "center":
            # Precompute the within-radius adjacency once and move it to host in a
            # single transfer, then run the identical greedy acceptance in pure
            # Python. This avoids a GPU->CPU sync on every candidate (the previous
            # per-iteration ``bool((dist < r).any())``); the kept set is unchanged.
            within = (center_dist < nms_radius_m).cpu().tolist()  # [C][C] bool
            kept = []
            for cand_idx in range(candidate_count):
                row = within[cand_idx]
                if any(row[k] for k in kept):
                    continue
                kept.append(cand_idx)
                if len(kept) == num_proposals:
                    break
            return kept

        # Compute every potentially suppressing pair in batched GPU chunks,
        # transfer one compact adjacency matrix, then preserve the exact
        # score-ordered greedy acceptance logic on CPU. This removes the old
        # per-candidate GPU synchronization without changing kept indices.
        sizes = boxes[:, 3:5].clamp_min(0.1).sum(dim=-1)
        overlap_radius = 0.5 * (sizes[:, None] + sizes[None, :])
        lower_triangle = torch.tril(
            torch.ones(candidate_count, candidate_count, dtype=torch.bool, device=boxes.device),
            diagonal=-1,
        )
        pairs = ((center_dist < overlap_radius) & lower_triangle).nonzero(as_tuple=False)
        suppress = torch.zeros(
            candidate_count, candidate_count, dtype=torch.bool, device="cpu"
        )
        chunk_size = 65536
        for start in range(0, len(pairs), chunk_size):
            pair = pairs[start : start + chunk_size]
            iou = rotated_bev_iou(boxes[pair[:, 0]], boxes[pair[:, 1]])
            active = iou > nms_radius_m
            if active.any():
                selected = pair[active].cpu()
                suppress[selected[:, 0], selected[:, 1]] = True
        kept = []
        for cand_idx in range(candidate_count):
            if kept and bool(suppress[cand_idx, kept].any()):
                continue
            kept.append(cand_idx)
            if len(kept) == num_proposals:
                break
        return kept

    @torch.no_grad()
    def decode(
        self,
        head_outputs: dict[str, torch.Tensor],
        data_cfg: DataConfig,
        num_proposals: int,
        output_stride: int = 4,
        nms_radius_m: float = 0.1,
        default_z: float = 0.0,
        default_height: float = 1.7,
        center_mode: str = "regression",
        peak_filter: bool = True,
        nms_mode: str = "rotated",
    ) -> dict[str, torch.Tensor]:
        """Decode dense BEV predictions into initial object proposals.

        Args:
            head_outputs: Dict returned by ``forward``.
                - ``heatmap_logits``: ``[B,1,H,W]``.
                - ``box_regression``: ``[B,8,H,W]`` with channels
                  ``dx,dy,log_length,log_width,sin_heading,cos_heading,z,
                  log_height``.
            data_cfg: Data config containing ROI and raw BEV voxel size.
            num_proposals: Number of proposals ``N`` to return per batch item.
            output_stride: Feature-map stride relative to the raw BEV grid.
                With ``PaperLikeLidarEncoder`` fused stride-4 features this is
                ``4``.
            nms_radius_m: Rotated IoU NMS threshold. The name is preserved for
                CLI/checkpoint compatibility. Set ``0`` to disable suppression.
            default_z: Fallback z center if the head emits no z channel.
            default_height: Fallback box height if the head emits no log-height
                channel.
            center_mode: ``"regression"`` uses learned ``dx,dy`` offsets.
                ``"cell"`` ignores center offsets and decodes heatmap cell
                centers directly. This is useful to diagnose whether low
                proposal recall comes from heatmap ranking or center regression.
            peak_filter: If ``True``, keep only local heatmap maxima before
                top-k. If ``False``, top-k is taken over all heatmap cells. The
                latter is a diagnostic upper bound for dense heatmap recall.
            nms_mode: Suppression mode for selected proposals:
                ``"rotated"`` uses exact rotated BEV IoU and interprets
                ``nms_radius_m`` as an IoU threshold; ``"center"`` uses fast
                center-distance suppression and interprets ``nms_radius_m`` as
                a meter radius; ``"none"`` disables suppression.

        Returns:
            Dict with:
            - ``boxes``: ``[B,N,7]`` as
              ``x,y,z,length,width,height,heading``.
            - ``poses``: ``[B,N,3]`` as ``x,y,heading``.
            - ``scores``: ``[B,N]`` proposal confidence scores.

        Notes:
            Decoding is non-differentiable by design because top-k/NMS selects
            anchors. The raw dense outputs should still receive their own
            initial detector loss during training.
        """

        heatmap = head_outputs["heatmap_logits"].sigmoid()
        box_reg = head_outputs["box_regression"]
        b, _, h, w = heatmap.shape
        device = heatmap.device

        if peak_filter:
            pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
            peaks = heatmap * (heatmap == pooled)
        else:
            peaks = heatmap
        candidate_count = min(num_proposals * 4, h * w)
        scores, flat_idx = peaks.flatten(1).topk(candidate_count, dim=1)

        ys = torch.div(flat_idx, w, rounding_mode="floor")
        xs = flat_idx % w
        reg = box_reg.permute(0, 2, 3, 1)

        x_min, y_min, _, _ = data_cfg.roi_m
        meters_per_cell = data_cfg.voxel_size_m * output_stride
        if center_mode not in {"regression", "cell"}:
            raise ValueError("center_mode must be 'regression' or 'cell'")
        if nms_mode not in {"rotated", "center", "none"}:
            raise ValueError("nms_mode must be 'rotated', 'center', or 'none'")

        # --- Vectorized candidate box decode (all candidates at once). ---
        # Gather the per-candidate regression params: [B, C, channels].
        gather_idx = flat_idx[:, :, None].expand(b, candidate_count, reg.shape[-1])
        params = reg.reshape(b, h * w, reg.shape[-1]).gather(1, gather_idx)
        cell_x = x_min + (xs.float() + 0.5) * meters_per_cell
        cell_y = y_min + (ys.float() + 0.5) * meters_per_cell
        if center_mode == "regression":
            center_x = cell_x + params[..., 0]
            center_y = cell_y + params[..., 1]
        else:
            center_x = cell_x
            center_y = cell_y
        length = params[..., 2].exp().clamp_min(0.1)
        width = params[..., 3].exp().clamp_min(0.1)
        heading = torch.atan2(params[..., 4], params[..., 5])
        if params.shape[-1] >= 8:
            z = params[..., 6]
            height = params[..., 7].exp().clamp_min(0.1)
        else:
            z = torch.full_like(center_x, float(default_z))
            height = torch.full_like(center_x, float(default_height))
        cand_boxes = torch.stack(
            [center_x, center_y, z, length, width, height, heading], dim=-1
        )  # [B, C, 7]

        out_boxes = []
        out_scores = []
        for batch_idx in range(b):
            boxes_b = cand_boxes[batch_idx]
            scores_b = scores[batch_idx]
            if nms_mode == "none" or nms_radius_m <= 0:
                keep = list(range(min(num_proposals, candidate_count)))
            else:
                keep = self._greedy_nms(
                    boxes_b, candidate_count, num_proposals, nms_mode, nms_radius_m
                )

            if not keep:
                selected_boxes = torch.zeros(1, 7, device=device)
                selected_scores = torch.zeros(1, device=device)
            else:
                keep_idx = torch.as_tensor(keep, device=device, dtype=torch.long)
                selected_boxes = boxes_b[keep_idx]
                selected_scores = scores_b[keep_idx]

            # Pad to num_proposals by repeating the last box with score 0.
            count = selected_boxes.shape[0]
            if count < num_proposals:
                pad = num_proposals - count
                selected_boxes = torch.cat(
                    [selected_boxes, selected_boxes[-1:].expand(pad, 7)], dim=0
                )
                selected_scores = torch.cat(
                    [selected_scores, torch.zeros(pad, device=device)], dim=0
                )
            out_boxes.append(selected_boxes[:num_proposals])
            out_scores.append(selected_scores[:num_proposals])

        boxes = torch.stack(out_boxes)
        proposal_scores = torch.stack(out_scores)
        poses = boxes[..., [0, 1, 6]]
        return {"boxes": boxes, "poses": poses, "scores": proposal_scores}


class PaperProposalHead(ProposalHead):
    """Initial pose header following DeTra supplementary A.3.

    The BEV encoder already fuses the three multi-resolution maps into a single
    stride-4 ``feature_map`` (the paper's FPN output of shape ``(C, L/4, W/4)``).
    On top of that, A.3 specifies *four* convolution layers interleaved with
    BatchNorm + ReLU for the box-parameter heatmap, and a *separate* four-conv
    stack for the detection-score heatmap. This differs from :class:`ProposalHead`
    (one 3x3 + one 1x1 per branch).

    Subclasses :class:`ProposalHead` only to swap the two branch stacks; the
    top-k + greedy-NMS ``decode`` is inherited unchanged. The launcher should
    pair this head with rotated-IoU NMS at 0.1 (paper A.3) via
    ``--proposal-nms-mode rotated --proposal-nms-radius-m 0.1``. Selected by
    ``--proposal-head paper``.
    """

    def __init__(self, hidden_dim: int = 128, num_convs: int = 4, norm_type: str = "batch") -> None:
        nn.Module.__init__(self)
        self.heatmap = _conv_branch(hidden_dim, 1, num_convs, norm_type)
        self.box = _conv_branch(hidden_dim, 8, num_convs, norm_type)

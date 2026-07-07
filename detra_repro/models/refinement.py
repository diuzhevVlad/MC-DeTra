import torch
import torch.nn.functional as F
from torch import nn

from detra_repro.config import DataConfig, ModelConfig


class MultiScalePoseDeformableLidarAttention(nn.Module):
    """Multi-scale pose-anchored lidar attention.

    Inputs:
        ``queries``: ``[B, N, F, T, D]`` query volume.
        ``poses``: ``[B, N, F, T, 3]`` pose anchors in local-frame meters.
        ``features``: dict of BEV feature maps:
        ``{"stride4": [B,D,H4,W4], "stride8": [B,D,H8,W8],
        "stride16": [B,D,H16,W16]}``.

    Output:
        ``context``: ``[B, N, F, T, D]``.

    DeTra mapping:
        This is the paper's core lidar cross-attention idea. Each query predicts
        learned 2D offsets and attention weights around its current-time pose
        anchor. Features are bilinearly sampled from multi-resolution BEV maps
        and weighted together. The implementation uses ``grid_sample`` for
        readability rather than a custom CUDA deformable-attention operator.
    """

    def __init__(
        self,
        hidden_dim: int,
        data_cfg: DataConfig,
        num_points: int = 4,
        level_strides: tuple[int, ...] = (4, 8, 16),
        radius_m: float = 2.0,
        anchor_mode: str = "current",
    ) -> None:
        super().__init__()
        if anchor_mode not in {"current", "time"}:
            raise ValueError("anchor_mode must be 'current' or 'time'")
        self.data_cfg = data_cfg
        self.num_points = num_points
        self.level_strides = level_strides
        self.level_names = tuple(f"stride{s}" for s in level_strides)
        self.num_levels = len(level_strides)
        self.radius_m = radius_m
        self.anchor_mode = anchor_mode
        self.offsets = nn.Linear(hidden_dim, self.num_levels * num_points * 2)
        self.weights = nn.Linear(hidden_dim, self.num_levels * num_points)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.debug_enabled = False
        self.last_debug: dict[str, torch.Tensor] = {}

    def _metric_to_grid(
        self,
        xy: torch.Tensor,
        height: int,
        width: int,
        stride: int,
    ) -> torch.Tensor:
        """Convert metric local-frame XY into ``grid_sample`` coordinates.

        Args:
            xy: ``[B, Q, K, 2]`` in meters.
            height: Feature-map height.
            width: Feature-map width.
            stride: Feature-map stride relative to raw BEV voxel grid.

        Returns:
            Normalized grid tensor ``[B, Q*K, 1, 2]``.
        """

        x_min, y_min, _, _ = self.data_cfg.roi_m
        meters_per_cell = self.data_cfg.voxel_size_m * stride
        col = (xy[..., 0] - x_min) / meters_per_cell - 0.5
        row = (xy[..., 1] - y_min) / meters_per_cell - 0.5
        gx = 2.0 * col / max(width - 1, 1) - 1.0
        gy = 2.0 * row / max(height - 1, 1) - 1.0
        return torch.stack([gx, gy], dim=-1).flatten(1, 2).unsqueeze(2)

    def forward(
        self,
        queries: torch.Tensor,
        poses: torch.Tensor,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sample multi-scale lidar context around pose anchors.

        Args:
            queries: ``[B, N, F, T, D]``.
            poses: ``[B, N, F, T, 3]``.
            features: stride-4/8/16 feature maps.

        Returns:
            ``[B, N, F, T, D]`` context tensor.
        """

        b, n, f, t, d = queries.shape
        q = queries.reshape(b, n * f * t, d)
        offsets = torch.tanh(self.offsets(q)).view(
            b, n, f, t, self.num_levels, self.num_points, 2
        )
        offsets = offsets * self.radius_m
        weights = self.weights(q).view(b, n, f, t, self.num_levels, self.num_points)
        weights = weights.flatten(-2).softmax(dim=-1).view_as(weights)

        if self.anchor_mode == "current":
            anchor_xy = poses[:, :, :, 0, :2].unsqueeze(3).unsqueeze(4)
        else:
            anchor_xy = poses[..., :2].unsqueeze(4)
        context = queries.new_zeros(b, n, f, t, d)
        in_bounds_values = []

        for level_idx, (level_name, stride) in enumerate(zip(self.level_names, self.level_strides)):
            feature_map = features[level_name]
            _, _, height, width = feature_map.shape
            sample_xy = anchor_xy + offsets[:, :, :, :, level_idx]
            grid = self._metric_to_grid(
                sample_xy.reshape(b, n * f * t, self.num_points, 2),
                height,
                width,
                stride,
            )
            if self.debug_enabled:
                in_bounds = (
                    (grid[..., 0] >= -1.0)
                    & (grid[..., 0] <= 1.0)
                    & (grid[..., 1] >= -1.0)
                    & (grid[..., 1] <= 1.0)
                )
                in_bounds_values.append(in_bounds.float().mean())
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
                b, n, f, t, self.num_points, d
            )
            context = context + (
                sampled * weights[:, :, :, :, level_idx].unsqueeze(-1)
            ).sum(dim=-2)

        if self.debug_enabled:
            with torch.no_grad():
                flat_offsets = offsets.detach().reshape(-1, self.num_levels, self.num_points, 2)
                offset_norm = flat_offsets.norm(dim=-1)
                flat_weights = weights.detach().reshape(-1, self.num_levels, self.num_points)
                weight_flat = flat_weights.flatten(1)
                entropy = -(
                    weight_flat * weight_flat.clamp_min(1e-8).log()
                ).sum(dim=-1)
                self.last_debug = {
                    "offset_norm_mean": offset_norm.mean(),
                    "offset_norm_p95": torch.quantile(offset_norm.flatten(), 0.95),
                    "offset_norm_max": offset_norm.max(),
                    "weight_entropy_mean": entropy.mean(),
                    "weight_entropy_max": entropy.new_tensor(
                        float(self.num_levels * self.num_points)
                    ).log(),
                    "level_weight_mean": flat_weights.sum(dim=-1).mean(dim=0),
                    "point_weight_mean": flat_weights.mean(dim=(0, 1)),
                    "radius_m": offsets.new_tensor(float(self.radius_m)),
                }
                if in_bounds_values:
                    self.last_debug["sample_in_bounds_mean"] = torch.stack(
                        in_bounds_values
                    ).mean()

        return self.out(context)


class MultiScalePoseCenterLidarAttention(nn.Module):
    """Multi-scale pose-centered lidar attention ablation.

    Inputs:
        ``queries``: ``[B, N, F, T, D]`` query volume.
        ``poses``: ``[B, N, F, T, 3]`` pose anchors in local-frame meters.
        ``features``: dict of BEV feature maps.

    Output:
        ``context``: ``[B, N, F, T, D]``.

    Purpose:
        This removes learned deformable offsets and samples exactly at the
        current pose anchor on every feature level. If this behaves similarly to
        deformable attention, the current failure is less likely to be caused by
        offset learning itself.
    """

    def __init__(
        self,
        hidden_dim: int,
        data_cfg: DataConfig,
        level_strides: tuple[int, ...] = (4, 8, 16),
        anchor_mode: str = "current",
    ) -> None:
        super().__init__()
        if anchor_mode not in {"current", "time"}:
            raise ValueError("anchor_mode must be 'current' or 'time'")
        self.data_cfg = data_cfg
        self.level_strides = level_strides
        self.level_names = tuple(f"stride{s}" for s in level_strides)
        self.anchor_mode = anchor_mode
        self.level_logits = nn.Parameter(torch.zeros(len(level_strides)))
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def _metric_to_grid(
        self,
        xy: torch.Tensor,
        height: int,
        width: int,
        stride: int,
    ) -> torch.Tensor:
        """Convert metric local-frame XY into ``grid_sample`` coordinates."""

        x_min, y_min, _, _ = self.data_cfg.roi_m
        meters_per_cell = self.data_cfg.voxel_size_m * stride
        col = (xy[..., 0] - x_min) / meters_per_cell - 0.5
        row = (xy[..., 1] - y_min) / meters_per_cell - 0.5
        gx = 2.0 * col / max(width - 1, 1) - 1.0
        gy = 2.0 * row / max(height - 1, 1) - 1.0
        return torch.stack([gx, gy], dim=-1).unsqueeze(2)

    def forward(
        self,
        queries: torch.Tensor,
        poses: torch.Tensor,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sample lidar features at current pose centers on all scales."""

        b, n, f, t, d = queries.shape
        if self.anchor_mode == "current":
            anchor_xy = poses[:, :, :, 0, :2].unsqueeze(3).expand(b, n, f, t, 2)
        else:
            anchor_xy = poses[..., :2]
        level_weights = self.level_logits.softmax(dim=0)
        context = queries.new_zeros(b, n, f, t, d)

        for level_idx, (level_name, stride) in enumerate(zip(self.level_names, self.level_strides)):
            feature_map = features[level_name]
            _, _, height, width = feature_map.shape
            grid = self._metric_to_grid(
                anchor_xy.reshape(b, n * f * t, 2),
                height,
                width,
                stride,
            )
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, n, f, t, d)
            context = context + sampled * level_weights[level_idx]

        return self.out(context)


class MapKNNAttention(nn.Module):
    """Optional k-nearest-neighbor cross-attention to vector-map tokens.

    Inputs:
        ``queries``: ``[B, N, F, T, D]``.
        ``poses``: ``[B, N, F, T, 3]``.
        ``map_tokens``: ``[B, M, D]``.
        ``map_xy``: ``[B, M, 2]``.

    Output:
        ``context``: ``[B, N, F, T, D]``.

    DeTra mapping:
        The paper attends to the k nearest map tokens at time indices
        ``{0, floor((T-1)/2), T-1}``. This module is ready for those tokens, but
        the repo still needs an HD-map parser and LaneGCN/GoRela-style encoder.
    """

    def __init__(self, hidden_dim: int, k: int = 4) -> None:
        super().__init__()
        self.k = k
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        poses: torch.Tensor,
        map_tokens: torch.Tensor,
        map_xy: torch.Tensor,
        time_indices: tuple[int, ...],
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend selected query time slots to nearest map tokens."""

        b, n, f, _, d = queries.shape
        m = map_tokens.shape[1]
        k = min(self.k, m)
        if k == 0 or (map_mask is not None and not map_mask.any()):
            return torch.zeros_like(queries)

        context = torch.zeros_like(queries)
        if map_mask is not None:
            map_tokens = map_tokens * map_mask.unsqueeze(-1).to(map_tokens.dtype)
        keys = self.key(map_tokens)
        values = self.value(map_tokens)
        if map_mask is not None:
            valid_scale = map_mask.unsqueeze(-1).to(map_tokens.dtype)
            keys = keys * valid_scale
            values = values * valid_scale
        scale = d**-0.5

        for time_idx in time_indices:
            pose_xy = poses[:, :, :, time_idx, :2].reshape(b, n * f, 2)
            dist = torch.cdist(pose_xy, map_xy)
            if map_mask is not None:
                dist = dist.masked_fill(~map_mask[:, None], float("inf"))
            knn = dist.topk(k, largest=False).indices
            gather_idx = knn[..., None].expand(b, n * f, k, d)
            local_keys = keys[:, None].expand(b, n * f, m, d).gather(2, gather_idx)
            local_values = values[:, None].expand(b, n * f, m, d).gather(2, gather_idx)
            q = self.query(queries[:, :, :, time_idx].reshape(b, n * f, d))
            attn = (q[:, :, None] * local_keys).sum(dim=-1).mul(scale).softmax(dim=-1)
            local_context = (attn[..., None] * local_values).sum(dim=2)
            context[:, :, :, time_idx] = self.out(local_context).reshape(b, n, f, d)

        if map_mask is not None:
            context = context * map_mask.any(dim=1).view(b, 1, 1, 1, 1).to(context.dtype)
        return context


class MapTokenEncoder(nn.Module):
    """Project cached vector-map tokens into DeTra map-token embeddings.

    Inputs:
        ``map_features``: ``[B, M, C_map]`` cached geometry/type features.
        ``map_mask``: optional bool tensor ``[B, M]``.

    Output:
        ``map_tokens``: ``[B, M, D]``. Invalid padded entries are zeroed.

    DeTra mapping:
        The paper uses a lane-graph encoder. This is a lightweight replacement
        that keeps the downstream interface identical: local map tokens with
        coordinates are consumed by k-nearest-neighbor map attention.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        map_features: torch.Tensor,
        map_xy: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode map features into tokens.

        ``map_xy`` is accepted but ignored, so this MLP encoder is a drop-in for
        :class:`MapGraphEncoder` (same call signature).
        """

        tokens = self.net(map_features)
        if map_mask is not None:
            tokens = tokens * map_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens


class MapGraphEncoder(nn.Module):
    """Runtime kNN-graph (LaneGCN/GoRela-style) map-token encoder.

    Inputs:
        ``map_features``: ``[B, M, C_map]`` cached geometry/type features.
        ``map_xy``: ``[B, M, 2]`` ego-frame token coordinates.
        ``map_mask``: optional bool ``[B, M]`` (True = valid token).

    Output:
        ``map_tokens``: ``[B, M, D]`` with invalid entries zeroed. Same interface
        as :class:`MapTokenEncoder`, so :class:`MapKNNAttention` and the
        ``DeTraMini`` wiring are unchanged.

    DeTra mapping:
        The paper uses a lane-graph encoder over connected polylines. The cache
        only stores a flat, unordered set of ego-frame tokens (no polyline ids),
        so this builds a kNN proximity graph from ``map_xy`` at runtime and runs
        a few GraphSAGE-style message-passing layers with a relative-position
        edge feature. This adds the spatial context the per-token MLP lacks. A
        fully faithful lane-graph would need polyline grouping stored in the
        cache (documented as a future step).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        graph_k: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.graph_k = graph_k
        self.num_layers = num_layers
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Per-layer message MLP (self + mean-neighbor + max-neighbor + edge) and norm.
        self.edge = nn.ModuleList(nn.Linear(2, hidden_dim) for _ in range(num_layers))
        self.message = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.norm = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))

    def forward(
        self,
        map_features: torch.Tensor,
        map_xy: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode map tokens with kNN-graph message passing."""

        b, m, _ = map_features.shape
        tokens = self.input(map_features)  # [B, M, D]
        if map_mask is not None:
            mask_f = map_mask.to(tokens.dtype)
            tokens = tokens * mask_f.unsqueeze(-1)
        if map_xy is None or m < 2:
            return tokens

        # kNN neighbor indices from token coordinates (invalid tokens pushed away).
        dist = torch.cdist(map_xy, map_xy)  # [B, M, M]
        if map_mask is not None:
            invalid = ~map_mask
            dist = dist.masked_fill(invalid[:, None, :], float("inf"))
            dist = dist.masked_fill(invalid[:, :, None], float("inf"))
        k = min(self.graph_k, m)
        neigh = dist.topk(k, dim=-1, largest=False).indices  # [B, M, k]

        batch_idx = torch.arange(b, device=tokens.device)[:, None, None]
        for layer in range(self.num_layers):
            neigh_feat = tokens[batch_idx, neigh]  # [B, M, k, D]
            rel = map_xy[batch_idx, neigh] - map_xy[:, :, None, :]  # [B, M, k, 2]
            msg = neigh_feat + self.edge[layer](rel)
            agg = torch.cat([tokens, msg.mean(dim=2), msg.max(dim=2).values], dim=-1)
            update = self.message[layer](agg)
            tokens = self.norm[layer](tokens + update)
            if map_mask is not None:
                tokens = tokens * mask_f.unsqueeze(-1)
        return tokens


def build_map_encoder(cfg) -> nn.Module:
    """Build the map-token encoder selected by ``cfg.map_encoder_type``.

    ``"graph"`` (default) -> :class:`MapGraphEncoder` (runtime kNN graph);
    ``"mlp"`` -> :class:`MapTokenEncoder` (per-token MLP, the legacy encoder).
    Both share the ``(map_features, map_xy, map_mask)`` forward signature.
    """

    encoder_type = getattr(cfg, "map_encoder_type", "graph")
    if encoder_type == "mlp":
        return MapTokenEncoder(cfg.map_feature_dim, cfg.hidden_dim)
    if encoder_type == "graph":
        return MapGraphEncoder(
            cfg.map_feature_dim,
            cfg.hidden_dim,
            graph_k=getattr(cfg, "map_graph_k", 8),
            num_layers=getattr(cfg, "map_graph_layers", 2),
        )
    raise ValueError("map_encoder_type must be 'graph' or 'mlp'")


class RefinementBlock(nn.Module):
    """One DeTra refinement transformer block.

    Inputs:
        ``queries``: ``[B, N, F, T, D]``.
        ``poses``: ``[B, N, F, T, 3]``.
        ``lidar_features``: stride-4/8/16 BEV maps.
        optional ``map_tokens``/``map_xy``.

    Attention layers follow the paper (Fig. 3 / Eq. 2). Each specialized
    attention is its own transformer sub-layer: a residual + LayerNorm around the
    attention, followed by a residual + LayerNorm around a per-attention FFN
    (paper Eq. 2 applies LN after every attention layer A; the standard
    transformer FFN sub-layer follows each). The pose update ``U`` is a separate
    module (:class:`PoseUpdate`) owned by :class:`DeTraMini`, so this block only
    refines the query features. Attention order:
        1. multi-scale deformable lidar attention (anchored at the t=0 pose)
        2. optional k-NN map attention
        3. time self-attention
        4. mode self-attention
        5. object self-attention

    Each of the above has its own ``norm_attn_*`` (post-attention LN), ``ffn_*``,
    and ``norm_ffn_*`` (post-FFN LN). This replaces the earlier single shared
    LayerNorm reused for every residual plus one shared FFN at the very end,
    which under-parameterized the normalization/FFN capacity vs the paper
    (CHANGELOG 2026-06-21 gap #6).
    """

    def __init__(
        self,
        hidden_dim: int,
        data_cfg: DataConfig,
        num_heads: int = 4,
        deformable_points: int = 4,
        deformable_radius_m: float = 2.0,
        attention_type: str = "deformable",
        lidar_attention_anchor: str = "current",
        map_k: int = 4,
    ) -> None:
        super().__init__()
        if attention_type == "deformable":
            self.lidar_attn = MultiScalePoseDeformableLidarAttention(
                hidden_dim,
                data_cfg,
                num_points=deformable_points,
                radius_m=deformable_radius_m,
                anchor_mode=lidar_attention_anchor,
            )
        elif attention_type == "center":
            self.lidar_attn = MultiScalePoseCenterLidarAttention(
                hidden_dim,
                data_cfg,
                anchor_mode=lidar_attention_anchor,
            )
        else:
            raise ValueError(
                f"Unsupported attention_type={attention_type!r}; use deformable or center"
            )
        self.map_attn = MapKNNAttention(hidden_dim, k=map_k)
        self.time_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mode_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.object_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Per-attention post-attention LayerNorms (paper Eq. 2: LN after each A).
        self._sublayers = ("lidar", "map", "time", "mode", "object")
        self.norm_attn = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in self._sublayers}
        )
        # Per-attention FFN sub-layer (residual + LN), one per specialized
        # attention rather than a single shared FFN at the block tail.
        self.ffn = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for name in self._sublayers
            }
        )
        self.norm_ffn = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in self._sublayers}
        )

    def _ffn_sublayer(self, name: str, queries: torch.Tensor) -> torch.Tensor:
        """Apply the per-attention FFN sub-layer (residual + LayerNorm)."""

        return self.norm_ffn[name](queries + self.ffn[name](queries))

    def forward(
        self,
        queries: torch.Tensor,
        poses: torch.Tensor,
        lidar_features: dict[str, torch.Tensor],
        map_tokens: torch.Tensor | None = None,
        map_xy: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine query features using the pose-anchored attention stack.

        Args:
            queries: ``[B, N, F, T, D]``.
            poses: ``[B, N, F, T, 3]`` pose anchors (used only as attention
                anchors here; the pose update happens in :class:`PoseUpdate`).
            lidar_features: stride-4/8/16 BEV maps.
            map_tokens / map_xy: optional map tokens for k-NN attention.

        Returns:
            Refined queries ``[B, N, F, T, D]``.
        """

        b, n, f, t, d = queries.shape

        # 1. LiDAR cross-attention sub-layer.
        queries = self.norm_attn["lidar"](
            queries + self.lidar_attn(queries, poses, lidar_features)
        )
        queries = self._ffn_sublayer("lidar", queries)

        # 2. Optional map cross-attention sub-layer.
        if map_tokens is not None and map_xy is not None:
            map_time_indices = tuple(sorted({0, (t - 1) // 2, t - 1}))
            queries = self.norm_attn["map"](
                queries + self.map_attn(
                    queries,
                    poses,
                    map_tokens,
                    map_xy,
                    map_time_indices,
                    map_mask=map_mask,
                )
            )
            queries = self._ffn_sublayer("map", queries)

        # 3. Time self-attention sub-layer.
        x = queries.reshape(b * n * f, t, d)
        x, _ = self.time_attn(x, x, x)
        queries = self.norm_attn["time"](queries + x.reshape(b, n, f, t, d))
        queries = self._ffn_sublayer("time", queries)

        # 4. Mode self-attention sub-layer.
        x = queries.permute(0, 1, 3, 2, 4).reshape(b * n * t, f, d)
        x, _ = self.mode_attn(x, x, x)
        queries = self.norm_attn["mode"](
            queries + x.reshape(b, n, t, f, d).permute(0, 1, 3, 2, 4)
        )
        queries = self._ffn_sublayer("mode", queries)

        # 5. Object self-attention sub-layer.
        x = queries.permute(0, 2, 3, 1, 4).reshape(b * f * t, n, d)
        x, _ = self.object_attn(x, x, x)
        queries = self.norm_attn["object"](
            queries + x.reshape(b, f, t, n, d).permute(0, 3, 1, 2, 4)
        )
        queries = self._ffn_sublayer("object", queries)
        return queries


class PoseUpdate(nn.Module):
    """DeTra pose-update function ``U`` applied after each refinement block.

    Given refined queries ``Q`` and the current pose volume ``P``, this produces
    the next pose volume and the per-block prediction dict, exactly as described
    in the paper (Sec. 3.2, "Pose update"):

    - Detection (t=0): ``D^(i+1) = D^(i) + MLP(mean_f Q_{t=0,f})`` where
      ``D = (dx, dy, dlog_l, dlog_w, dtheta)`` is a residual on the current box.
      The updated ``(x, y, theta)`` become ``P_{t=0}`` for all modes.
    - Future (t>0): a single bi-directional GRU processes the per-(object,mode)
      temporal query features; a per-step MLP predicts an isotropic Laplacian
      ``(mu_x, mu_y, log sigma_x, log sigma_y)``. Future positions are
      ``P_{t>0}.xy = P_{t=0}.xy + mu`` and headings come from finite differences
      of the waypoints. Mode logits come from an MLP on the temporal mean of the
      GRU hidden states.

    The same head weights are reused at every block (the losses supervise every
    block's output for deep supervision).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 3,
        predict_past: bool = False,
        past_horizon: int = 0,
    ) -> None:
        super().__init__()
        # Detection residual: dx, dy, dz, dlog_length, dlog_width, dlog_height,
        # dheading. x,y enter the pose; z,l,w,h,heading enter the box residual.
        # z is regressed so 3D detection metrics are achievable (otherwise the
        # box z stays frozen at the proposal default and 3D IoU AP collapses).
        self.box_head = nn.Linear(hidden_dim, 7)
        self.object_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.velocity_head = nn.Linear(hidden_dim, 2)
        self.future_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.future_mode_head = nn.Linear(hidden_dim * 2, 1)
        # Past-trajectory head (M1): a single-mode uni-directional GRU rolls the
        # mode-averaged current query backward to reconstruct the observed past
        # track. Past is observed/near-deterministic, so it is single-mode (no F)
        # and regresses xy offsets only (SmoothL1, no Laplace). A learned
        # per-past-step embedding seeds each step so the head does not collapse to
        # the current point. ``past_xy`` is a decoded *output target*, never an
        # input -> inference-safe.
        self.predict_past = predict_past
        self.past_horizon = past_horizon
        if predict_past and past_horizon > 0:
            self.past_time_embed = nn.Parameter(
                torch.randn(1, 1, past_horizon, hidden_dim) * 0.02
            )
            self.past_gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.past_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.past_time_embed = None
            self.past_gru = None
            self.past_head = None
        # Start as a near-identity update so early blocks do not jump.
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    @staticmethod
    def _headings_from_waypoints(
        current_xy: torch.Tensor,
        future_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute future headings via temporal finite differences.

        Args:
            current_xy: ``[B, N, F, 2]`` current (t=0) positions.
            future_xy: ``[B, N, F, T-1, 2]`` future waypoints.

        Returns:
            ``[B, N, F, T-1]`` headings in radians. The heading at each future
            step points along the displacement from the previous waypoint
            (``t=0`` for the first future step).
        """

        prev = torch.cat([current_xy.unsqueeze(3), future_xy[..., :-1, :]], dim=3)
        delta = future_xy - prev
        return torch.atan2(delta[..., 1], delta[..., 0])

    def forward(
        self,
        queries: torch.Tensor,
        poses: torch.Tensor,
        base_box: torch.Tensor,
        rollout_velocity: torch.Tensor | None = None,
        use_learned_velocity: bool = False,
        learned_velocity_frame: str = "ego",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Apply the pose update and produce a prediction dict.

        Args:
            queries: ``[B, N, F, T, D]`` refined queries.
            poses: ``[B, N, F, T, 3]`` current pose anchors.
            base_box: ``[B, N, 7]`` current detection box (residual base).
            rollout_velocity: Optional per-query velocity ``[B,N,2]`` in
                meters per future step. When provided, future predictions are
                residuals around constant-velocity rollout.
            use_learned_velocity: If true, use the internal velocity head as
                the rollout velocity instead of ``rollout_velocity``.
            learned_velocity_frame: ``ego`` predicts ego-frame velocity
                directly. ``object`` predicts longitudinal/lateral velocity in
                the predicted current box frame and rotates it into ego frame.

        Returns:
            ``(next_poses, next_box, pred)`` where ``next_poses`` is
            ``[B,N,F,T,3]``, ``next_box`` is ``[B,N,7]``, and ``pred`` is the
            prediction dict consumed by the losses.
        """

        b, n, f, t, d = queries.shape

        # --- Detection head (t=0), averaged over modes (paper: mean_f Q_{t=0}). ---
        det_query = queries[:, :, :, 0].mean(dim=2)  # [B, N, D]
        det_residual = self.box_head(det_query)  # [B, N, 7]
        pred_velocity_raw = self.velocity_head(det_query)  # [B, N, 2]
        dxy = det_residual[..., :2]
        dz = det_residual[..., 2]
        dlog_size = det_residual[..., 3:6].clamp(-4.0, 4.0)
        dtheta = det_residual[..., 6]

        next_box = base_box.clone()
        next_box[..., :2] = base_box[..., :2] + dxy
        next_box[..., 2] = base_box[..., 2] + dz
        next_box[..., 3:6] = base_box[..., 3:6].clamp_min(1e-2) * dlog_size.exp()
        next_box[..., 6] = base_box[..., 6] + dtheta

        # Current pose (t=0) for every mode comes from the updated detection.
        current_pose = torch.stack(
            [next_box[..., 0], next_box[..., 1], next_box[..., 6]], dim=-1
        )  # [B, N, 3]
        current_pose_modes = current_pose[:, :, None, :].expand(b, n, f, 3)

        if learned_velocity_frame == "ego":
            pred_velocity = pred_velocity_raw
            pred_velocity_local = None
        elif learned_velocity_frame == "object":
            heading = next_box[..., 6]
            c = torch.cos(heading)
            s = torch.sin(heading)
            v_long = pred_velocity_raw[..., 0]
            v_lat = pred_velocity_raw[..., 1]
            pred_velocity = torch.stack(
                [v_long * c - v_lat * s, v_long * s + v_lat * c],
                dim=-1,
            )
            pred_velocity_local = pred_velocity_raw
        else:
            raise ValueError("learned_velocity_frame must be 'ego' or 'object'")

        # --- Future head (t>0): bi-directional GRU per (object, mode). ---
        future_features = queries[:, :, :, 1:].reshape(b * n * f, t - 1, d)
        future_context, _ = self.future_gru(future_features)  # [B*N*F, T-1, 2D]
        future_params = self.future_head(future_context).reshape(b, n, f, t - 1, 4)
        future_offsets = future_params[..., :2]
        # Cap the upper bound so the Laplace WTA loss cannot be minimized by
        # inflating sigma instead of learning motion (sigma <= ~3 m). The lower
        # bound bounds the NLL ratio abs_error/sigma: a -5 floor allowed
        # sigma_min ~= 0.0067 m, so a large-error winner mode produced
        # abs_error/sigma ~= O(100s) whose d/d(log_sigma) = -abs_error/sigma
        # exploded into NaN grads in the refinement/forecast head (observed
        # clip_frac ~0.85 on the 2026-06-20 runs, gnorm_refine dominating). A
        # -2 floor (sigma_min ~= 0.135 m) bounds that ratio without constraining
        # real forecasts (observed median sigma ~= 0.21 m). See PLAN Tier 0.1.
        future_log_scale = future_params[..., 2:].clamp(-2.0, 1.1)

        current_xy_modes = current_pose_modes[..., :2]  # [B, N, F, 2]
        if use_learned_velocity:
            rollout_velocity = pred_velocity
        if rollout_velocity is None:
            future_base_xy = current_xy_modes.unsqueeze(3)
        else:
            steps = torch.arange(
                1,
                t,
                device=queries.device,
                dtype=queries.dtype,
            ).view(1, 1, 1, t - 1, 1)
            velocity_modes = rollout_velocity[:, :, None, None, :].expand(b, n, f, t - 1, 2)
            future_base_xy = current_xy_modes.unsqueeze(3) + steps * velocity_modes
        future_xy = future_base_xy + future_offsets  # [B, N, F, T-1, 2]
        future_headings = self._headings_from_waypoints(current_xy_modes, future_xy)

        # Assemble the full pose volume: t=0 from detection, t>0 from the GRU.
        next_poses = poses.new_zeros((b, n, f, t, 3))
        next_poses[:, :, :, 0, :] = current_pose_modes
        next_poses[:, :, :, 1:, :2] = future_xy
        next_poses[:, :, :, 1:, 2] = future_headings

        future_mode_features = future_context.reshape(b, n, f, t - 1, d * 2).mean(dim=3)
        # box_params kept for ``current_boxes_from_pred``: residual vs base_box as
        # (dz, dlog_l, dlog_w, dlog_h, dheading); x,y/theta already live in poses.
        box_params = torch.stack(
            [
                next_box[..., 2] - base_box[..., 2],
                (next_box[..., 3].clamp_min(1e-2) / base_box[..., 3].clamp_min(1e-2)).log(),
                (next_box[..., 4].clamp_min(1e-2) / base_box[..., 4].clamp_min(1e-2)).log(),
                (next_box[..., 5].clamp_min(1e-2) / base_box[..., 5].clamp_min(1e-2)).log(),
                next_box[..., 6] - base_box[..., 6],
            ],
            dim=-1,
        )
        pred = {
            "poses": next_poses,
            "future_xy": future_xy,
            "future_base_xy": future_base_xy.expand(b, n, f, t - 1, 2),
            "future_log_scale": future_log_scale,
            "object_logits": self.object_head(det_query).squeeze(-1),
            "class_logits": self.class_head(det_query),
            "mode_logits": self.future_mode_head(future_mode_features).squeeze(-1),
            "pred_velocity": pred_velocity,
            "pred_velocity_raw": pred_velocity_raw,
            "box_params": box_params,
            "base_boxes": base_box,
        }
        if pred_velocity_local is not None:
            pred["pred_velocity_local"] = pred_velocity_local

        # --- Past head (M1), single-mode, decoded from the current query. ---
        if self.past_gru is not None:
            h = self.past_horizon
            past_seed = det_query[:, :, None, :].expand(b, n, h, d) + self.past_time_embed
            past_ctx, _ = self.past_gru(past_seed.reshape(b * n, h, d))
            past_off = self.past_head(past_ctx).reshape(b, n, h, 2)
            # Anchored at the (mode-averaged) current center; slots run
            # newest->oldest (slot 0 = t=-1). Kept out of ``next_poses`` so eval
            # and decode are untouched.
            pred["past_xy"] = current_pose[..., None, :2] + past_off
        return next_poses, next_box, pred


class DeTraMini(nn.Module):
    """DeTra-style detector/forecaster core (paper-faithful).

    Inputs:
        ``lidar_features``: multi-scale dict from ``PaperLikeLidarEncoder``.
        ``initial_poses``: ``[B, N, 3]`` current pose proposals.
        ``initial_boxes``: ``[B, N, 7]`` proposal boxes (detector output, the
            residual base for detection; not a ground-truth cue).
        optional ``map_tokens``: ``[B, M, D]`` and ``map_xy``: ``[B, M, 2]``.

    Outputs:
        - ``poses``: ``[B, N, F, T, 3]`` refined pose trajectories.
        - ``object_logits``: ``[B, N]``.
        - ``mode_logits``: ``[B, N, F]``.
        - ``future_xy`` / ``future_log_scale``: GRU Laplacian forecasts.
        - ``box_params`` / ``base_boxes``: detection-box residual + base.
        - ``aux_outputs``: one prediction dict per refinement block for deep
          supervision.

    Query initialization (paper Sec. 3.2):
        Queries are the sum of learnable mode params ``F in R^{1xFx1xd}`` and time
        params ``T in R^{1x1xTxd}`` broadcast to ``N`` objects. All objects start
        from identical query features and are differentiated only by their
        initial poses, which guide the local attention. No lidar-sampled,
        pose, box, or actor-history cues are injected.

    Remaining gap to the exact paper:
        The map branch is wired as an optional interface only; the repo still
        needs Waymo map parsing and a vector-map encoder for final forecasting
        quality. The BEV gIoU used by the losses is axis-aligned.
    """

    def __init__(self, cfg: ModelConfig, data_cfg: DataConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg or DataConfig()
        # Larger init so modes start differentiated; with a tiny 0.02 init plus
        # averaging mode self-attention the 6 modes collapse to one under WTA.
        self.mode_embed = nn.Parameter(torch.randn(1, 1, cfg.num_modes, 1, cfg.hidden_dim) * 0.5)
        self.time_embed = nn.Parameter(torch.randn(1, 1, 1, cfg.num_times, cfg.hidden_dim) * 0.02)
        self.velocity_embed = nn.Linear(2, cfg.hidden_dim) if cfg.use_query_velocity else None
        self.blocks = nn.ModuleList(
            RefinementBlock(
                cfg.hidden_dim,
                self.data_cfg,
                deformable_points=cfg.deformable_points,
                deformable_radius_m=cfg.deformable_radius_m,
                attention_type=cfg.attention_type,
                lidar_attention_anchor=cfg.lidar_attention_anchor,
                map_k=cfg.map_k,
            )
            for _ in range(cfg.num_refinement_blocks)
        )
        self.pose_update = PoseUpdate(
            cfg.hidden_dim,
            num_classes=cfg.num_classes,
            predict_past=getattr(cfg, "predict_past", False),
            past_horizon=getattr(cfg, "past_horizon", 0),
        )
        if cfg.forecast_base_mode not in {"current", "constant_velocity", "learned_velocity"}:
            raise ValueError(
                "forecast_base_mode must be 'current', 'constant_velocity', or 'learned_velocity'"
            )
        if getattr(cfg, "forecast_velocity_frame", "ego") not in {"ego", "object"}:
            raise ValueError("forecast_velocity_frame must be 'ego' or 'object'")

    def _default_box(self, initial_poses: torch.Tensor) -> torch.Tensor:
        """Build a fallback box when no detector proposal boxes are supplied."""

        b, n, _ = initial_poses.shape
        box = initial_poses.new_zeros((b, n, 7))
        box[..., :2] = initial_poses[..., :2]
        box[..., 3:6] = initial_poses.new_tensor([4.5, 2.0, 1.7])
        box[..., 6] = initial_poses[..., 2]
        return box

    def _as_multiscale(self, lidar_features: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Accept a stride-4 tensor for tests, or a full multi-scale dict."""

        if isinstance(lidar_features, dict):
            return lidar_features
        return {
            "stride4": lidar_features,
            "stride8": F.avg_pool2d(lidar_features, kernel_size=2, stride=2, ceil_mode=True),
            "stride16": F.avg_pool2d(lidar_features, kernel_size=4, stride=4, ceil_mode=True),
        }

    def forward(
        self,
        lidar_features: dict[str, torch.Tensor] | torch.Tensor,
        initial_poses: torch.Tensor,
        initial_boxes: torch.Tensor | None = None,
        map_tokens: torch.Tensor | None = None,
        map_xy: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
        query_velocity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run iterative DeTra refinement.

        Args:
            lidar_features: stride-4/8/16 feature maps or one stride-4 map.
            initial_poses: ``[B, N, 3]`` as ``x,y,heading``.
            initial_boxes: optional proposal boxes ``[B,N,7]`` (detector output)
                used as the base for detection-box residuals.
            map_tokens: optional ``[B, M, D]``.
            map_xy: optional ``[B, M, 2]``.
            query_velocity: optional diagnostic velocity ``[B,N,2]`` in
                current-frame meters per future target step. Used only when
                ``ModelConfig.use_query_velocity=True``.

        Returns:
            Prediction dict documented in the class docstring.
        """

        lidar_features = self._as_multiscale(lidar_features)
        b, n, _ = initial_poses.shape
        f, t, d = self.cfg.num_modes, self.cfg.num_times, self.cfg.hidden_dim

        # Paper query init: mode params + time params, identical across objects.
        queries = (self.mode_embed + self.time_embed).expand(b, n, f, t, d).contiguous()
        if self.cfg.forecast_base_mode == "constant_velocity" and query_velocity is None:
            query_velocity = initial_poses.new_zeros((b, n, 2))
        if self.velocity_embed is not None:
            if query_velocity is None:
                query_velocity = initial_poses.new_zeros((b, n, 2))
            velocity_context = self.velocity_embed(query_velocity).view(b, n, 1, 1, d)
            queries = queries + velocity_context

        # Static pose initialization: same pose over modes and time (paper).
        poses = initial_poses[:, :, None, None, :].expand(b, n, f, t, 3).contiguous()
        box = initial_boxes.clone() if initial_boxes is not None else self._default_box(initial_poses)

        aux_outputs = []
        for block in self.blocks:
            block_poses = poses.detach() if self.cfg.detach_pose_refinement else poses
            block_box = box.detach() if self.cfg.detach_pose_refinement else box
            queries = block(
                queries,
                block_poses,
                lidar_features,
                map_tokens=map_tokens,
                map_xy=map_xy,
                map_mask=map_mask,
            )
            rollout_velocity = (
                query_velocity if self.cfg.forecast_base_mode == "constant_velocity" else None
            )
            poses, box, pred = self.pose_update(
                queries,
                block_poses,
                block_box,
                rollout_velocity=rollout_velocity,
                use_learned_velocity=self.cfg.forecast_base_mode == "learned_velocity",
                learned_velocity_frame=getattr(self.cfg, "forecast_velocity_frame", "ego"),
            )
            aux_outputs.append(pred)

        pred = dict(aux_outputs[-1])
        pred["aux_outputs"] = aux_outputs
        return pred

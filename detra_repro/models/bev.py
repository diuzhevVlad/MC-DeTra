import torch
import torch.nn.functional as F
from torch import nn

from detra_repro.config import DataConfig


def _make_norm(channels: int, norm_type: str = "batch") -> nn.Module:
    """Build a 2D normalization layer for BEV experiments."""

    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type == "group":
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type={norm_type!r}; use batch, group, or instance")


class SimpleBEVEncoder(nn.Module):
    """Small BEV CNN encoder used before DeTra refinement.

    Input:
        ``bev``: float tensor ``[B, C_in, H, W]``.

        Minimal channel convention:
        - channel 0: occupancy or point count
        - channel 1: max height
        - channel 2: mean height
        - channel 3: optional temporal statistic

    Output:
        ``tokens``: float tensor ``[B, H_out * W_out, D]``.
        ``token_xy``: float tensor ``[H_out * W_out, 2]`` if supplied by caller
            later, or ``None`` in this minimal class.
        ``feature_map``: float tensor ``[B, D, H_out, W_out]``.

    Paper mapping:
        The full DeTra lidar encoder uses PointNet voxel features and a stronger
        multi-resolution backbone. This class is intentionally replaceable: keep
        the output token shape ``[B, L, D]`` and downstream refinement code does
        not need to change.
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode BEV raster into tokens and a feature map.

        Args:
            bev: ``[B, C_in, H, W]`` float tensor.

        Returns:
            ``tokens``: ``[B, L, D]`` where ``L = H_out * W_out``.
            ``feature_map``: ``[B, D, H_out, W_out]``.
        """

        feature_map = self.net(bev)
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens, feature_map


class SqueezeExcite(nn.Module):
    """Squeeze-excitation block used inside the paper-like residual backbone.

    Input and output:
        ``x``: ``[B, C, H, W]``.

    Paper mapping:
        DeTra's supplementary uses residual blocks with dynamic convolution,
        batch norm, ReLU, squeeze-excitation, and dropout. This scaffold keeps
        the squeeze-excitation part and uses standard convolutions so the code
        stays easy to read and extend.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        inner = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, inner)
        self.fc2 = nn.Linear(inner, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = x.mean(dim=(2, 3))
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale)).view(b, c, 1, 1)
        return x * scale


class DynamicConv2d(nn.Module):
    """Input-adaptive mixture of convolution kernels.

    Input:
        ``x``: ``[B, C_in, H, W]``.

    Output:
        ``y``: ``[B, C_out, H_out, W_out]``.

    DeTra mapping:
        The supplementary describes residual blocks using dynamic convolution.
        This implementation keeps ``K`` expert kernels and predicts per-sample
        mixture weights from global average pooled input features. It is a
        readable approximation of the dynamic-conv building block used in the
        paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        experts: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.experts = experts
        self.weight = nn.Parameter(
            torch.empty(experts, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(experts, out_channels))
        self.router = nn.Sequential(
            nn.Linear(in_channels, max(8, in_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, in_channels // 4), experts),
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        routing = self.router(x.mean(dim=(2, 3))).softmax(dim=-1)
        weight = torch.einsum("be,eocij->bocij", routing, self.weight)
        bias = torch.einsum("be,eo->bo", routing, self.bias)

        y = F.conv2d(
            x.reshape(1, b * c, h, w),
            weight.reshape(b * self.out_channels, c, self.kernel_size, self.kernel_size),
            bias=bias.reshape(b * self.out_channels),
            stride=self.stride,
            padding=self.padding,
            groups=b,
        )
        return y.reshape(b, self.out_channels, y.shape[-2], y.shape[-1])


class ResBlock(nn.Module):
    """Residual BEV block with optional downsampling.

    Input:
        ``x``: ``[B, C, H, W]``.

    Output:
        ``y``: ``[B, C, H / stride, W / stride]``.

    Paper mapping:
        The block uses dynamic convolution, batch norm, ReLU, dynamic
        convolution, batch norm, squeeze-excitation, and dropout, matching the
        supplementary description at scaffold scale.
    """

    def __init__(
        self,
        channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        dynamic_conv_experts: int = 4,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.conv1 = DynamicConv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            experts=dynamic_conv_experts,
        )
        self.bn1 = _make_norm(channels, norm_type)
        self.conv2 = DynamicConv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            experts=dynamic_conv_experts,
        )
        self.bn2 = _make_norm(channels, norm_type)
        self.se = SqueezeExcite(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj = (
            nn.Conv2d(channels, channels, kernel_size=1, stride=stride)
            if stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y = self.dropout(self.se(y))
        return F.relu(y + residual, inplace=True)


class PaperLikeLidarEncoder(nn.Module):
    """PointNet scatter + multi-scale BEV backbone closer to DeTra.

    Input:
        ``points``: float tensor ``[B, P, C_point]``.

        Default point features are ``x, y, z, dt`` with ``C_point=4``. Intensity
        is intentionally not used.

        ``point_mask``: optional bool tensor ``[B, P]``. ``True`` marks valid
        points when batches are padded. If omitted, all points are valid.

    Output dict:
        - ``tokens``: fused high-resolution BEV tokens ``[B, L4, D]``.
        - ``feature_map``: fused BEV map ``[B, D, H/4, W/4]``.
        - ``multi_scale_features``:
          ``{"stride4": [B,D,H/4,W/4], "stride8": [B,D,H/8,W/8],
          "stride16": [B,D,H/16,W/16]}``.
        - ``token_xy``: center coordinates for fused tokens ``[L4, 2]`` in
          meters in the local/current frame.

    Paper mapping:
        1. Point MLP encodes each point feature.
        2. Encoded points are scatter-added into a BEV grid.
        3. A dynamic-conv residual backbone produces multi-resolution maps.
        4. The refinement transformer consumes ``multi_scale_features`` for
           multi-scale deformable lidar attention. ``feature_map``/``tokens``
           are still returned for proposal decoding and simple debugging.

    Practical note:
        Full Waymo at 0.1 m over [-80,80] creates a 1600x1600 grid, which is
        expensive. The API supports it, but start with 0.4 m for iteration.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        point_feature_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        dynamic_conv_experts: int = 4,
        norm_type: str = "batch",
        history_occupancy_channels: int = 0,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.hidden_dim = hidden_dim
        self.history_occupancy_channels = history_occupancy_channels
        self.point_mlp = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.history_occupancy_net = (
            nn.Sequential(
                nn.Conv2d(history_occupancy_channels, hidden_dim, kernel_size=3, padding=1),
                _make_norm(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                _make_norm(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
            )
            if history_occupancy_channels > 0
            else None
        )
        self.stem = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            _make_norm(hidden_dim, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            _make_norm(hidden_dim, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            _make_norm(hidden_dim, norm_type),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for block_idx in range(10):
            stride = 2 if block_idx in {0, 2, 4} else 1
            blocks.append(
                ResBlock(
                    hidden_dim,
                    stride=stride,
                    dropout=dropout,
                    dynamic_conv_experts=dynamic_conv_experts,
                    norm_type=norm_type,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.fuse_stride8 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.fuse_stride16 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.motion_aux_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            _make_norm(hidden_dim, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, history_occupancy_channels * 3, kernel_size=1),
        ) if history_occupancy_channels > 0 else None
        self.motion_context_proj = (
            nn.Conv2d(history_occupancy_channels * 3, hidden_dim, kernel_size=1)
            if history_occupancy_channels > 0
            else None
        )
        if self.motion_context_proj is not None:
            nn.init.zeros_(self.motion_context_proj.weight)
            nn.init.zeros_(self.motion_context_proj.bias)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return raw BEV grid shape ``(H, W)`` before CNN downsampling."""

        x_min, y_min, x_max, y_max = self.data_cfg.roi_m
        width = int(round((x_max - x_min) / self.data_cfg.voxel_size_m))
        height = int(round((y_max - y_min) / self.data_cfg.voxel_size_m))
        return height, width

    def _scatter_points(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Scatter point features into a BEV grid with sum aggregation.

        Args:
            points: ``[B, P, C_point]``. First two channels must be ``x,y`` in
                current-frame meters.
            point_mask: optional ``[B, P]`` bool.

        Returns:
            BEV feature grid ``[B, D, H, W]``.
        """

        b, p, _ = points.shape
        h, w = self.grid_shape
        x_min, y_min, x_max, y_max = self.data_cfg.roi_m
        voxel = self.data_cfg.voxel_size_m

        xy = points[..., :2]
        ix = torch.floor((xy[..., 0] - x_min) / voxel).long()
        iy = torch.floor((xy[..., 1] - y_min) / voxel).long()
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        if point_mask is not None:
            valid = valid & point_mask

        batch_ids = torch.arange(b, device=points.device)[:, None].expand(b, p)
        flat_index = batch_ids * (h * w) + iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)

        # Run the per-point MLP only on valid (non-padding) points. The MLP is
        # purely per-point (Linear/ReLU, no cross-point ops), so gathering before
        # it is numerically identical to masking after it, but avoids the MLP
        # FLOPs and the dense [B,P,D] intermediate for padded points.
        valid_points = points[valid]
        valid_features = self.point_mlp(valid_points)

        flat_grid = points.new_zeros((b * h * w, self.hidden_dim))
        flat_grid.index_add_(0, flat_index[valid], valid_features)
        return flat_grid.view(b, h, w, self.hidden_dim).permute(0, 3, 1, 2).contiguous()

    def _token_xy(self, height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
        """Return BEV token centers with shape ``[height * width, 2]``."""

        x_min, y_min, _, _ = self.data_cfg.roi_m
        step = self.data_cfg.voxel_size_m * stride
        ys = y_min + (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * step
        xs = x_min + (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * step
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        history_occupancy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Encode raw points into paper-like multi-scale BEV tokens.

        Args:
            points: ``[B, P, C_point]`` float tensor.
            point_mask: optional ``[B, P]`` bool tensor.
            history_occupancy: optional GT-history oracle BEV occupancy
                ``[B,H_hist,H,W]``. Used only for diagnostics.

        Returns:
            Output dict documented in the class docstring.
        """

        x = self._scatter_points(points, point_mask)
        if self.history_occupancy_net is not None:
            if history_occupancy is None:
                b, _, h, w = x.shape
                history_occupancy = x.new_zeros((b, self.history_occupancy_channels, h, w))
            x = x + self.history_occupancy_net(history_occupancy.to(dtype=x.dtype))
        x = self.stem(x)

        stride4 = stride8 = stride16 = None
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == 1:
                stride4 = x
            elif idx == 3:
                stride8 = x
            elif idx == 9:
                stride16 = x

        assert stride4 is not None and stride8 is not None and stride16 is not None
        stride8_up = F.interpolate(
            self.fuse_stride8(stride8),
            size=stride4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        stride16_up = F.interpolate(
            self.fuse_stride16(stride16),
            size=stride4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = stride4 + stride8_up + stride16_up
        motion = None
        if self.motion_aux_head is not None and self.motion_context_proj is not None:
            motion_flat = self.motion_aux_head(fused)
            fused = fused + self.motion_context_proj(motion_flat)
            b, _, h4, w4 = fused.shape
            motion = motion_flat.view(
                b,
                self.history_occupancy_channels,
                3,
                h4,
                w4,
            )
        tokens = fused.flatten(2).transpose(1, 2).contiguous()
        token_xy = self._token_xy(fused.shape[-2], fused.shape[-1], stride=4, device=fused.device)
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "tokens": tokens,
            "feature_map": fused,
            "multi_scale_features": {
                "stride4": stride4,
                "stride8": stride8,
                "stride16": stride16,
            },
            "token_xy": token_xy,
        }
        if motion is not None:
            out["motion_aux"] = {
                "occupancy_logits": motion[:, :, 0],
                "flow": motion[:, :, 1:3],
            }
        return out

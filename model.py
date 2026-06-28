from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.ops import FeaturePyramidNetwork


@dataclass
class TableLineModelConfig:
    """Configuration for the table line detector."""

    pretrained_backbone: bool = True
    fpn_channels: int = 256
    hidden_dim: int = 256
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    feedforward_dim: int = 1024
    dropout: float = 0.1

    num_horizontal_queries: int = 100
    num_vertical_queries: int = 100
    num_control_points: int = 16

    # Each FPN level is pooled to this spatial size before transformer decoding.
    # This bounds memory consumption for high-resolution document images.
    decoder_token_grid: Tuple[int, int] = (16, 16)

    # Output channels: endpoint, L-junction, T-junction, X-junction.
    num_junction_classes: int = 4


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResNet34Backbone(nn.Module):
    """ResNet-34 feature extractor returning C2-C5 feature maps."""

    out_channels: Dict[str, int] = {
        "c2": 64,
        "c3": 128,
        "c4": 256,
        "c5": 512,
    }

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        net = resnet34(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class TableFPN(nn.Module):
    """Torchvision FPN producing P2-P5 with equal channel count."""

    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        super().__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=out_channels,
        )

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pyramids = self.fpn(features)
        return {
            "p2": pyramids["c2"],
            "p3": pyramids["c3"],
            "p4": pyramids["c4"],
            "p5": pyramids["c5"],
        }


class PositionEmbeddingSine2D(nn.Module):
    """Two-dimensional sine/cosine positional embedding."""

    def __init__(self, hidden_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4")
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        device = x.device
        dtype = x.dtype

        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x_coord = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x_coord, indexing="ij")

        num_freq = self.hidden_dim // 4
        dim_t = torch.arange(num_freq, device=device, dtype=dtype)
        dim_t = self.temperature ** (dim_t / max(num_freq - 1, 1))

        pos_x = xx[..., None] / dim_t
        pos_y = yy[..., None] / dim_t
        pos_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)
        pos_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
        pos = torch.cat([pos_y, pos_x], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        return pos.expand(batch_size, -1, -1, -1)


class MultiScaleTokenEncoder(nn.Module):
    """Converts P2-P5 maps to a bounded multi-scale transformer memory."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        token_grid: Tuple[int, int],
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        self.token_grid = token_grid
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, hidden_dim, kernel_size=1) for _ in range(num_levels)]
        )
        self.level_embeddings = nn.Parameter(torch.randn(num_levels, hidden_dim) * 0.02)
        self.position_embedding = PositionEmbeddingSine2D(hidden_dim)

    def forward(self, features: List[Tensor]) -> Tensor:
        memory_tokens: List[Tensor] = []

        for level, (feature, projection) in enumerate(zip(features, self.projections)):
            pooled = F.adaptive_avg_pool2d(feature, self.token_grid)
            projected = projection(pooled)
            projected = projected + self.position_embedding(projected)
            projected = projected + self.level_embeddings[level][None, :, None, None]

            # Transformer expects sequence-first memory: [S, B, C].
            tokens = projected.flatten(2).permute(2, 0, 1)
            memory_tokens.append(tokens)

        return torch.cat(memory_tokens, dim=0)


class GeometryHead(nn.Module):
    """Predicts global deskew and perspective geometry.

    Outputs:
        angle_vector: normalized [sin(theta), cos(theta)]
        corners: four normalized table corners in image coordinates
        homography_delta: residual 3x3 projective transform around identity
    """

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvNormAct(in_channels, hidden_dim, 3),
            ConvNormAct(hidden_dim, hidden_dim, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.angle_head = nn.Linear(hidden_dim, 2)
        self.corner_head = nn.Linear(hidden_dim, 8)
        self.homography_head = nn.Linear(hidden_dim, 8)

        nn.init.zeros_(self.homography_head.weight)
        nn.init.zeros_(self.homography_head.bias)

    def forward(self, p5: Tensor) -> Dict[str, Tensor]:
        feature = self.encoder(p5)

        angle = self.angle_head(feature)
        angle = F.normalize(angle, dim=-1, eps=1e-6)

        corners = self.corner_head(feature).sigmoid().view(-1, 4, 2)

        residual = self.homography_head(feature)
        batch_size = residual.shape[0]
        homography = torch.zeros(
            batch_size, 3, 3, device=residual.device, dtype=residual.dtype
        )
        homography[:, 0, 0] = 1.0 + residual[:, 0]
        homography[:, 0, 1] = residual[:, 1]
        homography[:, 0, 2] = residual[:, 2]
        homography[:, 1, 0] = residual[:, 3]
        homography[:, 1, 1] = 1.0 + residual[:, 4]
        homography[:, 1, 2] = residual[:, 5]
        homography[:, 2, 0] = residual[:, 6]
        homography[:, 2, 1] = residual[:, 7]
        homography[:, 2, 2] = 1.0

        return {
            "angle_vector": angle,
            "corners": corners,
            "homography": homography,
        }


class LinePredictionHead(nn.Module):
    """Maps decoded query embeddings to line attributes."""

    def __init__(self, hidden_dim: int, num_control_points: int) -> None:
        super().__init__()
        self.num_control_points = num_control_points

        self.class_head = nn.Linear(hidden_dim, 2)  # no-line / line
        self.points_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_control_points * 2),
        )
        self.visibility_head = nn.Linear(hidden_dim, num_control_points)
        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_control_points),
        )

    def forward(self, query_features: Tensor) -> Dict[str, Tensor]:
        # query_features: [B, N, C]
        batch_size, num_queries, _ = query_features.shape

        class_logits = self.class_head(query_features)
        control_points = self.points_head(query_features).sigmoid()
        control_points = control_points.view(
            batch_size, num_queries, self.num_control_points, 2
        )
        visibility_logits = self.visibility_head(query_features)
        widths = F.softplus(self.width_head(query_features))

        return {
            "class_logits": class_logits,
            "control_points": control_points,
            "visibility_logits": visibility_logits,
            "widths": widths,
        }


class LineDecoder(nn.Module):
    """DETR-style line query decoder.

    It predicts ordered polylines rather than bounding boxes, so crossing horizontal
    and vertical lines do not suppress one another.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_control_points: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        orientation: Literal["horizontal", "vertical"],
    ) -> None:
        super().__init__()
        self.orientation = orientation
        self.query_embedding = nn.Embedding(num_queries, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.prediction_head = LinePredictionHead(hidden_dim, num_control_points)

        axis = torch.linspace(0.0, 1.0, num_control_points)
        self.register_buffer("canonical_axis", axis, persistent=False)

    def forward(self, memory: Tensor) -> Dict[str, Tensor]:
        # memory: [S, B, C]
        batch_size = memory.shape[1]
        queries = self.query_embedding.weight[:, None, :].expand(-1, batch_size, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        decoded = decoded.permute(1, 0, 2)

        output = self.prediction_head(decoded)
        points = output["control_points"]

        # The logical orientation is defined in rectified table coordinates.
        # One coordinate is initialized to an ordered canonical axis; the network
        # still predicts both coordinates, allowing perspective and curvature.
        axis = self.canonical_axis[None, None, :]
        if self.orientation == "horizontal":
            points_x = 0.5 * points[..., 0] + 0.5 * axis
            points_y = points[..., 1]
        else:
            points_x = points[..., 0]
            points_y = 0.5 * points[..., 1] + 0.5 * axis

        output["control_points"] = torch.stack([points_x, points_y], dim=-1)
        output["query_features"] = decoded
        return output


class PixelPredictionHeads(nn.Module):
    """High-resolution segmentation and junction prediction heads."""

    def __init__(self, channels: int, num_junction_classes: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            ConvNormAct(channels, channels, 3),
            ConvNormAct(channels, channels // 2, 3),
        )
        reduced = channels // 2

        self.line_mask_head = nn.Sequential(
            ConvNormAct(reduced, reduced, 3),
            nn.Conv2d(reduced, 2, kernel_size=1),
        )
        self.junction_head = nn.Sequential(
            ConvNormAct(reduced, reduced, 3),
            nn.Conv2d(reduced, num_junction_classes, kernel_size=1),
        )
        self.distance_head = nn.Sequential(
            ConvNormAct(reduced, reduced, 3),
            nn.Conv2d(reduced, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, p2: Tensor, output_size: Tuple[int, int]) -> Dict[str, Tensor]:
        feature = self.shared(p2)
        line_masks = self.line_mask_head(feature)
        junctions = self.junction_head(feature)
        distances = self.distance_head(feature)

        line_masks = F.interpolate(
            line_masks, size=output_size, mode="bilinear", align_corners=False
        )
        junctions = F.interpolate(
            junctions, size=output_size, mode="bilinear", align_corners=False
        )
        distances = F.interpolate(
            distances, size=output_size, mode="bilinear", align_corners=False
        )

        return {
            "line_mask_logits": line_masks,
            "junction_logits": junctions,
            "distance_maps": distances,
        }


class TableLineDetector(nn.Module):
    """Table separator detector for skewed and perspective-distorted tables.

    Input:
        images: float tensor [B, 3, H, W]. Values may be in [0, 1] or normalized
        externally according to the selected backbone weights.

    Output dictionary:
        horizontal: DETR-style horizontal line predictions
        vertical: DETR-style vertical line predictions
        geometry: deskew/perspective predictions
        line_mask_logits: [B, 2, H, W]
        junction_logits: [B, J, H, W]
        distance_maps: [B, 2, H, W]
        pyramid: P2-P5 feature maps, useful for auxiliary losses/debugging
    """

    def __init__(self, config: Optional[TableLineModelConfig] = None) -> None:
        super().__init__()
        self.config = config or TableLineModelConfig()

        self.backbone = ResNet34Backbone(
            pretrained=self.config.pretrained_backbone
        )
        self.fpn = TableFPN(
            in_channels=list(self.backbone.out_channels.values()),
            out_channels=self.config.fpn_channels,
        )
        self.token_encoder = MultiScaleTokenEncoder(
            channels=self.config.fpn_channels,
            hidden_dim=self.config.hidden_dim,
            token_grid=self.config.decoder_token_grid,
        )

        self.geometry_head = GeometryHead(
            in_channels=self.config.fpn_channels,
            hidden_dim=self.config.hidden_dim,
        )

        decoder_kwargs = dict(
            hidden_dim=self.config.hidden_dim,
            num_control_points=self.config.num_control_points,
            num_layers=self.config.num_decoder_layers,
            num_heads=self.config.num_attention_heads,
            feedforward_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
        )

        self.horizontal_decoder = LineDecoder(
            num_queries=self.config.num_horizontal_queries,
            orientation="horizontal",
            **decoder_kwargs,
        )
        self.vertical_decoder = LineDecoder(
            num_queries=self.config.num_vertical_queries,
            orientation="vertical",
            **decoder_kwargs,
        )

        self.pixel_heads = PixelPredictionHeads(
            channels=self.config.fpn_channels,
            num_junction_classes=self.config.num_junction_classes,
        )

    def forward(self, images: Tensor) -> Dict[str, object]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")

        output_size = (images.shape[-2], images.shape[-1])

        backbone_features = self.backbone(images)
        pyramid = self.fpn(backbone_features)
        pyramid_list = [pyramid["p2"], pyramid["p3"], pyramid["p4"], pyramid["p5"]]

        memory = self.token_encoder(pyramid_list)
        horizontal = self.horizontal_decoder(memory)
        vertical = self.vertical_decoder(memory)
        geometry = self.geometry_head(pyramid["p5"])
        pixel_outputs = self.pixel_heads(pyramid["p2"], output_size)

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "geometry": geometry,
            **pixel_outputs,
            "pyramid": pyramid,
        }


if __name__ == "__main__":
    config = TableLineModelConfig(pretrained_backbone=False)
    model = TableLineDetector(config)
    model.eval()

    images = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        outputs = model(images)

    print("Horizontal logits:", outputs["horizontal"]["class_logits"].shape)
    print("Horizontal points:", outputs["horizontal"]["control_points"].shape)
    print("Vertical logits:", outputs["vertical"]["class_logits"].shape)
    print("Vertical points:", outputs["vertical"]["control_points"].shape)
    print("Line masks:", outputs["line_mask_logits"].shape)
    print("Junctions:", outputs["junction_logits"].shape)
    print("Corners:", outputs["geometry"]["corners"].shape)

import torch
import torch.nn as nn

from data_challenge.models.cnn_baseline import build_feature_backbone


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GatedFeatureHead(nn.Module):
    """Lightweight channel-attention head for flat backbone features."""

    def __init__(self, in_features: int, hidden_dim: int, dropout: float, out_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_features),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x * self.gate(x))


def build_prediction_head(
    in_features: int,
    out_dim: int,
    head_type: str = "mlp",
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
) -> nn.Module:
    hidden_dims = hidden_dims or [256]

    if head_type == "linear":
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, out_dim))

    if head_type == "mlp":
        layers: list[nn.Module] = [nn.Dropout(dropout)]
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    if head_type == "residual_mlp":
        hidden_dim = hidden_dims[0]
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    if head_type == "gated":
        return GatedFeatureHead(
            in_features=in_features,
            hidden_dim=hidden_dims[0],
            dropout=dropout,
            out_dim=out_dim,
        )

    raise ValueError("Unknown head_type '%s'. Choose from: linear, mlp, residual_mlp, gated" % head_type)


class MultiTaskOcclusionModel(nn.Module):
    """Shared image backbone with occlusion regression and gender classification heads."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        regression_head: dict | None = None,
        gender_head: dict | None = None,
    ):
        super().__init__()
        self.backbone, in_features = build_feature_backbone(backbone, pretrained=pretrained)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        regression_head = regression_head or {}
        gender_head = gender_head or {}
        self.occlusion_head = build_prediction_head(
            in_features=in_features,
            out_dim=1,
            head_type=regression_head.get("type", "mlp"),
            hidden_dims=regression_head.get("hidden_dims", [256]),
            dropout=regression_head.get("dropout", 0.3),
        )
        self.gender_head = build_prediction_head(
            in_features=in_features,
            out_dim=1,
            head_type=gender_head.get("type", "mlp"),
            hidden_dims=gender_head.get("hidden_dims", [128]),
            dropout=gender_head.get("dropout", 0.2),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        return {
            "occlusion": torch.sigmoid(self.occlusion_head(features)).squeeze(1),
            "gender_logits": self.gender_head(features).squeeze(1),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

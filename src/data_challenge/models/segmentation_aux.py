import torch
import torch.nn as nn

from data_challenge.models.cnn_baseline import build_feature_backbone


class SegmentationAuxModel(nn.Module):
    """Predict face occlusion from RGB image features plus a generated mask branch."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        mask_feature_dim: int = 64,
        hidden_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone, image_feature_dim = build_feature_backbone(backbone, pretrained=pretrained)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, mask_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mask_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_feature_dim + mask_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, masks: torch.Tensor):
        if self.freeze_backbone:
            with torch.no_grad():
                image_features = self.backbone(images)
        else:
            image_features = self.backbone(images)
        mask_features = self.mask_encoder(masks)
        features = torch.cat([image_features, mask_features], dim=1)
        return self.head(features).squeeze(1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

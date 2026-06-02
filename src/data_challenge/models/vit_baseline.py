import torch.nn as nn
import timm


_TIMM_BACKBONES = {
    "efficientvit_b3": "efficientvit_b3.r288_in1k",
}


def _setup_backbone(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone not in _TIMM_BACKBONES:
        raise ValueError(f"Unknown ViT backbone '{backbone}'. Choose from: {list(_TIMM_BACKBONES)}")

    model_id = _TIMM_BACKBONES[backbone] if pretrained else backbone
    model = timm.create_model(model_id, pretrained=pretrained, num_classes=0)
    in_features = model.head.classifier[0].out_features
    return model, in_features


class ViTBaseline(nn.Module):
    """Pretrained ViT backbone (timm) with a regression head for face occlusion prediction."""

    def __init__(self, backbone: str = "efficientvit_b3", pretrained: bool = True, dropout: float = 0.35):
        super().__init__()
        base, in_features = _setup_backbone(backbone, pretrained)

        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)

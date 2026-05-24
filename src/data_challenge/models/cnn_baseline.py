import torch.nn as nn
from torchvision import models
import timm

# --- torchvision setup functions ---

def _setup_fc(m):
    in_features = m.fc.in_features
    m.fc = nn.Identity()
    return in_features


def _setup_efficientnet(m):
    in_features = m.classifier[1].in_features
    m.classifier = nn.Identity()
    return in_features


def _setup_convnext(m):
    in_features = m.classifier[2].in_features
    m.classifier[2] = nn.Identity()  # keep LayerNorm + Flatten, drop only the Linear
    return in_features


def _setup_vit(m):
    in_features = m.heads.head.in_features
    m.heads.head = nn.Identity()
    return in_features


def _setup_swin(m):
    in_features = m.head.in_features
    m.head = nn.Identity()
    return in_features


# (factory, weights, setup_fn)
_TORCHVISION_REGISTRY = {
    "resnet18":        (models.resnet18,        models.ResNet18_Weights.DEFAULT,        _setup_fc),
    "resnet50":        (models.resnet50,         models.ResNet50_Weights.DEFAULT,        _setup_fc),
    "efficientnet_b0": (models.efficientnet_b0,  models.EfficientNet_B0_Weights.DEFAULT, _setup_efficientnet),
    "efficientnet_b3": (models.efficientnet_b3,  models.EfficientNet_B3_Weights.DEFAULT, _setup_efficientnet),
    "convnext_tiny":   (models.convnext_tiny,    models.ConvNeXt_Tiny_Weights.DEFAULT,   _setup_convnext),
    "convnext_small":  (models.convnext_small,   models.ConvNeXt_Small_Weights.DEFAULT,  _setup_convnext),
    "vit_b_16":        (models.vit_b_16,         models.ViT_B_16_Weights.DEFAULT,        _setup_vit),
    "swin_t":          (models.swin_t,            models.Swin_T_Weights.DEFAULT,          _setup_swin),
    "swin_s":          (models.swin_s,            models.Swin_S_Weights.DEFAULT,          _setup_swin),
}

# timm model names - head is removed via num_classes=0, in_features via num_features
_TIMM_REGISTRY = {
    "twins_svt_small",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
}


class CNNBaseline(nn.Module):
    """Pretrained backbone with a regression head for face occlusion prediction.

    Supports both torchvision and timm backbones.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, dropout: float = 0.3, img_size: int = 224):
        super().__init__()
        all_backbones = list(_TORCHVISION_REGISTRY) + list(_TIMM_REGISTRY)
        if backbone not in all_backbones:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {all_backbones}")

        if backbone in _TIMM_REGISTRY:
            base = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=img_size)
            in_features = base.num_features
        else:
            factory, weights_enum, setup_fn = _TORCHVISION_REGISTRY[backbone]
            weights = weights_enum if pretrained else None
            base = factory(weights=weights)
            in_features = setup_fn(base)

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

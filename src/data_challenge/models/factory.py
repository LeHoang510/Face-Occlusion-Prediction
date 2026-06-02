import torch.nn as nn

from data_challenge.models.cnn_baseline import CNNBaseline
from data_challenge.models.vit_baseline import ViTBaseline


def build_model(model_cfg: dict) -> nn.Module:
    """Build a model from config. model_cfg must include 'type' (cnn | vit)."""
    model_type = model_cfg.get("type", "cnn")
    backbone = model_cfg["backbone"]
    pretrained = model_cfg["pretrained"]
    dropout = model_cfg.get("dropout", 0.3)

    if model_type == "cnn":
        return CNNBaseline(backbone=backbone, pretrained=pretrained, dropout=dropout)
    if model_type == "vit":
        return ViTBaseline(backbone=backbone, pretrained=pretrained, dropout=dropout)

    raise ValueError(f"Unknown model type '{model_type}'. Choose from: cnn, vit")

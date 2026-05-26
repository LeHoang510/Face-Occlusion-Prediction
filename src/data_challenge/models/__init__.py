"""Model factory.

Routes config -> nn.Module. Keys read from `cfg["model"]`:
    name: "cnn_baseline" | "dinov3"

Backward-compatible: configs without `model.name` default to "cnn_baseline".
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "cnn_baseline")

    if name == "cnn_baseline":
        from .cnn_baseline import CNNBaseline

        return CNNBaseline(
            backbone=model_cfg["backbone"],
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.3),
            img_size=cfg.get("data", {}).get("img_size", 224),
        )

    if name == "dinov3":
        from .dinov3 import DinoV3Regressor

        return DinoV3Regressor(
            model_id=model_cfg.get("model_id", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            lora=model_cfg.get("lora"),
            head_dropout=model_cfg.get("head_dropout", 0.1),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
        )

    raise ValueError(f"Unknown model name: {name!r}")

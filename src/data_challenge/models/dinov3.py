"""DINOv3 (or DINOv2) backbone with optional LoRA + MLP regression head.

Default config corresponds to:
    DINOv3-L (ViT-L/16) frozen, LoRA r=16 on Q/V projections,
    LayerNorm + MLP(512) + Sigmoid head producing FaceOcclusion in [0, 1].

Requires `transformers` and (optionally) `peft`. Both are declared in pyproject.
The model id can be pointed to any HuggingFace ViT-style model exposing
`last_hidden_state` with a CLS token at index 0 (DINOv2 / DINOv3 / Sapiens ViT).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _build_backbone(model_id: str, freeze: bool, trust_remote_code: bool) -> nn.Module:
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
    return backbone


def _apply_lora(backbone: nn.Module, lora_cfg: dict[str, Any]) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        bias=lora_cfg.get("bias", "none"),
    )
    return get_peft_model(backbone, cfg)


def _embed_dim(backbone: nn.Module) -> int:
    """Best-effort retrieval of the embedding dim from a HF or PEFT-wrapped model."""
    if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_size"):
        return int(backbone.config.hidden_size)
    if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "config"):
        return int(backbone.base_model.config.hidden_size)
    raise ValueError("Cannot infer hidden_size from backbone; pass it explicitly.")


class DinoV3Regressor(nn.Module):
    """DINOv3/v2 backbone + optional LoRA + regression head for face occlusion.

    Args:
        model_id: HuggingFace model id. Default: facebook/dinov3-vitl16-pretrain-lvd1689m
        freeze_backbone: if True, freezes all backbone parameters before (optional) LoRA
            adapters are injected. Recommended True when using LoRA.
        lora: dict {enabled, r, alpha, dropout, target_modules, bias}. If None or
            enabled=False, no LoRA is applied (pure linear probing if freeze_backbone=True).
        head_dropout: dropout in the MLP head.
        hidden_dim: hidden width of the MLP head. None = single Linear from embed_dim to 1.
        trust_remote_code: forwarded to AutoModel.from_pretrained (needed for some repos).
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        lora: dict[str, Any] | None = None,
        head_dropout: float = 0.1,
        hidden_dim: int | None = 512,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        backbone = _build_backbone(model_id, freeze=freeze_backbone, trust_remote_code=trust_remote_code)
        if lora and lora.get("enabled", False):
            backbone = _apply_lora(backbone, lora)
        self.backbone = backbone

        embed = _embed_dim(self.backbone)

        layers: list[nn.Module] = [nn.LayerNorm(embed), nn.Dropout(head_dropout)]
        if hidden_dim:
            layers += [nn.Linear(embed, hidden_dim), nn.GELU(), nn.Dropout(head_dropout)]
            in_feat = hidden_dim
        else:
            in_feat = embed
        layers += [nn.Linear(in_feat, 1), nn.Sigmoid()]
        self.head = nn.Sequential(*layers)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(1)

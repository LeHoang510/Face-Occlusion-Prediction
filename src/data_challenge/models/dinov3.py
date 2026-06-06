"""DINOv3 (or DINOv2) backbone with optional LoRA + MLP regression head.

Default config corresponds to:
    DINOv3-L (ViT-L/16) frozen, LoRA r=16 on q_proj/v_proj,
    LayerNorm + MLP(512) + Sigmoid head producing FaceOcclusion in [0, 1].

Requires `transformers` and (optionally) `peft`. Both are declared in pyproject.
The model id can be pointed to any HuggingFace ViT-style model exposing
`last_hidden_state` with a CLS token at index 0 (DINOv2 / DINOv3 / Sapiens ViT).
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn

from data_challenge.utils import torch_compat  # noqa: F401 — patch FP8 dtypes before HF import


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
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
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

    # ---- utilities used by the trainer for full fine-tuning ----

    def enable_gradient_checkpointing(self) -> None:
        """Turn on HF gradient checkpointing on the underlying backbone.

        Walks through any PEFT wrapper to reach the actual HF model.
        No-op if the backbone does not expose the API.
        """
        target = self.backbone
        # Drill: PEFT wraps as PeftModel.base_model.model -> the original HF backbone
        for attr in ("base_model", "model"):
            if hasattr(target, attr):
                inner = getattr(target, attr)
                # Only descend if inner has the API or further wrappers
                if hasattr(inner, "gradient_checkpointing_enable") or hasattr(inner, "base_model"):
                    target = inner
        if hasattr(target, "gradient_checkpointing_enable"):
            try:
                target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                # Older transformers signatures
                target.gradient_checkpointing_enable()


# ---------------------------------------------------------------------------
# Parameter-group builder for full fine-tuning
# ---------------------------------------------------------------------------

# Match the block index in a parameter name across the various HF / timm
# transformer naming conventions.
_LAYER_PATTERNS = [
    re.compile(r"encoder\.layer\.(\d+)"),
    re.compile(r"encoder\.layers\.(\d+)"),
    re.compile(r"\bblocks\.(\d+)"),
    re.compile(r"(?:^|\.)layer\.(\d+)"),
]


def _layer_index(name: str, n_layers: int) -> int:
    """Return the transformer block index inferred from a parameter name.

    Convention:
        -1               : pre-transformer (patch / positional embeddings, cls token)
        0..n_layers-1    : transformer block index
        n_layers         : post-transformer (final norm, anything else)
    """
    for pat in _LAYER_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    if any(s in name for s in ("embed", "patch_embed", "pos_embed", "cls_token")):
        return -1
    return n_layers


def build_param_groups(
    model: "DinoV3Regressor",
    *,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    llrd_decay: float = 1.0,
    n_layers: int = 24,
    no_decay_keywords: tuple[str, ...] = ("bias", "LayerNorm", "layernorm", "ln_", "norm.weight"),
) -> list[dict[str, Any]]:
    """AdamW param groups for full fine-tuning of a ViT-style backbone.

    Splits parameters into:
      - head (lr=lr_head)
      - backbone per-layer (lr = lr_backbone * llrd_decay**(n_layers - layer_idx))

    Bias and LayerNorm parameters get weight_decay=0 (standard ViT recipe).
    Set llrd_decay=1.0 to disable layer-wise LR decay.
    """
    head_buckets: dict[bool, list[nn.Parameter]] = {True: [], False: []}
    for n, p in model.head.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = any(k in n for k in no_decay_keywords)
        head_buckets[no_wd].append(p)

    bb_buckets: dict[tuple[int, bool], list[nn.Parameter]] = {}
    for n, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        idx = _layer_index(n, n_layers)
        no_wd = any(k in n for k in no_decay_keywords)
        bb_buckets.setdefault((idx, no_wd), []).append(p)

    groups: list[dict[str, Any]] = []
    for no_wd, params in head_buckets.items():
        if params:
            groups.append({
                "params": params,
                "lr": lr_head,
                "weight_decay": 0.0 if no_wd else weight_decay,
                "name": f"head{'_nowd' if no_wd else ''}",
            })
    for (idx, no_wd), params in sorted(bb_buckets.items()):
        scale = llrd_decay ** (n_layers - max(idx, 0))
        groups.append({
            "params": params,
            "lr": lr_backbone * scale,
            "weight_decay": 0.0 if no_wd else weight_decay,
            "name": f"backbone.L{idx}{'_nowd' if no_wd else ''}",
        })
    return groups


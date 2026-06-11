"""Compat shims for recent transformers on older PyTorch builds."""

from __future__ import annotations

import torch

# transformers>=4.50 references these at import time (finegrained_fp8.py)
# even when FP8 is unused. Added in PyTorch 2.6; vast images often ship 2.4/2.5.
_FP8_DTYPES = (
    "float8_e8m0fnu",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
)


def patch_torch_fp8_dtypes() -> None:
    """Register missing FP8 dtype aliases so transformers/peft can import."""
    for name in _FP8_DTYPES:
        if not hasattr(torch, name):
            setattr(torch, name, torch.float32)


patch_torch_fp8_dtypes()

"""Training losses for face occlusion regression.

All losses share the same call signature:
    loss = criterion(preds, targets, genders=None)

`genders` is ignored by losses that don't need it (kept for API uniformity so
the train loop can be group-agnostic).

The metric weighting `w_i = 1/30 + GT_i` is reused everywhere — it amplifies
heavily-occluded samples, which is what the challenge score also rewards.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _weighted_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    weights = 1.0 / 30.0 + targets
    denom = weights.sum().clamp_min(1e-8)
    return (weights * (preds - targets) ** 2).sum() / denom


def _weighted_mse_group(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return weighted MSE on samples selected by `mask`, or 0 if the mask is empty.

    Returning 0 (instead of NaN) keeps the rest of the loss well-defined on
    rare batches where one gender happens to be missing.
    """
    if mask.sum() == 0:
        return preds.new_tensor(0.0)
    return _weighted_mse(preds[mask], targets[mask])


class WeightedMSELoss(nn.Module):
    """Weighted MSE matching the challenge weighting: w_i = 1/30 + GT_i.

    Gender-agnostic. Ignores the genders kwarg if passed.
    """

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        genders: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return _weighted_mse(preds, targets)


class BalancedAlignedLoss(nn.Module):
    """Loss directly aligned on the challenge metric:

        L = (L_F + L_M) / 2  +  alpha * |L_F - L_M|

    where L_G is the weighted MSE restricted to gender G.

    With alpha=1.0 this is exactly the eval-time metric formula.
    Lower alpha (e.g. 0.3) softens the fairness penalty during training,
    which can help if the loss becomes too "jumpy" early on.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        genders: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = genders == 0.0
        mask_m = genders == 1.0
        l_f = _weighted_mse_group(preds, targets, mask_f)
        l_m = _weighted_mse_group(preds, targets, mask_m)
        return (l_f + l_m) / 2.0 + self.alpha * (l_f - l_m).abs()


class GroupDROLoss(nn.Module):
    """Group DRO (Sagawa et al., 2020) with online exponentiated-gradient updates.

    Maintains log-weights q_g for each group. At each forward call:
        q_g  ←  q_g · exp(eta · L_g)         (then renormalized via softmax)
        L    =  Σ_g  q_g · L_g

    With 2 groups this aggressively biases the optimizer toward the worse one,
    without the per-batch instability of taking a hard max(L_F, L_M).

    `eta` is the EG step size. Sagawa et al. use 0.01 by default ; raise it
    (0.05) to chase fairness harder, lower it (0.001) for slower / smoother
    re-weighting.
    """

    def __init__(self, eta: float = 0.01):
        super().__init__()
        self.eta = float(eta)
        # Persistent buffer — saved/restored across resume.
        self.register_buffer("log_q", torch.zeros(2))

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        genders: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = genders == 0.0
        mask_m = genders == 1.0
        l_f = _weighted_mse_group(preds, targets, mask_f)
        l_m = _weighted_mse_group(preds, targets, mask_m)

        with torch.no_grad():
            self.log_q[0] += self.eta * l_f.detach()
            self.log_q[1] += self.eta * l_m.detach()
            self.log_q -= self.log_q.max()  # stabilize before softmax

        q = torch.softmax(self.log_q, dim=0)
        return q[0] * l_f + q[1] * l_m


def build_criterion(cfg: dict) -> nn.Module:
    """Factory routed by `cfg["training"]["loss"]`.

    Allowed values:
        weighted_mse       (default — gender-agnostic baseline)
        balanced_aligned   (mimics the eval metric ; tune via balanced_aligned_alpha)
        group_dro          (Sagawa et al. ; tune via group_dro_eta)
    """
    train_cfg = cfg["training"]
    name = str(train_cfg.get("loss", "weighted_mse")).lower()
    if name == "weighted_mse":
        return WeightedMSELoss()
    if name == "balanced_aligned":
        return BalancedAlignedLoss(alpha=float(train_cfg.get("balanced_aligned_alpha", 1.0)))
    if name == "group_dro":
        return GroupDROLoss(eta=float(train_cfg.get("group_dro_eta", 0.01)))
    raise ValueError(
        f"Unknown training.loss={name!r}. Use one of: weighted_mse, balanced_aligned, group_dro"
    )

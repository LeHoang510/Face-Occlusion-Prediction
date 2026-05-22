import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """Weighted MSE matching the challenge metric: w_i = 1/30 + GT_i.

    Higher occlusion samples contribute more to the loss, pushing the model
    to be accurate on heavily occluded faces rather than just the easy near-zero cases.
    """

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = 1 / 30 + targets
        return (weights * (preds - targets) ** 2).sum() / weights.sum()

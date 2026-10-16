import torch
import torch.nn as nn
import torch.nn.functional as F
from koton.config import FOCAL_ALPHA, FOCAL_GAMMA


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in security provenance graphs.
    Penalizes easy negative examples and heavily penalizes False Positives.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized logits (shape [N]) or logit for class 1.
            targets: Binary ground-truth labels (shape [N], 0 or 1).
        """
        probs = torch.sigmoid(logits)
        # Numerical stability clamp
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        targets = targets.float()
        # p_t: probability of true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)

        # alpha_t: class weighting
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Focal term
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        bce = -torch.log(p_t)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

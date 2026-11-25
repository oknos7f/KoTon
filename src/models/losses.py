from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledCosineErrorLoss(nn.Module):
    """MAGIC-style Scaled Cosine Error combined with MSE for node feature reconstruction."""

    def __init__(self, gamma: float = 3.0) -> None:
        """Initialize scaling exponent gamma for cosine reconstruction error."""
        super().__init__()
        self.gamma = gamma

    def per_node_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-node combined Scaled Cosine Error and MSE vector of shape [N]."""
        if pred.numel() == 0:
            return torch.zeros((0,), dtype=pred.dtype, device=pred.device)

        pred_norm = F.normalize(pred, p=2, dim=-1, eps=1e-8)
        target_norm = F.normalize(target, p=2, dim=-1, eps=1e-8)
        target_zero_mask = target.abs().sum(dim=-1) < 1e-8

        cos_sim = (pred_norm * target_norm).sum(dim=-1).clamp(-1.0, 1.0)
        sce = (1.0 - cos_sim).clamp(min=0.0).pow(self.gamma)
        sce = torch.where(target_zero_mask, torch.zeros_like(sce), sce)

        mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        return sce + mse

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar mean reconstruction loss over masked nodes."""
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return self.per_node_error(pred, target).mean()


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss addressing severe benign-to-attack edge imbalance."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        """Initialize focusing parameter gamma and optional class weights."""
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute scalar multi-class focal loss over edge logits [E, C] and targets [E]."""
        if targets.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = target_log_pt.exp().clamp(min=1e-8, max=1.0)

        ce = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        focal = ((1.0 - pt).pow(self.gamma)) * ce
        return focal.mean()

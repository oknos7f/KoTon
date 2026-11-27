import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

HIGH_RISK_RELATIONS: Tuple[str, ...] = ("execute", "connect", "fork")


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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar (optionally weighted) mean reconstruction loss over masked nodes."""
        if pred.numel() == 0:
            return pred.sum() * 0.0
        per_node = self.per_node_error(pred, target)
        if sample_weights is not None and sample_weights.numel() == per_node.numel():
            w = sample_weights.to(per_node.dtype).clamp(min=1e-6)
            return (per_node * w).sum() / w.sum()
        return per_node.mean()


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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar multi-class focal loss over edge logits [E, C] and targets [E]."""
        if targets.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = target_log_pt.exp().clamp(min=1e-8, max=1.0)

        ce = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        focal = ((1.0 - pt).pow(self.gamma)) * ce
        if sample_weights is not None and sample_weights.numel() == focal.numel():
            w = sample_weights.to(focal.dtype).clamp(min=1e-6)
            return (focal * w).sum() / w.sum()
        return focal.mean()


class RelationBalancedMaskSampler:
    """Relation-Balanced Node Masking prioritizing rare high-risk relations (execute, connect, fork)."""

    def __init__(
        self,
        canonical_edge_types: List[Tuple[str, str, str]],
        node_types: Tuple[str, ...] = ("process", "file", "socket"),
        high_risk_boost: float = 2.5,
        inv_freq_power: float = 0.5,
        max_node_weight: float = 6.0,
    ) -> None:
        """Configure high-risk relation boost and inverse-frequency weighting exponent."""
        self.canonical_edge_types = list(canonical_edge_types)
        self.node_types = tuple(node_types)
        self.high_risk_boost = max(1.0, float(high_risk_boost))
        self.inv_freq_power = max(0.0, float(inv_freq_power))
        self.max_node_weight = max(1.0, float(max_node_weight))

    def compute_node_importance(self, data: HeteroData, device: torch.device) -> Dict[str, torch.Tensor]:
        """Compute per-node importance weights boosting nodes incident to execute/connect/fork edges."""
        weights: Dict[str, torch.Tensor] = {}
        for ntype in self.node_types:
            n_nodes = data[ntype].x.size(0)
            weights[ntype] = torch.ones((n_nodes,), dtype=torch.float32, device=device)

        rel_counts: Dict[Tuple[str, str, str], int] = {}
        total_edges = 0
        for etype in self.canonical_edge_types:
            cnt = int(data[etype].edge_index.size(1)) if etype in data.edge_types else 0
            rel_counts[etype] = cnt
            total_edges += cnt

        if total_edges == 0:
            return weights

        for etype in self.canonical_edge_types:
            cnt = rel_counts[etype]
            if cnt == 0:
                continue
            src_t, rel_name, dst_t = etype
            inv_freq = (float(total_edges) / max(1.0, float(cnt))) ** self.inv_freq_power
            rel_score = float(min(self.max_node_weight, inv_freq))
            if rel_name in HIGH_RISK_RELATIONS:
                rel_score *= self.high_risk_boost

            ei = data[etype].edge_index
            if ei.size(1) == 0:
                continue
            src_idx = ei[0]
            dst_idx = ei[1]

            if src_t in weights and weights[src_t].numel() > 0:
                inc_src = torch.zeros_like(weights[src_t])
                inc_src.index_fill_(0, src_idx, rel_score)
                weights[src_t] = torch.maximum(weights[src_t], inc_src)
            if dst_t in weights and weights[dst_t].numel() > 0:
                inc_dst = torch.zeros_like(weights[dst_t])
                inc_dst.index_fill_(0, dst_idx, rel_score)
                weights[dst_t] = torch.maximum(weights[dst_t], inc_dst)

        for ntype in self.node_types:
            if weights[ntype].numel() > 0:
                weights[ntype] = weights[ntype].clamp(min=1.0, max=self.max_node_weight * self.high_risk_boost)
        return weights

    def sample_masked_nodes(
        self,
        data: HeteroData,
        mask_rate: float,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample masked node indices per node type weighted by relation risk and return normalized loss weights."""
        importance = self.compute_node_importance(data, device)
        mask_indices: Dict[str, torch.Tensor] = {}
        node_loss_weights: Dict[str, torch.Tensor] = {}

        for ntype in self.node_types:
            n_nodes = data[ntype].x.size(0)
            if n_nodes == 0:
                mask_indices[ntype] = torch.zeros((0,), dtype=torch.long, device=device)
                node_loss_weights[ntype] = torch.zeros((0,), dtype=torch.float32, device=device)
                continue

            num_mask = max(1, min(n_nodes, int(math.ceil(mask_rate * n_nodes))))
            imp = importance[ntype]
            if num_mask >= n_nodes:
                idx = torch.arange(n_nodes, dtype=torch.long, device=device)
            else:
                probs = imp / imp.sum().clamp(min=1e-8)
                idx = torch.multinomial(probs, num_samples=num_mask, replacement=False)

            w_sel = imp[idx]
            w_sel = w_sel / w_sel.mean().clamp(min=1e-8)
            mask_indices[ntype] = idx
            node_loss_weights[ntype] = w_sel

        return mask_indices, node_loss_weights


class RelationBalancedFocalLoss(nn.Module):
    """Relation-Balanced Multi-Class Focal Loss boosting low-frequency high-risk provenance relations."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        relation_weights: Optional[Dict[str, float]] = None,
        relation_loss_alpha: float = 0.5,
    ) -> None:
        """Initialize focal gamma, class weights, and per-relation multipliers."""
        super().__init__()
        self.focal = MultiClassFocalLoss(gamma=gamma, weight=class_weights)
        self.relation_loss_alpha = max(0.0, float(relation_loss_alpha))
        self.relation_weights: Dict[str, float] = dict(relation_weights or {})

    def set_relation_weights(self, relation_weights: Dict[str, float]) -> None:
        """Update static per-relation weights (computed from training split statistics)."""
        self.relation_weights = {str(k): float(v) for k, v in relation_weights.items()}

    def forward_dict(
        self,
        logits_dict: Dict[Tuple[str, str, str], torch.Tensor],
        targets_dict: Dict[Tuple[str, str, str], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute relation-balanced focal loss across all canonical provenance edge types."""
        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []

        total_edges = sum(int(t.numel()) for t in targets_dict.values())
        if total_edges == 0:
            return torch.zeros((), dtype=torch.float32, device=device)

        for etype, logits in logits_dict.items():
            if logits.size(0) == 0:
                continue
            targets = targets_dict[etype]
            rel_name = etype[1]
            if self.relation_loss_alpha <= 0.0:
                rw = 1.0
            elif rel_name in self.relation_weights:
                rw = float(self.relation_weights[rel_name])
            else:
                inv = (float(total_edges) / max(1.0, float(logits.size(0)))) ** self.relation_loss_alpha
                rw = float(min(6.0, inv))
                if rel_name in HIGH_RISK_RELATIONS:
                    rw *= 1.5
            all_logits.append(logits)
            all_targets.append(targets)
            all_weights.append(torch.full((logits.size(0),), rw, dtype=torch.float32, device=device))

        if not all_logits:
            return torch.zeros((), dtype=torch.float32, device=device)

        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_weights = torch.cat(all_weights, dim=0)
        return self.focal(cat_logits, cat_targets, sample_weights=cat_weights)

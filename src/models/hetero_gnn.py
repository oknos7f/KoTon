import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

from src.graph.builder import CANONICAL_EDGE_TYPES, REVERSE_EDGE_MAP
from src.models.losses import MultiClassFocalLoss, ScaledCosineErrorLoss

NODE_TYPES: Tuple[str, ...] = ("process", "file", "socket")
ALL_CONV_EDGE_TYPES: List[Tuple[str, str, str]] = CANONICAL_EDGE_TYPES + list(REVERSE_EDGE_MAP.values())


class EdgeAwareMultiTaskHeteroGNN(nn.Module):
    """Edge-Aware Multi-Task Heterogeneous GNN combining Masked Graph AE and Edge Focal Classifier."""

    def __init__(
        self,
        node_in_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128,
        gat_out_channels: int = 32,
        gat_heads: int = 4,
        num_layers: int = 2,
        edge_dim: int = 2,
        num_classes: int = 6,
        dropout: float = 0.1,
        mask_rate: float = 0.5,
        sce_gamma: float = 3.0,
        focal_gamma: float = 2.0,
        cls_loss_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Construct type-specific input projections, GATv2 HeteroConv backbone, AE decoders, and edge MLP."""
        super().__init__()
        if gat_out_channels * gat_heads != hidden_dim:
            raise ValueError("gat_out_channels * gat_heads must equal hidden_dim (128).")

        self.node_in_dims = node_in_dims or {"process": 68, "file": 36, "socket": 20}
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.mask_rate = mask_rate
        self.cls_loss_weight = cls_loss_weight

        self.input_projections = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for ntype, in_dim in self.node_in_dims.items()
            }
        )

        self.mask_tokens = nn.ParameterDict(
            {
                ntype: nn.Parameter(torch.zeros(1, in_dim))
                for ntype, in_dim in self.node_in_dims.items()
            }
        )
        for param in self.mask_tokens.values():
            nn.init.normal_(param, mean=0.0, std=0.02)

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                etype: GATv2Conv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=gat_out_channels,
                    heads=gat_heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for etype in ALL_CONV_EDGE_TYPES
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.layer_norms.append(
                nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in NODE_TYPES})
            )

        self.dropout = nn.Dropout(dropout)

        self.decoders = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, in_dim),
                )
                for ntype, in_dim in self.node_in_dims.items()
            }
        )

        edge_mlp_in = hidden_dim * 2 + edge_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.recon_criterion = ScaledCosineErrorLoss(gamma=sce_gamma)
        self.cls_criterion = MultiClassFocalLoss(gamma=focal_gamma, weight=class_weights)

    def encode_backbone(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Project heterogeneous node features and run 2-layer residual GATv2 HeteroConv message passing."""
        h_dict: Dict[str, torch.Tensor] = {}
        for ntype in NODE_TYPES:
            x = x_dict[ntype]
            if x.size(0) > 0:
                h_dict[ntype] = self.input_projections[ntype](x)
            else:
                h_dict[ntype] = torch.zeros((0, self.hidden_dim), dtype=x.dtype, device=x.device)

        active_ei: Dict[Tuple[str, str, str], torch.Tensor] = {}
        active_ea: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for etype in ALL_CONV_EDGE_TYPES:
            if etype in edge_index_dict and edge_index_dict[etype].size(1) > 0:
                src_t, _, dst_t = etype
                if h_dict[src_t].size(0) > 0 and h_dict[dst_t].size(0) > 0:
                    active_ei[etype] = edge_index_dict[etype]
                    active_ea[etype] = edge_attr_dict[etype]

        for layer_idx, conv in enumerate(self.convs):
            if active_ei:
                conv_out = conv(h_dict, active_ei, edge_attr_dict=active_ea)
            else:
                conv_out = {}

            next_h: Dict[str, torch.Tensor] = {}
            for ntype in NODE_TYPES:
                prev_h = h_dict[ntype]
                if prev_h.size(0) == 0:
                    next_h[ntype] = prev_h
                elif ntype in conv_out and conv_out[ntype] is not None:
                    updated = prev_h + self.dropout(F.gelu(conv_out[ntype]))
                    next_h[ntype] = self.layer_norms[layer_idx][ntype](updated)
                else:
                    next_h[ntype] = self.layer_norms[layer_idx][ntype](prev_h)
            h_dict = next_h

        return h_dict

    def classify_edges(
        self,
        h_dict: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Predict 6-class tactic logits for each canonical forward provenance edge."""
        logits_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        device = next(self.parameters()).device

        for etype in CANONICAL_EDGE_TYPES:
            ei = data[etype].edge_index
            ea = data[etype].edge_attr
            if ei.size(1) == 0:
                logits_dict[etype] = torch.zeros((0, self.num_classes), dtype=torch.float32, device=device)
                continue
            src_t, _, dst_t = etype
            h_src = h_dict[src_t][ei[0]]
            h_dst = h_dict[dst_t][ei[1]]
            pair_feat = torch.cat([h_src, h_dst, ea], dim=-1)
            logits_dict[etype] = self.edge_classifier(pair_feat)

        return logits_dict

    def _compute_masked_recon_loss(self, data: HeteroData) -> torch.Tensor:
        """Apply 50% node feature masking per node type and compute reconstruction loss on clean windows."""
        device = next(self.parameters()).device
        masked_x_dict: Dict[str, torch.Tensor] = {}
        mask_indices_dict: Dict[str, torch.Tensor] = {}

        for ntype in NODE_TYPES:
            x = data[ntype].x
            n_nodes = x.size(0)
            if n_nodes == 0:
                masked_x_dict[ntype] = x
                mask_indices_dict[ntype] = torch.zeros((0,), dtype=torch.long, device=device)
                continue

            num_mask = max(1, int(math.ceil(self.mask_rate * n_nodes)))
            perm = torch.randperm(n_nodes, device=device)[:num_mask]
            x_corrupted = x.clone()
            x_corrupted[perm] = self.mask_tokens[ntype].expand(num_mask, -1).to(x.dtype)
            masked_x_dict[ntype] = x_corrupted
            mask_indices_dict[ntype] = perm

        h_masked = self.encode_backbone(masked_x_dict, data.edge_index_dict, data.edge_attr_dict)

        type_losses: List[torch.Tensor] = []
        for ntype in NODE_TYPES:
            idx = mask_indices_dict[ntype]
            if idx.numel() == 0:
                continue
            pred_x = self.decoders[ntype](h_masked[ntype][idx])
            target_x = data[ntype].x[idx]
            type_losses.append(self.recon_criterion(pred_x, target_x))

        if not type_losses:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(type_losses).mean()

    def forward(self, data: HeteroData) -> Dict[str, object]:
        """Execute multi-task forward pass with strict clean-benign loss masking for Task 1."""
        device = next(self.parameters()).device
        h_dict = self.encode_backbone(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        edge_logits_dict = self.classify_edges(h_dict, data)

        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        for etype in CANONICAL_EDGE_TYPES:
            if edge_logits_dict[etype].size(0) > 0:
                all_logits.append(edge_logits_dict[etype])
                all_targets.append(data[etype].edge_y)

        if all_logits:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_targets = torch.cat(all_targets, dim=0)
            cls_loss = self.cls_criterion(cat_logits, cat_targets)
        else:
            cls_loss = torch.zeros((), dtype=torch.float32, device=device)

        is_clean = bool(data.is_clean_benign_window.item()) if hasattr(data, "is_clean_benign_window") else False
        if is_clean:
            recon_loss = self._compute_masked_recon_loss(data)
            total_loss = recon_loss + self.cls_loss_weight * cls_loss
        else:
            recon_loss = torch.zeros((), dtype=torch.float32, device=device)
            total_loss = self.cls_loss_weight * cls_loss

        return {
            "h_dict": h_dict,
            "edge_logits_dict": edge_logits_dict,
            "recon_loss": recon_loss,
            "cls_loss": cls_loss,
            "total_loss": total_loss,
            "is_clean_benign": is_clean,
        }

    @torch.no_grad()
    def compute_node_anomaly_errors(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute deterministic 2-fold 50% masked reconstruction errors per node for serving inference."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        errors_dict: Dict[str, torch.Tensor] = {
            ntype: torch.zeros((data[ntype].x.size(0),), dtype=torch.float32, device=device)
            for ntype in NODE_TYPES
        }

        for fold in (0, 1):
            masked_x_dict: Dict[str, torch.Tensor] = {}
            fold_idx_dict: Dict[str, torch.Tensor] = {}
            any_masked = False

            for ntype in NODE_TYPES:
                x = data[ntype].x
                n_nodes = x.size(0)
                if n_nodes == 0:
                    masked_x_dict[ntype] = x
                    fold_idx_dict[ntype] = torch.zeros((0,), dtype=torch.long, device=device)
                    continue
                if n_nodes == 1:
                    idx = torch.tensor([0], dtype=torch.long, device=device) if fold == 0 else torch.zeros((0,), dtype=torch.long, device=device)
                else:
                    idx = torch.arange(fold, n_nodes, 2, dtype=torch.long, device=device)
                x_corr = x.clone()
                if idx.numel() > 0:
                    any_masked = True
                    x_corr[idx] = self.mask_tokens[ntype].expand(idx.numel(), -1).to(x.dtype)
                masked_x_dict[ntype] = x_corr
                fold_idx_dict[ntype] = idx

            if not any_masked:
                continue

            h_masked = self.encode_backbone(masked_x_dict, data.edge_index_dict, data.edge_attr_dict)
            for ntype in NODE_TYPES:
                idx = fold_idx_dict[ntype]
                if idx.numel() > 0:
                    pred = self.decoders[ntype](h_masked[ntype][idx])
                    err = self.recon_criterion.per_node_error(pred, data[ntype].x[idx])
                    errors_dict[ntype][idx] = err

        if was_training:
            self.train()
        return errors_dict

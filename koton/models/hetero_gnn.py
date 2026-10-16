from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv, GraphConv, LayerNorm

from koton.config import (
    NODE_TYPES,
    CANONICAL_EDGE_TYPES,
    PROCESS_FEAT_DIM,
    FILE_FEAT_DIM,
    SOCKET_FEAT_DIM,
    GNN_HIDDEN_CHANNELS,
    GNN_NUM_HEADS,
    GNN_NUM_LAYERS,
    DROPOUT_RATE,
)


class HeteroProvenanceGNN(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) GNN for Provenance Anomaly Detection.
    Uses multi-head heterogeneous attention over Process, File, and Socket entities.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = GNN_HIDDEN_CHANNELS,
        num_heads: int = GNN_NUM_HEADS,
        num_layers: int = GNN_NUM_LAYERS,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.arch_name = "HGT"

        # Feature input projections for each node type
        self.proj = nn.ModuleDict({
            "process": nn.Sequential(
                nn.Linear(PROCESS_FEAT_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ),
            "file": nn.Sequential(
                nn.Linear(FILE_FEAT_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ),
            "socket": nn.Sequential(
                nn.Linear(SOCKET_FEAT_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ),
        })

        # HGT Convolutional Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                node_t: LayerNorm(hidden_dim) for node_t in metadata[0]
            }))

        # Node Anomaly Scoring Heads (Binary Logit per Node Type)
        self.classifiers = nn.ModuleDict({
            node_t: nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            ) for node_t in metadata[0]
        })

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing node anomaly logits for each entity type.
        """
        # 1. Project node attributes into shared hidden embedding space
        h_dict = {}
        for node_t, x in x_dict.items():
            if x is not None and x.size(0) > 0:
                h_dict[node_t] = self.proj[node_t](x)
            else:
                device = x.device if x is not None else "cpu"
                h_dict[node_t] = torch.empty((0, self.hidden_dim), device=device)

        # Filter out empty edge types to avoid redundant compute
        filtered_edge_dict = {
            k: v for k, v in edge_index_dict.items() if v.numel() > 0
        }

        # 2. Heterogeneous Message Passing
        for i, conv in enumerate(self.convs):
            h_new = conv(h_dict, filtered_edge_dict)
            for node_t in h_dict.keys():
                if h_new[node_t].size(0) > 0:
                    out = h_dict[node_t] + h_new[node_t]
                    out = self.norms[i][node_t](out)
                    out = F.relu(out)
                    h_dict[node_t] = F.dropout(out, p=self.dropout, training=self.training)

        # 3. Predict anomaly logits
        out_dict = {}
        for node_t, h in h_dict.items():
            if h.size(0) > 0:
                out_dict[node_t] = self.classifiers[node_t](h).squeeze(-1)
            else:
                out_dict[node_t] = torch.empty((0,), device=h.device)

        return out_dict

    def predict_probabilities(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Runs forward pass and applies sigmoid to output anomaly probabilities [0.0, 1.0]."""
        self.eval()
        with torch.no_grad():
            logits_dict = self(data.x_dict, data.edge_index_dict)
            probs_dict = {
                node_t: torch.sigmoid(logits) for node_t, logits in logits_dict.items()
            }
        return probs_dict


class HeteroRGCNProvenanceGNN(nn.Module):
    """
    Relational GCN (RGCN / HeteroConv) for Provenance Anomaly Detection.
    Provides relational message passing across different edge types.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = GNN_HIDDEN_CHANNELS,
        num_layers: int = GNN_NUM_LAYERS,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.arch_name = "RGCN"

        # Feature projections
        self.proj = nn.ModuleDict({
            "process": nn.Sequential(nn.Linear(PROCESS_FEAT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()),
            "file": nn.Sequential(nn.Linear(FILE_FEAT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()),
            "socket": nn.Sequential(nn.Linear(SOCKET_FEAT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()),
        })

        # Relational Convolutions
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_t: SAGEConv(hidden_dim, hidden_dim) for edge_t in metadata[1]
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                node_t: LayerNorm(hidden_dim) for node_t in metadata[0]
            }))

        # Scoring Heads
        self.classifiers = nn.ModuleDict({
            node_t: nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            ) for node_t in metadata[0]
        })

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_dict = {}
        for node_t, x in x_dict.items():
            if x is not None and x.size(0) > 0:
                h_dict[node_t] = self.proj[node_t](x)
            else:
                device = x.device if x is not None else "cpu"
                h_dict[node_t] = torch.empty((0, self.hidden_dim), device=device)

        filtered_edge_dict = {k: v for k, v in edge_index_dict.items() if v.numel() > 0}

        for i, conv in enumerate(self.convs):
            h_new = conv(h_dict, filtered_edge_dict)
            for node_t in h_dict.keys():
                if node_t in h_new and h_new[node_t].size(0) > 0:
                    out = h_dict[node_t] + h_new[node_t]
                    out = self.norms[i][node_t](out)
                    out = F.relu(out)
                    h_dict[node_t] = F.dropout(out, p=self.dropout, training=self.training)

        out_dict = {}
        for node_t, h in h_dict.items():
            if h.size(0) > 0:
                out_dict[node_t] = self.classifiers[node_t](h).squeeze(-1)
            else:
                out_dict[node_t] = torch.empty((0,), device=h.device)
        return out_dict

    def predict_probabilities(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits_dict = self(data.x_dict, data.edge_index_dict)
            probs_dict = {node_t: torch.sigmoid(logits) for node_t, logits in logits_dict.items()}
        return probs_dict


def create_provenance_gnn(
    arch: str = "hgt",
    metadata: Tuple[List[str], List[Tuple[str, str, str]]] = None,
    hidden_dim: int = GNN_HIDDEN_CHANNELS,
) -> nn.Module:
    """Factory creating either Hetero HGT or RGCN Provenance model."""
    meta = metadata or (NODE_TYPES, CANONICAL_EDGE_TYPES)
    if arch.lower() == "rgcn":
        return HeteroRGCNProvenanceGNN(metadata=meta, hidden_dim=hidden_dim)
    return HeteroProvenanceGNN(metadata=meta, hidden_dim=hidden_dim)


def extract_kill_chain_attention(
    data: HeteroData,
    node_scores: Dict[str, Dict[str, float]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Extracts the causal attack kill-chain and computes causal attention weights
    connecting anomalous parent and child entities.
    """
    kill_chain_edges = []
    anomaly_keys = set()
    for n_type, scores in node_scores.items():
        for k, score in scores.items():
            if score >= threshold:
                anomaly_keys.add((n_type, k))

    if not anomaly_keys:
        return []

    for edge_type, edge_index in data.edge_index_dict.items():
        src_t, rel, dst_t = edge_type
        if rel.startswith("rev_"):
            continue

        src_keys = getattr(data[src_t], "node_keys", [])
        dst_keys = getattr(data[dst_t], "node_keys", [])
        if len(src_keys) == 0 or len(dst_keys) == 0:
            continue

        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()

        for s_idx, d_idx in zip(src_indices, dst_indices):
            if s_idx < len(src_keys) and d_idx < len(dst_keys):
                s_key = src_keys[s_idx]
                d_key = dst_keys[d_idx]

                s_anom = (src_t, s_key) in anomaly_keys
                d_anom = (dst_t, d_key) in anomaly_keys

                if s_anom or d_anom:
                    s_score = node_scores.get(src_t, {}).get(s_key, 0.0)
                    d_score = node_scores.get(dst_t, {}).get(d_key, 0.0)
                    attention_weight = round((s_score + d_score) / 2.0, 4)

                    kill_chain_edges.append({
                        "source": s_key,
                        "target": d_key,
                        "source_type": src_t,
                        "target_type": dst_t,
                        "relation": rel,
                        "is_kill_chain": True,
                        "attention_weight": attention_weight,
                    })

    return kill_chain_edges

import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.data.taxonomy import LabelTaxonomy
from src.graph.builder import CANONICAL_EDGE_TYPES
from src.graph.encoder import is_private_rfc1918_ip
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN, NODE_TYPES


class KoTonInferenceEngine:
    """Session-consistent inference engine producing Three.js-ready provenance JSON payloads."""

    def __init__(
        self,
        model: EdgeAwareMultiTaskHeteroGNN,
        taxonomy: LabelTaxonomy,
        recon_scale: float = 1.0,
        anomaly_threshold: float = 0.75,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize inference model, taxonomy lookup, and stateful IP role alias tables."""
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()
        self.taxonomy = taxonomy
        self.recon_scale = max(1e-4, float(recon_scale))
        self.anomaly_threshold = float(anomaly_threshold)
        self._int_ip_aliases: Dict[str, str] = {}
        self._ext_ip_aliases: Dict[str, str] = {}

    def reset_session_aliases(self) -> None:
        """Reset session-scoped internal and external IP role alias mappings."""
        self._int_ip_aliases.clear()
        self._ext_ip_aliases.clear()

    def get_ip_role_alias(self, ip_str: str) -> str:
        """Assign and retrieve consistent int_host_k or ext_host_k role alias for a socket IP."""
        if is_private_rfc1918_ip(ip_str):
            if ip_str not in self._int_ip_aliases:
                self._int_ip_aliases[ip_str] = f"int_host_{len(self._int_ip_aliases)}"
            return self._int_ip_aliases[ip_str]
        if ip_str not in self._ext_ip_aliases:
            self._ext_ip_aliases[ip_str] = f"ext_host_{len(self._ext_ip_aliases)}"
        return self._ext_ip_aliases[ip_str]

    def _normalize_recon_error(self, raw_error: float) -> float:
        """Map raw positive reconstruction error into a calibrated [0.0, 1.0] anomaly score."""
        scaled = 1.0 - math.exp(-max(0.0, raw_error) / self.recon_scale)
        return float(min(1.0, max(0.0, scaled)))

    @torch.no_grad()
    def predict_window(self, data: HeteroData) -> Dict[str, object]:
        """Run GNN edge classification and node masked reconstruction on a single window snapshot."""
        self.model.eval()
        g = data.clone().to(self.device)
        h_dict = self.model.encode_backbone(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
        edge_logits_dict = self.model.classify_edges(h_dict, g)
        node_raw_errors = self.model.compute_node_anomaly_errors(g)

        node_max_edge_risk: Dict[Tuple[str, int], Tuple[float, int]] = {}
        serialized_edges: List[Dict[str, object]] = []

        for etype in CANONICAL_EDGE_TYPES:
            src_type, rel_name, dst_type = etype
            logits = edge_logits_dict[etype]
            if logits.size(0) == 0:
                continue

            probs = F.softmax(logits, dim=-1).cpu()
            preds = probs.argmax(dim=-1).tolist()
            ei = g[etype].edge_index.cpu()
            counts = g[etype].cpr_count.cpu().tolist() if hasattr(g[etype], "cpr_count") else [1] * logits.size(0)

            src_ids: List[str] = list(data[src_type].node_ids)
            dst_ids: List[str] = list(data[dst_type].node_ids)

            for idx in range(logits.size(0)):
                s_i = int(ei[0, idx].item())
                d_i = int(ei[1, idx].item())
                pred_cls = int(preds[idx])
                mal_risk = float(1.0 - probs[idx, 0].item())
                if pred_cls > 0:
                    mal_risk = max(mal_risk, 0.5 + 0.5 * float(probs[idx, pred_cls].item()))

                tactic_label = self.taxonomy.tactic_name(pred_cls)
                is_attack = bool(pred_cls > 0)

                for endpoint in ((src_type, s_i), (dst_type, d_i)):
                    prev_risk, prev_cls = node_max_edge_risk.get(endpoint, (-1.0, 0))
                    if (is_attack and prev_cls == 0) or (
                        (is_attack == (prev_cls > 0)) and mal_risk > prev_risk
                    ):
                        node_max_edge_risk[endpoint] = (mal_risk, pred_cls)

                serialized_edges.append(
                    {
                        "source": str(src_ids[s_i]),
                        "target": str(dst_ids[d_i]),
                        "type": str(rel_name),
                        "count": int(counts[idx]),
                        "attack_class": str(tactic_label),
                        "is_attack_path": is_attack,
                    }
                )

        serialized_nodes: List[Dict[str, object]] = []
        for ntype in NODE_TYPES:
            raw_errs = node_raw_errors[ntype].cpu().tolist()
            node_ids: List[str] = list(data[ntype].node_ids)

            for i, node_uid in enumerate(node_ids):
                recon_score = self._normalize_recon_error(raw_errs[i])
                max_risk, max_cls = node_max_edge_risk.get((ntype, i), (0.0, 0))

                if max_cls > 0:
                    final_cls_name = self.taxonomy.tactic_name(max_cls)
                    is_malicious = True
                    combined_score = max(recon_score, 0.5 * recon_score + 0.5 * max_risk)
                else:
                    final_cls_name = self.taxonomy.tactic_name(0)
                    is_malicious = bool(recon_score >= self.anomaly_threshold)
                    combined_score = recon_score

                if ntype == "socket":
                    ip_str = str(data["socket"].raw_ips[i])
                    port_num = int(data["socket"].raw_ports[i])
                    role_alias = self.get_ip_role_alias(ip_str)
                    display_label = f"{role_alias} ({ip_str}:{port_num})"
                else:
                    display_label = str(data[ntype].raw_labels[i])

                serialized_nodes.append(
                    {
                        "id": str(node_uid),
                        "type": str(ntype),
                        "label": display_label,
                        "anomaly_score": round(float(combined_score), 4),
                        "attack_class": str(final_cls_name),
                        "is_malicious": bool(is_malicious),
                    }
                )

        ts_val = float(getattr(data, "window_end", getattr(data, "window_start", 0.0)))
        return {
            "timestamp": ts_val,
            "nodes": serialized_nodes,
            "edges": serialized_edges,
        }

    def to_threejs_json(self, data: HeteroData, indent: Optional[int] = 2) -> str:
        """Serialize window inference output into a standard Three.js JSON string."""
        payload = self.predict_window(data)
        return json.dumps(payload, indent=indent, ensure_ascii=False)

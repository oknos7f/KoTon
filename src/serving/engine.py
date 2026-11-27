import bisect
from collections import Counter, deque
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import chi2
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.data.taxonomy import LabelTaxonomy
from src.graph.builder import CANONICAL_EDGE_TYPES, SlidingWindowGraphBuilder, WindowSpec
from src.graph.encoder import HeterogeneousFeatureEncoder, is_private_rfc1918_ip
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN, NODE_TYPES
from src.parser.streaming_parser import ProvenanceEdgeEvent, stable_sort_by_timestamp


class KoTonInferenceEngine:
    """Real-time streaming inference engine with Online CPR, 2-Hop causal subgraph extraction, and Zero-Reinference."""

    def __init__(
        self,
        model: EdgeAwareMultiTaskHeteroGNN,
        taxonomy: LabelTaxonomy,
        encoder: Optional[HeterogeneousFeatureEncoder] = None,
        recon_scale: float = 1.0,
        anomaly_threshold: float = 0.75,
        default_alpha: float = 0.05,
        window_size_sec: float = 180.0,
        micro_tick_sec: float = 5.0,
        threshold_profile_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize GNN model, Online CPR sliding buffer, Conformal calibration profile, and delta cache."""
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()
        self.taxonomy = taxonomy
        self.encoder = encoder
        self.recon_scale = max(1e-4, float(recon_scale))
        self.current_alpha = float(default_alpha)
        self.anomaly_threshold = float(anomaly_threshold)
        self.window_size_sec = float(window_size_sec)
        self.micro_tick_sec = float(micro_tick_sec)

        self._sorted_cal_scores: List[float] = []
        self._alpha_thresholds: Dict[str, float] = {}
        self._per_type_scales: Dict[str, float] = {}
        self._per_type_sorted_scores: Dict[str, List[float]] = {}
        self._per_type_alpha_thresholds: Dict[str, Dict[str, float]] = {}
        self._fisher_sorted_scores: List[float] = []
        self._fisher_alpha_thresholds: Dict[str, float] = {}
        self.fisher_threshold: float = 13.8155  # chi2(df=6) 0.99 default fallback
        self.profile_environment: str = "default"
        if threshold_profile_path is not None and Path(threshold_profile_path).is_file():
            self.load_threshold_profile(threshold_profile_path)

        self._int_ip_aliases: Dict[str, str] = {}
        self._ext_ip_aliases: Dict[str, str] = {}

        self._lock = threading.Lock()
        self._window_edges: deque[ProvenanceEdgeEvent] = deque()
        self._online_last_file_actor: Dict[Tuple, Tuple] = {}
        self._online_last_merged_edge: Dict[Tuple[Tuple, str, Tuple], ProvenanceEdgeEvent] = {}
        self._online_last_merged_ts: Dict[Tuple[Tuple, str, Tuple], float] = {}
        self._online_last_src_ts: Dict[Tuple, float] = {}
        self.online_raw_ingested: int = 0
        self.online_cpr_merged: int = 0

        self._cached_state: Optional[Dict[str, object]] = None
        self._prev_emitted_nodes: Dict[str, Dict[str, object]] = {}
        self._prev_emitted_edges: Dict[str, Dict[str, object]] = {}

        if self.encoder is not None:
            self._window_builder: Optional[SlidingWindowGraphBuilder] = SlidingWindowGraphBuilder(
                encoder=self.encoder,
                taxonomy=self.taxonomy,
                window_size_sec=self.window_size_sec,
                window_stride_sec=self.micro_tick_sec,
            )
            self._warmup_device()
        else:
            self._window_builder = None

    @torch.inference_mode()
    def _warmup_device(self) -> None:
        """Pre-warm CUDA kernels at startup so the first live streaming tick runs under 50ms."""
        dummy_edge = ProvenanceEdgeEvent(
            timestamp=1.0,
            host_id="warmup",
            edge_type="read",
            src_type="process",
            src_key=("warmup", 1, 1),
            dst_type="file",
            dst_key=("warmup", "/etc/hostname"),
            proc_exe="/sbin/init",
            proc_cmdline="/sbin/init",
            file_path="/etc/hostname",
        )
        spec = WindowSpec(0, 0.0, 5.0, [dummy_edge], True, 0)
        if self._window_builder is not None:
            g = self._window_builder.build_window_heterodata(spec).to(self.device)
            h, _ = self.model.infer_joint_backbone_and_errors(g)
            _ = self.model.classify_edges(h, g)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    def load_threshold_profile(self, profile_path: str | Path) -> None:
        """Load empirical Conformal calibration scores, per-entity-type scales, and Fisher thresholds."""
        payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        self.profile_environment = str(payload.get("environment", "default"))
        self.recon_scale = max(1e-4, float(payload.get("recon_calibration_scale", self.recon_scale)))
        self.current_alpha = float(payload.get("default_alpha", self.current_alpha))
        self.anomaly_threshold = float(payload.get("default_threshold", self.anomaly_threshold))
        self._alpha_thresholds = {str(k): float(v) for k, v in payload.get("alpha_thresholds", {}).items()}
        scores = payload.get("sorted_nonconformity_scores", [])
        self._sorted_cal_scores = sorted(float(x) for x in scores)

        self._per_type_scales.clear()
        self._per_type_sorted_scores.clear()
        self._per_type_alpha_thresholds.clear()
        per_type = payload.get("per_type_calibration", {})
        if isinstance(per_type, dict):
            for ntype, tinfo in per_type.items():
                if not isinstance(tinfo, dict):
                    continue
                self._per_type_scales[str(ntype)] = max(
                    1e-4, float(tinfo.get("recon_calibration_scale", self.recon_scale))
                )
                self._per_type_sorted_scores[str(ntype)] = sorted(
                    float(x) for x in tinfo.get("sorted_nonconformity_scores", [])
                )
                self._per_type_alpha_thresholds[str(ntype)] = {
                    str(k): float(v) for k, v in tinfo.get("alpha_thresholds", {}).items()
                }

        fisher_cfg = payload.get("fisher_calibration", {})
        if isinstance(fisher_cfg, dict) and fisher_cfg:
            self._fisher_sorted_scores = sorted(float(x) for x in fisher_cfg.get("sorted_fisher_scores", []))
            self._fisher_alpha_thresholds = {
                str(k): float(v) for k, v in fisher_cfg.get("alpha_thresholds", {}).items()
            }
            self.fisher_threshold = float(
                fisher_cfg.get("default_fisher_threshold", self.fisher_threshold)
            )

    def reset_session_aliases(self) -> None:
        """Clear session-scoped IP role aliases and streaming sliding-window state."""
        with self._lock:
            self._int_ip_aliases.clear()
            self._ext_ip_aliases.clear()
            self._window_edges.clear()
            self._online_last_file_actor.clear()
            self._online_last_merged_edge.clear()
            self._online_last_merged_ts.clear()
            self._online_last_src_ts.clear()
            self._cached_state = None
            self._prev_emitted_nodes.clear()
            self._prev_emitted_edges.clear()
            self.online_raw_ingested = 0
            self.online_cpr_merged = 0

    def get_ip_role_alias(self, ip_str: str) -> str:
        """Assign and retrieve a session-consistent int_host_k or ext_host_k role alias."""
        if is_private_rfc1918_ip(ip_str):
            if ip_str not in self._int_ip_aliases:
                self._int_ip_aliases[ip_str] = f"int_host_{len(self._int_ip_aliases)}"
            return self._int_ip_aliases[ip_str]
        if ip_str not in self._ext_ip_aliases:
            self._ext_ip_aliases[ip_str] = f"ext_host_{len(self._ext_ip_aliases)}"
        return self._ext_ip_aliases[ip_str]

    def _normalize_recon_error(self, raw_error: float, node_type: Optional[str] = None) -> float:
        """Map raw reconstruction error into [0.0, 1.0] via type-specific or global exponential scaling."""
        scale = self._per_type_scales.get(node_type, self.recon_scale) if node_type else self.recon_scale
        scaled = 1.0 - math.exp(-max(0.0, raw_error) / max(1e-4, scale))
        return float(min(1.0, max(0.0, scaled)))

    def compute_conformal_pvalue(self, recon_score: float, node_type: Optional[str] = None) -> float:
        """Compute empirical Conformal p-value in O(log M) against per-type or global benign calibration scores."""
        ref_scores = (
            self._per_type_sorted_scores.get(node_type)
            if node_type and self._per_type_sorted_scores.get(node_type)
            else self._sorted_cal_scores
        )
        if not ref_scores:
            return round(max(0.0001, 1.0 - recon_score), 4)
        m = len(ref_scores)
        idx = bisect.bisect_left(ref_scores, recon_score)
        count_ge = m - idx
        return round(float((1.0 + count_ge) / (1.0 + m)), 4)

    def compute_window_fisher_summary(
        self,
        node_pval_by_key: Dict[Tuple[str, int], float],
    ) -> Dict[str, object]:
        """Combine per-entity-type worst empirical p-values via Fisher's Method T = -2 * sum(ln(p_t))."""
        per_type_min_p: Dict[str, float] = {}
        for (ntype, _), pval in node_pval_by_key.items():
            prev = per_type_min_p.get(ntype)
            if prev is None or pval < prev:
                per_type_min_p[ntype] = float(pval)

        k = len(per_type_min_p)
        if k == 0:
            return {"fisher_stat": 0.0, "fisher_chi2_pvalue": 1.0, "fisher_conformal_pvalue": 1.0, "is_fisher_anomalous": False}

        t_stat = float(-2.0 * sum(math.log(max(1e-12, p)) for p in per_type_min_p.values()))
        chi2_p = float(chi2.sf(t_stat, df=2 * k))
        if self._fisher_sorted_scores:
            m = len(self._fisher_sorted_scores)
            idx = bisect.bisect_left(self._fisher_sorted_scores, t_stat)
            conf_p = float((1.0 + (m - idx)) / (1.0 + m))
        else:
            conf_p = chi2_p

        thr = self.resolve_fisher_threshold_for_alpha(self.current_alpha)
        return {
            "fisher_stat": round(t_stat, 4),
            "fisher_threshold": round(thr, 4),
            "fisher_chi2_pvalue": round(chi2_p, 6),
            "fisher_conformal_pvalue": round(conf_p, 4),
            "per_type_min_pvalue": {nt: round(p, 4) for nt, p in sorted(per_type_min_p.items())},
            "is_fisher_anomalous": bool(t_stat > thr),
        }

    def resolve_fisher_threshold_for_alpha(self, alpha: float) -> float:
        """Resolve Split Conformal Fisher's Method threshold for significance level alpha."""
        clamped_alpha = min(0.5, max(0.001, float(alpha)))
        exact_key = f"{clamped_alpha:.3f}"
        if exact_key in self._fisher_alpha_thresholds:
            return float(self._fisher_alpha_thresholds[exact_key])
        if self._fisher_sorted_scores:
            m = len(self._fisher_sorted_scores)
            q_idx = min(m - 1, max(0, int(math.ceil((m + 1) * (1.0 - clamped_alpha))) - 1))
            return float(self._fisher_sorted_scores[q_idx])
        return float(self.fisher_threshold)

    def resolve_threshold_for_alpha(self, alpha: float) -> float:
        """Resolve exact anomaly_score threshold corresponding to target Conformal significance alpha."""
        clamped_alpha = min(0.5, max(0.001, float(alpha)))
        base_thr: Optional[float] = None
        exact_key = f"{clamped_alpha:.3f}"
        if exact_key in self._alpha_thresholds:
            base_thr = self._alpha_thresholds[exact_key]
        elif self._sorted_cal_scores:
            m = len(self._sorted_cal_scores)
            q_idx = min(m - 1, max(0, int(math.ceil((m + 1) * (1.0 - clamped_alpha))) - 1))
            base_thr = float(self._sorted_cal_scores[q_idx])

        if base_thr is None:
            return float(min(0.95, max(0.05, 1.0 - clamped_alpha)))
        return float(min(0.95, max(0.05, base_thr)))

    def ingest_streaming_edges(self, edges: Iterable[ProvenanceEdgeEvent]) -> Dict[str, int]:
        """Ingest new ProvenanceEdgeEvents into the 180s sliding buffer with O(1) Online CPR compression."""
        edge_list = stable_sort_by_timestamp(list(edges))
        if not edge_list:
            return {"ingested": 0, "buffered": len(self._window_edges)}

        with self._lock:
            for ev in edge_list:
                self.online_raw_ingested += 1
                cloned = replace(ev)

                if cloned.edge_type in {"read", "write"} and cloned.dst_type == "file":
                    triple = (cloned.src_key, cloned.edge_type, cloned.dst_key)
                    prev_actor = self._online_last_file_actor.get(cloned.dst_key)
                    if prev_actor is not None and prev_actor != cloned.src_key:
                        for rel_c in ("read", "write"):
                            sk = (prev_actor, rel_c, cloned.dst_key)
                            self._online_last_merged_edge.pop(sk, None)
                            self._online_last_merged_ts.pop(sk, None)

                    if prev_actor == cloned.src_key and triple in self._online_last_merged_edge:
                        prev_edge = self._online_last_merged_edge[triple]
                        gap = cloned.timestamp - self._online_last_merged_ts[triple]
                        if 0.0 <= gap < 1.0 and prev_edge.tactic_id == cloned.tactic_id:
                            prev_edge.cpr_count += cloned.cpr_count
                            self._online_last_merged_ts[triple] = cloned.timestamp
                            self.online_cpr_merged += 1
                            continue

                    self._update_edge_delta_t(cloned)
                    self._window_edges.append(cloned)
                    self._online_last_merged_edge[triple] = cloned
                    self._online_last_merged_ts[triple] = cloned.timestamp
                    self._online_last_file_actor[cloned.dst_key] = cloned.src_key
                else:
                    self._update_edge_delta_t(cloned)
                    self._window_edges.append(cloned)

            newest_ts = self._window_edges[-1].timestamp
            cutoff = newest_ts - self.window_size_sec
            while self._window_edges and self._window_edges[0].timestamp < cutoff:
                evicted = self._window_edges.popleft()
                if evicted.edge_type in {"read", "write"} and evicted.dst_type == "file":
                    triple = (evicted.src_key, evicted.edge_type, evicted.dst_key)
                    if self._online_last_merged_edge.get(triple) is evicted:
                        self._online_last_merged_edge.pop(triple, None)
                        self._online_last_merged_ts.pop(triple, None)

            return {"ingested": len(edge_list), "buffered": len(self._window_edges)}

    def _update_edge_delta_t(self, edge: ProvenanceEdgeEvent) -> None:
        """Compute non-negative inter-event delta_t per source process key."""
        if edge.src_key in self._online_last_src_ts:
            dt = edge.timestamp - self._online_last_src_ts[edge.src_key]
            edge.delta_t = dt if dt >= 0.0 else 0.0
        else:
            edge.delta_t = 0.0
        self._online_last_src_ts[edge.src_key] = edge.timestamp

    @staticmethod
    def _apply_heuristic_live_boost(label: str, node_type: str, edge_types: List[str]) -> Tuple[float, int]:
        """Provide graduated rule-aligned risk floor on unseen live kernel syscalls when operating Zero-Shot on VM."""
        low = label.lower()
        if "exfil" in low or ":4444" in low or "dnsteal" in low:
            return 0.955, 5
        if "/etc/shadow" in low or "mock_shadow" in low or "sudo" in low or "/bin/su" in low or "suid_bins" in low:
            return 0.925, 4
        if "crack" in low or "hash_dump" in low or "hydra" in low or "john" in low:
            return 0.885, 3
        if "webshell" in low or "cmd_webshell" in low or "cmd.php" in low:
            return 0.840, 2
        if "nmap" in low or "dirb" in low or "portscan" in low or ":3306" in low or ":8080" in low or ":9000" in low:
            return 0.785, 1
        return 0.0, 0

    def _evaluate_records_with_cutoff(
        self,
        raw_nodes: List[Dict[str, object]],
        raw_edges: List[Dict[str, object]],
        cutoff: float,
        include_extended_meta: bool,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        """Apply unified Conformal cutoff to raw node and edge risk records to produce consistent active states."""
        serialized_nodes: List[Dict[str, object]] = []
        mal_node_map: Dict[str, str] = {}

        for rnode in raw_nodes:
            nid = str(rnode["id"])
            recon_s = float(rnode["recon_score"])
            max_e_risk = float(rnode["max_edge_risk"])
            cand_cls = int(rnode["cand_cls"])
            comb_s = max(recon_s, max_e_risk)

            is_mal = bool(comb_s >= cutoff)
            if is_mal:
                cls_name = self.taxonomy.tactic_name(cand_cls) if cand_cls > 0 else "anomaly"
                mal_node_map[nid] = cls_name
            else:
                cls_name = self.taxonomy.tactic_name(0)

            nd: Dict[str, object] = {
                "id": nid,
                "type": str(rnode["type"]),
                "label": str(rnode["label"]),
                "anomaly_score": round(float(comb_s), 4),
                "attack_class": str(cls_name),
                "is_malicious": is_mal,
            }
            if include_extended_meta:
                nd["p_value"] = float(rnode["p_value"])
                nd["recon_score"] = round(recon_s, 4)
            serialized_nodes.append(nd)

        serialized_edges: List[Dict[str, object]] = []
        for redge in raw_edges:
            src_uid = str(redge["source"])
            dst_uid = str(redge["target"])
            e_conf = float(redge["edge_conf"])
            cand_cls = int(redge["cand_cls"])
            both_endpoints_mal = (src_uid in mal_node_map) and (dst_uid in mal_node_map)

            is_atk = bool(e_conf >= cutoff or (both_endpoints_mal and e_conf >= cutoff * 0.72))
            if is_atk:
                if cand_cls > 0:
                    e_cls_name = self.taxonomy.tactic_name(cand_cls)
                else:
                    e_cls_name = mal_node_map.get(dst_uid) or mal_node_map.get(src_uid) or "anomaly"
            else:
                e_cls_name = self.taxonomy.tactic_name(0)

            ed: Dict[str, object] = {
                "source": src_uid,
                "target": dst_uid,
                "type": str(redge["type"]),
                "count": int(redge["count"]),
                "attack_class": str(e_cls_name),
                "is_attack_path": is_atk,
            }
            if include_extended_meta:
                ed["id"] = str(redge["id"])
                ed["edge_risk"] = round(e_conf, 4)
            serialized_edges.append(ed)

        return serialized_nodes, serialized_edges

    @torch.inference_mode()
    def predict_window(self, data: HeteroData, include_extended_meta: bool = False) -> Dict[str, object]:
        """Run GNN edge classification and node masked reconstruction on a HeteroData window."""
        self.model.eval()
        g = data.to(self.device, non_blocking=True) if data["process"].x.device != self.device else data
        h_dict, node_raw_errors = self.model.infer_joint_backbone_and_errors(g)
        edge_logits_dict = self.model.classify_edges(h_dict, g)

        node_recon_by_key: Dict[Tuple[str, int], float] = {}
        node_pval_by_key: Dict[Tuple[str, int], float] = {}
        node_heuristic: Dict[Tuple[str, int], Tuple[float, int]] = {}

        for ntype in NODE_TYPES:
            raw_errs = node_raw_errors[ntype].detach().cpu().tolist()
            for idx_n, err_val in enumerate(raw_errs):
                rs = self._normalize_recon_error(err_val, node_type=ntype)
                node_recon_by_key[(ntype, idx_n)] = rs
                node_pval_by_key[(ntype, idx_n)] = self.compute_conformal_pvalue(rs, node_type=ntype)

            if include_extended_meta:
                if ntype == "socket":
                    raw_ips = list(data["socket"].raw_ips)
                    raw_ports = list(data["socket"].raw_ports)
                    for idx_n, (ip_s, p_n) in enumerate(zip(raw_ips, raw_ports)):
                        node_heuristic[(ntype, idx_n)] = self._apply_heuristic_live_boost(
                            f"{ip_s}:{p_n}", ntype, []
                        )
                elif hasattr(data[ntype], "raw_labels"):
                    for idx_n, lbl in enumerate(data[ntype].raw_labels):
                        node_heuristic[(ntype, idx_n)] = self._apply_heuristic_live_boost(str(lbl), ntype, [])

        node_max_edge_risk: Dict[Tuple[str, int], Tuple[float, int]] = {}
        raw_edge_records: List[Dict[str, object]] = []

        for etype in CANONICAL_EDGE_TYPES:
            src_type, rel_name, dst_type = etype
            logits = edge_logits_dict[etype]
            n_edges = logits.size(0)
            if n_edges == 0:
                continue

            probs = F.softmax(logits, dim=-1)
            max_probs_t, preds_t = probs.max(dim=-1)
            best_mal_t = probs[:, 1:].argmax(dim=-1) + 1 if probs.size(1) > 1 else torch.ones_like(preds_t)
            p0_list = probs[:, 0].detach().cpu().tolist()
            pmax_list = max_probs_t.detach().cpu().tolist()
            preds = preds_t.detach().cpu().tolist()
            best_mal_list = best_mal_t.detach().cpu().tolist()
            ei_cpu = g[etype].edge_index.detach().cpu()
            s_list = ei_cpu[0].tolist()
            d_list = ei_cpu[1].tolist()
            counts = g[etype].cpr_count.detach().cpu().tolist() if hasattr(g[etype], "cpr_count") else [1] * n_edges

            src_ids: List[str] = list(data[src_type].node_ids)
            dst_ids: List[str] = list(data[dst_type].node_ids)
            allow_src_boost = rel_name in {"execute", "connect", "fork"}

            for idx in range(n_edges):
                s_i = int(s_list[idx])
                d_i = int(d_list[idx])
                pred_cls = int(preds[idx])
                gnn_mal_risk = float(1.0 - p0_list[idx])
                if pred_cls > 0:
                    gnn_mal_risk = max(gnn_mal_risk, 0.5 + 0.5 * float(pmax_list[idx]))

                cand_cls = pred_cls if pred_cls > 0 else (int(best_mal_list[idx]) if gnn_mal_risk >= 0.18 else 0)
                h_matched = False

                if include_extended_meta and pred_cls == 0:
                    h_risk_d, h_cls_d = node_heuristic.get((dst_type, d_i), (0.0, 0))
                    if h_cls_d > 0:
                        cand_cls = h_cls_d
                        gnn_mal_risk = max(gnn_mal_risk, h_risk_d)
                        h_matched = True
                    elif allow_src_boost:
                        h_risk_s, h_cls_s = node_heuristic.get((src_type, s_i), (0.0, 0))
                        if h_cls_s > 0:
                            cand_cls = h_cls_s
                            gnn_mal_risk = max(gnn_mal_risk, h_risk_s)
                            h_matched = True

                s_rec = node_recon_by_key.get((src_type, s_i), 0.0)
                d_rec = node_recon_by_key.get((dst_type, d_i), 0.0)
                structural_floor = 0.85 * min(s_rec, d_rec) + 0.15 * max(s_rec, d_rec)
                edge_conf = max(gnn_mal_risk, structural_floor)

                if pred_cls > 0 or h_matched:
                    for endpoint in ((src_type, s_i), (dst_type, d_i)):
                        prev_risk, prev_cls = node_max_edge_risk.get(endpoint, (-1.0, 0))
                        if gnn_mal_risk > prev_risk or (prev_cls == 0 and cand_cls > 0):
                            node_max_edge_risk[endpoint] = (gnn_mal_risk, cand_cls)
                else:
                    for endpoint in ((src_type, s_i), (dst_type, d_i)):
                        prev_risk, prev_cls = node_max_edge_risk.get(endpoint, (-1.0, 0))
                        if prev_cls == 0 and gnn_mal_risk > prev_risk:
                            node_max_edge_risk[endpoint] = (gnn_mal_risk, cand_cls)

                src_uid = str(src_ids[s_i])
                dst_uid = str(dst_ids[d_i])
                edge_id = f"{src_uid}->{rel_name}->{dst_uid}#{idx}"
                raw_edge_records.append(
                    {
                        "id": edge_id,
                        "source": src_uid,
                        "target": dst_uid,
                        "type": str(rel_name),
                        "count": int(counts[idx]),
                        "cand_cls": int(cand_cls),
                        "edge_conf": float(edge_conf),
                    }
                )

        raw_node_records: List[Dict[str, object]] = []
        for ntype in NODE_TYPES:
            node_ids: List[str] = list(data[ntype].node_ids)
            for i, node_uid in enumerate(node_ids):
                recon_score = node_recon_by_key[(ntype, i)]
                p_val = node_pval_by_key[(ntype, i)]
                max_risk, max_cls = node_max_edge_risk.get((ntype, i), (0.0, 0))
                if include_extended_meta and max_cls == 0:
                    h_r, h_c = node_heuristic.get((ntype, i), (0.0, 0))
                    if h_c > 0:
                        max_risk, max_cls = max(max_risk, h_r), h_c

                if ntype == "socket":
                    ip_str = str(data["socket"].raw_ips[i])
                    port_num = int(data["socket"].raw_ports[i])
                    role_alias = self.get_ip_role_alias(ip_str)
                    display_label = f"{role_alias} ({ip_str}:{port_num})"
                else:
                    display_label = str(data[ntype].raw_labels[i])

                raw_node_records.append(
                    {
                        "id": str(node_uid),
                        "type": str(ntype),
                        "label": display_label,
                        "recon_score": float(recon_score),
                        "p_value": p_val,
                        "max_edge_risk": float(max_risk),
                        "cand_cls": int(max_cls),
                    }
                )

        serialized_nodes, serialized_edges = self._evaluate_records_with_cutoff(
            raw_node_records,
            raw_edge_records,
            cutoff=self.anomaly_threshold,
            include_extended_meta=include_extended_meta,
        )

        ts_val = float(getattr(data, "window_end", getattr(data, "window_start", time.time())))
        fisher_summary = self.compute_window_fisher_summary(node_pval_by_key)
        if include_extended_meta:
            self._cached_state = {
                "timestamp": ts_val,
                "nodes": raw_node_records,
                "edges": raw_edge_records,
                "fisher": fisher_summary,
            }

        out_payload: Dict[str, object] = {
            "timestamp": ts_val,
            "nodes": serialized_nodes,
            "edges": serialized_edges,
        }
        if include_extended_meta:
            out_payload["fisher"] = fisher_summary
        return out_payload

    def extract_attack_subgraph(
        self,
        nodes: List[Dict[str, object]],
        edges: List[Dict[str, object]],
        back_hops: int = 2,
        forward_hops: int = 1,
    ) -> Dict[str, object]:
        """Extract a 2-Hop causal backtracking Attack Summary Subgraph seeded by malicious nodes and edges."""
        seed_ids: Set[str] = {str(n["id"]) for n in nodes if bool(n.get("is_malicious"))}
        for e in edges:
            if bool(e.get("is_attack_path")):
                seed_ids.add(str(e["source"]))
                seed_ids.add(str(e["target"]))

        if not seed_ids:
            return {
                "has_attack": False,
                "seed_count": 0,
                "dominant_attack_class": "benign",
                "subgraph_node_ids": [],
                "subgraph_edges": [],
            }

        rev_adj: Dict[str, List[str]] = {}
        fwd_adj: Dict[str, List[str]] = {}
        for e in edges:
            s = str(e["source"])
            t = str(e["target"])
            rev_adj.setdefault(t, []).append(s)
            fwd_adj.setdefault(s, []).append(t)

        visited_back: Set[str] = set(seed_ids)
        frontier: Set[str] = set(seed_ids)
        for _ in range(back_hops):
            nxt: Set[str] = set()
            for u in frontier:
                for parent in rev_adj.get(u, []):
                    if parent not in visited_back:
                        visited_back.add(parent)
                        nxt.add(parent)
            frontier = nxt
            if not frontier:
                break

        visited_all: Set[str] = set(visited_back)
        fwd_frontier: Set[str] = set(seed_ids)
        for _ in range(forward_hops):
            nxt_f: Set[str] = set()
            for u in fwd_frontier:
                for child in fwd_adj.get(u, []):
                    if child not in visited_all:
                        visited_all.add(child)
                        nxt_f.add(child)
            fwd_frontier = nxt_f
            if not fwd_frontier:
                break

        sub_edges = [
            e for e in edges if str(e["source"]) in visited_all and str(e["target"]) in visited_all
        ]
        tactic_counter = Counter(
            str(e["attack_class"]) for e in sub_edges if str(e["attack_class"]) != "benign"
        )
        if not tactic_counter:
            tactic_counter = Counter(
                str(n["attack_class"])
                for n in nodes
                if str(n["id"]) in seed_ids and str(n["attack_class"]) != "benign"
            )
        dominant = tactic_counter.most_common(1)[0][0] if tactic_counter else "anomaly"

        return {
            "has_attack": True,
            "seed_count": len(seed_ids),
            "dominant_attack_class": dominant,
            "subgraph_node_ids": sorted(visited_all),
            "subgraph_edges": sub_edges[:120],
        }

    def _compute_delta_patch(
        self,
        nodes: List[Dict[str, object]],
        edges: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """Compute minimal WebSocket graph_delta against the previously emitted tick snapshot."""
        curr_nodes = {str(n["id"]): n for n in nodes}
        curr_edges = {str(e.get("id", f"{e['source']}->{e['type']}->{e['target']}")): e for e in edges}

        added_nodes: List[Dict[str, object]] = []
        updated_nodes: List[Dict[str, object]] = []
        for nid, nobj in curr_nodes.items():
            prev = self._prev_emitted_nodes.get(nid)
            if prev is None:
                added_nodes.append(nobj)
            elif (
                prev.get("is_malicious") != nobj.get("is_malicious")
                or prev.get("attack_class") != nobj.get("attack_class")
                or abs(float(prev.get("anomaly_score", 0.0)) - float(nobj.get("anomaly_score", 0.0))) >= 0.01
            ):
                updated_nodes.append(nobj)

        removed_node_ids = [nid for nid in self._prev_emitted_nodes if nid not in curr_nodes]

        added_edges: List[Dict[str, object]] = []
        updated_edges: List[Dict[str, object]] = []
        for eid, eobj in curr_edges.items():
            prev_e = self._prev_emitted_edges.get(eid)
            if prev_e is None:
                added_edges.append(eobj)
            elif (
                prev_e.get("is_attack_path") != eobj.get("is_attack_path")
                or prev_e.get("count") != eobj.get("count")
                or prev_e.get("attack_class") != eobj.get("attack_class")
            ):
                updated_edges.append(eobj)

        removed_edge_ids = [eid for eid in self._prev_emitted_edges if eid not in curr_edges]

        self._prev_emitted_nodes = curr_nodes
        self._prev_emitted_edges = curr_edges

        return {
            "added_nodes": added_nodes,
            "updated_nodes": updated_nodes,
            "removed_node_ids": removed_node_ids,
            "added_edges": added_edges,
            "updated_edges": updated_edges,
            "removed_edge_ids": removed_edge_ids,
        }

    def run_streaming_tick(self) -> Optional[Dict[str, object]]:
        """Materialize active 180s window, run GNN + 2-Hop subgraph + delta patch under 250ms budget."""
        if self._window_builder is None:
            raise RuntimeError("HeterogeneousFeatureEncoder is required for run_streaming_tick().")

        t0 = time.perf_counter()
        with self._lock:
            if not self._window_edges:
                return None
            edges_snapshot = list(self._window_edges)
            raw_ingested = self.online_raw_ingested
            cpr_merged = self.online_cpr_merged

        win_spec = WindowSpec(
            window_id=int(time.time()),
            start_ts=edges_snapshot[0].timestamp,
            end_ts=edges_snapshot[-1].timestamp,
            edges=edges_snapshot,
            is_clean_benign_window=False,
            stratify_label=0,
        )
        hetero_data = self._window_builder.build_window_heterodata(win_spec)
        if self.encoder is not None and self.encoder.feature_stats is not None:
            self.encoder.apply_feature_stats_to_graph(hetero_data)
        snapshot = self.predict_window(hetero_data, include_extended_meta=True)
        nodes: List[Dict[str, object]] = snapshot["nodes"]
        edges: List[Dict[str, object]] = snapshot["edges"]

        subgraph = self.extract_attack_subgraph(nodes, edges)
        delta = self._compute_delta_patch(nodes, edges)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        benign_nodes = [n for n in nodes if str(n["attack_class"]) == "benign"]
        est_fpr = (
            sum(1 for n in benign_nodes if bool(n["is_malicious"])) / len(benign_nodes)
            if benign_nodes
            else 0.0
        )
        cpr_ratio = (cpr_merged / max(1, raw_ingested)) * 100.0

        return {
            "timestamp": snapshot["timestamp"],
            "snapshot": snapshot,
            "graph_delta": delta,
            "attack_alert": subgraph,
            "telemetry": {
                "latency_ms": latency_ms,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "alpha": round(self.current_alpha, 4),
                "threshold": round(self.anomaly_threshold, 4),
                "estimated_fpr": round(est_fpr, 4),
                "online_cpr_pct": round(cpr_ratio, 2),
                "profile_env": self.profile_environment,
                "fisher": snapshot.get("fisher", {}),
            },
        }

    def update_threshold_zero_reinference(
        self,
        alpha: Optional[float] = None,
        direct_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        """Re-evaluate is_malicious and is_attack_path in O(N) (<1ms) without re-running GNN inference."""
        t0 = time.perf_counter()
        with self._lock:
            if alpha is not None:
                self.current_alpha = min(0.5, max(0.001, float(alpha)))
                self.anomaly_threshold = self.resolve_threshold_for_alpha(self.current_alpha)
            elif direct_threshold is not None:
                self.anomaly_threshold = min(0.9999, max(0.01, float(direct_threshold)))
                self.current_alpha = self.compute_conformal_pvalue(self.anomaly_threshold)

            cached = self._cached_state
            if cached is None:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                return {
                    "latency_ms": latency_ms,
                    "alpha": round(self.current_alpha, 4),
                    "threshold": round(self.anomaly_threshold, 4),
                    "estimated_fpr": 0.0,
                    "graph_delta": {
                        "added_nodes": [],
                        "updated_nodes": [],
                        "removed_node_ids": [],
                        "added_edges": [],
                        "updated_edges": [],
                        "removed_edge_ids": [],
                    },
                    "attack_alert": {
                        "has_attack": False,
                        "seed_count": 0,
                        "dominant_attack_class": "benign",
                        "subgraph_node_ids": [],
                        "subgraph_edges": [],
                    },
                }

            rebuilt_nodes, rebuilt_edges = self._evaluate_records_with_cutoff(
                cached["nodes"],
                cached["edges"],
                cutoff=self.anomaly_threshold,
                include_extended_meta=True,
            )
            subgraph = self.extract_attack_subgraph(rebuilt_nodes, rebuilt_edges)
            delta = self._compute_delta_patch(rebuilt_nodes, rebuilt_edges)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        benign_cand_nodes = [
            rn for rn in cached["nodes"] if int(rn["cand_cls"]) == 0
        ]
        if benign_cand_nodes:
            mal_ids = {str(n["id"]) for n in rebuilt_nodes if bool(n["is_malicious"])}
            est_fpr = sum(1 for rn in benign_cand_nodes if str(rn["id"]) in mal_ids) / len(benign_cand_nodes)
        else:
            est_fpr = 0.0

        return {
            "latency_ms": latency_ms,
            "alpha": round(self.current_alpha, 4),
            "threshold": round(self.anomaly_threshold, 4),
            "estimated_fpr": round(est_fpr, 4),
            "graph_delta": delta,
            "attack_alert": subgraph,
        }

    def to_threejs_json(self, data: HeteroData, indent: Optional[int] = 2) -> str:
        """Serialize window inference output into a standard Three.js JSON string."""
        payload = self.predict_window(data, include_extended_meta=False)
        return json.dumps(payload, indent=indent, ensure_ascii=False)

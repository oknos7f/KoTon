from dataclasses import replace
import logging
from typing import Dict, List, Tuple

from src.parser.streaming_parser import ProvenanceEdgeEvent, stable_sort_by_timestamp

logger = logging.getLogger(__name__)


class CausalityPreservingReducer:
    """Causality-Preserving Reduction (CPR) merging burst read/write edges without cross-process interference."""

    def __init__(self, max_delta_sec: float = 1.0) -> None:
        """Initialize CPR time threshold in seconds."""
        self.max_delta_sec = max_delta_sec

    def compress(self, events: List[ProvenanceEdgeEvent]) -> Tuple[List[ProvenanceEdgeEvent], float]:
        """Compress causally sorted provenance edges and return (reduced_edges, compression_ratio_pct)."""
        if not events:
            return [], 0.0

        sorted_events = stable_sort_by_timestamp(events)
        reduced: List[ProvenanceEdgeEvent] = []
        last_file_actor: Dict[Tuple, Tuple] = {}
        active_edge_idx: Dict[Tuple[Tuple, str, Tuple], int] = {}
        last_merged_ts: Dict[Tuple[Tuple, str, Tuple], float] = {}

        for ev in sorted_events:
            cloned = replace(ev)
            if cloned.edge_type in {"read", "write"} and cloned.dst_type == "file":
                triple_key = (cloned.src_key, cloned.edge_type, cloned.dst_key)
                prev_actor = last_file_actor.get(cloned.dst_key)

                if prev_actor is not None and prev_actor != cloned.src_key:
                    stale_keys = [k for k in active_edge_idx if k[2] == cloned.dst_key]
                    for sk in stale_keys:
                        active_edge_idx.pop(sk, None)
                        last_merged_ts.pop(sk, None)

                if prev_actor == cloned.src_key and triple_key in active_edge_idx:
                    idx = active_edge_idx[triple_key]
                    prev_ev = reduced[idx]
                    gap = cloned.timestamp - last_merged_ts[triple_key]
                    if 0.0 <= gap < self.max_delta_sec and prev_ev.tactic_id == cloned.tactic_id:
                        prev_ev.cpr_count += cloned.cpr_count
                        last_merged_ts[triple_key] = cloned.timestamp
                        last_file_actor[cloned.dst_key] = cloned.src_key
                        continue

                new_idx = len(reduced)
                reduced.append(cloned)
                active_edge_idx[triple_key] = new_idx
                last_merged_ts[triple_key] = cloned.timestamp
                last_file_actor[cloned.dst_key] = cloned.src_key
            else:
                reduced.append(cloned)

        last_src_ts: Dict[Tuple, float] = {}
        for edge in reduced:
            if edge.src_key in last_src_ts:
                dt = edge.timestamp - last_src_ts[edge.src_key]
                edge.delta_t = dt if dt >= 0.0 else 0.0
            else:
                edge.delta_t = 0.0
            last_src_ts[edge.src_key] = edge.timestamp

        raw_n = len(sorted_events)
        red_n = len(reduced)
        ratio_pct = (1.0 - (red_n / raw_n)) * 100.0 if raw_n > 0 else 0.0
        logger.info(
            "CPR compressed %d raw edges -> %d edges (compression rate: %.2f%%)",
            raw_n,
            red_n,
            ratio_pct,
        )
        return reduced, ratio_pct

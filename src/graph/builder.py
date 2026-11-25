from collections import Counter
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import HeteroData

from src.data.taxonomy import LabelTaxonomy
from src.graph.encoder import HeterogeneousFeatureEncoder
from src.parser.streaming_parser import ProvenanceEdgeEvent, stable_sort_by_timestamp

logger = logging.getLogger(__name__)

CANONICAL_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("process", "execute", "file"),
    ("process", "read", "file"),
    ("process", "write", "file"),
    ("process", "fork", "process"),
    ("process", "connect", "socket"),
]

REVERSE_EDGE_MAP: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {
    ("process", "execute", "file"): ("file", "rev_execute", "process"),
    ("process", "read", "file"): ("file", "rev_read", "process"),
    ("process", "write", "file"): ("file", "rev_write", "process"),
    ("process", "fork", "process"): ("process", "rev_fork", "process"),
    ("process", "connect", "socket"): ("socket", "rev_connect", "process"),
}


@dataclass
class WindowSpec:
    """Intermediate sliding-window edge slice prior to feature materialization."""

    window_id: int
    start_ts: float
    end_ts: float
    edges: List[ProvenanceEdgeEvent]
    is_clean_benign_window: bool
    stratify_label: int


def format_node_uid(node_type: str, key: Tuple) -> str:
    """Format a deterministic Three.js-compatible string node identifier from a typed tuple key."""
    if node_type == "process":
        host_id, pid, ver = key
        return f"{host_id}_proc_{pid}_v{ver}"
    if node_type == "file":
        host_id, path = key
        safe_path = path.strip("/").replace("/", "_").replace(".", "_") or "root"
        return f"{host_id}_file_{safe_path[:64]}"
    host_id, ip, port = key
    safe_ip = ip.replace(".", "_").replace(":", "_")
    return f"{host_id}_sock_{safe_ip}_{port}"


class SlidingWindowGraphBuilder:
    """Constructs snapshot-local HeteroData graphs with stratified train/val/test splitting."""

    def __init__(
        self,
        encoder: HeterogeneousFeatureEncoder,
        taxonomy: LabelTaxonomy,
        window_size_sec: float = 180.0,
        window_stride_sec: float = 60.0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Initialize sliding window parameters, feature encoder, and stratified split ratios."""
        self.encoder = encoder
        self.taxonomy = taxonomy
        self.window_size_sec = window_size_sec
        self.window_stride_sec = window_stride_sec
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def slice_windows(self, events: List[ProvenanceEdgeEvent]) -> List[WindowSpec]:
        """Slice causally sorted CPR edges into non-empty sliding windows with clean-benign flags."""
        if not events:
            return []

        sorted_events = stable_sort_by_timestamp(events)
        n_events = len(sorted_events)
        min_ts = sorted_events[0].timestamp
        max_ts = sorted_events[-1].timestamp

        windows: List[WindowSpec] = []
        left_idx = 0
        win_start = min_ts
        win_id = 0

        while win_start <= max_ts:
            win_end = win_start + self.window_size_sec
            while left_idx < n_events and sorted_events[left_idx].timestamp < win_start:
                left_idx += 1

            if left_idx >= n_events:
                break

            next_ts = sorted_events[left_idx].timestamp
            if next_ts >= win_end:
                jump_steps = int((next_ts - win_start) // self.window_stride_sec)
                win_start += max(1, jump_steps) * self.window_stride_sec
                continue

            right_idx = left_idx
            while right_idx < n_events and sorted_events[right_idx].timestamp < win_end:
                right_idx += 1

            slice_edges = sorted_events[left_idx:right_idx]
            if slice_edges:
                mal_tactics = [e.tactic_id for e in slice_edges if e.tactic_id > 0]
                has_malicious = len(mal_tactics) > 0
                phase_clean = (
                    self.taxonomy.is_clean_benign_timestamp(win_start)
                    and self.taxonomy.is_clean_benign_timestamp(win_end - 1e-3)
                    and all(self.taxonomy.is_clean_benign_timestamp(e.timestamp) for e in slice_edges)
                )
                is_clean_benign = bool(phase_clean and not has_malicious)
                if mal_tactics:
                    tactic_freq = Counter(mal_tactics)
                    rare_priority = (2, 4, 3, 1, 5)
                    strat_lbl = next((t for t in rare_priority if t in tactic_freq), mal_tactics[0])
                else:
                    strat_lbl = 0

                windows.append(
                    WindowSpec(
                        window_id=win_id,
                        start_ts=win_start,
                        end_ts=win_end,
                        edges=slice_edges,
                        is_clean_benign_window=is_clean_benign,
                        stratify_label=strat_lbl,
                    )
                )
                win_id += 1

            win_start += self.window_stride_sec

        return windows

    def stratified_split_windows(
        self,
        windows: List[WindowSpec],
    ) -> Tuple[List[WindowSpec], List[WindowSpec], List[WindowSpec]]:
        """Split chronologically ordered sliding windows per tactic into contiguous train/val/test blocks."""
        if len(windows) < 6:
            return windows, windows[:1], windows[-1:]

        by_class: Dict[int, List[int]] = {}
        for idx, win in enumerate(windows):
            by_class.setdefault(win.stratify_label, []).append(idx)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        total_ratio = max(1e-6, self.train_ratio + self.val_ratio + self.test_ratio)
        tr_frac = self.train_ratio / total_ratio
        va_frac = self.val_ratio / total_ratio

        for _, idx_list in sorted(by_class.items()):
            n = len(idx_list)
            if n == 1:
                train_indices.extend(idx_list)
            elif n == 2:
                train_indices.append(idx_list[0])
                test_indices.append(idx_list[1])
            else:
                n_val = max(1, int(round(n * va_frac)))
                n_test = max(1, int(round(n * (1.0 - tr_frac - va_frac))))
                if n_val + n_test >= n:
                    n_val = 1
                    n_test = 1
                n_train = n - n_val - n_test
                train_indices.extend(idx_list[:n_train])
                val_indices.extend(idx_list[n_train : n_train + n_val])
                test_indices.extend(idx_list[n_train + n_val :])

        train_list = [windows[i] for i in sorted(train_indices)]
        val_list = [windows[i] for i in sorted(val_indices)]
        test_list = [windows[i] for i in sorted(test_indices)]
        return train_list, val_list, test_list

    def build_window_heterodata(self, window: WindowSpec) -> HeteroData:
        """Materialize a single WindowSpec into a snapshot-local PyG HeteroData graph."""
        proc_index: Dict[Tuple, int] = {}
        file_index: Dict[Tuple, int] = {}
        sock_index: Dict[Tuple, int] = {}

        proc_records: List[Dict] = []
        file_records: List[Dict] = []
        sock_records: List[Dict] = []

        def ensure_proc(key: Tuple, exe: str, cmdline: str, uid: int) -> int:
            if key not in proc_index:
                proc_index[key] = len(proc_records)
                proc_records.append(
                    {
                        "key": key,
                        "exe": exe or "/sbin/init",
                        "cmdline": cmdline or exe or "/sbin/init",
                        "uid": uid,
                        "in_deg": 0,
                        "out_deg": 0,
                    }
                )
            else:
                rec = proc_records[proc_index[key]]
                if exe and rec["exe"] == "/sbin/init":
                    rec["exe"] = exe
                if cmdline and len(cmdline) > len(rec["cmdline"]):
                    rec["cmdline"] = cmdline
            return proc_index[key]

        def ensure_file(key: Tuple, path: str) -> int:
            if key not in file_index:
                file_index[key] = len(file_records)
                file_records.append(
                    {
                        "key": key,
                        "path": path or key[1],
                        "in_deg": 0,
                        "out_deg": 0,
                    }
                )
            return file_index[key]

        def ensure_sock(key: Tuple, ip: str, port: int) -> int:
            if key not in sock_index:
                sock_index[key] = len(sock_records)
                sock_records.append(
                    {
                        "key": key,
                        "ip": ip or key[1],
                        "port": int(port),
                        "in_deg": 0,
                        "out_deg": 0,
                        "conn_freq": 0,
                    }
                )
            return sock_index[key]

        edge_buckets: Dict[Tuple[str, str, str], Dict[str, List]] = {
            et: {"src": [], "dst": [], "attr": [], "y": [], "count": [], "fine_label": []}
            for et in CANONICAL_EDGE_TYPES
        }

        last_local_src_ts: Dict[Tuple, float] = {}

        for ev in stable_sort_by_timestamp(window.edges):
            s_idx = ensure_proc(ev.src_key, ev.proc_exe, ev.proc_cmdline, ev.proc_uid)
            proc_records[s_idx]["out_deg"] += 1

            if ev.dst_type == "file":
                d_idx = ensure_file(ev.dst_key, ev.file_path)
                file_records[d_idx]["in_deg"] += 1
            elif ev.dst_type == "socket":
                d_idx = ensure_sock(ev.dst_key, ev.sock_ip, ev.sock_port)
                sock_records[d_idx]["in_deg"] += 1
                sock_records[d_idx]["conn_freq"] += ev.cpr_count
            else:
                d_idx = ensure_proc(
                    ev.dst_key,
                    ev.dst_proc_exe or ev.proc_exe,
                    ev.dst_proc_cmdline or ev.proc_cmdline,
                    ev.dst_proc_uid,
                )
                proc_records[d_idx]["in_deg"] += 1

            if ev.src_key in last_local_src_ts:
                dt = ev.timestamp - last_local_src_ts[ev.src_key]
                if dt < 0.0:
                    dt = 0.0
            else:
                dt = max(0.0, ev.delta_t)
            last_local_src_ts[ev.src_key] = ev.timestamp

            triplet = (ev.src_type, ev.edge_type, ev.dst_type)
            bucket = edge_buckets[triplet]
            bucket["src"].append(s_idx)
            bucket["dst"].append(d_idx)
            bucket["attr"].append([math.log1p(dt), math.log1p(ev.cpr_count)])
            bucket["y"].append(int(ev.tactic_id))
            bucket["count"].append(int(ev.cpr_count))
            bucket["fine_label"].append(ev.fine_label)

        data = HeteroData()
        data["process"].x = torch.from_numpy(self.encoder.encode_process_nodes(proc_records))
        data["file"].x = torch.from_numpy(self.encoder.encode_file_nodes(file_records))
        data["socket"].x = torch.from_numpy(self.encoder.encode_socket_nodes(sock_records))

        data["process"].node_ids = [format_node_uid("process", r["key"]) for r in proc_records]
        data["process"].raw_labels = [f"{r['exe']}" for r in proc_records]
        data["process"].host_ids = [str(r["key"][0]) for r in proc_records]

        data["file"].node_ids = [format_node_uid("file", r["key"]) for r in file_records]
        data["file"].raw_labels = [str(r["path"]) for r in file_records]
        data["file"].host_ids = [str(r["key"][0]) for r in file_records]

        data["socket"].node_ids = [format_node_uid("socket", r["key"]) for r in sock_records]
        data["socket"].raw_ips = [str(r["ip"]) for r in sock_records]
        data["socket"].raw_ports = [int(r["port"]) for r in sock_records]
        data["socket"].host_ids = [str(r["key"][0]) for r in sock_records]

        for triplet in CANONICAL_EDGE_TYPES:
            bucket = edge_buckets[triplet]
            rev_triplet = REVERSE_EDGE_MAP[triplet]
            if bucket["src"]:
                edge_index = torch.tensor([bucket["src"], bucket["dst"]], dtype=torch.long)
                edge_attr = torch.tensor(bucket["attr"], dtype=torch.float32)
                edge_y = torch.tensor(bucket["y"], dtype=torch.long)
                edge_count = torch.tensor(bucket["count"], dtype=torch.long)
                rev_edge_index = torch.tensor([bucket["dst"], bucket["src"]], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 2), dtype=torch.float32)
                edge_y = torch.zeros((0,), dtype=torch.long)
                edge_count = torch.zeros((0,), dtype=torch.long)
                rev_edge_index = torch.zeros((2, 0), dtype=torch.long)

            data[triplet].edge_index = edge_index
            data[triplet].edge_attr = edge_attr
            data[triplet].edge_y = edge_y
            data[triplet].cpr_count = edge_count

            data[rev_triplet].edge_index = rev_edge_index
            data[rev_triplet].edge_attr = edge_attr.clone()

        data.window_id = int(window.window_id)
        data.window_start = float(window.start_ts)
        data.window_end = float(window.end_ts)
        data.is_clean_benign_window = torch.tensor([window.is_clean_benign_window], dtype=torch.bool)
        data.window_tactic = torch.tensor([window.stratify_label], dtype=torch.long)
        return data

    def build_and_save_dataset(
        self,
        cpr_edges: List[ProvenanceEdgeEvent],
        output_dir: str | Path,
        embedder_artifact_path: str | Path,
        cpr_ratio_pct: float = 0.0,
        fasttext_min_count: int = 1,
        fasttext_window: int = 5,
        fasttext_epochs: int = 10,
    ) -> Dict[str, List[HeteroData]]:
        """Slice windows, fit FastText on train split only, materialize HeteroData splits, and save .pt files."""
        windows = self.slice_windows(cpr_edges)
        train_specs, val_specs, test_specs = self.stratified_split_windows(windows)

        train_cmds: List[str] = []
        for w in train_specs:
            for e in w.edges:
                if e.proc_cmdline:
                    train_cmds.append(e.proc_cmdline)
                if e.dst_proc_cmdline:
                    train_cmds.append(e.dst_proc_cmdline)

        self.encoder.fit_cmdline_embedder(
            train_cmds,
            min_count=fasttext_min_count,
            window=fasttext_window,
            epochs=fasttext_epochs,
        )
        self.encoder.save_cmdline_embedder(embedder_artifact_path)

        splits = {
            "train": [self.build_window_heterodata(w) for w in train_specs],
            "val": [self.build_window_heterodata(w) for w in val_specs],
            "test": [self.build_window_heterodata(w) for w in test_specs],
        }

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for split_name, graphs in splits.items():
            torch.save(graphs, out_path / f"{split_name}.pt")

        meta = {
            "scenario": self.taxonomy.scenario,
            "num_cpr_edges": len(cpr_edges),
            "cpr_compression_pct": round(cpr_ratio_pct, 4),
            "num_windows_total": len(windows),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "out_of_phase_jsonl_fallbacks": self.taxonomy.out_of_phase_fallback_count,
        }
        with open(out_path / "metadata.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

        logger.info("Saved processed dataset splits to %s: %s", out_path, meta["split_sizes"])
        return splits

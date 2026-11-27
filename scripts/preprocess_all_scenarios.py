"""Preprocess and serialize all 7 AIT-LDS v2.0 scenarios (read-only dataset/ integrity preserved).

Produces per scenario under processed/<scenario>/:
  - train.pt, val.pt, test.pt (scenario-local HeteroData splits)
  - windows_raw.pt (CPR-compressed WindowSpec list for Leakage-Free 7-Fold LOOCV)
  - metadata.json (edge counts, CPR compression, tactic & false_positive_test stats)
"""

from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.data.taxonomy import LabelTaxonomy
from src.graph.builder import CANONICAL_EDGE_TYPES, SlidingWindowGraphBuilder
from src.graph.cpr import CausalityPreservingReducer
from src.graph.encoder import HeterogeneousFeatureEncoder
from src.graph.sanitizer import PathSanitizer
from src.parser.streaming_parser import StatefulAuditParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_all")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def snapshot_scenario_readonly(scenario_dir: Path) -> Dict[str, Tuple[int, str]]:
    """Lightweight read-only integrity manifest of scenario directory top-level & metadata."""
    manifest: Dict[str, Tuple[int, str]] = {}
    top_entries = sorted(p.relative_to(scenario_dir).as_posix() for p in scenario_dir.glob("*"))
    manifest["__entries__"] = (
        len(top_entries),
        hashlib.sha256("\n".join(top_entries).encode()).hexdigest(),
    )
    ds_yaml = scenario_dir / "dataset.yaml"
    if ds_yaml.is_file():
        manifest["dataset.yaml"] = (
            ds_yaml.stat().st_size,
            hashlib.sha256(ds_yaml.read_bytes()).hexdigest(),
        )
    return manifest


def preprocess_scenario(
    scenario: str,
    cfg: Dict,
    sanitizer: PathSanitizer,
    force: bool = False,
) -> Dict[str, object]:
    """Parse, CPR-compress, slice windows, and serialize a single AIT-LDS v2.0 scenario."""
    t0 = time.perf_counter()
    seed = int(cfg["project"]["seed"])
    scenario_dir = PROJECT_ROOT / cfg["paths"]["data_dir"] / scenario
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"] / scenario
    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]

    assert scenario_dir.is_dir(), f"Missing scenario directory: {scenario_dir}"

    raw_win_path = processed_dir / "windows_raw.pt"
    meta_path = processed_dir / "metadata.json"
    if (
        not force
        and raw_win_path.is_file()
        and (processed_dir / "train.pt").is_file()
        and (processed_dir / "val.pt").is_file()
        and (processed_dir / "test.pt").is_file()
        and meta_path.is_file()
    ):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "false_positive_test_windows" in meta:
            logger.info("[%s] Already preprocessed with windows_raw.pt; skipping.", scenario)
            return meta

    before_manifest = snapshot_scenario_readonly(scenario_dir)
    taxonomy = LabelTaxonomy(PROJECT_ROOT / "configs" / "taxonomy.yaml", scenario=scenario)
    parser = StatefulAuditParser(
        chunk_size=int(cfg["parser"]["chunk_size"]),
        carryover_sec=float(cfg["parser"]["carryover_sec"]),
        include_supplementary_logs=bool(cfg["parser"]["include_supplementary_logs"]),
        apache_dirb_sample_stride=int(cfg["parser"]["apache_dirb_sample_stride"]),
        dnsmasq_benign_sample_stride=int(cfg["parser"]["dnsmasq_benign_sample_stride"]),
    )

    logger.info("[%s] Parsing auditd + supplementary host logs from %s ...", scenario, scenario_dir)
    raw_edges = parser.parse_scenario(scenario_dir, taxonomy)
    assert len(raw_edges) > 0, f"[{scenario}] Parsed zero edges!"

    reducer = CausalityPreservingReducer(max_delta_sec=float(cfg["graph"]["cpr_delta_t_sec"]))
    cpr_edges, cpr_ratio_pct = reducer.compress(raw_edges)

    encoder = HeterogeneousFeatureEncoder(
        sanitizer=sanitizer,
        proc_exe_dim=int(cfg["features"]["process_exe_hfh_dim"]),
        proc_cmd_dim=int(cfg["features"]["process_cmd_emb_dim"]),
        file_path_dim=int(cfg["features"]["file_path_hfh_dim"]),
        sock_ip_dim=int(cfg["features"]["socket_ip_hfh_dim"]),
        sensitive_prefixes=cfg["features"]["sensitive_prefixes"],
        seed=seed,
    )
    builder = SlidingWindowGraphBuilder(
        encoder=encoder,
        taxonomy=taxonomy,
        window_size_sec=float(cfg["graph"]["window_size_sec"]),
        window_stride_sec=float(cfg["graph"]["window_stride_sec"]),
        train_ratio=float(cfg["graph"]["train_ratio"]),
        val_ratio=float(cfg["graph"]["val_ratio"]),
        test_ratio=float(cfg["graph"]["test_ratio"]),
        seed=seed,
    )

    windows = builder.slice_windows(cpr_edges)
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(windows, raw_win_path)

    splits = builder.build_and_save_dataset(
        cpr_edges=cpr_edges,
        output_dir=processed_dir,
        embedder_artifact_path=artifacts_dir / f"{scenario}_cmd_fasttext.model",
        cpr_ratio_pct=cpr_ratio_pct,
        fasttext_min_count=int(cfg["features"]["fasttext_min_count"]),
        fasttext_window=int(cfg["features"]["fasttext_window"]),
        fasttext_epochs=int(cfg["features"]["fasttext_epochs"]),
    )

    after_manifest = snapshot_scenario_readonly(scenario_dir)
    assert before_manifest == after_manifest, f"Read-only violation detected on {scenario_dir}!"

    edge_class_counts: Counter = Counter()
    edge_rel_counts: Counter = Counter()
    clean_count = 0
    fp_test_count = 0
    for split_graphs in splits.values():
        for g in split_graphs:
            if bool(g.is_clean_benign_window.item()):
                clean_count += 1
            if hasattr(g, "is_false_positive_test") and bool(g.is_false_positive_test.item()):
                fp_test_count += 1
            for et in CANONICAL_EDGE_TYPES:
                ey = g[et].edge_y
                if ey.numel() > 0:
                    edge_class_counts.update(ey.tolist())
                    edge_rel_counts[et[1]] += int(ey.numel())

    elapsed = round(time.perf_counter() - t0, 2)
    meta = {
        "scenario": scenario,
        "num_raw_edges": len(raw_edges),
        "num_cpr_edges": len(cpr_edges),
        "cpr_compression_pct": round(cpr_ratio_pct, 4),
        "num_windows_total": len(windows),
        "clean_benign_windows": clean_count,
        "false_positive_test_windows": fp_test_count,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "edge_class_counts": {str(k): int(v) for k, v in sorted(edge_class_counts.items())},
        "edge_relation_counts": dict(sorted(edge_rel_counts.items())),
        "out_of_phase_jsonl_fallbacks": taxonomy.out_of_phase_fallback_count,
        "elapsed_sec": elapsed,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "[%s DONE in %.1fs] CPR=%.2f%% | Windows=%d (clean=%d, fp_test=%d) | Classes=%s | Rels=%s",
        scenario,
        elapsed,
        cpr_ratio_pct,
        len(windows),
        clean_count,
        fp_test_count,
        meta["edge_class_counts"],
        meta["edge_relation_counts"],
    )
    return meta


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Preprocess all 7 AIT-LDS v2.0 scenarios into processed/<scenario>/")
    ap.add_argument("--force", action="store_true", help="Re-process even if cached .pt files exist.")
    ap.add_argument("--scenarios", nargs="*", default=None, help="Optional subset of scenarios to preprocess.")
    args = ap.parse_args()

    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    set_all_seeds(int(cfg["project"]["seed"]))
    enabled: List[str] = args.scenarios or list(cfg["scenarios"]["enabled"])
    excluded: List[str] = list(cfg["scenarios"]["excluded"])
    for sc in enabled:
        assert sc not in excluded, f"Scenario '{sc}' is excluded!"

    sanitizer = PathSanitizer(scenarios=list(cfg["scenarios"]["enabled"]) + excluded)
    summary: Dict[str, object] = {}
    t_all = time.perf_counter()

    for sc in enabled:
        summary[sc] = preprocess_scenario(sc, cfg, sanitizer, force=args.force)

    summary_path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "all_scenarios_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "ALL %d SCENARIOS PREPROCESSED in %.1fs -> %s",
        len(enabled),
        time.perf_counter() - t_all,
        summary_path,
    )


if __name__ == "__main__":
    main()

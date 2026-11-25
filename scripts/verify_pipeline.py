from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.data.taxonomy import LabelTaxonomy
from src.graph.builder import CANONICAL_EDGE_TYPES, REVERSE_EDGE_MAP, SlidingWindowGraphBuilder
from src.graph.cpr import CausalityPreservingReducer
from src.graph.encoder import HeterogeneousFeatureEncoder
from src.graph.sanitizer import PathSanitizer
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN
from src.models.trainer import (
    KoTonTrainer,
    build_stratified_mini_trainset,
    compute_balanced_class_weights,
)
from src.parser.streaming_parser import ProvenanceEdgeEvent, StatefulAuditParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_pipeline")


def set_all_seeds(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducible verification."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_scenario_directory(scenario_dir: Path) -> Dict[str, Tuple[int, str]]:
    """Compute a read-only integrity manifest strictly inside dataset/<scenario>/ without touching sibling folders."""
    manifest: Dict[str, Tuple[int, str]] = {}
    top_entries = sorted(p.relative_to(scenario_dir).as_posix() for p in scenario_dir.glob("*"))
    manifest["__scenario_entries__"] = (
        len(top_entries),
        hashlib.sha256("\n".join(top_entries).encode()).hexdigest(),
    )

    tracked_patterns = (
        "dataset.yaml",
        "rules/*",
        "labels/**/*",
        "gather/*/logs/audit/*",
        "gather/*/logs/apache2/*",
        "gather/*/logs/dnsmasq.log",
        "gather/*/logs/auth.log",
        "gather/*/logs/openvpn.log",
        "gather/*/logs/logstash/**/*",
    )
    for pat in tracked_patterns:
        for p in sorted(scenario_dir.glob(pat)):
            if p.is_file():
                rel = p.relative_to(scenario_dir).as_posix()
                size = p.stat().st_size
                if size <= 5_000_000:
                    digest = hashlib.sha256(p.read_bytes()).hexdigest()
                else:
                    with open(p, "rb") as fp:
                        head = fp.read(65536)
                    digest = hashlib.sha256(head).hexdigest() + f":{size}"
                manifest[rel] = (size, digest)
    return manifest


def verify_cpr_causality_unit() -> None:
    """Verify CPR merges uninterrupted <1s read/write bursts and blocks merge upon third-party intervention."""
    reducer = CausalityPreservingReducer(max_delta_sec=1.0)
    proc_a = ("host0", 100, 1)
    proc_b = ("host0", 200, 1)
    target_file = ("host0", "/etc/passwd")

    burst_uninterrupted = [
        ProvenanceEdgeEvent(10.0, "host0", "read", "process", proc_a, "file", target_file),
        ProvenanceEdgeEvent(10.3, "host0", "read", "process", proc_a, "file", target_file),
        ProvenanceEdgeEvent(10.7, "host0", "read", "process", proc_a, "file", target_file),
    ]
    merged, _ = reducer.compress(burst_uninterrupted)
    assert len(merged) == 1 and merged[0].cpr_count == 3, "Uninterrupted <1s read burst must merge into cpr_count=3."

    burst_intervened = [
        ProvenanceEdgeEvent(20.0, "host0", "read", "process", proc_a, "file", target_file),
        ProvenanceEdgeEvent(20.2, "host0", "write", "process", proc_b, "file", target_file),
        ProvenanceEdgeEvent(20.4, "host0", "read", "process", proc_a, "file", target_file),
    ]
    unmerged, _ = reducer.compress(burst_intervened)
    assert len(unmerged) == 3, "Intervening read/write by another process must prevent CPR merging."


def verify_cross_host_and_pid_isolation_unit(taxonomy: LabelTaxonomy) -> None:
    """Verify StatefulAuditParser isolates identical PIDs across hosts and increments exec_version on execve."""
    parser = StatefulAuditParser()
    lines_host_a = [
        (1, 'type=SYSCALL msg=audit(1642205000.100:10): arch=c000003e syscall=59 success=yes pid=4242 ppid=1 uid=0 exe="/bin/bash" comm="bash"\n'),
        (2, 'type=EXECVE msg=audit(1642205000.100:10): argc=2 a0="/bin/bash" a1="-c"\n'),
        (3, 'type=EOE msg=audit(1642205000.100:10):\n'),
        (4, 'type=SYSCALL msg=audit(1642205001.200:11): arch=c000003e syscall=59 success=yes pid=4242 ppid=1 uid=0 exe="/usr/bin/id" comm="id"\n'),
        (5, 'type=EOE msg=audit(1642205001.200:11):\n'),
    ]
    lines_host_b = [
        (1, 'type=SYSCALL msg=audit(1642205000.100:10): arch=c000003e syscall=59 success=yes pid=4242 ppid=1 uid=0 exe="/bin/bash" comm="bash"\n'),
        (2, 'type=EOE msg=audit(1642205000.100:10):\n'),
    ]
    ev_a = parser._process_chunk("hostA", lines_host_a, {}, is_eof=True)
    ev_b = parser._process_chunk("hostB", lines_host_b, {}, is_eof=True)
    edges_a = [e for ev in ev_a for e in parser.audit_event_to_edges(ev, taxonomy) if e.edge_type == "execute"]
    edges_b = [e for ev in ev_b for e in parser.audit_event_to_edges(ev, taxonomy) if e.edge_type == "execute"]

    assert edges_a[0].src_key != edges_b[0].src_key, "Cross-host identical PIDs must have distinct keys."
    assert edges_a[0].src_key[2] < edges_a[1].src_key[2], "execve must increment process exec_version."


def main() -> None:
    """Execute full 6-stage end-to-end verification and Conformal calibration strictly on dataset/fox."""
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    seed = int(cfg["project"]["seed"])
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running verification on device=%s (CUDA available=%s)", device, torch.cuda.is_available())

    scenario = cfg["scenarios"]["default_scenario"]
    assert scenario == "fox", "Only 'fox' scenario may be referenced while other scenarios are downloading."
    scenario_dir = PROJECT_ROOT / cfg["paths"]["data_dir"] / scenario
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"] / scenario
    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    checkpoints_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]

    logger.info("[Check 1/6] Taking pre-run read-only snapshot of %s ...", scenario_dir)
    before_manifest = snapshot_scenario_directory(scenario_dir)

    logger.info("[Check 3/6] Verifying PathSanitizer boundary regex rules ...")
    sanitizer = PathSanitizer(scenarios=cfg["scenarios"]["enabled"] + cfg["scenarios"]["excluded"])
    s_fox = sanitizer.sanitize_path("/var/log/apache2/fox-access.log")
    s_host = sanitizer.sanitize_path("/var/www/intranet.price.fox.org/wp-login.php")
    s_firefox = sanitizer.sanitize_path("/usr/bin/firefox")
    assert s_fox == "/var/log/apache2/-access.log", f"Unexpected sanitized path: {s_fox}"
    assert "fox" not in s_host.lower(), f"Scenario token leaked in path: {s_host}"
    assert s_firefox == "/usr/bin/firefox", f"/usr/bin/firefox was corrupted: {s_firefox}"

    taxonomy = LabelTaxonomy(PROJECT_ROOT / "configs" / "taxonomy.yaml", scenario=scenario)
    verify_cpr_causality_unit()
    verify_cross_host_and_pid_isolation_unit(taxonomy)

    parser = StatefulAuditParser(
        chunk_size=int(cfg["parser"]["chunk_size"]),
        carryover_sec=float(cfg["parser"]["carryover_sec"]),
        include_supplementary_logs=bool(cfg["parser"]["include_supplementary_logs"]),
        apache_dirb_sample_stride=int(cfg["parser"]["apache_dirb_sample_stride"]),
        dnsmasq_benign_sample_stride=int(cfg["parser"]["dnsmasq_benign_sample_stride"]),
    )
    raw_edges = parser.parse_scenario(scenario_dir, taxonomy)
    assert len(raw_edges) > 0, "Parsed edge list is empty."

    for idx in range(1, len(raw_edges)):
        assert raw_edges[idx].timestamp >= raw_edges[idx - 1].timestamp, "Stable sort timestamp monotonicity violated."

    reducer = CausalityPreservingReducer(max_delta_sec=float(cfg["graph"]["cpr_delta_t_sec"]))
    cpr_edges, cpr_ratio_pct = reducer.compress(raw_edges)
    assert cpr_ratio_pct > 0.0, f"Expected positive CPR compression ratio, got {cpr_ratio_pct:.2f}%"
    assert all(e.delta_t >= 0.0 for e in cpr_edges), "Negative delta_t found on CPR edges."

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

    splits = builder.build_and_save_dataset(
        cpr_edges=cpr_edges,
        output_dir=processed_dir,
        embedder_artifact_path=artifacts_dir / f"{scenario}_cmd_fasttext.model",
        cpr_ratio_pct=cpr_ratio_pct,
        fasttext_min_count=int(cfg["features"]["fasttext_min_count"]),
        fasttext_window=int(cfg["features"]["fasttext_window"]),
        fasttext_epochs=int(cfg["features"]["fasttext_epochs"]),
    )

    after_manifest = snapshot_scenario_directory(scenario_dir)
    assert before_manifest == after_manifest, "CRITICAL VIOLATION: dataset/fox directory was modified!"
    for pt_name in ("train.pt", "val.pt", "test.pt", "metadata.json"):
        assert (processed_dir / pt_name).is_file(), f"Missing output file: {processed_dir / pt_name}"
    logger.info("[Check 1/6 PASSED] dataset/fox read-only integrity preserved; processed/%s/*.pt created.", scenario)

    for split_name in ("train", "val", "test"):
        split_tactics = {int(g.window_tactic.item()) for g in splits[split_name]}
        logger.info("Split '%s' (%d windows) tactics present: %s", split_name, len(splits[split_name]), sorted(split_tactics))
        assert split_tactics == {0, 1, 2, 3, 4, 5}, f"Missing tactics in {split_name}: {split_tactics}"

    all_graphs = splits["train"] + splits["val"] + splits["test"]
    expected_dims = {"process": 68, "file": 36, "socket": 20}
    observed_classes: Counter = Counter()
    clean_window_mal_edges = 0
    clean_window_count = 0

    for g in all_graphs:
        host_uid_map: Dict[str, str] = {}
        for ntype, exp_dim in expected_dims.items():
            x = g[ntype].x
            assert x.dim() == 2 and x.size(1) == exp_dim, f"Invalid {ntype} dim: {x.shape}"
            assert torch.isfinite(x).all(), f"NaN/Inf detected in {ntype}.x"
            for uid, hid in zip(g[ntype].node_ids, g[ntype].host_ids):
                if uid in host_uid_map:
                    assert host_uid_map[uid] == hid, f"Cross-host node ID collision: {uid}"
                host_uid_map[uid] = hid

        is_clean = bool(g.is_clean_benign_window.item())
        if is_clean:
            clean_window_count += 1

        for etype in CANONICAL_EDGE_TYPES:
            ea = g[etype].edge_attr
            ey = g[etype].edge_y
            assert ea.dim() == 2 and ea.size(1) == 2
            assert torch.isfinite(ea).all()
            if ea.size(0) > 0:
                assert (ea[:, 0] >= 0.0).all()
            observed_classes.update(ey.tolist())
            if is_clean and ey.numel() > 0:
                clean_window_mal_edges += int((ey > 0).sum().item())

        for rev_etype in REVERSE_EDGE_MAP.values():
            rev_ea = g[rev_etype].edge_attr
            assert rev_ea.dim() == 2 and rev_ea.size(1) == 2
            assert torch.isfinite(rev_ea).all()

    logger.info("[Check 2/6 PASSED] All delta_t >= 0, node/edge dims (68/36/20/2) exact, zero NaN/Inf.")
    mal_classes_found: Set[int] = {c for c in observed_classes if c > 0}
    assert mal_classes_found == {1, 2, 3, 4, 5}
    assert clean_window_count > 0 and clean_window_mal_edges == 0
    logger.info(
        "[Check 4/6 PASSED] CPR compression=%.2f%% | Edge classes=%s | Clean windows=%d (contamination=0).",
        cpr_ratio_pct,
        dict(sorted(observed_classes.items())),
        clean_window_count,
    )

    mini_train_set = build_stratified_mini_trainset(
        splits["train"],
        per_mal_tactic_cap=35,
        rare_oversample_min=8,
        clean_cap=120,
    )
    mini_edge_dist: Counter = Counter()
    for g in mini_train_set:
        for et in CANONICAL_EDGE_TYPES:
            mini_edge_dist.update(g[et].edge_y.tolist())
    logger.info("Balanced mini_train_set (%d windows) edge_y dist: %s", len(mini_train_set), dict(sorted(mini_edge_dist.items())))
    assert {1, 2, 3, 4, 5}.issubset(set(mini_edge_dist.keys())), "Mini-train set missing malicious tactic classes!"

    cls_weights = compute_balanced_class_weights(mini_train_set, num_classes=6, device=device)
    model = EdgeAwareMultiTaskHeteroGNN(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        gat_out_channels=int(cfg["model"]["gat_out_channels"]),
        gat_heads=int(cfg["model"]["gat_heads"]),
        num_layers=int(cfg["model"]["num_layers"]),
        edge_dim=int(cfg["model"]["edge_dim"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
        mask_rate=float(cfg["model"]["mask_rate"]),
        sce_gamma=float(cfg["model"]["sce_gamma"]),
        focal_gamma=float(cfg["model"]["focal_gamma"]),
        cls_loss_weight=float(cfg["model"]["cls_loss_weight"]),
        class_weights=cls_weights,
    )
    trainer = KoTonTrainer(
        model=model,
        lr=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
        device=device,
        seed=seed,
    )

    epoch_losses: List[float] = []
    for ep in range(1, 11):
        stats = trainer.train_epoch(mini_train_set, epoch_idx=ep)
        epoch_losses.append(stats["total_loss"])
        assert stats["mal_window_recon_leak"] == 0.0, "Strict Loss Masking violated!"
        logger.info(
            "Epoch %02d/10 | L_total=%.4f | L_recon(clean)=%.4f | L_cls(6-class)=%.4f | mal_recon_leak=%.4f",
            ep,
            stats["total_loss"],
            stats["recon_loss"],
            stats["cls_loss"],
            stats["mal_window_recon_leak"],
        )

    assert epoch_losses[-1] < epoch_losses[0], "Total loss did not converge!"

    val_clean = [g for g in splits["val"] if bool(g.is_clean_benign_window.item())][:150]
    profile = trainer.calibrate_conformal_profile(
        val_clean,
        quantile=float(cfg["model"]["anomaly_threshold_quantile"]),
        profile_output_path=artifacts_dir / "threshold_profile.json",
    )
    recon_scale = float(profile["recon_calibration_scale"])
    trainer.save_checkpoint(checkpoints_dir / f"{scenario}_hetero_gnn.pt")

    val_metrics = trainer.evaluate_graphs(splits["val"], max_clean_eval=100)
    test_metrics = trainer.evaluate_graphs(splits["test"], max_clean_eval=100)
    logger.info("Val Metrics  | Macro-F1=%.4f | Weighted-F1=%.4f | Per-Class Recall=%s", val_metrics["macro_f1"], val_metrics["weighted_f1"], val_metrics["per_class_recall"])
    logger.info("Test Metrics | Macro-F1=%.4f | Weighted-F1=%.4f | Per-Class Recall=%s", test_metrics["macro_f1"], test_metrics["weighted_f1"], test_metrics["per_class_recall"])
    logger.info("[Check 5/6 PASSED] 10-Epoch 6-Class training + Conformal threshold_profile.json saved (recon_scale=%.4f).", recon_scale)

    from src.serving.engine import KoTonInferenceEngine

    engine = KoTonInferenceEngine(
        model=model,
        taxonomy=taxonomy,
        recon_scale=recon_scale,
        anomaly_threshold=float(profile["default_threshold"]),
        device=device,
    )

    valid_tactic_names = set(taxonomy.tactic_classes.values())
    target_demo_graph = next(
        (
            g
            for g in splits["test"]
            if g["socket"].x.size(0) > 0
            and any((g[et].edge_y > 0).any().item() for et in CANONICAL_EDGE_TYPES if g[et].edge_y.numel() > 0)
        ),
        splits["test"][0],
    )

    json_str = engine.to_threejs_json(target_demo_graph, indent=2)
    parsed_json = json.loads(json_str)
    assert {"timestamp", "nodes", "edges"}.issubset(set(parsed_json.keys()))
    assert len(parsed_json["nodes"]) > 0 and len(parsed_json["edges"]) > 0
    for node_obj in parsed_json["nodes"]:
        assert node_obj["attack_class"] in valid_tactic_names
    for edge_obj in parsed_json["edges"]:
        assert edge_obj["attack_class"] in valid_tactic_names

    sample_json_path = artifacts_dir / "sample_threejs_graph.json"
    sample_json_path.write_text(json_str, encoding="utf-8")
    logger.info("[Check 6/6 PASSED] Three.js JSON verified -> %s. ALL 6 CHECKS PASSED!", sample_json_path)


if __name__ == "__main__":
    main()

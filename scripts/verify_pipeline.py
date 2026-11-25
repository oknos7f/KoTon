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
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN, NODE_TYPES
from src.models.trainer import KoTonTrainer, compute_balanced_class_weights
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


def snapshot_data_directory(data_dir: Path) -> Dict[str, Tuple[int, str]]:
    """Compute a read-only integrity manifest over scenario metadata, audit logs, labels, and rules in data/."""
    manifest: Dict[str, Tuple[int, str]] = {}
    top_entries = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.glob("*/*"))
    manifest["__top_entries__"] = (len(top_entries), hashlib.sha256("\n".join(top_entries).encode()).hexdigest())

    tracked_patterns = (
        "*/dataset.yaml",
        "*/rules/*",
        "*/labels/**/*",
        "*/gather/*/logs/audit/*",
        "*/gather/*/logs/apache2/*",
        "*/gather/*/logs/dnsmasq.log",
        "*/gather/*/logs/auth.log",
        "*/gather/*/logs/openvpn.log",
        "*/gather/*/logs/logstash/**/*",
    )
    for pat in tracked_patterns:
        for p in sorted(data_dir.glob(pat)):
            if p.is_file():
                rel = p.relative_to(data_dir).as_posix()
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
    """Execute full 6-stage end-to-end verification of the KoTon core AI pipeline."""
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    seed = int(cfg["project"]["seed"])
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running verification on device=%s (CUDA available=%s)", device, torch.cuda.is_available())

    scenario = cfg["scenarios"]["default_scenario"]
    data_root = PROJECT_ROOT / cfg["paths"]["data_dir"]
    scenario_dir = data_root / scenario
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"] / scenario
    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    checkpoints_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]

    logger.info("[Check 1/6] Taking pre-run read-only snapshot of %s ...", data_root)
    before_manifest = snapshot_data_directory(data_root)

    logger.info("[Check 3/6] Verifying PathSanitizer boundary regex rules ...")
    sanitizer = PathSanitizer(scenarios=cfg["scenarios"]["enabled"] + cfg["scenarios"]["excluded"])
    s_fox = sanitizer.sanitize_path("/var/log/apache2/fox-access.log")
    s_host = sanitizer.sanitize_path("/var/www/intranet.price.fox.org/wp-login.php")
    s_firefox = sanitizer.sanitize_path("/usr/bin/firefox")
    assert s_fox == "/var/log/apache2/-access.log", f"Unexpected sanitized path: {s_fox}"
    assert "fox" not in s_host.lower(), f"Scenario token leaked in path: {s_host}"
    assert s_firefox == "/usr/bin/firefox", f"/usr/bin/firefox was corrupted: {s_firefox}"
    logger.info("PathSanitizer verified: fox-access.log -> %s | /usr/bin/firefox -> %s", s_fox, s_firefox)

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

    after_manifest = snapshot_data_directory(data_root)
    assert before_manifest == after_manifest, "CRITICAL VIOLATION: data/ directory was modified!"
    for pt_name in ("train.pt", "val.pt", "test.pt", "metadata.json"):
        assert (processed_dir / pt_name).is_file(), f"Missing output file: {processed_dir / pt_name}"
    logger.info("[Check 1/6 PASSED] data/ read-only integrity preserved; processed/%s/*.pt created.", scenario)

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
            assert ea.dim() == 2 and ea.size(1) == 2, f"Invalid edge_attr shape on {etype}: {ea.shape}"
            assert torch.isfinite(ea).all(), f"NaN/Inf detected in {etype}.edge_attr"
            if ea.size(0) > 0:
                assert (ea[:, 0] >= 0.0).all(), f"Negative log1p(delta_t) in {etype}"
            observed_classes.update(ey.tolist())
            if is_clean and ey.numel() > 0:
                clean_window_mal_edges += int((ey > 0).sum().item())

        for rev_etype in REVERSE_EDGE_MAP.values():
            rev_ea = g[rev_etype].edge_attr
            assert rev_ea.dim() == 2 and rev_ea.size(1) == 2
            assert torch.isfinite(rev_ea).all()

    logger.info("[Check 2/6 PASSED] All delta_t >= 0, node/edge dims (68/36/20/2) exact, zero NaN/Inf, zero host/PID collisions.")

    mal_classes_found: Set[int] = {c for c in observed_classes if c > 0}
    assert mal_classes_found == {1, 2, 3, 4, 5}, f"Expected all malicious classes 1..5, got {mal_classes_found}"
    assert clean_window_count > 0, "No clean benign windows found."
    assert clean_window_mal_edges == 0, f"Contamination in clean_benign_window: {clean_window_mal_edges} malicious edges!"
    logger.info(
        "[Check 4/6 PASSED] CPR compression=%.2f%% | Edge classes=%s | Clean windows=%d (malicious contamination=%d).",
        cpr_ratio_pct,
        dict(sorted(observed_classes.items())),
        clean_window_count,
        clean_window_mal_edges,
    )

    mal_train = [g for g in splits["train"] if not bool(g.is_clean_benign_window.item())]
    clean_train = [g for g in splits["train"] if bool(g.is_clean_benign_window.item())]
    sampled_mal = mal_train[: min(len(mal_train), 120)]
    sampled_clean = clean_train[: min(len(clean_train), max(60, len(sampled_mal)))]
    mini_train_set = sampled_mal + sampled_clean

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
        assert stats["mal_window_recon_leak"] == 0.0, "Strict Loss Masking violated on malicious window!"
        logger.info(
            "Epoch %02d/10 | L_total=%.4f | L_recon(clean)=%.4f | L_cls=%.4f | mal_recon_leak=%.4f",
            ep,
            stats["total_loss"],
            stats["recon_loss"],
            stats["cls_loss"],
            stats["mal_window_recon_leak"],
        )

    assert epoch_losses[-1] < epoch_losses[0], (
        f"Total loss did not converge: epoch1={epoch_losses[0]:.4f} -> epoch10={epoch_losses[-1]:.4f}"
    )
    recon_scale = trainer.calibrate_anomaly_scale(
        sampled_clean,
        quantile=float(cfg["model"]["anomaly_threshold_quantile"]),
    )
    trainer.save_checkpoint(checkpoints_dir / f"{scenario}_hetero_gnn.pt")
    logger.info(
        "[Check 5/6 PASSED] 10-Epoch L_total converged %.4f -> %.4f with Strict Loss Masking (recon_scale=%.4f).",
        epoch_losses[0],
        epoch_losses[-1],
        recon_scale,
    )

    from src.serving.engine import KoTonInferenceEngine

    engine = KoTonInferenceEngine(
        model=model,
        taxonomy=taxonomy,
        recon_scale=recon_scale,
        device=device,
    )

    valid_tactic_names = set(taxonomy.tactic_classes.values())
    target_demo_graph = None
    for g in all_graphs:
        has_sock = g["socket"].x.size(0) > 0
        has_attack = any((g[et].edge_y > 0).any().item() for et in CANONICAL_EDGE_TYPES if g[et].edge_y.numel() > 0)
        if has_sock and has_attack:
            target_demo_graph = g
            break
    if target_demo_graph is None:
        target_demo_graph = mal_train[0] if mal_train else all_graphs[0]

    json_str = engine.to_threejs_json(target_demo_graph, indent=2)
    parsed_json = json.loads(json_str)
    assert set(parsed_json.keys()) == {"timestamp", "nodes", "edges"}, "Invalid Three.js top-level keys."
    assert isinstance(parsed_json["timestamp"], float)
    assert len(parsed_json["nodes"]) > 0 and len(parsed_json["edges"]) > 0

    found_alias = False
    for node_obj in parsed_json["nodes"]:
        assert set(node_obj.keys()) == {"id", "type", "label", "anomaly_score", "attack_class", "is_malicious"}
        assert node_obj["attack_class"] in valid_tactic_names, f"Invalid node attack_class: {node_obj['attack_class']}"
        assert 0.0 <= float(node_obj["anomaly_score"]) <= 1.0
        if node_obj["type"] == "socket":
            if node_obj["label"].startswith("int_host_") or node_obj["label"].startswith("ext_host_"):
                found_alias = True

    for edge_obj in parsed_json["edges"]:
        assert set(edge_obj.keys()) == {"source", "target", "type", "count", "attack_class", "is_attack_path"}
        assert edge_obj["attack_class"] in valid_tactic_names, f"Invalid edge attack_class: {edge_obj['attack_class']}"

    assert found_alias, "Expected dynamic IP role alias (int_host_X or ext_host_X) in socket nodes."

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sample_json_path = artifacts_dir / "sample_threejs_graph.json"
    sample_json_path.write_text(json_str, encoding="utf-8")
    logger.info(
        "[Check 6/6 PASSED] Three.js JSON verified (%d nodes, %d edges) -> saved to %s.",
        len(parsed_json["nodes"]),
        len(parsed_json["edges"]),
        sample_json_path,
    )
    logger.info("ALL 6 END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

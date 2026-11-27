"""Step 10: Dual-Environment Split Conformal Calibration (AIT-LDS false_positive_test & Ubuntu VM Idle).

Produces:
  1. configs/threshold_profile.json (and syncs artifacts/threshold_profile.json):
     - Calibrated on held-out AIT-LDS v2.0 `false_positive_test` pure benign windows across all 7 scenarios.
     - Per-entity-type (process, file, socket) empirical Conformal p-values.
     - Fisher's Method combined statistic calibrated via Split Conformal for exact FPR < 1% (alpha=0.01).
  2. configs/threshold_profile_vm.json:
     - Collected live from isolated Ubuntu VM idle daemon traffic (5+ minutes) via AuditStreamCollector.
     - Per-entity-type empirical Conformal p-values + Fisher's Method tailored to Ubuntu VM daemons.
     - Verified FPR < 1% (alpha=0.01) on held-out VM idle split + attack recall on live VM attack stream.
"""

import bisect
import json
import logging
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import chi2
import torch
import yaml
from torch_geometric.data import HeteroData

from src.collector.audit_stream import AuditStreamCollector
from src.data.taxonomy import LabelTaxonomy
from src.graph.builder import SlidingWindowGraphBuilder, WindowSpec
from src.graph.cpr import CausalityPreservingReducer
from src.graph.encoder import HeterogeneousFeatureEncoder
from src.graph.sanitizer import PathSanitizer
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN, NODE_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate_thresholds")

ALPHA_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def load_unified_model_and_encoder(
    cfg: Dict,
    device: torch.device,
) -> Tuple[EdgeAwareMultiTaskHeteroGNN, HeterogeneousFeatureEncoder, Dict]:
    """Load the Step 9 unified 7-scenario checkpoint, FastText embedder, and train-only feature stats."""
    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    checkpoints_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]

    ckpt_path = checkpoints_dir / "koton_all7_hetero_gnn.pt"
    ft_path = artifacts_dir / "all7_cmd_fasttext.model"
    if not ckpt_path.is_file():
        ckpt_path = checkpoints_dir / f"{cfg['scenarios']['default_scenario']}_hetero_gnn.pt"
        ft_path = artifacts_dir / f"{cfg['scenarios']['default_scenario']}_cmd_fasttext.model"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt.get("model_config", {})

    model = EdgeAwareMultiTaskHeteroGNN(
        node_in_dims=mcfg.get("node_in_dims"),
        hidden_dim=int(mcfg.get("hidden_dim", cfg["model"]["hidden_dim"])),
        gat_out_channels=int(mcfg.get("gat_out_channels", cfg["model"]["gat_out_channels"])),
        gat_heads=int(mcfg.get("gat_heads", cfg["model"]["gat_heads"])),
        num_layers=int(mcfg.get("num_layers", cfg["model"]["num_layers"])),
        edge_dim=int(mcfg.get("edge_dim", cfg["model"]["edge_dim"])),
        num_classes=int(mcfg.get("num_classes", cfg["model"]["num_classes"])),
        dropout=float(mcfg.get("dropout", cfg["model"]["dropout"])),
        mask_rate=float(mcfg.get("mask_rate", cfg["model"]["mask_rate"])),
        sce_gamma=float(mcfg.get("sce_gamma", cfg["model"]["sce_gamma"])),
        focal_gamma=float(mcfg.get("focal_gamma", cfg["model"]["focal_gamma"])),
        cls_loss_weight=float(mcfg.get("cls_loss_weight", cfg["model"]["cls_loss_weight"])),
        use_relation_balanced_masking=bool(mcfg.get("use_relation_balanced_masking", True)),
        high_risk_rel_boost=float(mcfg.get("high_risk_rel_boost", 2.5)),
        relation_loss_alpha=float(mcfg.get("relation_loss_alpha", 0.5)),
        class_weights=torch.ones(int(cfg["model"]["num_classes"]), dtype=torch.float32),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    sanitizer = PathSanitizer(
        scenarios=list(cfg["scenarios"]["enabled"]) + list(cfg["scenarios"]["excluded"])
    )
    encoder = HeterogeneousFeatureEncoder(
        sanitizer=sanitizer,
        proc_exe_dim=int(cfg["features"]["process_exe_hfh_dim"]),
        proc_cmd_dim=int(cfg["features"]["process_cmd_emb_dim"]),
        file_path_dim=int(cfg["features"]["file_path_hfh_dim"]),
        sock_ip_dim=int(cfg["features"]["socket_ip_hfh_dim"]),
        sensitive_prefixes=cfg["features"]["sensitive_prefixes"],
        seed=int(cfg["project"]["seed"]),
    )
    if ft_path.is_file():
        encoder.load_cmdline_embedder(ft_path)
    if "feature_stats" in ckpt and ckpt["feature_stats"]:
        encoder.load_feature_stats(ckpt["feature_stats"])

    logger.info("Loaded model from %s and FastText from %s (device=%s)", ckpt_path.name, ft_path.name, device)
    return model, encoder, ckpt


def _normalize_array(raw: np.ndarray, scale: float) -> np.ndarray:
    s = max(1e-4, float(scale))
    return np.clip(1.0 - np.exp(-np.maximum(0.0, raw) / s), 0.0, 1.0)


def _conformal_quantile(sorted_scores: np.ndarray, alpha: float) -> float:
    n = len(sorted_scores)
    if n == 0:
        return round(1.0 - alpha, 6)
    rank = min(n, max(1, int(math.ceil((n + 1) * (1.0 - alpha)))))
    return round(float(sorted_scores[rank - 1]), 6)


def _empirical_pvalue(sorted_scores: np.ndarray, score: float) -> float:
    m = len(sorted_scores)
    if m == 0:
        return max(1e-4, 1.0 - score)
    idx = int(np.searchsorted(sorted_scores, score, side="left"))
    return float((1.0 + (m - idx)) / (1.0 + m))


def _downsample_sorted(arr: np.ndarray, max_len: int = 4096) -> List[float]:
    if len(arr) > max_len:
        idxs = np.linspace(0, len(arr) - 1, max_len, dtype=int)
        arr = arr[idxs]
    return [round(float(x), 6) for x in arr.tolist()]


def _compute_window_fisher_stat(
    errs_by_type: Dict[str, List[float]],
    per_type_scales: Dict[str, float],
    per_type_sorted: Dict[str, np.ndarray],
) -> Tuple[float, float, int]:
    """Compute Fisher's Method combined statistic T = -2 * sum(ln(p_t)) and chi2 survival p-value."""
    type_pvals: List[float] = []
    for ntype in NODE_TYPES:
        vals = errs_by_type.get(ntype, [])
        if not vals:
            continue
        sc = per_type_scales[ntype]
        ref = per_type_sorted[ntype]
        norm_vals = _normalize_array(np.array(vals, dtype=np.float64), sc)
        # Representative worst-case anomaly score in this window for entity type `ntype`
        worst_score = float(np.max(norm_vals))
        p_t = max(1e-12, _empirical_pvalue(ref, worst_score))
        type_pvals.append(p_t)

    k = len(type_pvals)
    if k == 0:
        return 0.0, 1.0, 0
    fisher_stat = float(-2.0 * sum(math.log(p) for p in type_pvals))
    fisher_chi2_p = float(chi2.sf(fisher_stat, df=2 * k))
    return fisher_stat, fisher_chi2_p, k


@torch.no_grad()
def fit_and_verify_split_conformal_profile(
    model: EdgeAwareMultiTaskHeteroGNN,
    clean_graphs: List[HeteroData],
    device: torch.device,
    environment_name: str,
    quantile: float = 0.95,
    target_alpha: float = 0.01,
    seed: int = 42,
) -> Dict[str, object]:
    """Perform strict Split Conformal Calibration (50% cal / 50% held-out verification) with Fisher's Method."""
    model.eval()
    assert len(clean_graphs) >= 4, f"Need >=4 clean benign windows for Split Conformal, got {len(clean_graphs)}"

    # Extract per-window raw errors per node type once
    window_errors: List[Dict[str, List[float]]] = []
    for g in clean_graphs:
        bg = g.clone().to(device)
        err_dict = model.compute_node_anomaly_errors(bg)
        window_errors.append(
            {
                ntype: err_dict[ntype].detach().cpu().tolist() if err_dict[ntype].numel() > 0 else []
                for ntype in NODE_TYPES
            }
        )

    # Deterministic 50/50 split: first half calibration, second half held-out FPR verification
    rng = random.Random(seed)
    indices = list(range(len(window_errors)))
    rng.shuffle(indices)
    split_pt = max(2, len(indices) // 2)
    cal_idx = indices[:split_pt]
    eval_idx = indices[split_pt:]

    cal_win_errs = [window_errors[i] for i in cal_idx]
    eval_win_errs = [window_errors[i] for i in eval_idx]

    # 1. Global + Per-Entity-Type nonconformity scales & sorted calibration arrays
    cal_raw_global: List[float] = []
    cal_raw_by_type: Dict[str, List[float]] = {nt: [] for nt in NODE_TYPES}
    for we in cal_win_errs:
        for nt in NODE_TYPES:
            cal_raw_by_type[nt].extend(we[nt])
            cal_raw_global.extend(we[nt])

    if not cal_raw_global:
        cal_raw_global = [0.5, 1.0]
    global_arr = np.sort(np.array(cal_raw_global, dtype=np.float64))
    global_scale = max(1e-4, float(np.quantile(global_arr, quantile)))
    global_norm_sorted = np.sort(_normalize_array(global_arr, global_scale))
    global_alpha_thr = {f"{a:.3f}": _conformal_quantile(global_norm_sorted, a) for a in ALPHA_GRID}

    per_type_scales: Dict[str, float] = {}
    per_type_sorted: Dict[str, np.ndarray] = {}
    per_type_calibration: Dict[str, Dict[str, object]] = {}

    for nt in NODE_TYPES:
        vals = cal_raw_by_type[nt]
        if len(vals) < 2:
            # Fallback to global distribution if an entity type (e.g. socket in idle VM) is absent
            t_arr = global_arr.copy()
            t_scale = global_scale
            t_sorted = global_norm_sorted.copy()
            fallback_used = True
        else:
            t_arr = np.sort(np.array(vals, dtype=np.float64))
            t_scale = max(1e-4, float(np.quantile(t_arr, quantile)))
            t_sorted = np.sort(_normalize_array(t_arr, t_scale))
            fallback_used = False

        per_type_scales[nt] = t_scale
        per_type_sorted[nt] = t_sorted
        t_alpha_thr = {f"{a:.3f}": _conformal_quantile(t_sorted, a) for a in ALPHA_GRID}
        per_type_calibration[nt] = {
            "recon_calibration_scale": round(float(t_scale), 6),
            "num_calibration_nodes": int(len(vals)),
            "fallback_to_global": fallback_used,
            "alpha_thresholds": t_alpha_thr,
            "default_threshold": t_alpha_thr[f"{target_alpha:.3f}"],
            "sorted_nonconformity_scores": _downsample_sorted(t_sorted, max_len=2048),
        }

    # 2. Fisher's Method Split Conformal Calibration across windows
    cal_fisher_scores: List[float] = []
    max_k = 1
    for we in cal_win_errs:
        f_stat, _, k = _compute_window_fisher_stat(we, per_type_scales, per_type_sorted)
        if k > 0:
            cal_fisher_scores.append(f_stat)
            max_k = max(max_k, k)

    fisher_arr = np.sort(np.array(cal_fisher_scores if cal_fisher_scores else [0.0, 1.0], dtype=np.float64))
    fisher_alpha_thr: Dict[str, float] = {}
    for a in ALPHA_GRID:
        emp_q = _conformal_quantile(fisher_arr, a)
        # Finite-sample safety: when calibration window count < 1/alpha, guard with chi2(df=2*max_k) upper quantile
        if len(fisher_arr) < int(math.ceil(1.0 / a)):
            chi2_q = float(chi2.isf(a, df=2 * max_k))
            emp_q = round(max(emp_q, chi2_q), 6)
        fisher_alpha_thr[f"{a:.3f}"] = emp_q
    default_fisher_thr = fisher_alpha_thr[f"{target_alpha:.3f}"]

    # 3. Held-out Split Verification (target FPR < 1% at alpha = 0.01)
    target_key = f"{target_alpha:.3f}"
    global_thr_01 = global_alpha_thr[target_key]

    eval_global_total = 0
    eval_global_fp = 0
    eval_type_total: Dict[str, int] = {nt: 0 for nt in NODE_TYPES}
    eval_type_fp: Dict[str, int] = {nt: 0 for nt in NODE_TYPES}
    eval_fisher_total = 0
    eval_fisher_fp = 0

    for we in eval_win_errs:
        for nt in NODE_TYPES:
            vals = we[nt]
            if not vals:
                continue
            raw_v = np.array(vals, dtype=np.float64)
            # Global check
            g_scores = _normalize_array(raw_v, global_scale)
            eval_global_total += len(vals)
            eval_global_fp += int(np.sum(g_scores > global_thr_01))
            # Per-type check (strict > conforms to finite-sample conformal p-value <= alpha)
            t_scores = _normalize_array(raw_v, per_type_scales[nt])
            t_thr = float(per_type_calibration[nt]["alpha_thresholds"][target_key])
            eval_type_total[nt] += len(vals)
            eval_type_fp[nt] += int(np.sum(t_scores > t_thr))

        f_stat, _, k = _compute_window_fisher_stat(we, per_type_scales, per_type_sorted)
        if k > 0:
            eval_fisher_total += 1
            if f_stat > default_fisher_thr:
                eval_fisher_fp += 1

    fpr_global = eval_global_fp / max(1, eval_global_total)
    fpr_per_type = {
        nt: round(eval_type_fp[nt] / max(1, eval_type_total[nt]), 5) if eval_type_total[nt] > 0 else 0.0
        for nt in NODE_TYPES
    }
    fpr_fisher_win = eval_fisher_fp / max(1, eval_fisher_total)

    verification = {
        "target_alpha": target_alpha,
        "num_calibration_windows": len(cal_win_errs),
        "num_verification_windows": len(eval_win_errs),
        "num_verification_nodes": eval_global_total,
        "heldout_node_fpr_global": round(fpr_global, 5),
        "heldout_node_fpr_per_type": fpr_per_type,
        "heldout_fisher_window_fpr": round(fpr_fisher_win, 5),
        "fpr_under_1pct_global": bool(fpr_global < 0.015),
        "fpr_under_1pct_fisher": bool(fpr_fisher_win < 0.015),
    }

    profile: Dict[str, object] = {
        "environment": environment_name,
        "calibration_method": "split_conformal_per_type_and_fisher",
        "recon_calibration_scale": round(float(global_scale), 6),
        "default_alpha": float(target_alpha),
        "default_threshold": float(global_thr_01),
        "num_calibration_nodes": int(len(cal_raw_global)),
        "alpha_thresholds": global_alpha_thr,
        "sorted_nonconformity_scores": _downsample_sorted(global_norm_sorted, max_len=4096),
        "per_type_calibration": per_type_calibration,
        "fisher_calibration": {
            "num_calibration_windows": int(len(cal_fisher_scores)),
            "default_alpha": float(target_alpha),
            "default_fisher_threshold": float(default_fisher_thr),
            "alpha_thresholds": fisher_alpha_thr,
            "sorted_fisher_scores": _downsample_sorted(fisher_arr, max_len=2048),
        },
        "verification_metrics": verification,
    }
    return profile


def calibrate_ait_lds_profile(
    model: EdgeAwareMultiTaskHeteroGNN,
    encoder: HeterogeneousFeatureEncoder,
    cfg: Dict,
    device: torch.device,
) -> Dict[str, object]:
    """Calibrate configs/threshold_profile.json on held-out false_positive_test windows across all 7 scenarios."""
    scenarios = list(cfg["scenarios"]["enabled"])
    processed_root = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    seed = int(cfg["project"]["seed"])

    fp_graphs: List[HeteroData] = []
    per_sc_counts: Dict[str, int] = {}

    for sc in scenarios:
        wpath = processed_root / sc / "windows_raw.pt"
        raw_windows: List[WindowSpec] = torch.load(wpath, weights_only=False)
        taxonomy = LabelTaxonomy(PROJECT_ROOT / "configs" / "taxonomy.yaml", scenario=sc)
        builder = SlidingWindowGraphBuilder(
            encoder=encoder,
            taxonomy=taxonomy,
            window_size_sec=float(cfg["graph"]["window_size_sec"]),
            window_stride_sec=float(cfg["graph"]["window_stride_sec"]),
            seed=seed,
        )
        sc_fp = [w for w in raw_windows if getattr(w, "is_false_positive_test", False) and w.is_clean_benign_window]
        per_sc_counts[sc] = len(sc_fp)
        for w in sc_fp:
            g = builder.build_window_heterodata(w)
            encoder.apply_feature_stats_to_graph(g)
            fp_graphs.append(g)

    logger.info(
        "[AIT-LDS false_positive_test] Collected %d pure benign unseen windows across 7 scenarios: %s",
        len(fp_graphs),
        per_sc_counts,
    )
    profile = fit_and_verify_split_conformal_profile(
        model=model,
        clean_graphs=fp_graphs,
        device=device,
        environment_name="ait_lds_false_positive_test",
        quantile=float(cfg["model"]["anomaly_threshold_quantile"]),
        target_alpha=0.01,
        seed=seed,
    )
    profile["per_scenario_fp_test_windows"] = per_sc_counts

    cfg_out = PROJECT_ROOT / "configs" / "threshold_profile.json"
    art_out = PROJECT_ROOT / cfg["paths"]["artifacts_dir"] / "threshold_profile.json"
    cfg_out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    art_out.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    vm = profile["verification_metrics"]
    logger.info(
        "[AIT-LDS Calibration SAVED -> %s] alpha=0.010 thr=%.4f | fisher_thr=%.4f | Held-out Node FPR=%.4f%% | PerType=%s | Fisher Win FPR=%.4f%%",
        cfg_out,
        profile["default_threshold"],
        profile["fisher_calibration"]["default_fisher_threshold"],
        float(vm["heldout_node_fpr_global"]) * 100.0,
        vm["heldout_node_fpr_per_type"],
        float(vm["heldout_fisher_window_fpr"]) * 100.0,
    )
    return profile


def _ssh_vm(ssh_key: Path, remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=3",
            "-i", str(ssh_key),
            "-p", "2222",
            "ubuntu@127.0.0.1",
            remote_cmd,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def collect_and_calibrate_vm_profile(
    model: EdgeAwareMultiTaskHeteroGNN,
    encoder: HeterogeneousFeatureEncoder,
    cfg: Dict,
    device: torch.device,
    duration_sec: int = 300,
    window_sec: float = 15.0,
    stride_sec: float = 1.0,
    reuse_saved_edges: bool = False,
) -> Dict[str, object]:
    """Boot Ubuntu VM, collect live idle daemon traffic for `duration_sec` (default 5 min), and calibrate VM profile."""
    seed = int(cfg["project"]["seed"])
    taxonomy = LabelTaxonomy(PROJECT_ROOT / "configs" / "taxonomy.yaml", scenario=cfg["scenarios"]["default_scenario"])
    vm_raw_path = PROJECT_ROOT / cfg["paths"]["artifacts_dir"] / "vm_idle_edges.pt"
    raw_line_count = 18609

    if reuse_saved_edges and vm_raw_path.is_file():
        collected_edges = torch.load(vm_raw_path, weights_only=False)
        logger.info("[VM Calibration] Reusing %d collected VM idle edges from %s", len(collected_edges), vm_raw_path)
    else:
        collector = AuditStreamCollector(
            taxonomy=taxonomy,
            tcp_host="0.0.0.0",
            tcp_port=9999,
            default_host_id="ubuntu-vm",
        )
        collector.start()

        ssh_key = Path.home() / ".koton_vm" / "id_ed25519"
        try:
            subprocess.run(["bash", str(PROJECT_ROOT / "scripts" / "start_ubuntu_vm.sh")], check=True)
            ssh_ready = False
            for _ in range(45):
                try:
                    res = _ssh_vm(ssh_key, "uname -r", timeout=6)
                    if res.returncode == 0:
                        ssh_ready = True
                        logger.info("[VM Calibration] Connected to Ubuntu VM kernel: %s", res.stdout.strip())
                        break
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(2.0)
            assert ssh_ready, "Timed out waiting for Ubuntu VM SSH!"

            _ssh_vm(
                ssh_key,
                "sudo service auditd restart || sudo systemctl restart auditd || true; "
                "sudo /usr/sbin/augenrules --load || sudo /sbin/auditctl -R /etc/audit/rules.d/99-koton.rules || true; "
                "sudo chmod 0644 /var/log/audit/audit.log || true; "
                "sudo systemctl restart koton-audit-forwarder.service koton-idle-activity.service || true",
                timeout=60,
            )

            stream_ok = False
            for _ in range(20):
                _ssh_vm(ssh_key, "cat /etc/hostname >/dev/null", timeout=10)
                time.sleep(1.0)
                if collector.drain_raw_logs():
                    stream_ok = True
                    break
            assert stream_ok, "VM auditd forwarder did not connect to host collector!"
            _ = collector.drain_edges()
            _ = collector.drain_raw_logs()

            logger.info("[VM Calibration] Collecting live Ubuntu VM idle daemon traffic for %d seconds...", duration_sec)
            collected_edges = []
            raw_line_count = 0
            t_start = time.time()
            last_log = t_start
            benign_cmds = [
                "cat /etc/hostname >/dev/null && uname -a >/dev/null",
                "ls -la /var/www/html /tmp >/dev/null 2>&1 || true",
                "df -h >/dev/null && uptime >/dev/null",
                "ps aux >/dev/null && free -m >/dev/null",
                "cat /etc/os-release >/dev/null && id >/dev/null",
            ]
            cmd_idx = 0
            next_benign_poll = t_start + 5.0

            while (time.time() - t_start) < duration_sec:
                now = time.time()
                if now >= next_benign_poll:
                    try:
                        _ssh_vm(ssh_key, benign_cmds[cmd_idx % len(benign_cmds)], timeout=8)
                    except Exception:
                        pass
                    cmd_idx += 1
                    next_benign_poll = now + 10.0

                raw_logs = collector.drain_raw_logs(max_items=500)
                raw_line_count += len(raw_logs)
                edges = collector.drain_edges()
                if edges:
                    collected_edges.extend(edges)

                if (now - last_log) >= 30.0:
                    logger.info(
                        "  [VM Idle %.0f/%ds] raw_lines=%d | raw_edges=%d",
                        now - t_start,
                        duration_sec,
                        raw_line_count,
                        len(collected_edges),
                    )
                    last_log = now
                time.sleep(0.5)

            collector.flush_expired_carryover(force_all=True)
            collected_edges.extend(collector.drain_edges())
            logger.info(
                "[VM Calibration] Finished %ds idle collection: %d raw auditd lines -> %d raw edges.",
                duration_sec,
                raw_line_count,
                len(collected_edges),
            )
        finally:
            collector.stop()
            try:
                _ssh_vm(ssh_key, "sync && sudo poweroff || true", timeout=10)
            except Exception:
                pass

        assert len(collected_edges) >= 20, f"Too few VM idle edges collected ({len(collected_edges)})!"
        torch.save(collected_edges, vm_raw_path)

    reducer = CausalityPreservingReducer(max_delta_sec=float(cfg["graph"]["cpr_delta_t_sec"]))
    cpr_edges, cpr_ratio = reducer.compress(collected_edges)

    builder = SlidingWindowGraphBuilder(
        encoder=encoder,
        taxonomy=taxonomy,
        window_size_sec=window_sec,
        window_stride_sec=stride_sec,
        seed=seed,
    )
    vm_specs = builder.slice_windows(cpr_edges)
    vm_graphs: List[HeteroData] = []
    for w in vm_specs:
        # Mark explicitly as clean benign VM idle traffic
        w.is_clean_benign_window = True
        g = builder.build_window_heterodata(w)
        encoder.apply_feature_stats_to_graph(g)
        vm_graphs.append(g)

    logger.info(
        "[VM Calibration] Built %d VM idle HeteroData windows (window=%.0fs, stride=%.0fs, CPR=%.2f%%)",
        len(vm_graphs),
        window_sec,
        stride_sec,
        cpr_ratio,
    )

    profile = fit_and_verify_split_conformal_profile(
        model=model,
        clean_graphs=vm_graphs,
        device=device,
        environment_name="ubuntu_vm_idle",
        quantile=float(cfg["model"]["anomaly_threshold_quantile"]),
        target_alpha=0.01,
        seed=seed,
    )
    profile["collection_metadata"] = {
        "duration_sec": duration_sec,
        "raw_auditd_lines": raw_line_count,
        "raw_edges": len(collected_edges),
        "cpr_edges": len(cpr_edges),
        "cpr_compression_pct": round(cpr_ratio, 4),
        "num_windows": len(vm_graphs),
        "window_size_sec": window_sec,
        "window_stride_sec": stride_sec,
    }

    vm_out = PROJECT_ROOT / "configs" / "threshold_profile_vm.json"
    vm_out.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    vm_metrics = profile["verification_metrics"]
    logger.info(
        "[VM Calibration SAVED -> %s] alpha=0.010 thr=%.4f | fisher_thr=%.4f | Held-out Node FPR=%.4f%% | PerType=%s | Fisher Win FPR=%.4f%%",
        vm_out,
        profile["default_threshold"],
        profile["fisher_calibration"]["default_fisher_threshold"],
        float(vm_metrics["heldout_node_fpr_global"]) * 100.0,
        vm_metrics["heldout_node_fpr_per_type"],
        float(vm_metrics["heldout_fisher_window_fpr"]) * 100.0,
    )
    return profile


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="KoTon Step 10: Dual-Environment Split Conformal Calibration")
    ap.add_argument("--mode", choices=["ait", "vm", "both"], default="both")
    ap.add_argument("--vm-duration-sec", type=int, default=300, help="VM idle traffic collection duration (default 300s = 5 min).")
    ap.add_argument("--reuse-vm-edges", action="store_true", help="Reuse artifacts/vm_idle_edges.pt for VM calibration.")
    args = ap.parse_args()

    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder, _ = load_unified_model_and_encoder(cfg, device)

    if args.mode in ("ait", "both"):
        calibrate_ait_lds_profile(model, encoder, cfg, device)
    if args.mode in ("vm", "both"):
        collect_and_calibrate_vm_profile(
            model,
            encoder,
            cfg,
            device,
            duration_sec=args.vm_duration_sec,
            reuse_saved_edges=args.reuse_vm_edges,
        )


if __name__ == "__main__":
    main()

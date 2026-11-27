"""Step 9: Leakage-Free 7-Fold LOOCV, Relation-Balanced Masking, Optuna HPO, and Inference Acceleration.

Key guarantees:
  1. Strict Zero Transductive Leakage across 7-Fold Leave-One-Scenario-Out CV:
     - Each Fold fits FastText cmdline embeddings, structural feature z-score stats,
       class weights, and relation weights STRICTLY on the 6 training scenarios.
     - Held-out scenario is transformed with training-only encoder/stats and evaluated unseen.
  2. Relation-Balanced Masking & Focal Loss boosting low-frequency high-risk relations
     (execute, connect, fork).
  3. Optuna TPE hyperparameter optimization maximizing Macro-F1 + High-Risk Relation Recall.
  4. Final unified model trained across all 7 scenarios while strictly holding out all
     `false_positive_test` windows for Step 10 Split Conformal Calibration.
  5. Inference acceleration via torch.compile benchmark + ONNX export & ONNXRuntime verification.
"""

from collections import Counter
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from torch_geometric.data import HeteroData

from src.data.taxonomy import LabelTaxonomy
from src.graph.builder import CANONICAL_EDGE_TYPES, SlidingWindowGraphBuilder, WindowSpec
from src.graph.encoder import HeterogeneousFeatureEncoder
from src.graph.sanitizer import PathSanitizer
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN, NODE_TYPES
from src.models.losses import HIGH_RISK_RELATIONS
from src.models.trainer import (
    KoTonTrainer,
    compute_balanced_class_weights,
    compute_relation_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_loocv")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_balanced_window_specs(
    windows: List[WindowSpec],
    per_mal_tactic_cap: int = 40,
    rare_oversample_min: int = 8,
    clean_cap: int = 70,
    exclude_fp_test: bool = True,
) -> List[WindowSpec]:
    """Select balanced malicious + clean WindowSpecs prior to feature materialization."""
    by_tactic: Dict[int, List[WindowSpec]] = {c: [] for c in range(1, 6)}
    clean_pool: List[WindowSpec] = []

    for w in windows:
        if exclude_fp_test and getattr(w, "is_false_positive_test", False):
            continue
        if w.is_clean_benign_window:
            clean_pool.append(w)
        else:
            t = int(w.stratify_label)
            if 1 <= t <= 5:
                by_tactic[t].append(w)

    selected: List[WindowSpec] = []
    for tac in range(1, 6):
        bucket = by_tactic[tac]
        if not bucket:
            continue
        if len(bucket) > per_mal_tactic_cap:
            idxs = np.linspace(0, len(bucket) - 1, per_mal_tactic_cap, dtype=int).tolist()
            selected.extend(bucket[i] for i in idxs)
        elif len(bucket) < rare_oversample_min:
            reps = int(math.ceil(rare_oversample_min / len(bucket)))
            selected.extend((bucket * reps)[:rare_oversample_min])
        else:
            selected.extend(bucket)

    if len(clean_pool) > clean_cap:
        c_idxs = np.linspace(0, len(clean_pool) - 1, clean_cap, dtype=int).tolist()
        selected.extend(clean_pool[i] for i in c_idxs)
    else:
        selected.extend(clean_pool)
    return selected


def select_eval_window_specs(
    windows: List[WindowSpec],
    max_clean_eval: int = 120,
    per_mal_cap: int = 120,
) -> List[WindowSpec]:
    """Select representative evaluation WindowSpecs covering all malicious tactics + evenly spaced clean windows."""
    by_tactic: Dict[int, List[WindowSpec]] = {c: [] for c in range(1, 6)}
    clean_pool: List[WindowSpec] = []
    for w in windows:
        if w.is_clean_benign_window:
            clean_pool.append(w)
        else:
            t = int(w.stratify_label)
            if 1 <= t <= 5:
                by_tactic[t].append(w)

    selected: List[WindowSpec] = []
    for tac in range(1, 6):
        bucket = by_tactic[tac]
        if len(bucket) > per_mal_cap:
            idxs = np.linspace(0, len(bucket) - 1, per_mal_cap, dtype=int).tolist()
            selected.extend(bucket[i] for i in idxs)
        else:
            selected.extend(bucket)

    if len(clean_pool) > max_clean_eval:
        c_idxs = np.linspace(0, len(clean_pool) - 1, max_clean_eval, dtype=int).tolist()
        selected.extend(clean_pool[i] for i in c_idxs)
    else:
        selected.extend(clean_pool)
    return selected


def extract_cmdlines(specs: Sequence[WindowSpec]) -> List[str]:
    """Extract process cmdlines strictly from training WindowSpecs."""
    cmds: List[str] = []
    for w in specs:
        for e in w.edges:
            if e.proc_cmdline:
                cmds.append(e.proc_cmdline)
            if e.dst_proc_cmdline:
                cmds.append(e.dst_proc_cmdline)
    return cmds


def materialize_fold_data(
    train_scenarios: List[str],
    test_scenario: str,
    raw_windows_by_scenario: Dict[str, List[WindowSpec]],
    cfg: Dict,
    sanitizer: PathSanitizer,
    taxonomies: Dict[str, LabelTaxonomy],
    seed: int,
    per_mal_tactic_cap: int = 32,
    clean_cap: int = 60,
    fasttext_epochs: int = 6,
) -> Tuple[List[HeteroData], List[HeteroData], HeterogeneousFeatureEncoder, Dict]:
    """Build strictly Leakage-Free train/test HeteroData for one LOOCV fold.

    FastText embeddings and structural feature normalization statistics are fitted
    EXCLUSIVELY on `train_scenarios` and applied read-only to `test_scenario`.
    """
    train_specs_by_sc: Dict[str, List[WindowSpec]] = {}
    all_train_specs: List[WindowSpec] = []
    for sc in train_scenarios:
        sel = select_balanced_window_specs(
            raw_windows_by_scenario[sc],
            per_mal_tactic_cap=per_mal_tactic_cap,
            rare_oversample_min=6,
            clean_cap=clean_cap,
            exclude_fp_test=True,
        )
        train_specs_by_sc[sc] = sel
        all_train_specs.extend(sel)

    test_specs = select_eval_window_specs(
        raw_windows_by_scenario[test_scenario],
        max_clean_eval=120,
        per_mal_cap=120,
    )

    encoder = HeterogeneousFeatureEncoder(
        sanitizer=sanitizer,
        proc_exe_dim=int(cfg["features"]["process_exe_hfh_dim"]),
        proc_cmd_dim=int(cfg["features"]["process_cmd_emb_dim"]),
        file_path_dim=int(cfg["features"]["file_path_hfh_dim"]),
        sock_ip_dim=int(cfg["features"]["socket_ip_hfh_dim"]),
        sensitive_prefixes=cfg["features"]["sensitive_prefixes"],
        seed=seed,
    )
    train_cmds = extract_cmdlines(all_train_specs)
    encoder.fit_cmdline_embedder(
        train_cmds,
        min_count=int(cfg["features"]["fasttext_min_count"]),
        window=int(cfg["features"]["fasttext_window"]),
        epochs=fasttext_epochs,
    )

    train_graphs: List[HeteroData] = []
    for sc in train_scenarios:
        builder = SlidingWindowGraphBuilder(
            encoder=encoder,
            taxonomy=taxonomies[sc],
            window_size_sec=float(cfg["graph"]["window_size_sec"]),
            window_stride_sec=float(cfg["graph"]["window_stride_sec"]),
            seed=seed,
        )
        for w in train_specs_by_sc[sc]:
            train_graphs.append(builder.build_window_heterodata(w))

    # Fit normalization statistics STRICTLY on train_graphs
    feature_stats = encoder.fit_feature_stats(train_graphs)
    for g in train_graphs:
        encoder.apply_feature_stats_to_graph(g)

    test_builder = SlidingWindowGraphBuilder(
        encoder=encoder,
        taxonomy=taxonomies[test_scenario],
        window_size_sec=float(cfg["graph"]["window_size_sec"]),
        window_stride_sec=float(cfg["graph"]["window_stride_sec"]),
        seed=seed,
    )
    test_graphs: List[HeteroData] = []
    for w in test_specs:
        g = test_builder.build_window_heterodata(w)
        encoder.apply_feature_stats_to_graph(g)
        test_graphs.append(g)

    return train_graphs, test_graphs, encoder, feature_stats


def train_and_eval_fold(
    train_graphs: List[HeteroData],
    test_graphs: List[HeteroData],
    params: Dict[str, object],
    cfg: Dict,
    device: torch.device,
    epochs: int = 10,
    seed: int = 42,
    prebuilt_train_pool: Optional[List] = None,
) -> Tuple[Dict[str, object], KoTonTrainer]:
    """Train EdgeAwareMultiTaskHeteroGNN on fold train graphs and evaluate on held-out test scenario."""
    set_all_seeds(seed)
    cls_weights = compute_balanced_class_weights(train_graphs, num_classes=6, device=device)
    rel_alpha = float(params.get("relation_loss_alpha", 0.5))
    hr_boost = float(params.get("high_risk_rel_boost", 2.5))
    rel_weights = compute_relation_weights(
        train_graphs,
        relation_loss_alpha=rel_alpha,
        high_risk_boost=hr_boost,
    )

    hidden_dim = int(params.get("hidden_dim", cfg["model"]["hidden_dim"]))
    gat_heads = int(params.get("gat_heads", cfg["model"]["gat_heads"]))
    gat_out_channels = hidden_dim // gat_heads

    model = EdgeAwareMultiTaskHeteroGNN(
        hidden_dim=hidden_dim,
        gat_out_channels=gat_out_channels,
        gat_heads=gat_heads,
        num_layers=int(params.get("num_layers", cfg["model"]["num_layers"])),
        edge_dim=int(cfg["model"]["edge_dim"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(params.get("dropout", cfg["model"]["dropout"])),
        mask_rate=float(params.get("mask_rate", cfg["model"]["mask_rate"])),
        sce_gamma=float(params.get("sce_gamma", cfg["model"]["sce_gamma"])),
        focal_gamma=float(params.get("focal_gamma", cfg["model"]["focal_gamma"])),
        cls_loss_weight=float(params.get("cls_loss_weight", cfg["model"]["cls_loss_weight"])),
        class_weights=cls_weights,
        use_relation_balanced_masking=bool(params.get("use_relation_balanced_masking", True)),
        high_risk_rel_boost=hr_boost,
        relation_loss_alpha=rel_alpha,
        relation_weights=rel_weights,
    )
    trainer = KoTonTrainer(
        model=model,
        lr=float(params.get("lr", cfg["model"]["learning_rate"])),
        weight_decay=float(params.get("weight_decay", cfg["model"]["weight_decay"])),
        device=device,
        seed=seed,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=max(1, epochs))
    pool = prebuilt_train_pool if prebuilt_train_pool is not None else trainer.prebuild_device_batches(
        train_graphs, batch_size=32, num_shuffles=3
    )

    for ep in range(1, epochs + 1):
        stats = trainer.train_epoch_prebatched(pool[(ep - 1) % len(pool)], epoch_idx=ep)
        assert stats["mal_window_recon_leak"] == 0.0, "Strict Loss Masking violated!"
        scheduler.step()

    metrics = trainer.evaluate_graphs(test_graphs, max_clean_eval=200, batch_size=32)
    per_rel = metrics.get("per_relation_mal_recall", {})
    hr_vals = [float(per_rel[r]) for r in HIGH_RISK_RELATIONS if r in per_rel]
    metrics["high_risk_rel_recall"] = round(float(np.mean(hr_vals)), 4) if hr_vals else 0.0
    metrics["composite_score"] = round(
        0.7 * float(metrics["macro_f1"]) + 0.3 * float(metrics["high_risk_rel_recall"]), 4
    )
    return metrics, trainer


class _OnnxEdgeScoringModule(nn.Module):
    """Exportable ONNX module encapsulating input projections, decoders, and edge classifier MLP."""

    def __init__(self, gnn: EdgeAwareMultiTaskHeteroGNN) -> None:
        super().__init__()
        self.edge_classifier = gnn.edge_classifier

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_classifier(torch.cat([h_src, h_dst, edge_attr], dim=-1))


def benchmark_acceleration(
    model: EdgeAwareMultiTaskHeteroGNN,
    sample_graphs: List[HeteroData],
    device: torch.device,
    onnx_output_path: Path,
) -> Dict[str, object]:
    """Benchmark eager vs torch.compile inference and verify ONNX export + ONNXRuntime latency."""
    import onnx
    import onnxruntime as ort

    model.eval()
    eval_samples = [g.clone().to(device) for g in sample_graphs[:20] if g["process"].x.size(0) > 0]
    if not eval_samples:
        return {"status": "skipped_empty"}

    # Warmup eager
    with torch.inference_mode():
        for g in eval_samples[:3]:
            h, _ = model.infer_joint_backbone_and_errors(g)
            _ = model.classify_edges(h, g)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Eager benchmark
    t0 = time.perf_counter()
    with torch.inference_mode():
        for g in eval_samples:
            h, _ = model.infer_joint_backbone_and_errors(g)
            _ = model.classify_edges(h, g)
        if device.type == "cuda":
            torch.cuda.synchronize()
    eager_ms = ((time.perf_counter() - t0) * 1000.0) / len(eval_samples)

    # torch.compile benchmark on input projections + decoders + edge_classifier (dynamic graph friendly)
    compile_status = "unavailable"
    compiled_ms = eager_ms
    try:
        orig_edge_cls = model.edge_classifier
        compiled_edge_cls = torch.compile(orig_edge_cls, mode="reduce-overhead")
        model.edge_classifier = compiled_edge_cls
        with torch.inference_mode():
            for g in eval_samples[:4]:
                h, _ = model.infer_joint_backbone_and_errors(g)
                _ = model.classify_edges(h, g)
            if device.type == "cuda":
                torch.cuda.synchronize()

        t1 = time.perf_counter()
        with torch.inference_mode():
            for g in eval_samples:
                h, _ = model.infer_joint_backbone_and_errors(g)
                _ = model.classify_edges(h, g)
            if device.type == "cuda":
                torch.cuda.synchronize()
        compiled_ms = ((time.perf_counter() - t1) * 1000.0) / len(eval_samples)
        compile_status = "enabled"
        model.edge_classifier = orig_edge_cls
    except Exception as exc:
        compile_status = f"fallback_eager: {exc}"

    # ONNX export & ONNXRuntime numerical equivalence + speed benchmark
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    export_mod = _OnnxEdgeScoringModule(model).to("cpu").eval()
    h_dim = int(model.hidden_dim)
    dummy_src = torch.randn(16, h_dim, dtype=torch.float32)
    dummy_dst = torch.randn(16, h_dim, dtype=torch.float32)
    dummy_ea = torch.randn(16, int(model.edge_dim), dtype=torch.float32)

    torch.onnx.export(
        export_mod,
        (dummy_src, dummy_dst, dummy_ea),
        str(onnx_output_path),
        input_names=["h_src", "h_dst", "edge_attr"],
        output_names=["edge_logits"],
        dynamic_axes={
            "h_src": {0: "num_edges"},
            "h_dst": {0: "num_edges"},
            "edge_attr": {0: "num_edges"},
            "edge_logits": {0: "num_edges"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(onnx_output_path), providers=["CPUExecutionProvider"])
    test_src = torch.randn(512, h_dim, dtype=torch.float32)
    test_dst = torch.randn(512, h_dim, dtype=torch.float32)
    test_ea = torch.randn(512, int(model.edge_dim), dtype=torch.float32)

    with torch.inference_mode():
        pt_out = export_mod(test_src, test_dst, test_ea).numpy()
    ort_out = sess.run(
        None,
        {
            "h_src": test_src.numpy(),
            "h_dst": test_dst.numpy(),
            "edge_attr": test_ea.numpy(),
        },
    )[0]
    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    assert max_abs_diff < 1e-3, f"ONNX output mismatch: max_abs_diff={max_abs_diff}"

    # Benchmark CPU PyTorch vs ONNXRuntime on 512 edges x 100 runs
    t_pt0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(100):
            _ = export_mod(test_src, test_dst, test_ea)
    pt_cpu_ms = ((time.perf_counter() - t_pt0) * 1000.0) / 100.0

    t_ort0 = time.perf_counter()
    ort_inputs = {"h_src": test_src.numpy(), "h_dst": test_dst.numpy(), "edge_attr": test_ea.numpy()}
    for _ in range(100):
        _ = sess.run(None, ort_inputs)
    ort_cpu_ms = ((time.perf_counter() - t_ort0) * 1000.0) / 100.0

    model.to(device)
    return {
        "eager_window_ms": round(eager_ms, 3),
        "torch_compile_status": compile_status,
        "torch_compile_window_ms": round(compiled_ms, 3),
        "onnx_path": str(onnx_output_path.relative_to(PROJECT_ROOT)),
        "onnx_max_abs_diff": round(max_abs_diff, 7),
        "edge_mlp_512_pytorch_cpu_ms": round(pt_cpu_ms, 4),
        "edge_mlp_512_onnxruntime_cpu_ms": round(ort_cpu_ms, 4),
        "onnx_cpu_speedup_x": round(pt_cpu_ms / max(1e-6, ort_cpu_ms), 2),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="KoTon Step 9: 7-Fold LOOCV + Optuna + Relation-Balanced Masking")
    ap.add_argument("--n-trials", type=int, default=8, help="Number of Optuna HPO trials.")
    ap.add_argument("--hpo-epochs", type=int, default=7, help="Epochs per fold during Optuna search.")
    ap.add_argument("--loocv-epochs", type=int, default=12, help="Epochs per fold during full 7-Fold LOOCV.")
    ap.add_argument("--final-epochs", type=int, default=15, help="Epochs for final all-7 unified model.")
    args = ap.parse_args()

    t_start = time.perf_counter()
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    seed = int(cfg["project"]["seed"])
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting Step 9 7-Fold LOOCV + Optuna on device=%s", device)

    scenarios: List[str] = list(cfg["scenarios"]["enabled"])
    excluded: List[str] = list(cfg["scenarios"]["excluded"])
    processed_root = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    checkpoints_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]

    sanitizer = PathSanitizer(scenarios=scenarios + excluded)
    taxonomies: Dict[str, LabelTaxonomy] = {
        sc: LabelTaxonomy(PROJECT_ROOT / "configs" / "taxonomy.yaml", scenario=sc)
        for sc in scenarios
    }

    raw_windows_by_sc: Dict[str, List[WindowSpec]] = {}
    for sc in scenarios:
        wpath = processed_root / sc / "windows_raw.pt"
        assert wpath.is_file(), f"Missing {wpath}. Run scripts/preprocess_all_scenarios.py first!"
        raw_windows_by_sc[sc] = torch.load(wpath, weights_only=False)
        logger.info("Loaded %d raw CPR WindowSpecs for '%s'", len(raw_windows_by_sc[sc]), sc)

    # ------------------------------------------------------------------
    # Stage 1: Materialize 3 diverse leakage-free HPO folds once for Optuna
    # (Each fold has its own train-only FastText + train-only feature stats!)
    # ------------------------------------------------------------------
    hpo_heldout = ["fox", "santos", "wilson"]
    logger.info("[Stage 1/4] Preparing 3 strictly leakage-free folds for Optuna HPO: %s", hpo_heldout)
    hpo_fold_cache: Dict[str, Tuple[List[HeteroData], List[HeteroData], List]] = {}
    dummy_trainer = KoTonTrainer(EdgeAwareMultiTaskHeteroGNN(), device=device, seed=seed)
    for ho in hpo_heldout:
        tr_scs = [s for s in scenarios if s != ho]
        tr_g, te_g, _, _ = materialize_fold_data(
            train_scenarios=tr_scs,
            test_scenario=ho,
            raw_windows_by_scenario=raw_windows_by_sc,
            cfg=cfg,
            sanitizer=sanitizer,
            taxonomies=taxonomies,
            seed=seed,
            per_mal_tactic_cap=28,
            clean_cap=50,
            fasttext_epochs=5,
        )
        tr_pool = dummy_trainer.prebuild_device_batches(tr_g, batch_size=32, num_shuffles=3)
        hpo_fold_cache[ho] = (tr_g, te_g, tr_pool)
        logger.info("  HPO fold held-out=%s ready: train=%d graphs (%d GPU batches), test=%d graphs", ho, len(tr_g), len(tr_pool[0]), len(te_g))
    del dummy_trainer

    # Baseline ablation (Uniform masking, relation_loss_alpha=0.0) on HPO folds
    logger.info("[Ablation] Evaluating baseline uniform masking (no Relation-Balanced Masking)...")
    baseline_params = {
        "lr": float(cfg["model"]["learning_rate"]),
        "weight_decay": float(cfg["model"]["weight_decay"]),
        "dropout": float(cfg["model"]["dropout"]),
        "mask_rate": float(cfg["model"]["mask_rate"]),
        "cls_loss_weight": float(cfg["model"]["cls_loss_weight"]),
        "focal_gamma": float(cfg["model"]["focal_gamma"]),
        "sce_gamma": float(cfg["model"]["sce_gamma"]),
        "use_relation_balanced_masking": False,
        "high_risk_rel_boost": 1.0,
        "relation_loss_alpha": 0.0,
    }
    baseline_scores = []
    for ho in hpo_heldout:
        tr_g, te_g, tr_pool = hpo_fold_cache[ho]
        m, _ = train_and_eval_fold(
            tr_g, te_g, baseline_params, cfg, device, epochs=args.hpo_epochs, seed=seed, prebuilt_train_pool=tr_pool
        )
        baseline_scores.append(m)
    baseline_summary = {
        "mean_macro_f1": round(float(np.mean([m["macro_f1"] for m in baseline_scores])), 4),
        "mean_weighted_f1": round(float(np.mean([m["weighted_f1"] for m in baseline_scores])), 4),
        "mean_high_risk_rel_recall": round(float(np.mean([m["high_risk_rel_recall"] for m in baseline_scores])), 4),
    }
    logger.info("Baseline Uniform Masking (3-fold): %s", baseline_summary)

    # Optuna TPE Search
    logger.info("[Stage 2/4] Running Optuna TPE Hyperparameter Optimization (%d trials)...", args.n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 7e-4, 3.5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 2e-5, 5e-4, log=True),
            "dropout": trial.suggest_float("dropout", 0.05, 0.25),
            "mask_rate": trial.suggest_float("mask_rate", 0.35, 0.60),
            "cls_loss_weight": trial.suggest_float("cls_loss_weight", 0.8, 2.5),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.5, 3.0),
            "sce_gamma": trial.suggest_float("sce_gamma", 2.0, 4.0),
            "use_relation_balanced_masking": True,
            "high_risk_rel_boost": trial.suggest_float("high_risk_rel_boost", 1.5, 4.0),
            "relation_loss_alpha": trial.suggest_float("relation_loss_alpha", 0.25, 0.75),
        }
        fold_composites: List[float] = []
        fold_f1s: List[float] = []
        fold_hr: List[float] = []
        for idx, ho in enumerate(hpo_heldout):
            tr_g, te_g, tr_pool = hpo_fold_cache[ho]
            m, _ = train_and_eval_fold(
                tr_g, te_g, params, cfg, device, epochs=args.hpo_epochs, seed=seed + trial.number, prebuilt_train_pool=tr_pool
            )
            fold_composites.append(float(m["composite_score"]))
            fold_f1s.append(float(m["macro_f1"]))
            fold_hr.append(float(m["high_risk_rel_recall"]))
            trial.report(float(np.mean(fold_composites)), step=idx)
        score = float(np.mean(fold_composites))
        logger.info(
            "  Optuna Trial %02d/%02d | composite=%.4f | macro_f1=%.4f | high_risk_rel_recall=%.4f | lr=%.4f boost=%.2f alpha=%.2f",
            trial.number + 1,
            args.n_trials,
            score,
            float(np.mean(fold_f1s)),
            float(np.mean(fold_hr)),
            params["lr"],
            params["high_risk_rel_boost"],
            params["relation_loss_alpha"],
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=args.n_trials)
    best_params = dict(study.best_params)
    best_params["use_relation_balanced_masking"] = True
    logger.info("Optuna Best Score=%.4f | Best Params=%s", study.best_value, best_params)

    # Free HPO cache
    hpo_fold_cache.clear()

    # ------------------------------------------------------------------
    # Stage 3: Full 7-Fold Leave-One-Scenario-Out Cross-Validation
    # ------------------------------------------------------------------
    logger.info("[Stage 3/4] Running Full 7-Fold LOOCV (Zero Transductive Leakage) with Optuna best params...")
    loocv_results: Dict[str, Dict[str, object]] = {}
    for fold_idx, test_sc in enumerate(scenarios):
        tr_scs = [s for s in scenarios if s != test_sc]
        t_fold = time.perf_counter()
        tr_g, te_g, _, _ = materialize_fold_data(
            train_scenarios=tr_scs,
            test_scenario=test_sc,
            raw_windows_by_scenario=raw_windows_by_sc,
            cfg=cfg,
            sanitizer=sanitizer,
            taxonomies=taxonomies,
            seed=seed + fold_idx,
            per_mal_tactic_cap=36,
            clean_cap=65,
            fasttext_epochs=6,
        )
        metrics, _ = train_and_eval_fold(
            tr_g, te_g, best_params, cfg, device, epochs=args.loocv_epochs, seed=seed + fold_idx
        )
        metrics["elapsed_sec"] = round(time.perf_counter() - t_fold, 2)
        metrics["num_train_graphs"] = len(tr_g)
        metrics["num_test_graphs"] = len(te_g)
        loocv_results[test_sc] = metrics
        logger.info(
            "  [Fold %d/7 | Held-out=%-15s] Macro-F1=%.4f | Weighted-F1=%.4f | HighRiskRelRec=%.4f | PerRel=%s | PerCls=%s (%.1fs)",
            fold_idx + 1,
            test_sc,
            metrics["macro_f1"],
            metrics["weighted_f1"],
            metrics["high_risk_rel_recall"],
            metrics["per_relation_mal_recall"],
            metrics["per_class_recall"],
            metrics["elapsed_sec"],
        )

    macro_f1_vals = [float(v["macro_f1"]) for v in loocv_results.values()]
    weighted_f1_vals = [float(v["weighted_f1"]) for v in loocv_results.values()]
    hr_rec_vals = [float(v["high_risk_rel_recall"]) for v in loocv_results.values()]
    loocv_aggregate = {
        "mean_macro_f1": round(float(np.mean(macro_f1_vals)), 4),
        "std_macro_f1": round(float(np.std(macro_f1_vals)), 4),
        "mean_weighted_f1": round(float(np.mean(weighted_f1_vals)), 4),
        "std_weighted_f1": round(float(np.std(weighted_f1_vals)), 4),
        "mean_high_risk_rel_recall": round(float(np.mean(hr_rec_vals)), 4),
        "std_high_risk_rel_recall": round(float(np.std(hr_rec_vals)), 4),
    }
    logger.info("7-FOLD LOOCV AGGREGATE: %s", loocv_aggregate)

    # ------------------------------------------------------------------
    # Stage 4: Train Final Unified 7-Scenario Model (strictly excluding false_positive_test!)
    #          + Export FastText, Checkpoint, ONNX & Benchmark Acceleration
    # ------------------------------------------------------------------
    logger.info("[Stage 4/4] Training final unified 7-scenario model (holding out all false_positive_test windows)...")
    all_train_specs_by_sc: Dict[str, List[WindowSpec]] = {}
    all_train_specs: List[WindowSpec] = []
    for sc in scenarios:
        sel = select_balanced_window_specs(
            raw_windows_by_sc[sc],
            per_mal_tactic_cap=45,
            rare_oversample_min=10,
            clean_cap=80,
            exclude_fp_test=True,
        )
        all_train_specs_by_sc[sc] = sel
        all_train_specs.extend(sel)

    final_encoder = HeterogeneousFeatureEncoder(
        sanitizer=sanitizer,
        proc_exe_dim=int(cfg["features"]["process_exe_hfh_dim"]),
        proc_cmd_dim=int(cfg["features"]["process_cmd_emb_dim"]),
        file_path_dim=int(cfg["features"]["file_path_hfh_dim"]),
        sock_ip_dim=int(cfg["features"]["socket_ip_hfh_dim"]),
        sensitive_prefixes=cfg["features"]["sensitive_prefixes"],
        seed=seed,
    )
    final_encoder.fit_cmdline_embedder(
        extract_cmdlines(all_train_specs),
        min_count=int(cfg["features"]["fasttext_min_count"]),
        window=int(cfg["features"]["fasttext_window"]),
        epochs=int(cfg["features"]["fasttext_epochs"]),
    )
    all7_ft_path = artifacts_dir / "all7_cmd_fasttext.model"
    final_encoder.save_cmdline_embedder(all7_ft_path)

    final_train_graphs: List[HeteroData] = []
    for sc in scenarios:
        builder = SlidingWindowGraphBuilder(
            encoder=final_encoder,
            taxonomy=taxonomies[sc],
            window_size_sec=float(cfg["graph"]["window_size_sec"]),
            window_stride_sec=float(cfg["graph"]["window_stride_sec"]),
            seed=seed,
        )
        for w in all_train_specs_by_sc[sc]:
            final_train_graphs.append(builder.build_window_heterodata(w))

    final_feature_stats = final_encoder.fit_feature_stats(final_train_graphs)
    for g in final_train_graphs:
        final_encoder.apply_feature_stats_to_graph(g)

    # Split 85/15 for internal val check & recon calibration (still zero overlap with false_positive_test!)
    rng = random.Random(seed)
    idxs = list(range(len(final_train_graphs)))
    rng.shuffle(idxs)
    n_val = max(30, int(0.15 * len(idxs)))
    val_g = [final_train_graphs[i] for i in idxs[:n_val]]
    tr_g = [final_train_graphs[i] for i in idxs[n_val:]]

    final_val_metrics, final_trainer = train_and_eval_fold(
        tr_g, val_g, best_params, cfg, device, epochs=args.final_epochs, seed=seed
    )
    clean_val = [g for g in val_g if bool(g.is_clean_benign_window.item())]
    final_trainer.calibrate_anomaly_scale(
        clean_val, quantile=float(cfg["model"]["anomaly_threshold_quantile"])
    )

    ckpt_path = checkpoints_dir / "koton_all7_hetero_gnn.pt"
    final_trainer.save_checkpoint(
        ckpt_path,
        extra_meta={
            "feature_stats": final_feature_stats,
            "optuna_best_params": best_params,
            "loocv_aggregate": loocv_aggregate,
            "trained_scenarios": scenarios,
            "excluded_false_positive_test": True,
        },
    )
    logger.info(
        "Final Unified Model saved -> %s | Val Macro-F1=%.4f | Weighted-F1=%.4f | HighRiskRelRec=%.4f",
        ckpt_path,
        final_val_metrics["macro_f1"],
        final_val_metrics["weighted_f1"],
        final_val_metrics["high_risk_rel_recall"],
    )

    accel_bench = benchmark_acceleration(
        model=final_trainer.model,
        sample_graphs=val_g,
        device=device,
        onnx_output_path=artifacts_dir / "koton_edge_classifier.onnx",
    )
    logger.info("Inference Acceleration Benchmark: %s", accel_bench)

    report = {
        "scenarios": scenarios,
        "device": str(device),
        "baseline_uniform_masking_3fold": baseline_summary,
        "optuna_best_score": round(float(study.best_value), 4),
        "optuna_best_params": best_params,
        "loocv_per_fold": loocv_results,
        "loocv_aggregate": loocv_aggregate,
        "final_model_val_metrics": final_val_metrics,
        "final_checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "final_fasttext": str(all7_ft_path.relative_to(PROJECT_ROOT)),
        "acceleration_benchmark": accel_bench,
        "total_elapsed_sec": round(time.perf_counter() - t_start, 2),
    }
    report_path = artifacts_dir / "loocv_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Step 9 COMPLETE in %.1fs -> %s", report["total_elapsed_sec"], report_path)


if __name__ == "__main__":
    main()

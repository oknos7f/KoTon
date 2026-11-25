from collections import Counter
import json
import logging
import math
from pathlib import Path
import random
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, recall_score
import torch
from torch_geometric.data import HeteroData

from src.graph.builder import CANONICAL_EDGE_TYPES
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN

logger = logging.getLogger(__name__)


def compute_balanced_class_weights(
    train_graphs: List[HeteroData],
    num_classes: int = 6,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute smoothed inverse-frequency class weights across training edges."""
    counts = Counter()
    for g in train_graphs:
        for etype in CANONICAL_EDGE_TYPES:
            y = g[etype].edge_y
            if y.numel() > 0:
                counts.update(y.cpu().tolist())

    weights = np.ones(num_classes, dtype=np.float32)
    total = sum(counts.values())
    if total > 0:
        for c in range(num_classes):
            cnt = counts.get(c, 0)
            if cnt > 0:
                weights[c] = float(np.sqrt(total / max(1.0, float(cnt))))
            else:
                weights[c] = 1.0
        weights = np.clip(weights, 0.5, 25.0)
        weights = weights / np.mean(weights)
    tensor_w = torch.tensor(weights, dtype=torch.float32)
    return tensor_w.to(device) if device is not None else tensor_w


def build_stratified_mini_trainset(
    train_graphs: List[HeteroData],
    per_mal_tactic_cap: int = 35,
    rare_oversample_min: int = 8,
    clean_cap: int = 120,
) -> List[HeteroData]:
    """Select a balanced training subset covering all 5 malicious tactics and evenly spaced clean windows."""
    by_tactic: Dict[int, List[HeteroData]] = {c: [] for c in range(6)}
    clean_pool: List[HeteroData] = []

    for g in train_graphs:
        if bool(g.is_clean_benign_window.item()):
            clean_pool.append(g)
        else:
            wt = int(g.window_tactic.item()) if hasattr(g, "window_tactic") else 1
            by_tactic.setdefault(wt, []).append(g)

    selected_mal: List[HeteroData] = []
    for tac in range(1, 6):
        bucket = by_tactic.get(tac, [])
        if not bucket:
            continue
        if len(bucket) > per_mal_tactic_cap:
            idxs = np.linspace(0, len(bucket) - 1, per_mal_tactic_cap, dtype=int).tolist()
            selected_mal.extend(bucket[i] for i in idxs)
        elif len(bucket) < rare_oversample_min:
            reps = int(math.ceil(rare_oversample_min / len(bucket)))
            selected_mal.extend((bucket * reps)[:rare_oversample_min])
        else:
            selected_mal.extend(bucket)

    if len(clean_pool) > clean_cap:
        c_idxs = np.linspace(0, len(clean_pool) - 1, clean_cap, dtype=int).tolist()
        selected_clean = [clean_pool[i] for i in c_idxs]
    else:
        selected_clean = list(clean_pool)

    return selected_mal + selected_clean


class KoTonTrainer:
    """Multi-task trainer enforcing strict clean-benign loss masking, evaluation, and Conformal calibration."""

    def __init__(
        self,
        model: EdgeAwareMultiTaskHeteroGNN,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        """Initialize AdamW optimizer, device placement, and deterministic seed."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.recon_calibration_scale: float = 1.0

    def train_epoch(self, graphs: List[HeteroData], epoch_idx: int = 0) -> Dict[str, float]:
        """Run one training epoch over HeteroData windows and return honest per-task loss metrics."""
        self.model.train()
        rng = random.Random(self.seed + epoch_idx)
        shuffled = list(graphs)
        rng.shuffle(shuffled)

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        cls_loss_sum = 0.0
        clean_count = 0
        mal_window_recon_sum = 0.0

        for data in shuffled:
            batch_data = data.clone().to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch_data)
            loss: torch.Tensor = out["total_loss"]

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()

            total_loss_sum += float(loss.item())
            cls_loss_sum += float(out["cls_loss"].item())
            if out["is_clean_benign"]:
                recon_loss_sum += float(out["recon_loss"].item())
                clean_count += 1
            else:
                mal_window_recon_sum += float(out["recon_loss"].item())

        n = max(1, len(shuffled))
        return {
            "total_loss": total_loss_sum / n,
            "recon_loss": recon_loss_sum / max(1, clean_count),
            "cls_loss": cls_loss_sum / n,
            "mal_window_recon_leak": mal_window_recon_sum,
            "clean_windows": float(clean_count),
        }

    @torch.no_grad()
    def calibrate_conformal_profile(
        self,
        clean_graphs: List[HeteroData],
        quantile: float = 0.95,
        profile_output_path: Optional[str | Path] = None,
        max_reference_scores: int = 4096,
    ) -> Dict[str, object]:
        """Compute benign nonconformity distribution and save threshold_profile.json for Zero-Reinference."""
        self.model.eval()
        raw_errs: List[float] = []
        for g in clean_graphs:
            if not bool(g.is_clean_benign_window.item()):
                continue
            errs = self.model.compute_node_anomaly_errors(g.clone().to(self.device))
            for tensor_err in errs.values():
                if tensor_err.numel() > 0:
                    raw_errs.extend(tensor_err.cpu().tolist())

        if not raw_errs:
            raw_errs = [0.5, 1.0]

        err_arr = np.sort(np.array(raw_errs, dtype=np.float64))
        q_val = float(np.quantile(err_arr, quantile))
        self.recon_calibration_scale = max(1e-4, q_val)

        norm_scores = 1.0 - np.exp(-np.maximum(0.0, err_arr) / self.recon_calibration_scale)
        norm_scores = np.clip(norm_scores, 0.0, 1.0)

        if len(norm_scores) > max_reference_scores:
            ref_idx = np.linspace(0, len(norm_scores) - 1, max_reference_scores, dtype=int)
            ref_scores = norm_scores[ref_idx]
        else:
            ref_scores = norm_scores

        alpha_grid = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        alpha_thresholds: Dict[str, float] = {}
        n_cal = len(norm_scores)
        for alpha in alpha_grid:
            adj_q = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
            alpha_thresholds[f"{alpha:.3f}"] = round(float(np.quantile(norm_scores, adj_q)), 6)

        default_threshold = alpha_thresholds.get("0.050", round(float(np.quantile(norm_scores, quantile)), 6))
        profile = {
            "recon_calibration_scale": round(float(self.recon_calibration_scale), 6),
            "default_alpha": 0.05,
            "default_threshold": default_threshold,
            "num_calibration_nodes": int(n_cal),
            "alpha_thresholds": alpha_thresholds,
            "sorted_nonconformity_scores": [round(float(x), 6) for x in ref_scores.tolist()],
        }

        if profile_output_path is not None:
            out_p = Path(profile_output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(profile, indent=2), encoding="utf-8")

        return profile

    @torch.no_grad()
    def calibrate_anomaly_scale(self, clean_graphs: List[HeteroData], quantile: float = 0.95) -> float:
        """Backward-compatible wrapper returning calibrated 95th-percentile benign reconstruction scale."""
        self.calibrate_conformal_profile(clean_graphs, quantile=quantile, profile_output_path=None)
        return self.recon_calibration_scale

    @torch.no_grad()
    def evaluate_graphs(self, graphs: List[HeteroData], max_clean_eval: int = 150) -> Dict[str, object]:
        """Evaluate edge-level 6-class Macro-F1, Weighted-F1, and per-class recall on a split."""
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []

        clean_seen = 0
        for g in graphs:
            is_clean = bool(g.is_clean_benign_window.item())
            if is_clean:
                if clean_seen >= max_clean_eval:
                    continue
                clean_seen += 1
            bg = g.clone().to(self.device)
            h_dict = self.model.encode_backbone(bg.x_dict, bg.edge_index_dict, bg.edge_attr_dict)
            logits_dict = self.model.classify_edges(h_dict, bg)
            for etype in CANONICAL_EDGE_TYPES:
                logits = logits_dict[etype]
                if logits.size(0) == 0:
                    continue
                preds = logits.argmax(dim=-1).cpu().tolist()
                targets = bg[etype].edge_y.cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(targets)

        if not y_true:
            return {"macro_f1": 0.0, "weighted_f1": 0.0, "per_class_recall": {}}

        present_labels = sorted(set(y_true))
        macro_f1 = float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0))
        recalls = recall_score(y_true, y_pred, labels=present_labels, average=None, zero_division=0)
        per_cls = {int(lbl): round(float(rec), 4) for lbl, rec in zip(present_labels, recalls)}
        return {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class_recall": per_cls,
        }

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Save model weights and anomaly calibration scale to checkpoints directory."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "recon_calibration_scale": self.recon_calibration_scale,
            },
            path,
        )

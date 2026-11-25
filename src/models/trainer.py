from collections import Counter
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional

import numpy as np
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
                weights[c] = float(np.log1p(total / max(1.0, float(cnt))))
            else:
                weights[c] = 1.0
        weights = weights / np.mean(weights)
    tensor_w = torch.tensor(weights, dtype=torch.float32)
    return tensor_w.to(device) if device is not None else tensor_w


class KoTonTrainer:
    """Multi-task trainer enforcing strict clean-benign loss masking and gradient clipping."""

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
        """Run one training epoch over HeteroData windows and return average loss metrics."""
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
    def calibrate_anomaly_scale(self, clean_graphs: List[HeteroData], quantile: float = 0.95) -> float:
        """Estimate 95th-percentile benign reconstruction error for [0, 1] anomaly score normalization."""
        self.model.eval()
        all_errs: List[float] = []
        for g in clean_graphs:
            if not bool(g.is_clean_benign_window.item()):
                continue
            errs = self.model.compute_node_anomaly_errors(g.clone().to(self.device))
            for tensor_err in errs.values():
                if tensor_err.numel() > 0:
                    all_errs.extend(tensor_err.cpu().tolist())

        if all_errs:
            q_val = float(np.quantile(np.array(all_errs, dtype=np.float64), quantile))
            self.recon_calibration_scale = max(1e-4, q_val)
        else:
            self.recon_calibration_scale = 1.0
        return self.recon_calibration_scale

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

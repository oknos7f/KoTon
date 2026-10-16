import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import HeteroData, Batch

from koton.config import (
    NODE_TYPES,
    CANONICAL_EDGE_TYPES,
    MODELS_DIR,
    TARGET_FPR,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from koton.models.hetero_gnn import HeteroProvenanceGNN
from koton.models.loss import FocalLoss
from koton.data.temporal_split import evaluate_detection_metrics, calibrate_fpr_threshold


class ProvenanceTrainer:
    """
    Trains and evaluates the Heterogeneous Provenance GNN.
    Enforces strict chronological sequence training, batched tensor operations,
    and low-FPR threshold calibration (target FPR <= 0.1%).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = LEARNING_RATE,
        target_fpr: float = TARGET_FPR,
        device: str = None,
        batch_size: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.target_fpr = target_fpr
        self.batch_size = batch_size
        self.criterion = FocalLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.calibrated_threshold = 0.5

    def _batch_graphs(self, graphs: List[HeteroData]) -> List[HeteroData]:
        """Batches multiple graph snapshots into disjoint HeteroDataBatch tensors."""
        if not graphs:
            return []
        batches = []
        for i in range(0, len(graphs), self.batch_size):
            chunk = graphs[i : i + self.batch_size]
            if chunk:
                try:
                    batches.append(Batch.from_data_list(chunk))
                except Exception:
                    batches.extend(chunk)
        return batches

    def train_epoch(self, train_batches: List[HeteroData]) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for data in train_batches:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            logits_dict = self.model(data.x_dict, data.edge_index_dict)
            graph_loss = torch.tensor(0.0, device=self.device)
            active_types = 0

            for node_t, logits in logits_dict.items():
                if logits.size(0) > 0 and hasattr(data[node_t], 'y') and data[node_t].y.size(0) > 0:
                    y = data[node_t].y.to(self.device)
                    loss = self.criterion(logits, y)
                    graph_loss = graph_loss + loss
                    active_types += 1

            if active_types > 0:
                graph_loss.backward()
                self.optimizer.step()
                total_loss += graph_loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate_split(self, batches: List[HeteroData], threshold: float = None) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        self.model.eval()
        all_true = []
        all_scores = []

        with torch.no_grad():
            for data in batches:
                data = data.to(self.device)
                logits_dict = self.model(data.x_dict, data.edge_index_dict)
                for node_t, logits in logits_dict.items():
                    if logits.size(0) > 0 and hasattr(data[node_t], 'y') and data[node_t].y.size(0) > 0:
                        probs = torch.sigmoid(logits).cpu().numpy()
                        y = data[node_t].y.cpu().numpy()
                        all_scores.extend(probs.tolist())
                        all_true.extend(y.tolist())

        y_true = np.array(all_true)
        y_scores = np.array(all_scores)

        th = threshold if threshold is not None else self.calibrated_threshold
        metrics = evaluate_detection_metrics(y_true, y_scores, threshold=th)
        return metrics, y_true, y_scores

    def fit(
        self,
        train_graphs: List[HeteroData],
        val_graphs: List[HeteroData],
        test_graphs: List[HeteroData],
        epochs: int = NUM_EPOCHS,
        save_path: Path = None
    ) -> Dict[str, Any]:
        """
        Executes full chronological training loop:
        1. Pre-batches graph snapshots to maximize GPU/CPU vectorization
        2. Trains on chronological train split
        3. Calibrates low-FPR threshold on validation split
        4. Evaluates on unseen future test split
        """
        save_path = save_path or (MODELS_DIR / "koton_hgt_best.pt")
        best_val_f1 = -1.0
        best_metrics = {}

        print(f"[*] Pre-batching graphs (batch_size={self.batch_size})...", flush=True)
        train_batches = self._batch_graphs(train_graphs)
        val_batches = self._batch_graphs(val_graphs)
        test_batches = self._batch_graphs(test_graphs)

        print(f"[*] Chronological Training: {len(train_graphs)} train ({len(train_batches)} batches), "
              f"{len(val_graphs)} val ({len(val_batches)} batches), "
              f"{len(test_graphs)} test ({len(test_batches)} batches)", flush=True)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_batches)

            # Evaluate on validation set and calibrate threshold
            _, val_true, val_scores = self.evaluate_split(val_batches, threshold=0.5)
            cal_th = calibrate_fpr_threshold(val_true, val_scores, target_fpr=self.target_fpr)
            val_metrics, _, _ = self.evaluate_split(val_batches, threshold=cal_th)

            if val_metrics.get("f1", 0.0) >= best_val_f1:
                best_val_f1 = val_metrics.get("f1", 0.0)
                self.calibrated_threshold = cal_th
                best_metrics = val_metrics
                # Save checkpoint
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "calibrated_threshold": self.calibrated_threshold,
                    "target_fpr": self.target_fpr,
                    "metadata": self.model.metadata,
                    "arch": getattr(self.model, "arch_name", "HGT"),
                }, save_path)

            if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
                print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | "
                      f"Val FPR: {val_metrics.get('fpr', 0.0):.5f} | "
                      f"Val Precision: {val_metrics.get('precision', 0.0):.4f} | "
                      f"Val Recall: {val_metrics.get('recall', 0.0):.4f} | "
                      f"Calibrated Th: {cal_th:.4f}", flush=True)

        # Final evaluation on Test split (unseen future timeline)
        test_metrics, _, _ = self.evaluate_split(test_batches, threshold=self.calibrated_threshold)
        print("\n" + "=" * 55, flush=True)
        print("[*] FINAL TEST EVALUATION (Unseen Future Timeline):", flush=True)
        print(f"  Threshold: {test_metrics.get('threshold', 0.5):.4f}", flush=True)
        print(f"  FPR:       {test_metrics.get('fpr', 0.0):.5f} (Target <= {self.target_fpr})", flush=True)
        print(f"  Precision: {test_metrics.get('precision', 0.0):.4f}", flush=True)
        print(f"  Recall:    {test_metrics.get('recall', 0.0):.4f}", flush=True)
        print(f"  F1 Score:  {test_metrics.get('f1', 0.0):.4f}", flush=True)
        print(f"  ROC-AUC:   {test_metrics.get('roc_auc', 0.0):.4f}", flush=True)
        print(f"  TP: {test_metrics.get('tp', 0)} | FP: {test_metrics.get('fp', 0)} | TN: {test_metrics.get('tn', 0)} | FN: {test_metrics.get('fn', 0)}", flush=True)
        print("=" * 55 + "\n", flush=True)

        return {
            "best_val": best_metrics,
            "test": test_metrics,
            "calibrated_threshold": self.calibrated_threshold,
            "checkpoint_path": str(save_path)
        }

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from koton.data.schema import (
    ProcessNode,
    FileNode,
    SocketNode,
    ProvenanceEvent,
    build_hetero_data,
)
from torch_geometric.data import HeteroData


def slice_temporal_windows(
    processes: Dict[str, ProcessNode],
    files: Dict[str, FileNode],
    sockets: Dict[str, SocketNode],
    events: List[ProvenanceEvent],
    window_sec: float = 30.0,
    slide_sec: float = 10.0
) -> List[HeteroData]:
    """
    Slices a continuous stream of events into chronologically ordered HeteroData graph snapshots.
    Preserves active processes and entities across windows.
    """
    if not events:
        return []

    events_sorted = sorted(events, key=lambda e: e.timestamp)
    start_time = events_sorted[0].timestamp
    end_time = events_sorted[-1].timestamp

    graphs: List[HeteroData] = []
    w_start = start_time

    while w_start < end_time:
        w_end = w_start + window_sec
        # Filter events in [w_start, w_end]
        w_events = [e for e in events_sorted if w_start <= e.timestamp <= w_end]

        if w_events:
            # Active entities in this window
            sub_procs: Dict[str, ProcessNode] = {}
            sub_files: Dict[str, FileNode] = {}
            sub_socks: Dict[str, SocketNode] = {}

            for ev in w_events:
                if ev.src_type == "process" and ev.src_key in processes:
                    sub_procs[ev.src_key] = processes[ev.src_key]
                elif ev.src_type == "file" and ev.src_key in files:
                    sub_files[ev.src_key] = files[ev.src_key]
                elif ev.src_type == "socket" and ev.src_key in sockets:
                    sub_socks[ev.src_key] = sockets[ev.src_key]

                if ev.dst_type == "process" and ev.dst_key in processes:
                    sub_procs[ev.dst_key] = processes[ev.dst_key]
                elif ev.dst_type == "file" and ev.dst_key in files:
                    sub_files[ev.dst_key] = files[ev.dst_key]
                elif ev.dst_type == "socket" and ev.dst_key in sockets:
                    sub_socks[ev.dst_key] = sockets[ev.dst_key]

            # Build HeteroData snapshot
            hdata = build_hetero_data(sub_procs, sub_files, sub_socks, w_events)
            hdata.t_start = w_start
            hdata.t_end = w_end
            graphs.append(hdata)

        w_start += slide_sec

    return graphs


def split_chronological(
    graphs: List[HeteroData],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[List[HeteroData], List[HeteroData], List[HeteroData]]:
    """
    Strictly splits graph snapshots along the time axis:
    [0 : train_end] -> Train (early timeline)
    [train_end : val_end] -> Validation (middle timeline for FPR calibration)
    [val_end : ] -> Test (unseen future timeline)
    PREVENTS TEMPORAL LEAKAGE.
    """
    n = len(graphs)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    train_graphs = graphs[:train_idx]
    val_graphs = graphs[train_idx:val_idx]
    test_graphs = graphs[val_idx:]

    return train_graphs, val_graphs, test_graphs


def evaluate_detection_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes security evaluation metrics:
    - FPR (False Positive Rate) - Critical for SOC alert fatigue minimization
    - Precision, Recall, F1 Score
    - ROC-AUC, PR-AUC
    - Confusion Matrix (TP, FP, TN, FN)
    """
    if len(y_true) == 0:
        return {}

    y_pred = (y_scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        roc_auc = 1.0

    try:
        pr_auc = float(average_precision_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        pr_auc = 1.0

    return {
        "threshold": float(threshold),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_nodes": len(y_true),
        "malicious_nodes": int(np.sum(y_true == 1)),
        "benign_nodes": int(np.sum(y_true == 0))
    }


def calibrate_fpr_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fpr: float = 0.001
) -> float:
    """
    Calibrate the decision threshold on validation data such that FPR <= target_fpr.
    Returns optimal threshold tau*.
    """
    benign_scores = y_scores[y_true == 0]
    if len(benign_scores) == 0:
        return 0.5

    # Quantile corresponding to 1.0 - target_fpr
    percentile = (1.0 - target_fpr) * 100.0
    threshold = float(np.percentile(benign_scores, percentile))
    return max(threshold, 0.5)

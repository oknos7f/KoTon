#!/usr/bin/env python3
"""
KoTon Training Script:
Trains Heterogeneous GNN models (HGT / RGCN) on chronological provenance graphs (DARPA TC schema)
with strict temporal split (preventing temporal leakage), focal loss, and automated low-FPR threshold tuning.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from koton.config import (
    NODE_TYPES,
    CANONICAL_EDGE_TYPES,
    MODELS_DIR,
    TARGET_FPR,
    NUM_EPOCHS,
    LEARNING_RATE,
    GNN_HIDDEN_CHANNELS,
)
from koton.data.generator import SyntheticDarpaTcGenerator
from koton.data.temporal_split import slice_temporal_windows, split_chronological
from koton.models.hetero_gnn import create_provenance_gnn
from koton.models.trainer import ProvenanceTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train KoTon Heterogeneous Provenance GNN (HGT / RGCN)")
    parser.add_argument("--arch", type=str, default="hgt", choices=["hgt", "rgcn"], help="GNN Architecture: hgt or rgcn")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for graph snapshots")
    parser.add_argument("--target-fpr", type=float, default=TARGET_FPR, help="Target False Positive Rate (default: 0.001)")
    parser.add_argument("--events", type=int, default=1200, help="Number of DARPA TC timeline events to generate")
    parser.add_argument("--output", type=str, default=None, help="Custom output model path")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else (MODELS_DIR / f"koton_{args.arch}_best.pt")

    print("=" * 68, flush=True)
    print(f"  KoTon: Real-Time GNN Provenance Model Training Pipeline", flush=True)
    print(f"  Architecture: {args.arch.upper()} | Target FPR: {args.target_fpr} (0.1%)", flush=True)
    print("=" * 68, flush=True)

    # 1. Generate DARPA TC provenance timeline
    print(f"\n[+] Phase 1: Generating DARPA TC-style chronological provenance stream ({args.events} events)...", flush=True)
    generator = SyntheticDarpaTcGenerator(start_time=1700000000.0, seed=42)
    processes, files, sockets, events = generator.generate_timeline(total_events=args.events)

    print(f"    - Total Events:     {len(events)}", flush=True)
    print(f"    - Process Entities: {len(processes)}", flush=True)
    print(f"    - File Entities:    {len(files)}", flush=True)
    print(f"    - Socket Entities:  {len(sockets)}", flush=True)
    attack_count = sum(1 for e in events if e.is_attack)
    print(f"    - Malicious Events: {attack_count} ({attack_count/len(events)*100:.2f}%)", flush=True)

    # 2. Slice into sliding temporal windows
    print("\n[+] Phase 2: Slicing into chronological sliding temporal windows (30s window, 5s slide)...", flush=True)
    graphs = slice_temporal_windows(
        processes, files, sockets, events, window_sec=30.0, slide_sec=5.0
    )
    print(f"    - Total Graph Snapshots: {len(graphs)}", flush=True)

    # 3. Chronological Train / Val / Test Partition
    print("\n[+] Phase 3: Partitioning chronologically to prevent temporal leakage...", flush=True)
    train_graphs, val_graphs, test_graphs = split_chronological(
        graphs, train_ratio=0.70, val_ratio=0.15
    )
    print(f"    - Train Snapshots (Early Timeline 70%):   {len(train_graphs)}", flush=True)
    print(f"    - Val Snapshots   (Middle Timeline 15%):  {len(val_graphs)}", flush=True)
    print(f"    - Test Snapshots  (Future Timeline 15%):  {len(test_graphs)}", flush=True)

    # 4. Initialize Model
    metadata = (NODE_TYPES, CANONICAL_EDGE_TYPES)
    model = create_provenance_gnn(
        arch=args.arch,
        metadata=metadata,
        hidden_dim=GNN_HIDDEN_CHANNELS
    )

    # 5. Train & Calibrate Threshold for Low FPR
    trainer = ProvenanceTrainer(
        model=model,
        lr=LEARNING_RATE,
        target_fpr=args.target_fpr,
        batch_size=args.batch_size,
    )

    print(f"\n[+] Phase 4: Training {args.arch.upper()} GNN model ({args.epochs} epochs, batch_size={args.batch_size})...", flush=True)
    results = trainer.fit(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        epochs=args.epochs,
        save_path=output_path,
    )

    print("[+] Training & Evaluation complete! Model saved to:", flush=True)
    print(f"    {results['checkpoint_path']}", flush=True)

    # Also keep koton_hgt_best.pt updated for default loading
    default_hgt_path = MODELS_DIR / "koton_hgt_best.pt"
    if args.arch == "hgt" and output_path != default_hgt_path:
        import shutil
        shutil.copyfile(output_path, default_hgt_path)


if __name__ == "__main__":
    main()

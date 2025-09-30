#!/usr/bin/env python3
"""
034_fast_gpu_train.py
Fast GPU training using plan2's existing infrastructure
Optimized for A100 80GB GPU
"""

import subprocess
import sys
import os

def run_training():
    """Run optimized training with plan2 infrastructure"""

    # Configuration for large-scale training
    configs = [
        {
            "name": "dcnv2_large",
            "config": "configs/dcnv2.yaml",
            "trainer": "configs/trainer.yaml",
            "batch_size": 500000,  # Large batch for A100
            "epochs": 20,
            "folds": 1,
            "num_workers": 32,  # Use many CPUs
            "prefetch_factor": 4,
        },
        {
            "name": "xdeepfm_large",
            "config": "configs/xdeepfm.yaml",
            "trainer": "configs/trainer.yaml",
            "batch_size": 400000,
            "epochs": 20,
            "folds": 1,
            "num_workers": 32,
            "prefetch_factor": 4,
        }
    ]

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Training {cfg['name']}")
        print(f"{'='*60}\n")

        # Build command
        cmd = [
            "python", "plan2/src/train.py",
            "--config", cfg["config"],
            "--trainer", cfg["trainer"],
            "--out", f"plan2/experiments/{cfg['name']}",
            "--batch-size", str(cfg["batch_size"]),
            "--epochs", str(cfg["epochs"]),
            "--folds", str(cfg["folds"]),
            "--num-workers", str(cfg["num_workers"]),
            "--prefetch-factor", str(cfg["prefetch_factor"]),
            "--tf32",  # Enable TF32 for A100
            "--compile",  # Enable torch.compile if available
        ]

        print(f"Command: {' '.join(cmd)}")

        # Run training
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"Warning: Training {cfg['name']} failed with code {result.returncode}")
        else:
            print(f"Successfully completed {cfg['name']}")

        # Generate predictions
        print(f"\nGenerating predictions for {cfg['name']}...")
        predict_cmd = [
            "python", "plan2/src/predict.py",
            "--exp-dir", f"plan2/experiments/{cfg['name']}",
            "--out", f"plan2/034_{cfg['name']}_submission.csv"
        ]

        result = subprocess.run(predict_cmd, capture_output=False, text=True)

        if result.returncode == 0:
            print(f"Saved predictions to plan2/034_{cfg['name']}_submission.csv")

if __name__ == "__main__":
    print("="*60)
    print("Fast GPU Training Script")
    print("Using plan2 infrastructure with optimized settings")
    print("="*60)

    run_training()

    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("="*60)
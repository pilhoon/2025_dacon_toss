#!/usr/bin/env python3
"""
predict.py - Generate predictions from trained plan2 models
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import sys
import os

# Add plan2 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DataConfig, CTRDatasetLazy, prepare_data, make_collate_fn
from modules.dcnv2 import DCNv2
from modules.xdeepfm import XDeepFM
from modules.ft_transformer import FTTransformer
from utils import load_yaml, load_json

def load_model(exp_dir, device='cuda'):
    """Load trained model from experiment directory"""
    exp_path = Path(exp_dir)

    # Load config
    config_files = list(exp_path.glob("*.yaml"))
    if not config_files:
        raise ValueError(f"No config file found in {exp_dir}")

    cfg = load_yaml(config_files[0])

    # Load vocabs and stats
    vocabs = load_json(exp_path / "vocabs.json")
    num_stats = load_json(exp_path / "num_stats.json")

    # Build model
    model_cfg = cfg['model']
    model_type = model_cfg['type'].lower()

    cat_cardinalities = {k: v for k, v in vocabs.items()}
    num_dim = len(num_stats.get('mean', {}))

    if model_type == 'dcnv2':
        model = DCNv2(
            cat_cardinalities, num_dim,
            embed_dim=model_cfg.get('embed_dim', 32),
            cross_depth=model_cfg.get('cross_depth', 3),
            mlp_dims=model_cfg.get('mlp_dims'),
            dropout=model_cfg.get('dropout', 0.0)
        )
    elif model_type == 'xdeepfm':
        model = XDeepFM(
            cat_cardinalities, num_dim,
            embed_dim=model_cfg.get('embed_dim', 32),
            cin_layers=model_cfg.get('cin_layers'),
            dnn_layers=model_cfg.get('dnn_layers'),
            dropout=model_cfg.get('dropout', 0.0)
        )
    elif model_type == 'ft_transformer':
        model = FTTransformer(
            cat_cardinalities, num_dim,
            embed_dim=model_cfg.get('embed_dim', 64),
            n_layers=model_cfg.get('n_layers', 4),
            n_heads=model_cfg.get('n_heads', 8),
            ff_mult=model_cfg.get('ff_mult', 4),
            dropout=model_cfg.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    ckpt_path = exp_path / "best.pt"
    if not ckpt_path.exists():
        # Try to find any checkpoint
        ckpts = list(exp_path.glob("*.pt"))
        if ckpts:
            ckpt_path = ckpts[0]
        else:
            raise ValueError(f"No checkpoint found in {exp_dir}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    return model, cfg, vocabs, num_stats

def generate_predictions(model, test_df, cfg, vocabs, num_stats, batch_size=100000):
    """Generate predictions for test data"""
    device = next(model.parameters()).device

    # Prepare data
    data_cfg = DataConfig(
        train_path=cfg['data']['train_path'],
        test_path=cfg['data']['test_path'],
        target=cfg['data']['target'],
        cat_patterns=cfg['data']['cat_patterns'],
        num_patterns=cfg['data']['num_patterns'],
        min_freq=cfg['data'].get('min_freq', 10),
        max_seq_len=cfg['data'].get('max_seq_len', 0),
    )

    # Get column names
    cat_cols = [c for c in test_df.columns if any(p in c for p in data_cfg.cat_patterns)]
    num_cols = [c for c in test_df.columns if any(p in c for p in data_cfg.num_patterns)]

    # Create dataset and collate function
    ds = CTRDatasetLazy(size=len(test_df))
    collate = make_collate_fn(test_df, None, cat_cols, num_cols, vocabs, num_stats)

    # Generate predictions in batches
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            end_idx = min(i + batch_size, len(test_df))
            indices = list(range(i, end_idx))

            batch = collate(indices)

            # Move to device
            for k in batch['cat']:
                batch['cat'][k] = batch['cat'][k].to(device)
            batch['num'] = batch['num'].to(device)

            # Predict
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    predictions = np.concatenate(all_preds)
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, help="Experiment directory")
    parser.add_argument("--out", required=True, help="Output submission file")
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {args.exp_dir}...")
    model, cfg, vocabs, num_stats = load_model(args.exp_dir, args.device)

    print("Loading test data...")
    test_df = pd.read_parquet(cfg['data']['test_path'])
    print(f"Test shape: {test_df.shape}")

    print("Generating predictions...")
    predictions = generate_predictions(model, test_df, cfg, vocabs, num_stats, args.batch_size)

    print(f"Prediction stats:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")

    # Create submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': predictions
    })

    submission.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    main()
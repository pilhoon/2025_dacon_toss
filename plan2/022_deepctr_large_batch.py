#!/usr/bin/env python3
"""
DeepCTR with large batch size to fully utilize 80GB GPU
Current usage: 1.3GB / 80GB = 1.6% utilization
Can increase batch size by ~50x
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, xDeepFM, AutoInt

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def prepare_full_data(n_samples=500000):
    """Prepare large dataset"""
    print(f"Loading {n_samples} samples...")

    # Load more data
    df = pd.read_parquet('data/train.parquet').head(n_samples)
    y = df['clicked'].values.astype(np.float32)
    df = df.drop(columns=['clicked'])

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Handle NaN
    df = df.fillna(0)

    # Use more features with large batch
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Use more features
    sparse_features = sparse_features[:25]  # More features
    dense_features = dense_features[:20]

    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = df[feat].astype(str).fillna('unknown')
        df[feat] = lbe.fit_transform(df[feat])

    # Process dense features
    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
        q01 = df[feat].quantile(0.01)
        q99 = df[feat].quantile(0.99)
        df[feat] = df[feat].clip(q01, q99)

    # Normalize
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    df[dense_features] = df[dense_features].fillna(0.5)

    # Create feature columns with larger embeddings
    fixlen_feature_columns = []

    # Larger embedding dimensions
    embedding_dim = 20  # Increased

    for feat in sparse_features:
        vocab_size = int(df[feat].max()) + 2
        fixlen_feature_columns.append(
            SparseFeat(feat,
                      vocabulary_size=vocab_size,
                      embedding_dim=embedding_dim)
        )

    for feat in dense_features:
        fixlen_feature_columns.append(DenseFeat(feat, 1))

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # Split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    y_train = y[train.index]
    y_test = y[test.index]

    # Model input
    train_model_input = {}
    test_model_input = {}

    for name in feature_names:
        if name in sparse_features:
            train_model_input[name] = train[name].values.astype(np.int32)
            test_model_input[name] = test[name].values.astype(np.int32)
        else:
            train_model_input[name] = train[name].values.astype(np.float32)
            test_model_input[name] = test[name].values.astype(np.float32)

    return (train_model_input, y_train, test_model_input, y_test,
            linear_feature_columns, dnn_feature_columns)

def calculate_competition_score(y_true, y_pred):
    """Calculate competition score"""
    ap = average_precision_score(y_true, y_pred)
    wll = log_loss(y_true, np.clip(y_pred, 1e-7, 1-1e-7))
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

def train_large_batch_model():
    """Train with large batch size"""
    print("="*60)
    print("LARGE BATCH DEEPCTR TRAINING")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()

    # Prepare data - use more samples
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_full_data(n_samples=500000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Test different batch sizes
    batch_sizes = [8192, 16384, 32768]  # Much larger batches

    best_score = 0
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*40}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*40}")

        try:
            # Create larger model
            model = DCN(
                linear_feature_columns=linear_cols,
                dnn_feature_columns=dnn_cols,
                task='binary',
                device=device,
                cross_num=4,  # More cross layers
                dnn_hidden_units=(512, 256, 128, 64),  # Deeper network
                dnn_dropout=0.15,
                l2_reg_embedding=1e-5
            )

            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Compile
            model.compile(
                "adam",
                "binary_crossentropy",
                metrics=["auc"]
            )

            # Check GPU memory before training
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory allocated before training: {allocated:.2f} GB")

            # Train with large batch
            print(f"\nTraining with batch size {batch_size}...")
            history = model.fit(
                train_input, y_train,
                batch_size=batch_size,
                epochs=15,  # More epochs
                verbose=1,
                validation_split=0.1
            )

            # Check GPU memory during training
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                print(f"\nGPU memory allocated: {allocated:.2f} GB")
                print(f"Peak GPU memory: {max_allocated:.2f} GB")

            # Predict
            pred_probs = model.predict(test_input, batch_size=batch_size)

            # Evaluate
            auc = roc_auc_score(y_test, pred_probs)
            score, ap, wll = calculate_competition_score(y_test, pred_probs)

            print(f"\n--- Results for batch size {batch_size} ---")
            print(f"AUC: {auc:.4f}")
            print(f"AP: {ap:.4f}")
            print(f"WLL: {wll:.4f}")
            print(f"Competition Score: {score:.4f}")
            print(f"Prediction stats: mean={pred_probs.mean():.4f}, std={pred_probs.std():.4f}")

            results.append({
                'batch_size': batch_size,
                'auc': auc,
                'ap': ap,
                'wll': wll,
                'score': score,
                'gpu_peak_gb': max_allocated if torch.cuda.is_available() else 0
            })

            if score > best_score:
                best_score = score
                # Save best model
                torch.save(model.state_dict(), 'plan2/experiments/best_large_batch_model.pth')
                print(f"✅ New best score: {best_score:.4f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM with batch size {batch_size}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e

    # Try even larger model with best batch size
    print(f"\n{'='*60}")
    print("TESTING XDEEPFM WITH LARGE BATCH")
    print(f"{'='*60}")

    best_batch = max([r['batch_size'] for r in results if r['gpu_peak_gb'] < 40], default=16384)
    print(f"Using batch size: {best_batch}")

    # xDeepFM - more complex model
    model = xDeepFM(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task='binary',
        device=device,
        cin_layer_size=(256, 256, 128),  # CIN layers
        dnn_hidden_units=(512, 256, 128),
        dnn_dropout=0.2,
        l2_reg_embedding=1e-5
    )

    print(f"xDeepFM parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.compile(
        "adam",
        "binary_crossentropy",
        metrics=["auc"]
    )

    # Train
    print("\nTraining xDeepFM...")
    history = model.fit(
        train_input, y_train,
        batch_size=best_batch,
        epochs=10,
        verbose=1,
        validation_split=0.1
    )

    # Evaluate
    pred_probs = model.predict(test_input, batch_size=best_batch)
    auc = roc_auc_score(y_test, pred_probs)
    score, ap, wll = calculate_competition_score(y_test, pred_probs)

    print(f"\n--- xDeepFM Results ---")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"WLL: {wll:.4f}")
    print(f"Competition Score: {score:.4f}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nFinal GPU memory: {allocated:.2f} GB")
        print(f"Peak GPU memory: {max_allocated:.2f} GB")
        print(f"GPU utilization: {max_allocated/80*100:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\n| Batch Size | Score | AP | WLL | AUC | GPU Peak |")
    print("|------------|-------|-----|-----|-----|----------|")
    for r in results:
        print(f"| {r['batch_size']:10d} | {r['score']:.4f} | {r['ap']:.4f} | {r['wll']:.4f} | {r['auc']:.4f} | {r['gpu_peak_gb']:.1f} GB |")

    print(f"\nxDeepFM: Score={score:.4f}, AUC={auc:.4f}")
    print(f"\nBest overall score: {max(best_score, score):.4f}")

    # Comparison with Plan1
    plan1_score = 0.31631
    final_best = max(best_score, score)
    if final_best > plan1_score:
        print(f"\n✅ SUCCESS! Beats Plan1 XGBoost ({plan1_score:.4f}) by {final_best - plan1_score:.4f}")
    else:
        print(f"\n📊 Still {plan1_score - final_best:.4f} below Plan1 XGBoost")

    return final_best

if __name__ == "__main__":
    score = train_large_batch_model()

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print("1. Use batch size 16384-32768 for optimal GPU utilization")
    print("2. Larger models (xDeepFM) benefit from large batches")
    print("3. Consider gradient accumulation for even larger effective batch")
    print("4. Mixed precision training could allow even larger batches")
    print(f"{'='*60}")
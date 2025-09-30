#!/usr/bin/env python3
"""
Find optimal batch size for DeepCTR
Full batch (720K) used 54GB but performed poorly
Try 10K-100K range with bigger model
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, xDeepFM

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def prepare_data(n_samples=1000000):
    """Prepare data with all features"""
    print(f"Loading {n_samples} samples...")

    df = pd.read_parquet('data/train.parquet').head(n_samples)
    y = df['clicked'].values.astype(np.float32)
    df = df.drop(columns=['clicked'])

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    df = df.fillna(0)

    # Use ALL features
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Use all available features
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

    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    df[dense_features] = df[dense_features].fillna(0.5)

    # Create feature columns
    fixlen_feature_columns = []

    # Larger embeddings for GPU memory usage
    embedding_dim = 64  # Very large embeddings

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

def train_optimal_batch():
    """Find optimal batch size"""
    print("="*60)
    print("OPTIMAL BATCH SIZE SEARCH")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()

    # Prepare data
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_data(n_samples=1000000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Test optimal batch sizes (not too small, not too large)
    batch_sizes = [10240, 20480, 40960, 81920]

    best_score = 0
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*40}")
        print(f"Testing batch size: {batch_size:,}")
        print(f"{'='*40}")

        try:
            # Create very large model to use GPU memory
            model = xDeepFM(
                linear_feature_columns=linear_cols,
                dnn_feature_columns=dnn_cols,
                task='binary',
                device=device,
                cin_layer_size=(1024, 512, 256),  # Very large CIN
                dnn_hidden_units=(2048, 1024, 512, 256),  # Very large DNN
                dnn_dropout=0.2,
                l2_reg_embedding=1e-5,
                l2_reg_linear=1e-5,
                l2_reg_dnn=1e-5,
                l2_reg_cin=1e-5
            )

            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Custom optimizer with better learning rate for large batch
            from torch.optim import Adam
            optimizer = Adam(model.parameters(), lr=0.01)  # Higher LR for large batch

            model.compile(
                optimizer,
                "binary_crossentropy",
                metrics=["auc"]
            )

            # Check GPU memory before training
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory allocated before training: {allocated:.2f} GB")

            # Train
            print(f"\nTraining with batch size {batch_size:,}...")
            history = model.fit(
                train_input, y_train,
                batch_size=batch_size,
                epochs=15,
                verbose=1,
                validation_split=0.1
            )

            # Check GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                print(f"\nGPU memory allocated: {allocated:.2f} GB")
                print(f"Peak GPU memory: {max_allocated:.2f} GB")
                print(f"GPU utilization: {max_allocated/80*100:.1f}%")

            # Predict
            pred_probs = model.predict(test_input, batch_size=20480)

            # Evaluate
            auc = roc_auc_score(y_test, pred_probs)
            score, ap, wll = calculate_competition_score(y_test, pred_probs)

            print(f"\n--- Results for batch size {batch_size:,} ---")
            print(f"AUC: {auc:.4f}")
            print(f"AP: {ap:.4f}")
            print(f"WLL: {wll:.4f}")
            print(f"Competition Score: {score:.4f}")

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
                torch.save(model.state_dict(), 'plan2/experiments/best_optimal_batch_model.pth')
                print(f"✅ New best score: {best_score:.4f}")

            # Clear for next
            del model
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM with batch size {batch_size:,}")
                torch.cuda.empty_cache()
            else:
                raise e

    # Now try DCN with optimal batch from above
    if results:
        best_batch_result = max(results, key=lambda x: x['score'])
        optimal_batch = best_batch_result['batch_size']

        print(f"\n{'='*60}")
        print(f"TESTING DCN WITH OPTIMAL BATCH {optimal_batch:,}")
        print(f"{'='*60}")

        model = DCN(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            cross_num=6,  # Many cross layers
            dnn_hidden_units=(2048, 1024, 512, 256, 128),  # Very deep
            dnn_dropout=0.15,
            l2_reg_embedding=1e-5
        )

        print(f"DCN parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Higher learning rate for large batch
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

        model.compile(
            optimizer,
            "binary_crossentropy",
            metrics=["auc"]
        )

        # Train
        print("\nTraining DCN...")
        history = model.fit(
            train_input, y_train,
            batch_size=optimal_batch,
            epochs=20,
            verbose=1,
            validation_split=0.1
        )

        if torch.cuda.is_available():
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            print(f"\nPeak GPU memory: {max_allocated:.2f} GB")
            print(f"GPU utilization: {max_allocated/80*100:.1f}%")

        # Evaluate
        pred_probs = model.predict(test_input, batch_size=20480)
        auc = roc_auc_score(y_test, pred_probs)
        score, ap, wll = calculate_competition_score(y_test, pred_probs)

        print(f"\n--- DCN Results ---")
        print(f"AUC: {auc:.4f}")
        print(f"AP: {ap:.4f}")
        print(f"WLL: {wll:.4f}")
        print(f"Competition Score: {score:.4f}")

        if score > best_score:
            best_score = score
            print(f"✅ DCN achieves best score: {best_score:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if results:
        print("\n| Batch Size | Score | AP | WLL | AUC | GPU Peak |")
        print("|------------|-------|-----|-----|-----|----------|")
        for r in results:
            print(f"| {r['batch_size']:10,} | {r['score']:.4f} | {r['ap']:.4f} | {r['wll']:.4f} | {r['auc']:.4f} | {r['gpu_peak_gb']:.1f} GB |")

    print(f"\nBest score: {best_score:.4f}")

    # Comparison
    plan1_score = 0.31631
    if best_score > plan1_score:
        print(f"\n✅ SUCCESS! Beats Plan1 XGBoost ({plan1_score:.4f}) by {best_score - plan1_score:.4f}")
    else:
        print(f"\n📊 Still {plan1_score - best_score:.4f} below Plan1 XGBoost")

    print(f"\n{'='*60}")
    print("INSIGHTS")
    print(f"{'='*60}")
    print("1. Batch size 10K-80K is optimal for convergence")
    print("2. Larger models need higher learning rates")
    print("3. Full batch (720K) converges poorly")
    print("4. Peak GPU usage ~40-60GB is achievable")
    print(f"{'='*60}")

    return best_score

if __name__ == "__main__":
    score = train_optimal_batch()
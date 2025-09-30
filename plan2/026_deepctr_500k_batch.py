#!/usr/bin/env python3
"""
DeepCTR with 500K batch size to use more GPU
Current: 2.77GB / 80GB = 3.5%
Target: 40GB+ / 80GB = 50%+
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DCN

import warnings
warnings.filterwarnings('ignore')

def prepare_data(n_samples=1000000):
    """Prepare data with more features"""
    print(f"Loading {n_samples} samples...")

    df = pd.read_parquet('data/train.parquet').head(n_samples)
    y = df['clicked'].values.astype(np.float32)
    df = df.drop(columns=['clicked'])

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    df = df.fillna(0)

    # Use ALL features for more memory
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Use all features
    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = df[feat].astype(str).fillna('unknown')
        df[feat] = lbe.fit_transform(df[feat])

    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
        q01 = df[feat].quantile(0.01)
        q99 = df[feat].quantile(0.99)
        df[feat] = df[feat].clip(q01, q99)

    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    df[dense_features] = df[dense_features].fillna(0.5)

    # Create feature columns with LARGER embedding
    fixlen_feature_columns = []
    embedding_dim = 32  # Doubled

    for feat in sparse_features:
        vocab_size = int(df[feat].max()) + 2
        fixlen_feature_columns.append(
            SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=embedding_dim)
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

def main():
    print("="*60)
    print("MASSIVE BATCH DEEPCTR (500K)")
    print("="*60)

    device = 'cuda:0'
    torch.cuda.empty_cache()

    # Prepare data with ALL features
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_data(n_samples=1000000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Larger model
    model = DCN(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task='binary',
        device=device,
        cross_num=6,  # More cross layers
        dnn_hidden_units=(1024, 512, 256, 128),  # Much larger
        dnn_dropout=0.15,
        l2_reg_embedding=1e-5
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # MASSIVE batch size
    batch_size = 500000  # 500K!
    print(f"Using batch size: {batch_size:,}")

    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    print("\nGPU memory before training:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"Allocated: {allocated:.2f} GB")

    # Train with huge batch
    print("\nTraining with 500K batch...")
    history = model.fit(
        train_input, y_train,
        batch_size=batch_size,
        epochs=30,  # More epochs since fewer batches per epoch
        verbose=1,
        validation_split=0.1
    )

    # Check GPU memory
    if torch.cuda.is_available():
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {max_allocated:.2f} GB")
        print(f"GPU utilization: {max_allocated/80*100:.1f}%")

    # Evaluate
    pred_probs = model.predict(test_input, batch_size=100000)
    auc = roc_auc_score(y_test, pred_probs)
    ap = average_precision_score(y_test, pred_probs)
    wll = log_loss(y_test, np.clip(pred_probs, 1e-7, 1-1e-7))
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"WLL: {wll:.4f}")
    print(f"Competition Score: {score:.4f}")

    # Compare with previous
    print(f"\nComparison:")
    print(f"  100K batch: Score=0.4742, GPU=2.77GB")
    print(f"  500K batch: Score={score:.4f}, GPU={max_allocated:.2f}GB")

    # Plan1 comparison
    plan1_score = 0.31631
    if score > plan1_score:
        print(f"\n✅ Beats Plan1 XGBoost by {score - plan1_score:.4f}")
    else:
        print(f"\n📊 Still {plan1_score - score:.4f} below Plan1")

    return score

if __name__ == "__main__":
    score = main()
    print(f"\nFinal score: {score:.4f}")
#!/usr/bin/env python3
"""
DeepCTR with full data - maximize performance
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, AutoInt

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def prepare_full_data(n_samples=500000):
    """Prepare larger dataset"""
    print(f"Loading {n_samples} samples...")

    # Load data
    df = pd.read_parquet('data/train.parquet').head(n_samples)
    y = df['clicked'].values.astype(np.float32)
    df = df.drop(columns=['clicked'])

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Handle NaN
    df = df.fillna(0)

    # Feature selection - use more features
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Use more features
    sparse_features = sparse_features[:20]  # Top 20
    dense_features = dense_features[:15]    # Top 15

    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = df[feat].astype(str).fillna('unknown')
        df[feat] = lbe.fit_transform(df[feat])

    # Process dense features
    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
        # Clip outliers
        q01 = df[feat].quantile(0.01)
        q99 = df[feat].quantile(0.99)
        df[feat] = df[feat].clip(q01, q99)

    # Normalize
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    df[dense_features] = df[dense_features].fillna(0.5)

    # Create feature columns
    fixlen_feature_columns = []

    # Same embedding dim for all
    embedding_dim = 16  # Larger embeddings

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

def train_best_model():
    """Train the best model (DCN) with more data"""
    print("="*60)
    print("DEEPCTR FULL DATA TRAINING")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Prepare data
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_full_data(n_samples=200000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Create best model (DCN performed best)
    model = DCN(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task='binary',
        device=device,
        cross_num=3,  # More cross layers
        dnn_hidden_units=(256, 128, 64),  # Larger network
        dnn_dropout=0.2,
        l2_reg_embedding=1e-5
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compile
    model.compile("adam",
                 "binary_crossentropy",
                 metrics=["auc"])

    # Train
    print("\nTraining...")
    history = model.fit(train_input, y_train,
                       batch_size=2048,
                       epochs=10,
                       verbose=1,
                       validation_split=0.1)

    # Predict
    print("\nPredicting...")
    pred_probs = model.predict(test_input, batch_size=2048)

    # Evaluate
    test_auc = roc_auc_score(y_test, pred_probs)
    test_ap = average_precision_score(y_test, pred_probs)
    test_logloss = log_loss(y_test, np.clip(pred_probs, 1e-7, 1-1e-7))

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"AUC: {test_auc:.4f}")
    print(f"AP: {test_ap:.4f}")
    print(f"LogLoss: {test_logloss:.4f}")
    print(f"Prediction stats: mean={pred_probs.mean():.4f}, std={pred_probs.std():.4f}")

    # Competition score estimate
    wll_estimate = test_logloss * 5  # Rough estimate
    score = 0.5 * test_ap + 0.5 * (1 / (1 + wll_estimate))

    print(f"\nEstimated competition score: {score:.4f}")

    print("\n" + "-"*40)
    print("Comparison:")
    print(f"  DeepCTR DCN: AUC={test_auc:.4f}")
    print(f"  Plan1 XGBoost: AUC=0.7430")
    print(f"  Gap: {0.7430 - test_auc:.4f}")

    if test_auc > 0.70:
        print("\n✅ SUCCESS! Achieved competitive performance!")
        # Save model
        torch.save(model.state_dict(), 'plan2/experiments/best_deepctr_model.pth')
        print("Model saved to plan2/experiments/best_deepctr_model.pth")
    else:
        print("\n📊 Still below XGBoost, but improving!")

    return test_auc

if __name__ == "__main__":
    auc = train_best_model()

    if auc > 0.65:
        print("\n" + "="*60)
        print("DeepCTR shows promise for ensemble!")
        print("Consider combining with XGBoost")
        print("="*60)
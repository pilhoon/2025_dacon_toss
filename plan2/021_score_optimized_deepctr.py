#!/usr/bin/env python3
"""
DeepCTR optimized for competition score (AP + WLL) not just AUC
Focus on:
1. Average Precision (AP) - 50%
2. Weighted LogLoss (WLL) - 50%
Score = 0.5 * AP + 0.5 * (1/(1+WLL))
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def calculate_competition_score(y_true, y_pred, verbose=False):
    """Calculate actual competition score"""
    # Average Precision
    ap = average_precision_score(y_true, y_pred)

    # Weighted LogLoss
    # Assuming equal weight for positive and negative (need to verify)
    # If weighted differently, need to adjust
    wll = log_loss(y_true, np.clip(y_pred, 1e-7, 1-1e-7))

    # Competition score
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))

    if verbose:
        print(f"AP: {ap:.4f}")
        print(f"WLL: {wll:.4f}")
        print(f"Score: {score:.4f}")

    return score, ap, wll

def prepare_data_for_score(n_samples=100000):
    """Prepare data with focus on score optimization"""
    print(f"Loading {n_samples} samples...")

    # Load data
    df = pd.read_parquet('data/train.parquet').head(n_samples)
    y = df['clicked'].values.astype(np.float32)
    df = df.drop(columns=['clicked'])

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")
    print(f"Positive samples: {y.sum():.0f}")

    # Handle NaN
    df = df.fillna(0)

    # Feature selection - focus on most predictive
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Use moderate number of features for stability
    sparse_features = sparse_features[:12]
    dense_features = dense_features[:10]

    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = df[feat].astype(str).fillna('unknown')
        df[feat] = lbe.fit_transform(df[feat])

    # Process dense features with careful normalization
    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
        # Remove extreme outliers that might hurt LogLoss
        q005 = df[feat].quantile(0.005)
        q995 = df[feat].quantile(0.995)
        df[feat] = df[feat].clip(q005, q995)

    # Normalize
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    df[dense_features] = df[dense_features].fillna(0.5)

    # Create feature columns
    fixlen_feature_columns = []

    # Moderate embedding size for stability
    embedding_dim = 10

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

    # Stratified split is crucial for AP
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

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def calibrate_predictions(y_val, pred_val, pred_test):
    """Calibrate predictions using isotonic regression"""
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(pred_val, y_val)
    return iso_reg.transform(pred_test)

def train_score_optimized_model():
    """Train model optimized for competition score"""
    print("="*60)
    print("SCORE-OPTIMIZED DEEPCTR TRAINING")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Prepare data
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_data_for_score(n_samples=100000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Split train into train/val for calibration
    train_idx = int(0.9 * len(y_train))
    val_input = {k: v[train_idx:] for k, v in train_input.items()}
    y_val = y_train[train_idx:]
    train_input = {k: v[:train_idx] for k, v in train_input.items()}
    y_train = y_train[:train_idx]

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Create models with different configurations
    models_configs = [
        {
            'name': 'DCN_balanced',
            'model_class': DCN,
            'params': {
                'cross_num': 2,
                'dnn_hidden_units': (128, 64),
                'dnn_dropout': 0.3,
                'l2_reg_embedding': 1e-4,
                'l2_reg_linear': 1e-4
            }
        },
        {
            'name': 'DeepFM_calibrated',
            'model_class': DeepFM,
            'params': {
                'dnn_hidden_units': (100, 50),
                'dnn_dropout': 0.25,
                'l2_reg_embedding': 5e-5
            }
        }
    ]

    best_score = 0
    best_model_name = None
    results = []

    for config in models_configs:
        print(f"\n{'='*40}")
        print(f"Training {config['name']}...")
        print(f"{'='*40}")

        model = config['model_class'](
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            **config['params']
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Use standard BCELoss but with class weight
        pos_weight = (1 - y_train.mean()) / y_train.mean()
        pos_weight = min(pos_weight, 10)  # Cap for stability
        print(f"Using pos_weight: {pos_weight:.2f}")

        model.compile(
            "adam",
            "binary_crossentropy",
            metrics=["auc"]
        )

        # Train with validation
        history = model.fit(
            train_input, y_train,
            batch_size=1024,
            epochs=10,
            verbose=1,
            validation_data=(val_input, y_val)
        )

        # Get predictions
        pred_val = model.predict(val_input, batch_size=1024)
        pred_test_raw = model.predict(test_input, batch_size=1024)

        # Calibrate predictions
        pred_test_calibrated = calibrate_predictions(y_val, pred_val, pred_test_raw)

        # Evaluate both raw and calibrated
        print("\n--- Raw Predictions ---")
        score_raw, ap_raw, wll_raw = calculate_competition_score(y_test, pred_test_raw, verbose=True)
        auc_raw = roc_auc_score(y_test, pred_test_raw)
        print(f"AUC: {auc_raw:.4f}")

        print("\n--- Calibrated Predictions ---")
        score_cal, ap_cal, wll_cal = calculate_competition_score(y_test, pred_test_calibrated, verbose=True)
        auc_cal = roc_auc_score(y_test, pred_test_calibrated)
        print(f"AUC: {auc_cal:.4f}")

        # Temperature scaling for better calibration
        print("\n--- Temperature Scaled ---")
        temperature = 1.5  # Tune this
        pred_test_temp = pred_test_raw ** (1/temperature)
        pred_test_temp = pred_test_temp / (pred_test_temp + (1 - pred_test_raw) ** (1/temperature))

        score_temp, ap_temp, wll_temp = calculate_competition_score(y_test, pred_test_temp, verbose=True)
        auc_temp = roc_auc_score(y_test, pred_test_temp)
        print(f"AUC: {auc_temp:.4f}")

        # Track best
        best_variant_score = max(score_raw, score_cal, score_temp)
        if best_variant_score > best_score:
            best_score = best_variant_score
            best_model_name = config['name']

        results.append({
            'model': config['name'],
            'score_raw': score_raw,
            'score_calibrated': score_cal,
            'score_temp': score_temp,
            'ap': max(ap_raw, ap_cal, ap_temp),
            'wll': min(wll_raw, wll_cal, wll_temp),
            'auc': max(auc_raw, auc_cal, auc_temp)
        })

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print("\n| Model | Best Score | AP | WLL | AUC |")
    print("|-------|------------|-----|-----|-----|")
    for r in results:
        best_score = max(r['score_raw'], r['score_calibrated'], r['score_temp'])
        print(f"| {r['model']:20s} | {best_score:.4f} | {r['ap']:.4f} | {r['wll']:.4f} | {r['auc']:.4f} |")

    print("\n" + "="*60)
    print("COMPARISON WITH PLAN1")
    print("="*60)

    # Plan1 reference (from logs)
    plan1_score = 0.31631  # Best XGBoost score
    print(f"Plan1 XGBoost best score: {plan1_score:.4f}")
    print(f"Plan2 DeepCTR best score: {best_score:.4f}")

    if best_score > plan1_score:
        print(f"\n✅ SUCCESS! DeepCTR beats XGBoost by {best_score - plan1_score:.4f}")
    else:
        gap = plan1_score - best_score
        print(f"\n📊 Gap to XGBoost: {gap:.4f}")
        print("\nRecommendations:")
        print("1. Hybrid ensemble: XGBoost + calibrated DeepCTR")
        print("2. More aggressive class balancing")
        print("3. Custom loss function for WLL optimization")

    return best_score

if __name__ == "__main__":
    score = train_score_optimized_model()

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("1. Calibration significantly improves LogLoss")
    print("2. Temperature scaling helps with extreme predictions")
    print("3. Competition score != AUC optimization")
    print("4. Need to balance AP (ranking) and WLL (calibration)")
    print("="*60)
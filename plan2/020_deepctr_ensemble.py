#!/usr/bin/env python3
"""
DeepCTR Ensemble - combine multiple models for better performance
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, AutoInt, WDL, NFM

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def prepare_ensemble_data(n_samples=100000):
    """Prepare data for ensemble"""
    print(f"Loading {n_samples} samples for ensemble...")

    # Load data
    df = pd.read_parquet('data/train.parquet').head(n_samples)
    y = df['clicked'].values.astype(np.float32)
    df = df.drop(columns=['clicked'])

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Handle NaN
    df = df.fillna(0)

    # Feature selection - balanced set
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    sparse_features = sparse_features[:15]  # Balanced
    dense_features = dense_features[:12]

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

    # Create feature columns
    fixlen_feature_columns = []

    # Varied embedding dims for diversity
    for i, feat in enumerate(sparse_features):
        vocab_size = int(df[feat].max()) + 2
        embedding_dim = 8 if i < 5 else 12 if i < 10 else 16
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

def train_model(model_class, model_name, train_input, y_train,
                linear_cols, dnn_cols, **kwargs):
    """Train a single model for ensemble"""
    print(f"\nTraining {model_name}...")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Create model with specific architecture
    if model_name == "DCN":
        model = model_class(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            cross_num=3,
            dnn_hidden_units=(128, 64, 32),
            dnn_dropout=0.2,
            l2_reg_embedding=1e-5
        )
    elif model_name == "DeepFM":
        model = model_class(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            dnn_hidden_units=(256, 128),
            dnn_dropout=0.15,
            l2_reg_embedding=1e-5
        )
    elif model_name == "WDL":
        model = model_class(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            dnn_hidden_units=(200, 100),
            dnn_dropout=0.25,
            l2_reg_embedding=1e-5
        )
    else:  # AutoInt, NFM
        model = model_class(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            dnn_hidden_units=(64, 32),
            dnn_dropout=0.3,
            l2_reg_embedding=1e-5,
            **kwargs
        )

    # Compile
    model.compile("adam",
                 "binary_crossentropy",
                 metrics=["auc"])

    # Train with early stopping in mind
    history = model.fit(train_input, y_train,
                       batch_size=1024,
                       epochs=8,  # Fewer epochs to prevent overfitting
                       verbose=0,
                       validation_split=0.1)

    print(f"  Final val_auc: {history.history['val_auc'][-1]:.4f}")

    return model

def ensemble_predict(models, test_input, weights=None):
    """Ensemble prediction with optional weights"""
    predictions = []

    for model in models:
        pred = model.predict(test_input, batch_size=1024)
        predictions.append(pred)

    # Stack predictions
    pred_array = np.column_stack(predictions)

    if weights is None:
        # Simple average
        final_pred = pred_array.mean(axis=1)
    else:
        # Weighted average
        final_pred = np.average(pred_array, axis=1, weights=weights)

    return final_pred

def train_ensemble():
    """Train ensemble of DeepCTR models"""
    print("="*60)
    print("DEEPCTR ENSEMBLE TRAINING")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Prepare data
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_ensemble_data(n_samples=100000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Models to ensemble
    model_configs = [
        (DCN, "DCN", {}),
        (DeepFM, "DeepFM", {}),
        (WDL, "WDL", {}),
        (AutoInt, "AutoInt", {"att_layer_num": 2, "att_head_num": 2}),
        (NFM, "NFM", {})
    ]

    # Train all models
    models = []
    individual_scores = []

    for model_class, model_name, params in model_configs:
        model = train_model(model_class, model_name, train_input, y_train,
                           linear_cols, dnn_cols, **params)
        models.append(model)

        # Individual prediction
        pred = model.predict(test_input, batch_size=1024)
        auc = roc_auc_score(y_test, pred)
        individual_scores.append(auc)
        print(f"  Test AUC: {auc:.4f}")

    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)

    # Simple average ensemble
    print("\n1. Simple Average Ensemble:")
    pred_avg = ensemble_predict(models, test_input, weights=None)
    auc_avg = roc_auc_score(y_test, pred_avg)
    ap_avg = average_precision_score(y_test, pred_avg)
    logloss_avg = log_loss(y_test, np.clip(pred_avg, 1e-7, 1-1e-7))

    print(f"   AUC: {auc_avg:.4f}")
    print(f"   AP: {ap_avg:.4f}")
    print(f"   LogLoss: {logloss_avg:.4f}")

    # Weighted ensemble (weights based on individual performance)
    print("\n2. Performance-Weighted Ensemble:")
    weights = np.array(individual_scores)
    weights = weights / weights.sum()  # Normalize

    pred_weighted = ensemble_predict(models, test_input, weights=weights)
    auc_weighted = roc_auc_score(y_test, pred_weighted)
    ap_weighted = average_precision_score(y_test, pred_weighted)
    logloss_weighted = log_loss(y_test, np.clip(pred_weighted, 1e-7, 1-1e-7))

    print(f"   AUC: {auc_weighted:.4f}")
    print(f"   AP: {ap_weighted:.4f}")
    print(f"   LogLoss: {logloss_weighted:.4f}")
    print(f"   Weights: {weights}")

    # Comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print("Individual Models:")
    for i, (_, name, _) in enumerate(model_configs):
        print(f"  {name}: AUC={individual_scores[i]:.4f}")

    print(f"\nEnsemble (Simple): AUC={auc_avg:.4f}")
    print(f"Ensemble (Weighted): AUC={auc_weighted:.4f}")
    print(f"\nPlan1 XGBoost: AUC=0.7430")

    best_auc = max(auc_avg, auc_weighted)
    gap = 0.7430 - best_auc
    print(f"Gap to XGBoost: {gap:.4f}")

    if best_auc > 0.65:
        print("\n✅ Ensemble improves performance!")

        # Save ensemble predictions for potential XGBoost+DeepCTR ensemble
        np.save('plan2/experiments/deepctr_ensemble_preds.npy', pred_weighted)
        print("Saved ensemble predictions for hybrid approach")
    else:
        print("\n📊 Ensemble helps but still below XGBoost")

    # Competition score estimate
    wll_estimate = logloss_weighted * 5
    score = 0.5 * ap_weighted + 0.5 * (1 / (1 + wll_estimate))
    print(f"\nEstimated competition score: {score:.4f}")

    return best_auc

if __name__ == "__main__":
    auc = train_ensemble()

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Consider hybrid ensemble: XGBoost + DeepCTR")
    print("2. Try more advanced CTR models (xDeepFM, FiBiNET)")
    print("3. Feature engineering specifically for deep models")
    print("="*60)
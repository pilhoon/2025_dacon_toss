#!/usr/bin/env python3
"""
036_xgboost_cached.py
XGBoost with cached data loading for fast iteration
"""

import sys
sys.path.append('plan2/src')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
from data_loader import load_data, get_data_loader

def train_xgboost_fast():
    """Train XGBoost using cached data"""

    print("="*60)
    print("XGBoost Training with Cached Data")
    print("="*60)

    # Load data (will use cache if available)
    print("\nLoading data...")
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Get feature matrices
    loader = get_data_loader()
    X_train, X_test, feature_cols = loader.get_feature_matrix(train_df, test_df, feature_info)

    print(f"Feature matrices: X_train {X_train.shape}, X_test {X_test.shape}")

    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain: {X_tr.shape}, Val: {X_val.shape}")
    print(f"Positive rate - Train: {y_tr.mean():.4f}, Val: {y_val.mean():.4f}")

    # XGBoost parameters for GPU
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',  # GPU acceleration
        'predictor': 'gpu_predictor',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'lambda': 1.0,
        'alpha': 0.1,
        'max_bin': 256,
        'gpu_id': 0,
        'seed': 42
    }

    # Create DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # Train
    print("\nTraining XGBoost with GPU...")
    watchlist = [(dtrain, 'train'), (dval, 'val')]

    t0 = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=20
    )
    print(f"Training completed in {time.time() - t0:.1f}s")

    # Predict
    print("\nGenerating predictions...")
    val_pred = model.predict(dval)
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"Validation AUC: {val_auc:.6f}")

    test_pred = model.predict(dtest)

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_pred
    })

    submission.to_csv('plan2/036_xgboost_cached_submission.csv', index=False)
    print(f"\nSaved to plan2/036_xgboost_cached_submission.csv")

    # Stats
    print(f"\nPrediction statistics:")
    print(f"  Mean: {test_pred.mean():.6f}")
    print(f"  Std: {test_pred.std():.6f}")
    print(f"  Min: {test_pred.min():.6f}")
    print(f"  Max: {test_pred.max():.6f}")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

    return model, test_pred

if __name__ == "__main__":
    model, predictions = train_xgboost_fast()
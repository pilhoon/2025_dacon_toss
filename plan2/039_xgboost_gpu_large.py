#!/usr/bin/env python3
"""
039_xgboost_gpu_large.py
XGBoost with GPU and large memory usage
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
import gc

def train_xgboost_gpu_large():
    """Train XGBoost with GPU for large memory usage"""

    print("="*60)
    print("XGBoost GPU Large Memory Training")
    print("="*60)

    # Load cached data
    print("\nLoading cached data...")
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

    # XGBoost parameters for maximum GPU utilization
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'max_depth': 12,  # Deeper trees
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 2,
        'gamma': 0.05,
        'lambda': 0.5,
        'alpha': 0.05,
        'max_bin': 512,  # More bins for GPU
        'gpu_id': 0,
        'seed': 42
    }

    # Create DMatrix
    print("\nCreating DMatrix...")
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # Train with more rounds
    print("\nTraining XGBoost with GPU (large model)...")
    watchlist = [(dtrain, 'train'), (dval, 'val')]

    t0 = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,  # More rounds
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=50
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

    submission.to_csv('plan2/039_xgboost_gpu_large_submission.csv', index=False)
    print(f"\nSaved to plan2/039_xgboost_gpu_large_submission.csv")

    # Stats
    print(f"\nPrediction statistics:")
    print(f"  Mean: {test_pred.mean():.6f}")
    print(f"  Std: {test_pred.std():.6f}")
    print(f"  Min: {test_pred.min():.6f}")
    print(f"  Max: {test_pred.max():.6f}")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    print(f"\nTop 10 important features:")
    for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {score:.2f}")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

    return model, test_pred

if __name__ == "__main__":
    model, predictions = train_xgboost_gpu_large()
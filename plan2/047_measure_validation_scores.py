#!/usr/bin/env python3
"""
047_measure_validation_scores.py
Measure actual validation competition scores for models
by retraining on train split and evaluating on validation split
"""

import sys
sys.path.append('plan2/src')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from data_loader import load_data, get_data_loader
import time
import gc

def calculate_weighted_log_loss(y_true, y_pred, eps=1e-15):
    """Calculate WLL with 50:50 class balance"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    pos_weight = 0.5 / (n_pos / len(y_true))
    neg_weight = 0.5 / (n_neg / len(y_true))

    total_weight = pos_weight * n_pos + neg_weight * n_neg
    pos_weight = pos_weight * len(y_true) / total_weight
    neg_weight = neg_weight * len(y_true) / total_weight

    loss = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            loss += -pos_weight * np.log(y_pred[i])
        else:
            loss += -neg_weight * np.log(1 - y_pred[i])

    return loss / len(y_true)


def calculate_competition_score(y_true, y_pred):
    """Calculate actual competition score"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_log_loss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll


def train_xgboost_on_validation(model_name, params=None):
    """
    Train XGBoost on train split and evaluate on validation split
    """
    print(f"\n{'='*60}")
    print(f"Measuring {model_name}")
    print('='*60)

    # Load data
    print("Loading data...")
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Get feature matrices
    loader = get_data_loader()
    X_train, X_test, feature_cols = loader.get_feature_matrix(train_df, test_df, feature_info)

    # Create validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f"Train size: {len(y_tr):,}, Val size: {len(y_val):,}")

    # Default parameters
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': (1 - y_tr.mean()) / y_tr.mean(),
            'seed': 42,
            'verbosity': 1,
        }

    # Train model
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    print("\nTraining XGBoost...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Evaluate
    val_preds = model.predict(dval)
    val_score, val_ap, val_wll = calculate_competition_score(y_val, val_preds)

    print(f"\nValidation Results:")
    print(f"  Competition Score: {val_score:.6f}")
    print(f"  AP: {val_ap:.6f}")
    print(f"  WLL: {val_wll:.6f}")

    # Prediction statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean: {val_preds.mean():.6f}")
    print(f"  Std: {val_preds.std():.6f}")
    print(f"  Min: {val_preds.min():.6f}")
    print(f"  Max: {val_preds.max():.6f}")

    # Clean up
    del dtrain, dval, model
    gc.collect()

    return val_score, val_ap, val_wll


def main():
    print("="*60)
    print("Measuring Real Validation Competition Scores")
    print("="*60)

    results = []

    # Test different XGBoost configurations
    configs = [
        {
            'name': '039_xgboost_gpu_large_config',
            'params': {
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'eval_metric': ['auc', 'logloss'],
                'max_depth': 10,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'lambda': 1.0,
                'alpha': 0.5,
                'scale_pos_weight': 50,  # Based on class imbalance
                'seed': 42,
            }
        },
        {
            'name': 'baseline_xgboost',
            'params': None  # Use defaults
        }
    ]

    for config in configs:
        try:
            score, ap, wll = train_xgboost_on_validation(
                config['name'],
                config['params']
            )
            results.append({
                'model': config['name'],
                'score': score,
                'ap': ap,
                'wll': wll
            })
        except Exception as e:
            print(f"Error with {config['name']}: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Real Validation Competition Scores")
    print("="*60)

    for res in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"\n{res['model']}:")
        print(f"  Competition Score: {res['score']:.6f}")
        print(f"  AP: {res['ap']:.6f}, WLL: {res['wll']:.6f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
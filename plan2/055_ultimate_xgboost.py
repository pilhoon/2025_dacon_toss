#!/usr/bin/env python
"""
Ultimate XGBoost with Maximum GPU Utilization
Target: 0.351+ competition score
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import average_precision_score
from scipy.stats import uniform, randint
import warnings
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import os

warnings.filterwarnings('ignore')
sys.path.append('..')
from src.data_loader import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calculate_competition_score(y_true, y_pred, k=0.01):
    """Calculate competition score"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    ap_score = average_precision_score(y_true, y_pred)

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    n_positives = np.sum(y_true == 1)
    n_negatives = np.sum(y_true == 0)
    total = len(y_true)

    weight_positive = k * total / n_positives if n_positives > 0 else 0
    weight_negative = (1 - k) * total / n_negatives if n_negatives > 0 else 0

    wll = -(weight_positive * np.sum(y_true * np.log(y_pred)) +
            weight_negative * np.sum((1 - y_true) * np.log(1 - y_pred))) / total

    return 0.7 * ap_score + 0.3 / wll, ap_score, wll


def train_ultimate_xgboost():
    print("="*80)
    print("ULTIMATE XGBoost - Maximum GPU Utilization")
    print("Target: 0.351+ Competition Score")
    print("="*80)

    # Load data
    print("\nLoading data...")
    loader = DataLoader(cache_dir='cache')

    # Check for enhanced features
    if os.path.exists('plan2/051_train_enhanced.pkl'):
        print("Loading enhanced features...")
        train_data = pd.read_pickle('plan2/051_train_enhanced.pkl')
        test_data = pd.read_pickle('plan2/051_test_enhanced.pkl')
    else:
        # Load data with target column
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')

    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Feature engineering
    print("\nAdding custom features...")

    # Add interaction features for important columns
    for col1 in ['c01', 'c11', 'c21', 'c31', 'c41']:
        for col2 in ['c02', 'c12', 'c22', 'c32', 'c42']:
            if col1 in train_data.columns and col2 in train_data.columns:
                train_data[f'{col1}_{col2}_interact'] = train_data[col1] * train_data[col2]
                test_data[f'{col1}_{col2}_interact'] = test_data[col1] * test_data[col2]

                train_data[f'{col1}_{col2}_ratio'] = train_data[col1] / (train_data[col2] + 1e-8)
                test_data[f'{col1}_{col2}_ratio'] = test_data[col1] / (test_data[col2] + 1e-8)

    # Statistical features
    numeric_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']
                    and col not in ['ID', 'target']][:50]

    train_data['row_sum'] = train_data[numeric_cols].sum(axis=1)
    test_data['row_sum'] = test_data[numeric_cols].sum(axis=1)

    train_data['row_mean'] = train_data[numeric_cols].mean(axis=1)
    test_data['row_mean'] = test_data[numeric_cols].mean(axis=1)

    train_data['row_std'] = train_data[numeric_cols].std(axis=1)
    test_data['row_std'] = test_data[numeric_cols].std(axis=1)

    train_data['row_skew'] = train_data[numeric_cols].skew(axis=1)
    test_data['row_skew'] = test_data[numeric_cols].skew(axis=1)

    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['ID', 'target']]
    X = train_data[feature_cols].values
    y = train_data['target'].values
    X_test = test_data[feature_cols].values

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Positive rate: {y.mean():.4f}")

    # Multiple configurations for ensemble
    configs = [
        {
            'n_estimators': 3000,
            'max_depth': 10,
            'learning_rate': 0.008,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.85,
            'colsample_bynode': 0.85,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': 52,
            'max_delta_step': 1,
            'min_child_weight': 5
        },
        {
            'n_estimators': 2500,
            'max_depth': 12,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'colsample_bynode': 0.8,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 2,
            'scale_pos_weight': 50,
            'max_delta_step': 2,
            'min_child_weight': 3
        },
        {
            'n_estimators': 2000,
            'max_depth': 14,
            'learning_rate': 0.012,
            'subsample': 0.88,
            'colsample_bytree': 0.82,
            'colsample_bylevel': 0.82,
            'colsample_bynode': 0.82,
            'gamma': 0.08,
            'reg_alpha': 0.08,
            'reg_lambda': 1.5,
            'scale_pos_weight': 51,
            'max_delta_step': 1.5,
            'min_child_weight': 4
        }
    ]

    # Train multiple models with different configs
    all_test_predictions = []

    for config_idx, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Training Configuration {config_idx}/{len(configs)}")
        print(f"{'='*80}")

        # Add GPU parameters
        config.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
            'random_state': 42 + config_idx,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 1,
            'nthread': -1,
            'max_bin': 256,
            'grow_policy': 'depthwise'
        })

        # 10-fold cross validation for robustness
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + config_idx)
        fold_predictions = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            print(f"\nFold {fold}/10...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create DMatrix for faster training
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test)

            # Train model
            model = xgb.train(
                config,
                dtrain,
                num_boost_round=config['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=100,
                verbose_eval=False
            )

            # Validate
            val_pred = model.predict(dval)
            score, ap, wll = calculate_competition_score(y_val, val_pred)
            print(f"  Validation: Score={score:.6f}, AP={ap:.4f}, WLL={wll:.4f}")

            # Predict on test
            test_pred = model.predict(dtest)
            fold_predictions.append(test_pred)

            # Clean up
            del model, dtrain, dval, dtest
            gc.collect()

        # Average fold predictions
        config_predictions = np.mean(fold_predictions, axis=0)
        all_test_predictions.append(config_predictions)

        print(f"\nConfig {config_idx} predictions: mean={config_predictions.mean():.6f}, "
              f"std={config_predictions.std():.6f}")

    # Final ensemble
    print("\n" + "="*80)
    print("Creating Final Ensemble")
    print("="*80)

    # Weighted average with emphasis on diversity
    weights = [0.4, 0.35, 0.25]  # Give more weight to first (most conservative) config
    final_predictions = np.average(all_test_predictions, axis=0, weights=weights)

    print(f"Final predictions: mean={final_predictions.mean():.6f}, std={final_predictions.std():.6f}")
    print(f"Min={final_predictions.min():.6f}, Max={final_predictions.max():.6f}")

    # Post-processing: Calibration
    print("\nApplying calibration...")

    # Shift predictions towards extremes for better discrimination
    def calibrate(p, power=1.2):
        """Power calibration to improve discrimination"""
        return np.power(p, power) / (np.power(p, power) + np.power(1-p, power))

    calibrated_predictions = calibrate(final_predictions, power=1.15)

    print(f"Calibrated: mean={calibrated_predictions.mean():.6f}, std={calibrated_predictions.std():.6f}")

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'target': calibrated_predictions
    })

    submission.to_csv('plan2/055_ultimate_xgboost_submission.csv', index=False)
    print("\nSaved to plan2/055_ultimate_xgboost_submission.csv")

    # Also save uncalibrated version
    submission_uncalibrated = pd.DataFrame({
        'ID': test_data['ID'],
        'target': final_predictions
    })
    submission_uncalibrated.to_csv('plan2/055_ultimate_xgboost_uncalibrated.csv', index=False)

    print("\n" + "="*80)
    print("ULTIMATE XGBoost Complete!")
    print("Target: 0.351+ Competition Score")
    print("="*80)

    return calibrated_predictions


if __name__ == "__main__":
    predictions = train_ultimate_xgboost()
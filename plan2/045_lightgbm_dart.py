#!/usr/bin/env python3
"""
045_lightgbm_dart.py
LightGBM with DART (Dropouts meet Multiple Additive Regression Trees)
Better generalization and less overfitting
"""

import sys
sys.path.append('plan2/src')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import time
from data_loader import load_data, get_data_loader
import gc
import warnings
warnings.filterwarnings('ignore')

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


def train_lightgbm_dart():
    """Train LightGBM with DART mode for better generalization"""

    print("="*60)
    print("LightGBM DART Model for Competition Score")
    print("DART: Dropouts meet Multiple Additive Regression Trees")
    print("="*60)

    # Load data
    print("\nLoading cached data...")
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Get feature matrices
    loader = get_data_loader()
    X_train, X_test, feature_cols = loader.get_feature_matrix(train_df, test_df, feature_info)

    # Identify categorical columns
    cat_cols = feature_info['cat_cols']
    cat_indices = [i for i, col in enumerate(feature_cols) if col in cat_cols]

    print(f"\nFeatures: {len(cat_indices)} categorical, {X_train.shape[1] - len(cat_indices)} numerical")
    print(f"Class distribution: {y_train.mean():.4f} positive")

    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f"\nTrain size: {len(y_tr):,}, Val size: {len(y_val):,}")

    # Class weight for imbalance
    pos_weight = (1 - y_tr.mean()) / y_tr.mean()
    print(f"Positive class weight: {pos_weight:.2f}")

    # LightGBM DART parameters
    params = {
        # DART specific
        'boosting_type': 'dart',
        'drop_rate': 0.1,  # Dropout rate
        'max_drop': 50,    # Max number of trees to drop
        'skip_drop': 0.5,  # Probability of skipping dropout
        'uniform_drop': False,  # Non-uniform dropout
        'xgboost_dart_mode': False,  # Use LightGBM's implementation

        # General parameters
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 127,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'min_gain_to_split': 0.001,

        # Regularization
        'lambda_l1': 0.5,
        'lambda_l2': 1.0,
        'min_child_weight': 10,
        'max_bin': 255,

        # Handle imbalance - use only one of these
        # 'is_unbalance': True,
        'scale_pos_weight': pos_weight,

        # Performance
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_threads': 32,

        'verbose': -1,
        'seed': 42,
    }

    # Create datasets
    lgb_train = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=cat_indices
    )

    lgb_val = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=cat_indices,
        reference=lgb_train
    )

    # Callbacks for monitoring
    def competition_score_callback(env):
        """Custom callback to track competition score"""
        if env.iteration % 100 == 0 and env.iteration > 0:
            val_preds = env.model.predict(X_val, num_iteration=env.iteration)
            score, ap, wll = calculate_competition_score(y_val, val_preds)
            print(f"  [Iter {env.iteration}] Competition Score: {score:.4f} "
                  f"(AP: {ap:.4f}, WLL: {wll:.4f})")

    # Train model
    print("\nTraining LightGBM DART model...")
    print("Note: DART is slower but generalizes better")
    print("-" * 60)

    # Callbacks list
    callbacks = [
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
    callbacks.append(competition_score_callback)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )

    # Evaluate on validation
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_score, val_ap, val_wll = calculate_competition_score(y_val, val_preds)

    print(f"\nValidation Results:")
    print(f"Competition Score: {val_score:.6f}")
    print(f"AP: {val_ap:.6f}")
    print(f"WLL: {val_wll:.6f}")
    print(f"Best iteration: {model.best_iteration}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    top_features_idx = np.argsort(importance)[-20:][::-1]

    print("\nTop 20 Features (by gain):")
    for idx in top_features_idx:
        print(f"  {feature_cols[idx]}: {importance[idx]:.2f}")

    # Generate test predictions
    print("\nGenerating test predictions...")
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)

    # Calibration
    train_positive_rate = y_train.mean()
    test_mean = test_preds.mean()

    print(f"\nPrediction distribution:")
    print(f"  Train positive rate: {train_positive_rate:.6f}")
    print(f"  Test mean (raw): {test_mean:.6f}")

    # Very light calibration for DART (it already regularizes well)
    if test_mean > 0 and abs(test_mean - train_positive_rate) > 0.01:
        calibration_factor = np.power(train_positive_rate / test_mean, 0.15)  # Very conservative
        test_preds_calibrated = test_preds * calibration_factor
        test_preds_calibrated = np.clip(test_preds_calibrated, 0.0001, 0.9999)
    else:
        test_preds_calibrated = test_preds

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds_calibrated
    })

    submission.to_csv('plan2/045_lightgbm_dart_submission.csv', index=False)
    print(f"\nSaved to plan2/045_lightgbm_dart_submission.csv")

    # Final stats
    print(f"\n" + "="*60)
    print(f"Final Results:")
    print(f"Validation Competition Score: {val_score:.6f}")
    print(f"Validation AP: {val_ap:.6f}")
    print(f"Validation WLL: {val_wll:.6f}")
    print(f"\nTest predictions (calibrated):")
    print(f"  Mean: {test_preds_calibrated.mean():.6f}")
    print(f"  Std: {test_preds_calibrated.std():.6f}")
    print(f"  Min: {test_preds_calibrated.min():.6f}")
    print(f"  Max: {test_preds_calibrated.max():.6f}")
    print(f"  >0.5: {(test_preds_calibrated > 0.5).sum()} "
          f"({(test_preds_calibrated > 0.5).mean()*100:.2f}%)")
    print("="*60)

    # Save model
    model.save_model('plan2/045_lightgbm_dart_model.txt')
    print("\nModel saved to plan2/045_lightgbm_dart_model.txt")

    # Clean up
    gc.collect()

    return model, test_preds_calibrated


if __name__ == "__main__":
    model, predictions = train_lightgbm_dart()
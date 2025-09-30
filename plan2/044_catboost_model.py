#!/usr/bin/env python3
"""
044_catboost_model.py
CatBoost with competition score optimization
Known for better handling of categorical features
"""

import sys
sys.path.append('plan2/src')

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import time
from data_loader import load_data, get_data_loader
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


def train_catboost():
    """Train CatBoost model optimized for competition score"""

    print("="*60)
    print("CatBoost Model for Competition Score")
    print("="*60)

    # Load data
    print("\nLoading cached data...")
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Get feature matrices
    loader = get_data_loader()
    X_train, X_test, feature_cols = loader.get_feature_matrix(train_df, test_df, feature_info)

    # Identify categorical columns indices
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

    # CatBoost parameters optimized for competition score
    params = {
        'iterations': 3000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3,
        'min_data_in_leaf': 50,
        'random_strength': 0.5,
        'bagging_temperature': 0.7,
        'border_count': 128,
        'grow_policy': 'Lossguide',
        'max_leaves': 64,

        # Handle imbalance
        'auto_class_weights': 'Balanced',
        'scale_pos_weight': pos_weight,

        # GPU settings
        'task_type': 'GPU',
        'devices': '0',

        # Regularization
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'random_seed': 42,

        # Optimization
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'od_wait': 100,

        'verbose': 100,
        'allow_writing_files': False,
        'thread_count': -1,
    }

    # Create pools
    # CatBoost categorical features need special handling
    # Since our data is already encoded, we'll treat everything as numerical
    # This avoids the error about floating point data with cat_features
    train_pool = Pool(
        X_tr, y_tr,
        cat_features=None  # Treat all as numerical since already encoded
    )

    val_pool = Pool(
        X_val, y_val,
        cat_features=None  # Treat all as numerical since already encoded
    )

    # Train model
    print("\nTraining CatBoost model...")
    print("-" * 60)

    model = CatBoostClassifier(**params)

    # Custom eval with competition score
    best_score = 0
    best_iteration = 0

    def competition_score_eval(pool):
        preds = model.predict_proba(pool)[:, 1]
        labels = pool.get_label()
        score, ap, wll = calculate_competition_score(labels, preds)
        return score

    # Fit with early stopping based on validation
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
        plot=False
    )

    # Evaluate on validation
    val_preds = model.predict_proba(X_val)[:, 1]
    val_score, val_ap, val_wll = calculate_competition_score(y_val, val_preds)

    print(f"\nValidation Results:")
    print(f"Competition Score: {val_score:.6f}")
    print(f"AP: {val_ap:.6f}")
    print(f"WLL: {val_wll:.6f}")
    print(f"Best iteration: {model.best_iteration_}")

    # Feature importance
    feature_importance = model.get_feature_importance()
    top_features_idx = np.argsort(feature_importance)[-20:][::-1]

    print("\nTop 20 Features:")
    for idx in top_features_idx:
        print(f"  {feature_cols[idx]}: {feature_importance[idx]:.2f}")

    # Generate test predictions
    print("\nGenerating test predictions...")
    test_pool = Pool(X_test, cat_features=None)  # Treat all as numerical
    test_preds = model.predict_proba(test_pool)[:, 1]

    # Calibration
    train_positive_rate = y_train.mean()
    test_mean = test_preds.mean()

    print(f"\nPrediction distribution:")
    print(f"  Train positive rate: {train_positive_rate:.6f}")
    print(f"  Test mean (raw): {test_mean:.6f}")

    # Light calibration
    if test_mean > 0 and abs(test_mean - train_positive_rate) > 0.005:
        calibration_factor = np.power(train_positive_rate / test_mean, 0.2)  # Very light
        test_preds_calibrated = test_preds * calibration_factor
        test_preds_calibrated = np.clip(test_preds_calibrated, 0.0001, 0.9999)
    else:
        test_preds_calibrated = test_preds

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds_calibrated
    })

    submission.to_csv('plan2/044_catboost_submission.csv', index=False)
    print(f"\nSaved to plan2/044_catboost_submission.csv")

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

    # Clean up
    del train_pool, val_pool, test_pool
    gc.collect()

    return model, test_preds_calibrated


if __name__ == "__main__":
    model, predictions = train_catboost()
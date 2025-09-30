#!/usr/bin/env python
"""
CatBoost GPU Model for Plan3
Utilizing GPU acceleration for gradient boosting
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, log_loss
import warnings
warnings.filterwarnings('ignore')
import gc

print("="*80)
print("CATBOOST GPU MODEL FOR PLAN3")
print("Target: 0.351+ competition score")
print("="*80)

# Check GPU availability
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                           capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip()
        print(f"GPU Available: {gpu_info}")
except:
    print("GPU check failed")

# Load data
print("\nLoading data...")
train = pd.read_parquet('data/train.parquet')
test = pd.read_parquet('data/test.parquet')

# Identify categorical columns
cat_features = []
for col in train.columns:
    if train[col].dtype == 'object':
        cat_features.append(col)
        # Convert to category for CatBoost
        train[col] = train[col].astype('category')

for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = test[col].astype('category')

print(f"Categorical features: {len(cat_features)}")

# Prepare data
X = train.drop(columns=['clicked'])
y = train['clicked']
X_test = test.drop(columns=['ID']) if 'ID' in test.columns else test.copy()

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Positive class ratio: {y.mean():.4f}")

# CatBoost parameters optimized for GPU
params = {
    'objective': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'devices': '0',

    # Model complexity - increased for better performance
    'iterations': 10000,
    'depth': 10,
    'learning_rate': 0.02,
    'l2_leaf_reg': 5,

    # Regularization
    'random_strength': 1,
    'bagging_temperature': 0.5,
    'border_count': 254,  # Max for GPU

    # GPU specific
    'gpu_ram_part': 0.95,  # Use 95% of GPU RAM
    'max_ctr_complexity': 4,  # Complex categorical features

    # Feature sampling
    'rsm': 0.8,  # Random subspace method
    'subsample': 0.8,

    # Training
    'use_best_model': True,
    'early_stopping_rounds': 200,
    'random_seed': 42,
    'verbose': 100,

    # Class weights for imbalance
    'auto_class_weights': 'Balanced',

    # Advanced options
    'grow_policy': 'Lossguide',
    'min_data_in_leaf': 50,
    'max_leaves': 64,
    'boosting_type': 'Plain',

    # Enable all GPU features
    'bootstrap_type': 'Bayesian',
    'posterior_sampling': True,
    'sampling_frequency': 'PerTree',
}

# 5-fold cross-validation
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_scores = []
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

print(f"\nTraining with {n_folds}-fold CV...")
print(f"Using GPU with {params['gpu_ram_part']*100:.0f}% RAM allocation")

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{n_folds}")
    print(f"{'='*60}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Create CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)

    # Train model
    model = CatBoostClassifier(**params)

    model.fit(
        train_pool,
        eval_set=valid_pool,
        plot=False
    )

    # Get best iteration
    print(f"Best iteration: {model.best_iteration_}")

    # Predictions
    valid_pred = model.predict_proba(valid_pool)[:, 1]
    test_pred = model.predict_proba(test_pool)[:, 1]

    # Store OOF predictions
    oof_preds[valid_idx] = valid_pred

    # Calculate scores
    ap_score = average_precision_score(y_valid, valid_pred)
    logloss = log_loss(y_valid, valid_pred)

    # Competition metric
    competition_score = 0.7 * ap_score + 0.3 / logloss
    cv_scores.append(competition_score)

    print(f"\nFold {fold} Results:")
    print(f"  AP Score: {ap_score:.6f}")
    print(f"  LogLoss: {logloss:.6f}")
    print(f"  Competition Score: {competition_score:.6f}")

    # Feature importance
    if fold == 1:
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        print("\nTop 10 Features:")
        for idx in top_features_idx:
            print(f"  {X.columns[idx]}: {feature_importance[idx]:.2f}")

    # Average test predictions
    test_preds += test_pred / n_folds

    # Clean up
    del model, train_pool, valid_pool, test_pool
    gc.collect()

# Final results
print("\n" + "="*80)
print("CATBOOST GPU RESULTS")
print("="*80)

# Overall OOF score
oof_ap = average_precision_score(y, oof_preds)
oof_logloss = log_loss(y, oof_preds)
oof_score = 0.7 * oof_ap + 0.3 / oof_logloss

print(f"Out-of-Fold AP: {oof_ap:.6f}")
print(f"Out-of-Fold LogLoss: {oof_logloss:.6f}")
print(f"Out-of-Fold Competition Score: {oof_score:.6f}")

print(f"\nCV Scores by Fold: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

# Create submission
print("\nCreating submission file...")
submission = pd.DataFrame()
submission['ID'] = [f'TEST_{i:07d}' for i in range(len(test))]
submission['clicked'] = test_preds

# Apply calibration
calibration_power = 1.06
submission['clicked'] = np.power(submission['clicked'], calibration_power)

# Save submission
submission.to_csv('plan3/034_catboost_gpu_submission.csv', index=False)
print(f"Submission saved to plan3/034_catboost_gpu_submission.csv")

# Save OOF predictions for stacking
oof_df = pd.DataFrame({
    'oof_catboost': oof_preds,
    'target': y
})
oof_df.to_csv('plan3/034_catboost_oof.csv', index=False)
print(f"OOF predictions saved to plan3/034_catboost_oof.csv")

# Print statistics
print(f"\nPrediction statistics:")
print(f"  Mean: {submission['clicked'].mean():.6f}")
print(f"  Std: {submission['clicked'].std():.6f}")
print(f"  Min: {submission['clicked'].min():.6f}")
print(f"  Max: {submission['clicked'].max():.6f}")

print("\n✓ CatBoost GPU model complete!")
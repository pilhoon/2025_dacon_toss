#!/usr/bin/env python
"""
Massive GPU XGBoost Model - Maximizing GPU Memory Usage
Using very large tree depth and number of rounds to fill GPU memory
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, log_loss
import warnings
warnings.filterwarnings('ignore')
import gc
from multiprocessing import cpu_count

print("="*80)
print("MASSIVE GPU XGBOOST MODEL FOR 0.351+ TARGET")
print("Maximizing GPU Memory Usage with Large Trees")
print("="*80)

# Check GPU
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
if result.returncode == 0:
    mem_total, mem_free = map(int, result.stdout.strip().split(', '))
    print(f"\nGPU Memory: {mem_total/1024:.1f}GB total, {mem_free/1024:.1f}GB free")

# Load data
print("\nLoading data...")
train = pd.read_parquet('data/train.parquet')
test = pd.read_parquet('data/test.parquet')

# Convert object dtype columns to numeric
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = pd.factorize(train[col])[0]
for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = pd.factorize(test[col])[0]

X = train.drop(columns=['clicked'])
y = train['clicked']
# Remove ID column from test if present
X_test = test.drop(columns=['ID']) if 'ID' in test.columns else test.copy()

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Positive class ratio: {y.mean():.4f}")

# XGBoost parameters for maximum GPU usage
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'logloss'],
    'tree_method': 'gpu_hist',
    'gpu_id': 0,

    # Reduced depth to avoid memory issues
    'max_depth': 12,  # Reduced from 20
    'max_leaves': 0,   # Unlimited leaves

    # More complexity
    'min_child_weight': 1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.9,
    'colsample_bynode': 0.9,

    # Learning parameters
    'learning_rate': 0.01,
    'gamma': 0.001,
    'lambda': 1.0,
    'alpha': 0.1,

    # GPU specific
    'predictor': 'gpu_predictor',
    'max_bin': 256,  # Reduced from 512

    'seed': 42,
    'nthread': cpu_count(),
    'verbosity': 1
}

# Use many boosting rounds
num_boost_round = 5000  # Many rounds to use GPU longer

# Cross-validation
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_scores = []
test_preds = np.zeros(len(X_test))

print(f"\nTraining with {n_folds}-fold CV...")
print(f"Max depth: {params['max_depth']}, Rounds: {num_boost_round}")

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nFold {fold}/{n_folds}...")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)

    # Train model with early stopping
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=200,
        verbose_eval=100
    )

    # Get best iteration
    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")

    # Predict
    valid_pred = model.predict(dvalid, iteration_range=(0, best_iter))
    test_pred = model.predict(dtest, iteration_range=(0, best_iter))

    # Calculate scores
    ap_score = average_precision_score(y_valid, valid_pred)
    logloss = log_loss(y_valid, valid_pred)

    # Competition metric
    competition_score = 0.7 * ap_score + 0.3 / logloss
    cv_scores.append(competition_score)

    print(f"Fold {fold} - AP: {ap_score:.6f}, LogLoss: {logloss:.6f}")
    print(f"Competition Score: {competition_score:.6f}")

    # Average test predictions
    test_preds += test_pred / n_folds

    # Check GPU memory usage
    if fold == 1:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            mem_used = int(result.stdout.strip())
            print(f"GPU Memory Used: {mem_used/1024:.1f}GB")

    # Clean up
    del dtrain, dvalid, dtest, model
    gc.collect()

print("\n" + "="*80)
print("MASSIVE GPU XGBOOST RESULTS")
print("="*80)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
print(f"Best Fold Score: {np.max(cv_scores):.6f}")

# Create submission
print("\nCreating submission file...")
submission = pd.DataFrame()
submission['ID'] = [f'TEST_{i:07d}' for i in range(len(test))]
submission['clicked'] = test_preds

# Apply calibration
power = 1.08
submission['clicked'] = np.power(submission['clicked'], power)

# Save submission
submission.to_csv('plan3/026_massive_gpu_xgboost_submission.csv', index=False)
print(f"Submission saved to plan3/026_massive_gpu_xgboost_submission.csv")

# Print statistics
print(f"\nPrediction statistics:")
print(f"  Mean: {submission['clicked'].mean():.6f}")
print(f"  Std: {submission['clicked'].std():.6f}")
print(f"  Min: {submission['clicked'].min():.6f}")
print(f"  Max: {submission['clicked'].max():.6f}")

print("\n✓ Massive GPU XGBoost model complete!")
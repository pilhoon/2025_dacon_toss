#!/usr/bin/env python
"""
Meta-Learning Model: Using out-of-fold predictions as features
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
print("META-LEARNING MODEL FOR 0.351+ TARGET")
print("="*80)

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
X_test = test.copy()

# Remove ID column from test if it exists
if 'ID' in X_test.columns:
    X_test = X_test.drop(columns=['ID'])

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Positive class ratio: {y.mean():.4f}")

# Create base models with different configurations
base_models = []

# XGBoost with different parameters
for max_depth in [8, 10, 12]:
    for learning_rate in [0.01, 0.02]:
        base_models.append({
            'name': f'xgb_d{max_depth}_lr{learning_rate}',
            'model': xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=500,
                tree_method='gpu_hist',
                gpu_id=0,
                n_jobs=cpu_count(),
                random_state=42,
                eval_metric='logloss'
            )
        })

print(f"\nCreated {len(base_models)} base models")

# Generate out-of-fold predictions
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store OOF predictions
oof_predictions = np.zeros((len(X), len(base_models)))
test_predictions = np.zeros((len(X_test), len(base_models)))

print("\nGenerating out-of-fold predictions...")
for model_idx, base_model_info in enumerate(base_models):
    print(f"\nModel {model_idx+1}/{len(base_models)}: {base_model_info['name']}")

    model_test_preds = np.zeros((len(X_test), n_folds))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold+1}/{n_folds}...", end=' ')

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]

        # Train base model
        model = base_model_info['model']
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y.iloc[valid_idx])],
            verbose=False
        )

        # Store OOF predictions
        oof_predictions[valid_idx, model_idx] = model.predict_proba(X_valid_fold)[:, 1]

        # Store test predictions for this fold
        model_test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

        print(f"AP: {average_precision_score(y.iloc[valid_idx], oof_predictions[valid_idx, model_idx]):.4f}")

        # Clean up
        del model
        gc.collect()

    # Average test predictions across folds
    test_predictions[:, model_idx] = model_test_preds.mean(axis=1)

    # Print model performance
    model_ap = average_precision_score(y, oof_predictions[:, model_idx])
    print(f"  Overall AP: {model_ap:.6f}")

print("\n" + "="*80)
print("TRAINING META-LEARNER")
print("="*80)

# Create meta features
print("\nCreating meta features...")
meta_train = pd.DataFrame(oof_predictions, columns=[f'base_{i}' for i in range(len(base_models))])
meta_test = pd.DataFrame(test_predictions, columns=[f'base_{i}' for i in range(len(base_models))])

# Add statistical features
meta_train['mean'] = meta_train.mean(axis=1)
meta_train['std'] = meta_train.std(axis=1)
meta_train['max'] = meta_train.max(axis=1)
meta_train['min'] = meta_train.min(axis=1)
meta_train['range'] = meta_train['max'] - meta_train['min']

meta_test['mean'] = meta_test.mean(axis=1)
meta_test['std'] = meta_test.std(axis=1)
meta_test['max'] = meta_test.max(axis=1)
meta_test['min'] = meta_test.min(axis=1)
meta_test['range'] = meta_test['max'] - meta_test['min']

print(f"Meta features shape: {meta_train.shape}")

# Train meta-learner
print("\nTraining meta-learner...")
meta_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    tree_method='gpu_hist',
    gpu_id=0,
    n_jobs=cpu_count(),
    random_state=42,
    eval_metric='logloss'
)

# Cross-validation for meta-learner
meta_scores = []
final_test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(meta_train, y)):
    print(f"\nMeta Fold {fold+1}/{n_folds}...")

    X_meta_train = meta_train.iloc[train_idx]
    y_meta_train = y.iloc[train_idx]
    X_meta_valid = meta_train.iloc[valid_idx]
    y_meta_valid = y.iloc[valid_idx]

    meta_model.fit(
        X_meta_train, y_meta_train,
        eval_set=[(X_meta_valid, y_meta_valid)],
        verbose=False
    )

    valid_pred = meta_model.predict_proba(X_meta_valid)[:, 1]
    fold_ap = average_precision_score(y_meta_valid, valid_pred)
    fold_logloss = log_loss(y_meta_valid, valid_pred)

    # Competition metric
    fold_score = 0.7 * fold_ap + 0.3 / fold_logloss
    meta_scores.append(fold_score)

    print(f"  AP: {fold_ap:.6f}")
    print(f"  LogLoss: {fold_logloss:.6f}")
    print(f"  Competition Score: {fold_score:.6f}")

    # Predict on test
    final_test_preds += meta_model.predict_proba(meta_test)[:, 1] / n_folds

print("\n" + "="*80)
print("META-LEARNING RESULTS")
print("="*80)
print(f"Average Competition Score: {np.mean(meta_scores):.6f} ± {np.std(meta_scores):.6f}")
print(f"Best Fold Score: {np.max(meta_scores):.6f}")

# Create submission
print("\nCreating submission file...")
submission = pd.DataFrame()
submission['ID'] = [f'TEST_{i:07d}' for i in range(len(test))]
submission['clicked'] = final_test_preds

# Apply calibration
power = 1.08
submission['clicked'] = np.power(submission['clicked'], power)

# Save submission
submission.to_csv('plan3/020_meta_learning_submission.csv', index=False)
print(f"Submission saved to plan3/020_meta_learning_submission.csv")

# Print statistics
print(f"\nPrediction statistics:")
print(f"  Mean: {submission['clicked'].mean():.6f}")
print(f"  Std: {submission['clicked'].std():.6f}")
print(f"  Min: {submission['clicked'].min():.6f}")
print(f"  Max: {submission['clicked'].max():.6f}")

print("\n✓ Meta-learning model complete!")
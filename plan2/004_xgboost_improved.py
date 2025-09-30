"""
Improved XGBoost based on plan1 learnings
Target: score > 0.349 (current best: 0.31631)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import LabelEncoder
import time
import json
from pathlib import Path

print("=" * 60)
print("XGBoost Improved - Target Score > 0.349")
print("=" * 60)

# Load data
print("Loading data...")
t0 = time.time()
train_df = pd.read_parquet('data/train.parquet', engine='pyarrow')
test_df = pd.read_parquet('data/test.parquet', engine='pyarrow')
print(f"Loaded in {time.time()-t0:.1f}s")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Target
y = train_df['clicked'].values
X = train_df.drop(columns=['clicked'])
X_test = test_df.drop(columns=['clicked'])

print(f"Positive rate: {y.mean():.4f}")

# Feature engineering
print("\nFeature engineering...")
# Combine train and test for encoding
all_data = pd.concat([X, X_test], axis=0, ignore_index=True)

# Encode categorical features
cat_cols = []
for col in X.columns:
    if col.startswith(('gender', 'age_group', 'inventory_id', 'seq', 'l_feat_', 'feat_')):
        cat_cols.append(col)

print(f"Encoding {len(cat_cols)} categorical columns...")
for col in cat_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))

# Add interaction features
print("Adding interaction features...")
# Key interactions from plan1 analysis
all_data['gender_age'] = all_data['gender'] * 100 + all_data['age_group']
all_data['hour_dow'] = all_data['hour'] * 10 + all_data['day_of_week']
all_data['inventory_age'] = all_data['inventory_id'] * 100 + all_data['age_group']

# History aggregations
history_cols = [c for c in all_data.columns if c.startswith('history_')]
all_data['history_sum'] = all_data[history_cols].sum(axis=1)
all_data['history_mean'] = all_data[history_cols].mean(axis=1)
all_data['history_std'] = all_data[history_cols].std(axis=1)
all_data['history_max'] = all_data[history_cols].max(axis=1)

# Split back
X = all_data.iloc[:len(X)].copy()
X_test = all_data.iloc[len(X):].copy()

print(f"Final feature count: {X.shape[1]}")

# XGBoost parameters - balanced for AP and WLL
# Key insights from plan1:
# - Need prediction std > 0.05 for good AP
# - Need prediction mean close to 0.0191 for good WLL
# - scale_pos_weight between 10-20 works best

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'max_depth': 8,  # Deeper than plan1's 4, but not too deep
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 15,  # Balanced between 10 and 20
    'reg_alpha': 0.05,
    'reg_lambda': 1.0,
    'min_child_weight': 10,
    'gamma': 0.1,
    'random_state': 42
}

print("\nXGBoost parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")

# 3-fold CV for robust evaluation
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(X_test))
models = []

print(f"\nStarting {n_folds}-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== FOLD {fold + 1} ===")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
        early_stopping_rounds=20
    )

    # Predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds

    # Test predictions
    test_preds += model.predict_proba(X_test)[:, 1] / n_folds

    # Validation metrics
    val_auc = roc_auc_score(y_val, val_preds)
    val_ap = average_precision_score(y_val, val_preds)
    val_logloss = log_loss(y_val, val_preds)

    print(f"\nFold {fold + 1} Results:")
    print(f"  AUC: {val_auc:.6f}")
    print(f"  AP: {val_ap:.6f}")
    print(f"  LogLoss: {val_logloss:.6f}")
    print(f"  Pred mean: {val_preds.mean():.6f}")
    print(f"  Pred std: {val_preds.std():.6f}")

    models.append(model)

# OOF evaluation
print("\n" + "=" * 60)
print("OUT-OF-FOLD RESULTS")
print("=" * 60)

oof_auc = roc_auc_score(y, oof_preds)
oof_ap = average_precision_score(y, oof_preds)
oof_logloss = log_loss(y, oof_preds)

# Weighted log loss calculation
epsilon = 1e-7
oof_preds_clipped = np.clip(oof_preds, epsilon, 1 - epsilon)
pos_weight = 15.0  # Same as training
wll = -np.mean(
    y * pos_weight * np.log(oof_preds_clipped) +
    (1 - y) * np.log(1 - oof_preds_clipped)
)

# Competition score
score = 0.5 * oof_ap + 0.5 * (1 / (1 + wll))

print(f"AUC: {oof_auc:.6f}")
print(f"AP: {oof_ap:.6f}")
print(f"LogLoss: {oof_logloss:.6f}")
print(f"WLL: {wll:.6f}")
print(f"Pred mean: {oof_preds.mean():.6f} (target: 0.0191)")
print(f"Pred std: {oof_preds.std():.6f} (target: >0.05)")
print(f"\n**COMPETITION SCORE: {score:.6f}** (target: >0.349)")

# Save results
output_dir = Path('plan2/experiments/006_xgboost_improved')
output_dir.mkdir(exist_ok=True, parents=True)

# Save metrics
metrics = {
    'oof_auc': float(oof_auc),
    'oof_ap': float(oof_ap),
    'oof_logloss': float(oof_logloss),
    'oof_wll': float(wll),
    'competition_score': float(score),
    'pred_mean': float(oof_preds.mean()),
    'pred_std': float(oof_preds.std()),
    'target_achieved': bool(score > 0.349)
}

with open(output_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save predictions
np.save(output_dir / 'oof_preds.npy', oof_preds)
np.save(output_dir / 'test_preds.npy', test_preds)

# Feature importance
print("\nTop 20 Important Features:")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': np.mean([m.feature_importances_ for m in models], axis=0)
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(20).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

# Create submission if score is good
if score > 0.32:  # Only create submission if reasonably good
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'clicked': test_preds
    })
    submission_path = output_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    print(f"Test predictions - mean: {test_preds.mean():.6f}, std: {test_preds.std():.6f}")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
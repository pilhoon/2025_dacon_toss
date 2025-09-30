"""
XGBoost with subset of data for faster testing
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import LabelEncoder
import time

print("Loading subset of data...")
t0 = time.time()

# Load smaller subset
n_rows = 500000
train_df = pd.read_parquet('data/train.parquet', engine='pyarrow').head(n_rows)
test_df = pd.read_parquet('data/test.parquet', engine='pyarrow').head(50000)

print(f"Loaded in {time.time()-t0:.1f}s")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Target
y = train_df['clicked'].values
X = train_df.drop(columns=['clicked'])
X_test = test_df.copy()
if 'clicked' in X_test.columns:
    X_test = X_test.drop(columns=['clicked'])

print(f"Positive rate: {y.mean():.4f}")

# Simple preprocessing
print("Preprocessing...")
all_data = pd.concat([X, X_test], axis=0)

# Encode categoricals
for col in X.columns:
    if col.startswith(('gender', 'age_group', 'inventory', 'seq', 'l_feat', 'feat')):
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))

X = all_data.iloc[:len(X)]
X_test = all_data.iloc[len(X):]

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 10,
    'random_state': 42
}

print("\nTraining XGBoost...")
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=20
)

# Predictions
val_preds = model.predict_proba(X_val)[:, 1]
test_preds = model.predict_proba(X_test)[:, 1]

# Metrics
val_auc = roc_auc_score(y_val, val_preds)
val_ap = average_precision_score(y_val, val_preds)
val_logloss = log_loss(y_val, val_preds)

print("\n" + "="*50)
print("VALIDATION RESULTS")
print("="*50)
print(f"AUC: {val_auc:.6f}")
print(f"AP: {val_ap:.6f}")
print(f"LogLoss: {val_logloss:.6f}")
print(f"Pred mean: {val_preds.mean():.6f} (target: ~0.02)")
print(f"Pred std: {val_preds.std():.6f} (target: >0.05)")

# Estimated competition score
wll = val_logloss * 10  # Rough estimate
score = 0.5 * val_ap + 0.5 * (1 / (1 + wll))
print(f"\nEstimated score: {score:.6f} (target: >0.349)")

print("\nDone!")
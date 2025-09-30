#!/usr/bin/env python3
"""
XGBoost WITHOUT any guardrail constraints.
Clean implementation focused purely on performance.
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from score import competition_score
from feature_engineering import FeatureEngineer

N_JOBS = os.cpu_count()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {N_JOBS} CPU cores")

# Best parameters from Optuna
BEST_PARAMS = {
    'max_depth': 4,
    'eta': 0.013780073764671054,
    'subsample': 0.6817427853448248,
    'colsample_bytree': 0.8862717025528952,
    'min_child_weight': 4,
    'gamma': 0.08572213685785915,
    'reg_alpha': 0.24498411648086793,
    'reg_lambda': 0.48467697930320736,
    'num_boost_round': 253
}


def main():
    print("="*70)
    print("XGBoost Training - NO GUARDRAILS")
    print("="*70)

    # Load data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    df_train = pd.read_parquet('../data/train.parquet')
    df_test = pd.read_parquet('../data/test.parquet')

    # Store IDs
    test_ids = df_test['ID'].copy() if 'ID' in df_test.columns else pd.Series([f'TEST_{i:07d}' for i in range(len(df_test))])

    print(f"  Train: {df_train.shape}")
    print(f"  Test: {df_test.shape}")
    print(f"  Target rate: {df_train['clicked'].mean():.4%}")

    # Separate target
    y_train = df_train['clicked'].values
    df_train = df_train.drop(columns=['clicked'])

    # Feature Engineering
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature engineering...")
    fe = FeatureEngineer(use_cache=True, verbose=True)
    df_train = fe.fit_transform(df_train)
    df_test = fe.transform(df_test)

    print(f"  After FE: Train {df_train.shape}, Test {df_test.shape}")

    # Handle categorical columns
    cat_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    cat_cols = [col for col in cat_cols if col in df_train.columns]

    if cat_cols:
        print(f"  Encoding {len(cat_cols)} categorical columns...")
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Fit on combined data
        combined = pd.concat([df_train[cat_cols], df_test[cat_cols]], ignore_index=True)
        encoder.fit(combined.fillna('missing'))

        # Transform
        df_train[cat_cols] = encoder.transform(df_train[cat_cols].fillna('missing'))
        df_test[cat_cols] = encoder.transform(df_test[cat_cols].fillna('missing'))

    # Remove ID columns
    for df in [df_train, df_test]:
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)

    # Convert all object columns to numeric
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Converting object columns to numeric...")
    for col in df_train.select_dtypes(include=['object']).columns:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)
    for col in df_test.select_dtypes(include=['object']).columns:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0)

    # Convert to float32 for memory efficiency
    df_train = df_train.astype(np.float32)
    df_test = df_test.astype(np.float32)

    # Cross-validation
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting 5-fold CV...")

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cpu',
        'nthread': N_JOBS,
        'seed': 42,
        'verbosity': 0,
        **{k: v for k, v in BEST_PARAMS.items() if k != 'num_boost_round'}
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    test_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y_train), 1):
        print(f"\n  Fold {fold}/5")

        X_tr, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(df_test)

        # Train model
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=BEST_PARAMS['num_boost_round'],
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predictions
        val_pred = model.predict(dval)
        test_pred = model.predict(dtest)

        # Evaluate
        score = competition_score(y_val, val_pred)
        cv_scores.append(score)
        test_preds.append(test_pred)

        print(f"    Score: {score:.5f}")
        print(f"    Val predictions: mean={val_pred.mean():.5f}, std={val_pred.std():.5f}")

    # Average predictions
    final_preds = np.mean(test_preds, axis=0)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training complete!")
    print(f"  CV Score: {np.mean(cv_scores):.5f} (±{np.std(cv_scores):.5f})")
    print(f"  Test predictions: mean={final_preds.mean():.5f}, std={final_preds.std():.5f}")

    # NO GUARDRAIL ADJUSTMENT - Use raw predictions
    print(f"\n  ✓ No guardrail applied - using raw predictions")

    # Save submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'clicked': final_preds
    })

    submission_file = '018_xgboost_no_guardrail.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n{'='*70}")
    print(f"✓ Submission saved to {submission_file}")
    print(f"✓ Mean: {final_preds.mean():.5f}, Std: {final_preds.std():.5f}")
    print(f"{'='*70}")

    return submission


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Pure XGBoost focused on performance without guardrail constraints.
Based on 007 but with all guardrail checks removed.
"""
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
import time
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path - fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from score import competition_score
from feature_engineering import FeatureEngineer

# CPU settings - use all available cores
N_JOBS = os.cpu_count()
print(f"Using {N_JOBS} CPU cores for parallel processing")

# Best parameters from Optuna (can be adjusted)
PARAMS = {
    'max_depth': 6,  # Increased depth
    'eta': 0.03,     # Slightly higher learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'num_boost_round': 300
}


def load_and_preprocess_data(use_cache=True):
    """Load and preprocess data with caching"""

    cache_file = 'plan4/preprocessed_data_cache.pkl'

    if use_cache and os.path.exists(cache_file):
        print("Loading preprocessed data from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Loading raw data...")
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    # Store test IDs before any processing
    test_ids = df_test['ID'].copy()

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    # Separate target
    y_train = df_train['clicked'].values
    df_train = df_train.drop(columns=['clicked'])

    # Feature Engineering
    print("\nApplying feature engineering...")
    fe = FeatureEngineer(cache_dir='plan4/fe_cache')

    # Fit on train, transform both
    df_train_fe = fe.fit_transform(df_train)
    df_test_fe = fe.transform(df_test)

    print(f"After FE - Train: {df_train_fe.shape}, Test: {df_test_fe.shape}")

    # Identify categorical columns
    cat_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    cat_cols = [col for col in cat_cols if col in df_train_fe.columns]

    if cat_cols:
        print(f"\nEncoding {len(cat_cols)} categorical columns...")
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Fit encoder on combined data to ensure consistency
        combined_cats = pd.concat([
            df_train_fe[cat_cols],
            df_test_fe[cat_cols]
        ], ignore_index=True)

        encoder.fit(combined_cats)

        # Transform
        df_train_fe[cat_cols] = encoder.transform(df_train_fe[cat_cols].fillna('missing'))
        df_test_fe[cat_cols] = encoder.transform(df_test_fe[cat_cols].fillna('missing'))

    # Remove ID if present
    for df in [df_train_fe, df_test_fe]:
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)

    # Convert to float32 for memory efficiency
    df_train_fe = df_train_fe.astype(np.float32)
    df_test_fe = df_test_fe.astype(np.float32)

    result = {
        'X_train': df_train_fe,
        'y_train': y_train,
        'X_test': df_test_fe,
        'test_ids': test_ids,
        'feature_names': df_train_fe.columns.tolist()
    }

    # Cache the preprocessed data
    if use_cache:
        print("Caching preprocessed data...")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

    return result


def train_xgboost_cv(X_train, y_train, X_test, params, n_folds=5):
    """Train XGBoost with cross-validation"""

    print(f"\nStarting {n_folds}-fold CV training...")

    # XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cpu',
        'nthread': N_JOBS,
        'seed': 42,
        'verbosity': 0,

        # Model parameters
        'max_depth': params.get('max_depth', 6),
        'eta': params.get('eta', 0.03),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'min_child_weight': params.get('min_child_weight', 3),
        'gamma': params.get('gamma', 0.1),
        'reg_alpha': params.get('reg_alpha', 0.1),
        'reg_lambda': params.get('reg_lambda', 1.0),
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\nFold {fold}/{n_folds}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        # Train with early stopping
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 300),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        models.append(model)

        # Predictions
        val_pred = model.predict(dval)
        test_pred = model.predict(dtest)

        # Store predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / n_folds

        # Evaluate
        score = competition_score(y_val, val_pred)
        cv_scores.append(score)

        print(f"  Fold {fold} score: {score:.5f}")
        print(f"  Fold {fold} prediction stats: mean={val_pred.mean():.5f}, std={val_pred.std():.5f}")

    # Overall CV score
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    oof_score = competition_score(y_train, oof_predictions)

    print(f"\nCV Score: {cv_mean:.5f} (±{cv_std:.5f})")
    print(f"OOF Score: {oof_score:.5f}")

    return {
        'models': models,
        'oof_predictions': oof_predictions,
        'test_predictions': test_predictions,
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'oof_score': oof_score
    }


def main():
    """Main execution"""

    print("=" * 60)
    print("XGBoost Pure Performance Training")
    print("=" * 60)

    # Load data
    data = load_and_preprocess_data(use_cache=True)

    # Train model
    results = train_xgboost_cv(
        data['X_train'],
        data['y_train'],
        data['X_test'],
        PARAMS,
        n_folds=5
    )

    # Prepare submission
    print("\n" + "=" * 60)
    print("Preparing submission...")

    submission = pd.DataFrame({
        'ID': data['test_ids'],
        'clicked': results['test_predictions']
    })

    # Save raw predictions (no guardrail adjustment)
    submission_file = 'plan4/013_xgboost_pure_performance.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\nSubmission saved to {submission_file}")
    print(f"Prediction stats:")
    print(f"  Mean: {results['test_predictions'].mean():.5f}")
    print(f"  Std:  {results['test_predictions'].std():.5f}")
    print(f"  Min:  {results['test_predictions'].min():.5f}")
    print(f"  Max:  {results['test_predictions'].max():.5f}")

    # Distribution analysis
    print(f"\nDistribution percentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        val = np.percentile(results['test_predictions'], p)
        print(f"  P{p:2d}: {val:.5f}")

    # Save model artifacts
    print("\nSaving model artifacts...")
    joblib.dump(results['models'], 'plan4/013_models.pkl')
    joblib.dump(results, 'plan4/013_results.pkl')

    print("\n✓ Training complete!")
    print(f"✓ CV Score: {results['cv_mean']:.5f}")
    print(f"✓ Submission ready: {submission_file}")

    return results


if __name__ == "__main__":
    main()
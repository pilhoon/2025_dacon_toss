#!/usr/bin/env python3
"""
LightGBM baseline model for better performance.
Focus on score without guardrail constraints.
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
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

# Use all CPU cores
N_JOBS = os.cpu_count()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {N_JOBS} CPU cores")


def load_and_preprocess_data():
    """Load and preprocess data"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")

    # Load data
    df_train = pd.read_parquet('../data/train.parquet')
    df_test = pd.read_parquet('../data/test.parquet')

    # Store IDs
    test_ids = df_test['ID'].copy()

    print(f"  Train shape: {df_train.shape}")
    print(f"  Test shape: {df_test.shape}")
    print(f"  Target rate: {df_train['clicked'].mean():.4%}")

    # Separate target
    y_train = df_train['clicked'].values
    df_train = df_train.drop(columns=['clicked'])

    # Feature Engineering
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature engineering...")
    fe = FeatureEngineer(use_cache=True)

    df_train_fe = fe.fit_transform(df_train)
    df_test_fe = fe.transform(df_test)

    print(f"  After FE: Train {df_train_fe.shape}, Test {df_test_fe.shape}")

    # Handle categorical columns
    cat_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    cat_cols = [col for col in cat_cols if col in df_train_fe.columns]

    if cat_cols:
        print(f"  Encoding {len(cat_cols)} categorical columns...")
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Fit on combined data
        combined = pd.concat([df_train_fe[cat_cols], df_test_fe[cat_cols]], ignore_index=True)
        encoder.fit(combined.fillna('missing'))

        # Transform
        df_train_fe[cat_cols] = encoder.transform(df_train_fe[cat_cols].fillna('missing'))
        df_test_fe[cat_cols] = encoder.transform(df_test_fe[cat_cols].fillna('missing'))

    # Remove ID columns
    for df in [df_train_fe, df_test_fe]:
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)

    return df_train_fe, y_train, df_test_fe, test_ids, cat_cols


def train_lightgbm(X_train, y_train, X_test, cat_features=None, n_folds=5):
    """Train LightGBM with cross-validation"""

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {n_folds}-fold CV with LightGBM...")

    # LightGBM parameters - optimized for CTR prediction
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',

        # Model complexity
        'num_leaves': 63,  # 2^6 - 1
        'max_depth': 7,
        'min_data_in_leaf': 100,
        'min_sum_hessian_in_leaf': 10.0,

        # Sampling
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,

        # Regularization
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0.01,

        # Training
        'learning_rate': 0.03,
        'num_threads': N_JOBS,
        'seed': 42,
        'verbose': -1,

        # Handle imbalance
        'is_unbalance': True,  # Auto-adjust for imbalanced data
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n  Fold {fold}/{n_folds}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_tr, label=y_tr,
            categorical_feature=cat_features
        )

        val_data = lgb.Dataset(
            X_val, label=y_val,
            reference=train_data,
            categorical_feature=cat_features
        )

        # Train model
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # Suppress output
            ]
        )

        models.append(model)

        # Predictions
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        # Store predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / n_folds

        # Evaluate
        score = competition_score(y_val, val_pred)
        cv_scores.append(score)

        print(f"    Score: {score:.5f}")
        print(f"    Best iteration: {model.best_iteration}")
        print(f"    Predictions: mean={val_pred.mean():.5f}, std={val_pred.std():.5f}")

    # Overall results
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    oof_score = competition_score(y_train, oof_predictions)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cross-validation complete")
    print(f"  CV Score: {cv_mean:.5f} (±{cv_std:.5f})")
    print(f"  OOF Score: {oof_score:.5f}")

    return {
        'models': models,
        'oof_predictions': oof_predictions,
        'test_predictions': test_predictions,
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'oof_score': oof_score
    }


def analyze_feature_importance(models, feature_names):
    """Analyze feature importance across all models"""

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Feature importance analysis...")

    # Aggregate importance
    importance_dict = {}

    for model in models:
        importance = model.feature_importance(importance_type='gain')
        for feat, imp in zip(feature_names, importance):
            if feat not in importance_dict:
                importance_dict[feat] = []
            importance_dict[feat].append(imp)

    # Average importance
    avg_importance = {
        feat: np.mean(imps) for feat, imps in importance_dict.items()
    }

    # Sort by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    print("  Top 20 features:")
    for i, (feat, imp) in enumerate(sorted_features[:20], 1):
        print(f"    {i:2d}. {feat:30s}: {imp:10.2f}")

    return sorted_features


def main():
    """Main execution"""

    print("=" * 70)
    print("LightGBM Baseline Model")
    print("=" * 70)

    # Load and preprocess data
    X_train, y_train, X_test, test_ids, cat_cols = load_and_preprocess_data()

    # Get categorical feature indices for LightGBM
    cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train.columns]

    # Train LightGBM
    results = train_lightgbm(
        X_train, y_train, X_test,
        cat_features=cat_indices if cat_indices else None,
        n_folds=5
    )

    # Feature importance
    feature_importance = analyze_feature_importance(
        results['models'],
        X_train.columns.tolist()
    )

    # Create submission
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Creating submission...")

    submission = pd.DataFrame({
        'ID': test_ids,
        'clicked': results['test_predictions']
    })

    submission_file = '015_lightgbm_submission.csv'
    submission.to_csv(submission_file, index=False)

    print(f"  Saved to {submission_file}")
    print(f"  Predictions: mean={results['test_predictions'].mean():.5f}, std={results['test_predictions'].std():.5f}")

    # Distribution analysis
    print(f"\n  Distribution percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(results['test_predictions'], p)
        print(f"    P{p:2d}: {val:.5f}")

    # Save models and results
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving artifacts...")
    joblib.dump(results, '015_lightgbm_results.pkl')
    joblib.dump(feature_importance, '015_feature_importance.pkl')

    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"✓ CV Score: {results['cv_mean']:.5f}")
    print(f"✓ Submission: {submission_file}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
CatBoost baseline model - handles categorical features natively.
Focus on performance without guardrail constraints.
"""
import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from score import competition_score
from feature_engineering import FeatureEngineer

# CPU settings
N_JOBS = os.cpu_count()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {N_JOBS} CPU cores")


def load_and_preprocess_data():
    """Load and preprocess data for CatBoost"""
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

    # Identify categorical columns for CatBoost
    cat_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    cat_cols = [col for col in cat_cols if col in df_train_fe.columns]

    # CatBoost handles categorical features directly - keep as strings
    print(f"  Keeping {len(cat_cols)} categorical features for CatBoost")

    # For CatBoost, fill missing values in categorical columns
    for col in cat_cols:
        df_train_fe[col] = df_train_fe[col].fillna('missing').astype(str)
        df_test_fe[col] = df_test_fe[col].fillna('missing').astype(str)

    # Remove ID columns
    for df in [df_train_fe, df_test_fe]:
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)

    # Get categorical column indices
    cat_indices = [df_train_fe.columns.get_loc(col) for col in cat_cols if col in df_train_fe.columns]

    return df_train_fe, y_train, df_test_fe, test_ids, cat_indices


def train_catboost(X_train, y_train, X_test, cat_indices=None, n_folds=5):
    """Train CatBoost with cross-validation"""

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {n_folds}-fold CV with CatBoost...")

    # CatBoost parameters
    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 7,

        # Regularization
        'l2_leaf_reg': 3.0,
        'min_data_in_leaf': 20,
        'random_strength': 1.0,
        'bagging_temperature': 1.0,

        # Performance
        'thread_count': N_JOBS,
        'use_best_model': True,
        'eval_metric': 'Logloss',
        'random_seed': 42,

        # Handle imbalance
        'auto_class_weights': 'Balanced',

        # Output
        'verbose': False,
        'early_stopping_rounds': 50,
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

        # Create CatBoost Pool objects
        train_pool = Pool(
            data=X_tr,
            label=y_tr,
            cat_features=cat_indices
        )

        val_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=cat_indices
        )

        test_pool = Pool(
            data=X_test,
            cat_features=cat_indices
        )

        # Train model
        model = CatBoostClassifier(**catboost_params)

        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=100  # Show progress every 100 iterations
        )

        models.append(model)

        # Predictions (get probabilities for positive class)
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]

        # Store predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / n_folds

        # Evaluate
        score = competition_score(y_val, val_pred)
        cv_scores.append(score)

        print(f"    Score: {score:.5f}")
        print(f"    Best iteration: {model.best_iteration_}")
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
    """Analyze CatBoost feature importance"""

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Feature importance analysis...")

    # Aggregate importance across models
    importance_sum = np.zeros(len(feature_names))

    for model in models:
        importance = model.feature_importances_
        importance_sum += importance

    # Average importance
    avg_importance = importance_sum / len(models)

    # Create sorted list
    feature_importance = list(zip(feature_names, avg_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("  Top 20 features:")
    for i, (feat, imp) in enumerate(feature_importance[:20], 1):
        print(f"    {i:2d}. {feat:30s}: {imp:10.2f}")

    return feature_importance


def main():
    """Main execution"""

    print("=" * 70)
    print("CatBoost Baseline Model")
    print("=" * 70)

    # Load and preprocess data
    X_train, y_train, X_test, test_ids, cat_indices = load_and_preprocess_data()

    # Train CatBoost
    results = train_catboost(
        X_train, y_train, X_test,
        cat_indices=cat_indices,
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

    submission_file = '016_catboost_submission.csv'
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
    joblib.dump(results, '016_catboost_results.pkl')
    joblib.dump(feature_importance, '016_catboost_importance.pkl')

    # Also save OOF predictions for ensemble
    np.save('016_oof_predictions.npy', results['oof_predictions'])
    np.save('016_test_predictions.npy', results['test_predictions'])

    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"✓ CV Score: {results['cv_mean']:.5f}")
    print(f"✓ Submission: {submission_file}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    main()
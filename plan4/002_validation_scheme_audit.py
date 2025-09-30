# plan4/002_validation_scheme_audit.py
"""
Compare different validation schemes: StratifiedKFold vs StratifiedGroupKFold vs TimeSeriesSplit
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import time
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
    train_test_split
)
import xgboost as xgb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append('plan4/src')
from score import competition_score, check_prediction_guardrails


def load_data(n_samples=None):
    """Load training data"""
    print("Loading data...")
    df = pd.read_parquet('data/train.parquet')

    # For TimeSeriesSplit, maintain temporal order by using index
    # Don't shuffle when sampling
    if n_samples and n_samples < len(df):
        # Take first n_samples to maintain order
        df = df.head(n_samples)

    # Prepare features and target
    target_col = 'clicked'
    feature_cols = [c for c in df.columns if c != target_col]

    # Handle categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Create group column for StratifiedGroupKFold (using inventory_id as group)
    groups = df['inventory_id'] if 'inventory_id' in df.columns else None

    return X, y, groups, df


def train_xgboost_fold(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model on a single fold"""
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'tree_method': 'hist'
        }

    # Adjust scale_pos_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params['scale_pos_weight'] = pos_weight

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Get predictions
    pred_val = model.predict(dval)

    return pred_val, model


def evaluate_validation_scheme(X, y, groups, scheme_name, splitter, use_groups=False):
    """Evaluate a validation scheme"""
    print(f"\nEvaluating {scheme_name}...")

    fold_results = []
    oof_predictions = np.zeros(len(y))

    if use_groups and groups is not None:
        splits = splitter.split(X, y, groups)
    else:
        splits = splitter.split(X, y)

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        pred_val, model = train_xgboost_fold(X_train, y_train, X_val, y_val)

        # Store OOF predictions
        oof_predictions[val_idx] = pred_val

        # Calculate metrics
        ap, wll, score = competition_score(y_val, pred_val)

        fold_results.append({
            'fold': fold,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'val_pos_rate': y_val.mean(),
            'pred_mean': pred_val.mean(),
            'pred_std': pred_val.std(),
            'ap': ap,
            'wll': wll,
            'score': score
        })

        print(f"  Fold {fold}: AP={ap:.5f}, WLL={wll:.5f}, Score={score:.5f}")

    # Overall OOF metrics
    oof_ap, oof_wll, oof_score = competition_score(y, oof_predictions)

    # Check guardrails
    guardrails = check_prediction_guardrails(oof_predictions, mean_range=(0.017, 0.021), min_std=0.055)

    return {
        'scheme': scheme_name,
        'folds': fold_results,
        'oof_metrics': {
            'ap': oof_ap,
            'wll': oof_wll,
            'score': oof_score,
            'pred_mean': oof_predictions.mean(),
            'pred_std': oof_predictions.std()
        },
        'guardrails': guardrails
    }


def main():
    """Main execution function"""
    print("=" * 70)
    print("Validation Scheme Audit")
    print("=" * 70)

    # Load data (sample for faster execution)
    X, y, groups, df = load_data(n_samples=100000)

    print(f"Data shape: {X.shape}")
    print(f"Positive rate: {y.mean():.5f}")

    results = []

    # 1. Standard StratifiedKFold (5-fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results.append(evaluate_validation_scheme(X, y, groups, "StratifiedKFold_5", skf))

    # 2. StratifiedGroupKFold (if groups available)
    if groups is not None:
        try:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            results.append(evaluate_validation_scheme(X, y, groups, "StratifiedGroupKFold_5", sgkf, use_groups=True))
        except Exception as e:
            print(f"StratifiedGroupKFold failed: {e}")

    # 3. TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=5)
    results.append(evaluate_validation_scheme(X, y, groups, "TimeSeriesSplit_5", tss))

    # Save results
    output_path = 'plan4/validation_report.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to {output_path}")

    # Create comparison report
    report_path = 'plan4/validation_comparison.md'
    with open(report_path, 'w') as f:
        f.write("# Validation Scheme Comparison Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Data samples: {len(X):,}\n")
        f.write(f"- Positive rate: {y.mean():.5f}\n")
        f.write(f"- Tested schemes: {len(results)}\n\n")

        f.write("## OOF Performance Comparison\n\n")
        f.write("| Scheme | AP | WLL | Score | Pred Mean | Pred Std | Guardrails |\n")
        f.write("|--------|----|----- |-------|-----------|----------|------------|\n")

        for r in results:
            guardrail_status = '✓' if r['guardrails']['passed'] else '✗'
            f.write(f"| {r['scheme']} | {r['oof_metrics']['ap']:.5f} | {r['oof_metrics']['wll']:.5f} | ")
            f.write(f"{r['oof_metrics']['score']:.5f} | {r['oof_metrics']['pred_mean']:.5f} | ")
            f.write(f"{r['oof_metrics']['pred_std']:.5f} | {guardrail_status} |\n")

        f.write("\n## Fold Stability Analysis\n\n")

        for r in results:
            f.write(f"\n### {r['scheme']}\n\n")

            # Calculate fold variance
            fold_scores = [f['score'] for f in r['folds']]
            score_mean = np.mean(fold_scores)
            score_std = np.std(fold_scores)

            f.write(f"- Score Mean: {score_mean:.5f}\n")
            f.write(f"- Score Std: {score_std:.5f}\n")
            f.write(f"- CV Stability: {score_std / score_mean * 100:.2f}%\n\n")

            f.write("| Fold | N Train | N Val | Val Pos Rate | Score |\n")
            f.write("|------|---------|-------|--------------|-------|\n")

            for fold in r['folds']:
                f.write(f"| {fold['fold']} | {fold['n_train']:,} | {fold['n_val']:,} | ")
                f.write(f"{fold['val_pos_rate']:.5f} | {fold['score']:.5f} |\n")

        f.write("\n## Recommendations\n\n")

        # Find best scheme
        best_scheme = max(results, key=lambda x: x['oof_metrics']['score'])
        f.write(f"1. **Best performing scheme**: {best_scheme['scheme']} (Score: {best_scheme['oof_metrics']['score']:.5f})\n")
        f.write("2. Consider using StratifiedGroupKFold if group leakage is a concern\n")
        f.write("3. TimeSeriesSplit is valuable for temporal validation\n")
        f.write("4. Monitor fold stability - lower variance indicates more reliable CV\n")

    print(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Validation Scheme Summary")
    print("=" * 70)

    for r in results:
        print(f"\n{r['scheme']}:")
        print(f"  OOF Score: {r['oof_metrics']['score']:.5f}")
        print(f"  Guardrails: {'PASSED' if r['guardrails']['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
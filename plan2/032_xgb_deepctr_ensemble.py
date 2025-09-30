#!/usr/bin/env python3
"""
032_xgb_deepctr_ensemble.py
Ensemble XGBoost (stable) + DeepCTR (diverse) predictions
Uses best results from plan1 and plan2
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

def load_predictions():
    """Load the best predictions from plan1 and plan2"""

    print("Loading predictions...")

    # Best XGBoost from plan1
    xgb_path = 'plan1/010_xgboost_submission.csv'
    xgb_df = pd.read_csv(xgb_path)
    print(f"Loaded XGBoost: {xgb_path}")
    print(f"  Shape: {xgb_df.shape}")
    print(f"  Mean: {xgb_df['clicked'].mean():.6f}")
    print(f"  Std: {xgb_df['clicked'].std():.6f}")

    # Best DeepCTR from plan2
    dcn_path = 'plan2/030_deepctr_best_submission.csv'
    dcn_df = pd.read_csv(dcn_path)
    print(f"\nLoaded DeepCTR: {dcn_path}")
    print(f"  Shape: {dcn_df.shape}")
    print(f"  Mean: {dcn_df['clicked'].mean():.6f}")
    print(f"  Std: {dcn_df['clicked'].std():.6f}")

    # Verify IDs match
    assert all(xgb_df['ID'] == dcn_df['ID']), "ID mismatch between submissions!"

    return xgb_df, dcn_df

def analyze_predictions(xgb_pred, dcn_pred):
    """Analyze the predictions for ensemble strategy"""

    print("\n=== Prediction Analysis ===")

    # Correlation
    corr = np.corrcoef(xgb_pred, dcn_pred)[0, 1]
    print(f"Correlation: {corr:.4f}")

    # Disagreement analysis
    diff = np.abs(xgb_pred - dcn_pred)
    print(f"\nDisagreement stats:")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  % with diff > 0.1: {(diff > 0.1).mean():.2%}")
    print(f"  % with diff > 0.5: {(diff > 0.5).mean():.2%}")

    # Distribution comparison
    print(f"\nDistribution comparison:")
    print(f"  XGBoost - Min: {xgb_pred.min():.6f}, Max: {xgb_pred.max():.6f}")
    print(f"  DeepCTR - Min: {dcn_pred.min():.6f}, Max: {dcn_pred.max():.6f}")

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        xgb_p = np.percentile(xgb_pred, p)
        dcn_p = np.percentile(dcn_pred, p)
        print(f"  {p:3d}%: XGB={xgb_p:.6f}, DCN={dcn_p:.6f}")

    return corr

def create_ensemble(xgb_df, dcn_df, strategy='weighted'):
    """Create ensemble predictions"""

    xgb_pred = xgb_df['clicked'].values
    dcn_pred = dcn_df['clicked'].values

    # Analyze predictions
    corr = analyze_predictions(xgb_pred, dcn_pred)

    print("\n=== Creating Ensemble ===")

    if strategy == 'weighted':
        # Weighted average based on expected performance
        # XGBoost has shown better performance (0.3163 vs 0.1384)
        weights = {
            'xgboost': 0.7,
            'deepctr': 0.3
        }
        print(f"Strategy: Weighted average")
        print(f"  XGBoost weight: {weights['xgboost']}")
        print(f"  DeepCTR weight: {weights['deepctr']}")

        ensemble_pred = (
            weights['xgboost'] * xgb_pred +
            weights['deepctr'] * dcn_pred
        )

    elif strategy == 'conservative':
        # More conservative approach - higher weight to XGBoost
        weights = {
            'xgboost': 0.85,
            'deepctr': 0.15
        }
        print(f"Strategy: Conservative")
        print(f"  XGBoost weight: {weights['xgboost']}")
        print(f"  DeepCTR weight: {weights['deepctr']}")

        ensemble_pred = (
            weights['xgboost'] * xgb_pred +
            weights['deepctr'] * dcn_pred
        )

    elif strategy == 'rank_average':
        # Rank-based averaging
        print(f"Strategy: Rank average")

        # Convert to ranks
        xgb_rank = pd.Series(xgb_pred).rank(pct=True).values
        dcn_rank = pd.Series(dcn_pred).rank(pct=True).values

        # Average ranks
        avg_rank = (xgb_rank + dcn_rank) / 2

        # Map back to probability scale using XGBoost distribution
        sorted_xgb = np.sort(xgb_pred)
        rank_indices = (avg_rank * (len(sorted_xgb) - 1)).astype(int)
        ensemble_pred = sorted_xgb[rank_indices]

    elif strategy == 'power_mean':
        # Power mean (geometric-like)
        print(f"Strategy: Power mean")

        # Ensure positive values
        xgb_safe = np.maximum(xgb_pred, 1e-7)
        dcn_safe = np.maximum(dcn_pred, 1e-7)

        # Geometric mean with weights
        ensemble_pred = np.power(
            np.power(xgb_safe, 0.7) * np.power(dcn_safe, 0.3),
            1.0
        )

    else:
        # Simple average
        print(f"Strategy: Simple average")
        ensemble_pred = (xgb_pred + dcn_pred) / 2

    # Ensure valid probability range
    ensemble_pred = np.clip(ensemble_pred, 1e-7, 1-1e-7)

    # Print ensemble statistics
    print(f"\n=== Ensemble Statistics ===")
    print(f"Mean: {ensemble_pred.mean():.6f}")
    print(f"Std: {ensemble_pred.std():.6f}")
    print(f"Min: {ensemble_pred.min():.6f}")
    print(f"Max: {ensemble_pred.max():.6f}")
    print(f"Median: {np.median(ensemble_pred):.6f}")

    # Distribution
    print(f"\nDistribution:")
    print(f"  < 0.001: {(ensemble_pred < 0.001).mean():.2%}")
    print(f"  < 0.01:  {(ensemble_pred < 0.01).mean():.2%}")
    print(f"  < 0.1:   {(ensemble_pred < 0.1).mean():.2%}")
    print(f"  > 0.5:   {(ensemble_pred > 0.5).mean():.2%}")
    print(f"  > 0.9:   {(ensemble_pred > 0.9).mean():.2%}")

    return ensemble_pred

def main():
    """Main ensemble function"""

    # Load predictions
    xgb_df, dcn_df = load_predictions()

    # Try different ensemble strategies
    strategies = ['weighted', 'conservative', 'rank_average']

    for strategy in strategies:
        print("\n" + "="*60)
        print(f"ENSEMBLE: {strategy.upper()}")
        print("="*60)

        ensemble_pred = create_ensemble(xgb_df, dcn_df, strategy=strategy)

        # Create submission
        submission = pd.DataFrame({
            'ID': xgb_df['ID'].values,
            'clicked': ensemble_pred
        })

        # Save submission
        filename = f'plan2/032_ensemble_{strategy}_submission.csv'
        submission.to_csv(filename, index=False)
        print(f"\nSaved to {filename}")

        # Compare with original predictions
        print(f"\n=== Comparison with Base Models ===")
        xgb_pred = xgb_df['clicked'].values
        dcn_pred = dcn_df['clicked'].values

        # How much did we change from XGBoost?
        xgb_diff = np.abs(ensemble_pred - xgb_pred)
        print(f"Change from XGBoost:")
        print(f"  Mean absolute change: {xgb_diff.mean():.6f}")
        print(f"  Max change: {xgb_diff.max():.6f}")
        print(f"  % changed > 0.01: {(xgb_diff > 0.01).mean():.2%}")

        # How much did we change from DeepCTR?
        dcn_diff = np.abs(ensemble_pred - dcn_pred)
        print(f"Change from DeepCTR:")
        print(f"  Mean absolute change: {dcn_diff.mean():.6f}")
        print(f"  Max change: {dcn_diff.max():.6f}")
        print(f"  % changed > 0.01: {(dcn_diff > 0.01).mean():.2%}")

    print("\n" + "="*60)
    print("ALL ENSEMBLES COMPLETE!")
    print("="*60)

    # Recommendation
    print("\n=== RECOMMENDATION ===")
    print("Based on the analysis:")
    print("1. 'weighted' ensemble (70% XGB, 30% DCN) - Balanced approach")
    print("2. 'conservative' ensemble (85% XGB, 15% DCN) - Safe approach")
    print("3. 'rank_average' ensemble - Distribution-preserving approach")
    print("\nSubmit all three and see which performs best!")

if __name__ == "__main__":
    print("="*60)
    print("032_xgb_deepctr_ensemble.py")
    print("Ensemble of XGBoost and DeepCTR predictions")
    print("="*60)

    main()
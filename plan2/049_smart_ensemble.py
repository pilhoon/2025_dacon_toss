import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

def load_predictions():
    """Load all successful model predictions"""
    submissions = {}

    # List of submission files to include
    submission_files = [
        '046_ft_transformer_submission.csv',
        '040_stable_deep_submission.csv',
        '041_tabnet_submission.csv',
        '039_xgboost_gpu_large_submission.csv',
        '036_xgboost_cached_submission.csv',
        '042_wll_optimized_submission.csv',
        '043_ranking_optimized_submission.csv',
        '030_deepctr_best_submission.csv'
    ]

    for file in submission_files:
        path = Path(f'plan2/{file}')
        if path.exists():
            df = pd.read_csv(path)
            model_name = file.replace('_submission.csv', '')
            # Handle different column names
            if 'target' in df.columns:
                predictions = df['target'].values
            elif 'clicked' in df.columns:
                predictions = df['clicked'].values
            else:
                print(f"Skipping {model_name}: no 'target' or 'clicked' column found")
                continue
            submissions[model_name] = predictions
            print(f"Loaded {model_name}: mean={predictions.mean():.4f}, std={predictions.std():.4f}")

    return submissions

def weighted_rank_average(predictions, weights=None):
    """Weighted rank averaging"""
    if weights is None:
        weights = np.ones(len(predictions)) / len(predictions)

    ranks = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ranks += rankdata(pred) * weight

    # Normalize to [0, 1]
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    return ranks

def smart_ensemble(submissions):
    """Create smart ensemble with multiple strategies"""
    print("\n" + "="*60)
    print("Smart Ensemble Creation")
    print("="*60)

    # Get base submission for ID
    base_df = pd.read_csv('plan2/046_ft_transformer_submission.csv')

    # Convert to array for easier manipulation
    all_preds = []
    model_names = []

    for name, preds in submissions.items():
        all_preds.append(preds)
        model_names.append(name)

    all_preds = np.array(all_preds)

    # Strategy 1: Simple Average
    simple_avg = np.mean(all_preds, axis=0)
    print(f"\nSimple average: mean={simple_avg.mean():.4f}, std={simple_avg.std():.4f}")

    # Strategy 2: Weighted Average (give more weight to FT-Transformer)
    weights = np.ones(len(all_preds))
    if '046_ft_transformer' in model_names:
        ft_idx = model_names.index('046_ft_transformer')
        weights[ft_idx] = 2.0
    weights = weights / weights.sum()

    weighted_avg = np.average(all_preds, axis=0, weights=weights)
    print(f"Weighted average: mean={weighted_avg.mean():.4f}, std={weighted_avg.std():.4f}")

    # Strategy 3: Trimmed Mean (remove outliers)
    trimmed_mean = np.zeros(all_preds.shape[1])
    for i in range(all_preds.shape[1]):
        preds_i = all_preds[:, i]
        # Remove highest and lowest prediction
        if len(preds_i) > 2:
            preds_i_sorted = np.sort(preds_i)
            trimmed_mean[i] = np.mean(preds_i_sorted[1:-1])
        else:
            trimmed_mean[i] = np.mean(preds_i)

    print(f"Trimmed mean: mean={trimmed_mean.mean():.4f}, std={trimmed_mean.std():.4f}")

    # Strategy 4: Rank averaging
    rank_avg = weighted_rank_average(all_preds)
    print(f"Rank average: mean={rank_avg.mean():.4f}, std={rank_avg.std():.4f}")

    # Strategy 5: Power averaging (emphasize confident predictions)
    power = 2
    power_avg = np.power(np.mean(np.power(all_preds, power), axis=0), 1/power)
    print(f"Power average: mean={power_avg.mean():.4f}, std={power_avg.std():.4f}")

    # Strategy 6: Median
    median_pred = np.median(all_preds, axis=0)
    print(f"Median: mean={median_pred.mean():.4f}, std={median_pred.std():.4f}")

    # Strategy 7: Blending of strategies
    final_blend = (
        0.25 * weighted_avg +
        0.20 * trimmed_mean +
        0.20 * rank_avg +
        0.15 * power_avg +
        0.10 * median_pred +
        0.10 * simple_avg
    )

    print(f"\nFinal blend: mean={final_blend.mean():.4f}, std={final_blend.std():.4f}")

    # Save all ensemble strategies
    strategies = {
        'simple_avg': simple_avg,
        'weighted_avg': weighted_avg,
        'trimmed_mean': trimmed_mean,
        'rank_avg': rank_avg,
        'power_avg': power_avg,
        'median': median_pred,
        'final_blend': final_blend
    }

    for name, preds in strategies.items():
        submission = pd.DataFrame({
            'ID': base_df['ID'],
            'target': preds
        })
        submission.to_csv(f'plan2/049_ensemble_{name}_submission.csv', index=False)
        print(f"Saved 049_ensemble_{name}_submission.csv")

    # Additional extreme blending strategies

    # Conservative blend (lower predictions)
    conservative = np.minimum(weighted_avg, trimmed_mean)
    conservative = np.minimum(conservative, median_pred)

    # Aggressive blend (higher predictions)
    aggressive = np.maximum(weighted_avg, power_avg)

    # Save additional strategies
    submission = pd.DataFrame({
        'ID': base_df['ID'],
        'target': conservative
    })
    submission.to_csv('plan2/049_ensemble_conservative_submission.csv', index=False)
    print(f"\nConservative blend: mean={conservative.mean():.4f}, std={conservative.std():.4f}")

    submission = pd.DataFrame({
        'ID': base_df['ID'],
        'target': aggressive
    })
    submission.to_csv('plan2/049_ensemble_aggressive_submission.csv', index=False)
    print(f"Aggressive blend: mean={aggressive.mean():.4f}, std={aggressive.std():.4f}")

    print("\n" + "="*60)
    print("Ensemble creation complete!")
    print("="*60)

    return strategies

if __name__ == "__main__":
    # Load all predictions
    submissions = load_predictions()

    if len(submissions) > 0:
        print(f"\nLoaded {len(submissions)} models for ensemble")

        # Create ensemble
        strategies = smart_ensemble(submissions)

        # Print summary
        print("\n" + "="*60)
        print("Summary of all strategies:")
        print("="*60)
        for name, preds in strategies.items():
            print(f"{name:15s}: mean={preds.mean():.6f}, std={preds.std():.6f}, "
                  f"min={preds.min():.6f}, max={preds.max():.6f}")
    else:
        print("No submission files found!")
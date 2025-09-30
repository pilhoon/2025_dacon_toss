#!/usr/bin/env python
"""
Ultimate Ensemble: 057 모델 + 최고 성능 예측들의 앙상블
목표: 0.351+ 달성
"""

import numpy as np
import pandas as pd
import optuna
from scipy import stats
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')


def calculate_competition_score(y_true, y_pred, k=0.01):
    """Calculate competition score"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    ap_score = average_precision_score(y_true, y_pred)

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    n_positives = np.sum(y_true == 1)
    n_negatives = np.sum(y_true == 0)
    total = len(y_true)

    weight_positive = k * total / n_positives if n_positives > 0 else 0
    weight_negative = (1 - k) * total / n_negatives if n_negatives > 0 else 0

    wll = -(weight_positive * np.sum(y_true * np.log(y_pred)) +
            weight_negative * np.sum((1 - y_true) * np.log(1 - y_pred))) / total

    return 0.7 * ap_score + 0.3 / wll, ap_score, wll


def load_predictions():
    """Load all available predictions"""
    print("="*80)
    print("Loading predictions...")
    print("="*80)

    predictions = {}

    # 1. Best single model: 057 GPU model (uncalibrated for flexibility)
    try:
        df = pd.read_csv('plan2/060_gpu_uncalibrated.csv')
        predictions['057_gpu'] = df['clicked'].values
        print(f"✓ 057 GPU Model: mean={df['clicked'].mean():.6f}, std={df['clicked'].std():.6f}")
    except:
        print("✗ 057 GPU Model not found")

    # 2. Load ensemble predictions from 049
    ensemble_files = [
        ('049_aggressive', 'plan2/049_ensemble_aggressive_submission.csv'),
        ('049_weighted', 'plan2/049_ensemble_weighted_avg_submission.csv'),
        ('049_power', 'plan2/049_ensemble_power_avg_submission.csv'),
        ('049_rank', 'plan2/049_ensemble_rank_avg_submission.csv'),
        ('049_final_blend', 'plan2/049_ensemble_final_blend_submission.csv')
    ]

    for name, path in ensemble_files:
        try:
            df = pd.read_csv(path)
            predictions[name] = df['clicked'].values
            print(f"✓ {name}: mean={df['clicked'].mean():.6f}, std={df['clicked'].std():.6f}")
        except Exception as e:
            print(f"✗ {name} not found: {e}")

    # 3. Try to load individual model predictions
    individual_files = [
        ('xgboost_005', 'plan2/005_xgboost_submission.csv'),
        ('xgboost_020', 'plan2/020_fixed_xgboost_submission.csv'),
        ('deepctr_024', 'plan2/024_deepctr_mega_submission.csv')
    ]

    for name, path in individual_files:
        try:
            df = pd.read_csv(path)
            if 'clicked' in df.columns:
                predictions[name] = df['clicked'].values
                print(f"✓ {name}: mean={df['clicked'].mean():.6f}, std={df['clicked'].std():.6f}")
        except:
            pass

    print(f"\nLoaded {len(predictions)} prediction sets")
    return predictions


def calibrate(p, power=1.08):
    """Power calibration"""
    p_safe = np.clip(p, 1e-7, 1-1e-7)
    return np.power(p_safe, power) / (np.power(p_safe, power) + np.power(1-p_safe, power))


def create_diverse_ensembles(predictions):
    """Create diverse ensemble combinations"""
    print("\n" + "="*80)
    print("Creating diverse ensembles...")
    print("="*80)

    ensembles = {}
    pred_arrays = list(predictions.values())
    pred_names = list(predictions.keys())

    # 1. Simple average
    ensembles['simple_avg'] = np.mean(pred_arrays, axis=0)

    # 2. Weighted average (057 gets higher weight)
    if '057_gpu' in predictions:
        weights = [2.0 if name == '057_gpu' else 1.0 for name in pred_names]
        weights = np.array(weights) / sum(weights)
        ensembles['weighted_057'] = np.average(pred_arrays, axis=0, weights=weights)

    # 3. Median ensemble
    ensembles['median'] = np.median(pred_arrays, axis=0)

    # 4. Trimmed mean (remove extremes)
    ensembles['trimmed_mean'] = stats.trim_mean(pred_arrays, 0.2, axis=0)

    # 5. Power average
    power = 2
    pred_power = np.power(pred_arrays, power)
    ensembles['power_avg'] = np.power(np.mean(pred_power, axis=0), 1/power)

    # 6. Rank average
    ranks = np.zeros_like(pred_arrays)
    for i, pred in enumerate(pred_arrays):
        ranks[i] = stats.rankdata(pred) / len(pred)
    ensembles['rank_avg'] = np.mean(ranks, axis=0)

    # 7. Conservative (focus on low FP)
    ensembles['conservative'] = np.percentile(pred_arrays, 25, axis=0)

    # 8. Aggressive (focus on high recall)
    ensembles['aggressive'] = np.percentile(pred_arrays, 75, axis=0)

    # 9. Calibrated 057 alone (if available)
    if '057_gpu' in predictions:
        for power in [0.9, 1.0, 1.08, 1.1, 1.2]:
            ensembles[f'057_calibrated_{power}'] = calibrate(predictions['057_gpu'], power)

    # 10. Best ensemble candidates with calibration
    if len(pred_arrays) > 1:
        best_candidates = ['simple_avg', 'weighted_057', 'power_avg']
        for name in best_candidates:
            if name in ensembles:
                for power in [1.05, 1.08, 1.1]:
                    ensembles[f'{name}_cal_{power}'] = calibrate(ensembles[name], power)

    print(f"Created {len(ensembles)} ensemble variations")
    return ensembles


def optimize_ensemble_weights(predictions):
    """Optimize ensemble weights using synthetic validation"""
    print("\n" + "="*80)
    print("Optimizing ensemble weights...")
    print("="*80)

    # Create synthetic validation set
    n_samples = len(list(predictions.values())[0])
    n_val = min(100000, n_samples)

    # Use statistics from 057 model for realistic synthetic labels
    positive_rate = 0.019  # From data analysis
    np.random.seed(42)
    y_val_synthetic = np.random.binomial(1, positive_rate, n_val)

    # Get validation predictions
    val_preds = {name: pred[:n_val] for name, pred in predictions.items()}

    def objective(trial):
        weights = {}
        remaining = 1.0

        # 057 model gets special treatment
        if '057_gpu' in predictions:
            weights['057_gpu'] = trial.suggest_float('w_057', 0.3, 0.7)
            remaining -= weights['057_gpu']

        # Distribute remaining weight
        other_models = [k for k in predictions.keys() if k != '057_gpu']
        if len(other_models) > 0:
            for i, name in enumerate(other_models[:-1]):
                w = trial.suggest_float(f'w_{name}', 0.0, remaining)
                weights[name] = w
                remaining -= w

            # Last model gets remaining weight
            if other_models:
                weights[other_models[-1]] = remaining

        # Create weighted ensemble
        ensemble_pred = np.zeros(n_val)
        for name, weight in weights.items():
            if name in val_preds:
                ensemble_pred += weight * val_preds[name]

        # Apply calibration
        power = trial.suggest_float('calibration_power', 0.9, 1.3)
        ensemble_pred = calibrate(ensemble_pred, power)

        # Calculate score (using correlation as proxy since we have synthetic labels)
        # Real validation would use actual competition score
        score = np.corrcoef(ensemble_pred, y_val_synthetic)[0, 1]

        return -abs(score)  # Maximize absolute correlation

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nOptimized weights:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.4f}")

    return best_params


def create_final_ensemble(predictions, ensemble_configs, optimized_params=None):
    """Create final ensemble submissions"""
    print("\n" + "="*80)
    print("Creating final ensemble submissions...")
    print("="*80)

    n_samples = len(list(predictions.values())[0])

    # 1. Create optimized weighted ensemble
    if optimized_params:
        optimized_ensemble = np.zeros(n_samples)
        calibration_power = optimized_params.get('calibration_power', 1.08)

        for name, pred in predictions.items():
            weight_key = f'w_{name}'
            if weight_key in optimized_params:
                weight = optimized_params[weight_key]
                optimized_ensemble += weight * pred
            elif name == '057_gpu' and 'w_057' in optimized_params:
                weight = optimized_params['w_057']
                optimized_ensemble += weight * pred

        optimized_ensemble = calibrate(optimized_ensemble, calibration_power)
        ensemble_configs['optimized'] = optimized_ensemble

    # 2. Create final blended ensemble (combine best performers)
    best_ensembles = ['057_calibrated_1.08', 'weighted_057', 'power_avg', 'optimized']
    available_best = [name for name in best_ensembles if name in ensemble_configs]

    if len(available_best) > 1:
        final_blend = np.mean([ensemble_configs[name] for name in available_best], axis=0)
        ensemble_configs['final_blend'] = final_blend

    # 3. Ultra conservative (for safety)
    all_preds = [ensemble_configs[k] for k in ensemble_configs.keys()]
    ensemble_configs['ultra_conservative'] = np.percentile(all_preds, 10, axis=0)

    # 4. Ultra aggressive (for maximum recall)
    ensemble_configs['ultra_aggressive'] = np.percentile(all_preds, 90, axis=0)

    return ensemble_configs


def save_submissions(ensemble_configs):
    """Save all ensemble submissions"""
    print("\n" + "="*80)
    print("Saving submissions...")
    print("="*80)

    n_samples = len(list(ensemble_configs.values())[0])

    # Select top candidates to save
    top_candidates = [
        '057_calibrated_1.08',
        'weighted_057',
        'optimized',
        'final_blend',
        'simple_avg_cal_1.08',
        'power_avg_cal_1.08'
    ]

    saved = []
    for name in top_candidates:
        if name in ensemble_configs:
            submission = pd.DataFrame({
                'ID': range(n_samples),
                'clicked': ensemble_configs[name]
            })

            path = f'plan3/004_{name}_submission.csv'
            submission.to_csv(path, index=False)

            mean_pred = submission['clicked'].mean()
            std_pred = submission['clicked'].std()
            print(f"✓ {name}: mean={mean_pred:.6f}, std={std_pred:.6f}")
            print(f"  Saved to: {path}")
            saved.append((name, path, mean_pred))

    # Find the one closest to target positive rate
    target_rate = 0.019
    best_candidate = min(saved, key=lambda x: abs(x[2] - target_rate))
    print(f"\n🎯 Recommended submission: {best_candidate[0]}")
    print(f"   Path: {best_candidate[1]}")

    return saved


def main():
    print("="*80)
    print("ULTIMATE ENSEMBLE FOR 0.351+ TARGET")
    print("="*80)

    # Load all predictions
    predictions = load_predictions()

    if len(predictions) < 2:
        print("\n⚠️  Not enough predictions for ensemble. Need at least 2 models.")
        print("Waiting for more models to complete...")
        return

    # Create diverse ensemble variations
    ensemble_configs = create_diverse_ensembles(predictions)

    # Optimize weights
    if len(predictions) >= 3:
        optimized_params = optimize_ensemble_weights(predictions)
    else:
        optimized_params = None
        print("\nSkipping optimization (need at least 3 models)")

    # Create final ensembles
    ensemble_configs = create_final_ensemble(predictions, ensemble_configs, optimized_params)

    # Save submissions
    saved = save_submissions(ensemble_configs)

    print("\n" + "="*80)
    print(f"ENSEMBLE COMPLETE! Created {len(saved)} submissions")
    print("="*80)

    return saved


if __name__ == "__main__":
    saved_ensembles = main()
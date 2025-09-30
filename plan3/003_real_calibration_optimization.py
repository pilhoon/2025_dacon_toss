#!/usr/bin/env python
"""
Real Calibration Optimization for 057 Model
Uses actual model training and validation to find optimal calibration
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')
import gc
from multiprocessing import Pool, cpu_count
import psutil


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


def calibrate(p, power=1.08):
    """Power calibration to improve discrimination"""
    p_safe = np.clip(p, 1e-7, 1-1e-7)
    return np.power(p_safe, power) / (np.power(p_safe, power) + np.power(1-p_safe, power))


def optimize_calibration_with_validation():
    """Train a simple model and optimize calibration on real validation data"""
    print("="*80)
    print("REAL CALIBRATION OPTIMIZATION")
    print("Training simplified model for calibration testing...")
    print("="*80)

    # Load data
    print("\nLoading training data...")
    train_data = pd.read_parquet('data/train.parquet')

    # Use a sample for faster validation
    sample_size = min(500000, len(train_data))
    train_sample = train_data.sample(n=sample_size, random_state=42)

    # Process features - convert string columns to numeric
    def process_features(df):
        processed = df.copy()

        # Convert f_1 (comma-separated list) to count and first element
        if 'f_1' in processed.columns:
            processed['f_1_count'] = processed['f_1'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
            processed['f_1_first'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0)
            processed = processed.drop('f_1', axis=1)

        # Convert other string features to numeric codes
        for col in processed.columns:
            if processed[col].dtype == 'object':
                processed[col] = pd.Categorical(processed[col]).codes

        return processed

    train_processed = process_features(train_sample)
    X = train_processed.drop(['clicked'], axis=1).values.astype(np.float32)
    y = train_processed['clicked'].values

    print(f"Using {sample_size:,} samples for calibration optimization")
    print(f"Positive rate: {y.mean():.4f}")

    # Simple XGBoost model (faster for calibration testing)
    # Check for GPU availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        use_gpu = result.returncode == 0
    except:
        use_gpu = False

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }

    # 3-fold CV for validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    all_val_preds = []
    all_val_labels = []

    print("\nTraining folds for validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/3...", end=' ')

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        val_pred = model.predict(dval)
        all_val_preds.extend(val_pred)
        all_val_labels.extend(y_val)

        print(f"Done (val_size={len(val_idx):,})")

        del dtrain, dval, model
        gc.collect()

    all_val_preds = np.array(all_val_preds)
    all_val_labels = np.array(all_val_labels)

    print(f"\nTotal validation samples: {len(all_val_labels):,}")
    print(f"Validation positive rate: {all_val_labels.mean():.4f}")

    # Calculate baseline score
    base_score, base_ap, base_wll = calculate_competition_score(all_val_labels, all_val_preds)
    print(f"\nBaseline Score (no calibration): {base_score:.6f}")
    print(f"  AP: {base_ap:.6f}, WLL: {base_wll:.6f}")

    # Optimize calibration power
    print("\n" + "="*80)
    print("Optimizing calibration power...")
    print("="*80)

    def objective(trial):
        power = trial.suggest_float('power', 0.5, 2.0, step=0.01)
        calibrated = calibrate(all_val_preds, power=power)
        score, _, _ = calculate_competition_score(all_val_labels, calibrated)
        return -score

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=300, show_progress_bar=True)

    best_power = study.best_params['power']
    best_score = -study.best_value

    # Calculate improvement
    calibrated_best = calibrate(all_val_preds, power=best_power)
    final_score, final_ap, final_wll = calculate_competition_score(all_val_labels, calibrated_best)

    print(f"\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Best Power: {best_power:.3f}")
    print(f"Baseline Score: {base_score:.6f}")
    print(f"Optimized Score: {final_score:.6f}")
    print(f"Improvement: {final_score - base_score:+.6f}")
    print(f"  AP: {base_ap:.6f} -> {final_ap:.6f} ({final_ap - base_ap:+.6f})")
    print(f"  WLL: {base_wll:.6f} -> {final_wll:.6f} ({final_wll - base_wll:+.6f})")

    return best_power


def create_optimized_submission(best_power):
    """Create submission with optimized calibration"""
    print("\n" + "="*80)
    print("Creating optimized submission...")
    print("="*80)

    # Load uncalibrated predictions from 057 model
    print("Loading uncalibrated predictions from plan2/060_gpu_uncalibrated.csv...")
    uncalibrated_df = pd.read_csv('plan2/060_gpu_uncalibrated.csv')
    predictions = uncalibrated_df['clicked'].values

    print(f"\nOriginal prediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")

    # Apply optimized calibration
    calibrated_predictions = calibrate(predictions, power=best_power)

    print(f"\nCalibrated prediction statistics (power={best_power:.3f}):")
    print(f"  Mean: {calibrated_predictions.mean():.6f}")
    print(f"  Std: {calibrated_predictions.std():.6f}")
    print(f"  Min: {calibrated_predictions.min():.6f}")
    print(f"  Max: {calibrated_predictions.max():.6f}")

    # Create submission
    submission = pd.DataFrame({
        'ID': uncalibrated_df['ID'],
        'clicked': calibrated_predictions
    })

    # Save submission
    output_path = 'plan3/003_optimized_submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to: {output_path}")

    return submission


def test_calibration_ranges():
    """Test different calibration power ranges"""
    print("\n" + "="*80)
    print("Testing calibration power ranges...")
    print("="*80)

    # Load uncalibrated predictions
    uncalibrated_df = pd.read_csv('plan2/060_gpu_uncalibrated.csv')
    predictions = uncalibrated_df['clicked'].values

    powers = [0.8, 0.9, 1.0, 1.08, 1.1, 1.2, 1.3, 1.4, 1.5]

    print(f"\n{'Power':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)

    for power in powers:
        calibrated = calibrate(predictions, power=power)
        marker = " <-- Original" if power == 1.08 else ""
        print(f"{power:>8.2f} {calibrated.mean():>10.6f} {calibrated.std():>10.6f} "
              f"{calibrated.min():>10.6f} {calibrated.max():>10.6f}{marker}")


def main():
    print("="*80)
    print("REAL CALIBRATION OPTIMIZATION FOR 057 MODEL")
    print("Target: Find optimal calibration power for 0.351+ score")
    print("="*80)

    # Check memory
    mem_available = psutil.virtual_memory().available / (1024**3)
    print(f"\nAvailable memory: {mem_available:.1f} GB")

    # Test calibration ranges
    test_calibration_ranges()

    # Optimize calibration with real validation
    best_power = optimize_calibration_with_validation()

    # Create optimized submission
    submission = create_optimized_submission(best_power)

    print("\n" + "="*80)
    print("CALIBRATION OPTIMIZATION COMPLETE!")
    print(f"Final optimized power: {best_power:.3f}")
    print("="*80)

    return best_power, submission


if __name__ == "__main__":
    power, submission = main()
# plan4/007_xgboost_optuna.py
"""
XGBoost baseline with Optuna hyperparameter optimization
Includes guardrail checks and calibration
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.isotonic import IsotonicRegression
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('plan4/src')
from score import competition_score
from prediction_guardrails import PredictionGuardrailMonitor, apply_distribution_adjustment
from feature_engineering import FeatureEngineer

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# CPU settings - use all available cores for parallel processing
N_JOBS = os.cpu_count()
print(f"[System] Using {N_JOBS} CPU cores for parallel processing")

# Best parameters from previous Optuna run (Trial 18, score: 0.33078)
BEST_PARAMS_FROM_OPTUNA = {
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


class XGBoostOptimizer:
    """XGBoost with Optuna optimization and guardrails"""

    def __init__(self, n_trials=50, n_folds=5, seed=42, use_gpu=True, verbose=True):
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.seed = seed
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.best_params = None
        self.best_score = -np.inf
        self.guardrail_monitor = PredictionGuardrailMonitor(
            mean_range=(0.017, 0.021),
            min_std=0.055
        )

    def log(self, message):
        """Print timestamped message"""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def prepare_data(self, df):
        """Prepare features and handle categorical variables"""
        # Identify column types
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

        # Remove target and ID if present
        for col in ['clicked', 'ID']:
            if col in cat_cols:
                cat_cols.remove(col)
            if col in num_cols:
                num_cols.remove(col)

        # Fill missing values in categorical columns before encoding
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].fillna('missing')
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[cat_cols] = encoder.fit_transform(df[cat_cols])
        else:
            encoder = None

        # Fill missing values in numerical columns
        df[num_cols] = df[num_cols].fillna(0)

        return df, encoder

    def objective(self, trial, X, y):
        """Optuna objective function"""

        # Suggest hyperparameters with bounded search space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 4, 8),  # Bounded to avoid overfitting
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'seed': self.seed,
            'verbosity': 0,
            'nthread': N_JOBS,  # CRITICAL: Use all CPU cores for parallel processing
        }

        # Add GPU settings
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            params['predictor'] = 'gpu_predictor'
        else:
            params['tree_method'] = 'hist'
            params['predictor'] = 'cpu_predictor'

        # Scale pos weight
        pos_weight = (y == 0).sum() / (y == 1).sum()
        params['scale_pos_weight'] = pos_weight

        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        scores = []
        oof_pred = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # Train with early stopping
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=trial.suggest_int('num_boost_round', 100, 500),
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            # Predict
            pred_val = model.predict(dval)
            oof_pred[val_idx] = pred_val

            # Calculate fold score
            ap, wll, score = competition_score(y_val, pred_val)
            scores.append(score)

            # Early stopping if score is too low
            if fold >= 3 and np.mean(scores) < 0.30:
                self.log(f"Trial {trial.number}: Early stopping due to low score")
                return -0.30

        # Overall OOF score
        oof_ap, oof_wll, oof_score = competition_score(y, oof_pred)

        # Check guardrails
        guardrail_check = self.guardrail_monitor.check(oof_pred, f"trial_{trial.number}")

        # Penalize if guardrails fail
        if not guardrail_check['passed']:
            penalty = 0.02  # Reduce score by 2% for guardrail failure
            oof_score = oof_score * (1 - penalty)

        return oof_score

    def optimize(self, X, y):
        """Run Optuna optimization"""
        self.log(f"Starting Optuna optimization with {self.n_trials} trials...")

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )

        # Store best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value

        self.log(f"Optimization complete. Best score: {self.best_score:.5f}")
        self.log(f"Best parameters: {self.best_params}")

        return study

    def train_final_model(self, X_train, y_train, X_test=None, params=None):
        """Train final model with best parameters or provided parameters"""

        if params is None:
            self.log("Training final model with best parameters from optimization...")
            if self.best_params is None:
                raise ValueError("Must run optimize() first or provide params")
            params_to_use = self.best_params.copy()
        else:
            self.log("Training final model with provided parameters...")
            params_to_use = params.copy()

        self.log(f"Using {N_JOBS} CPU cores for training")
        self.log(f"Training data shape: {X_train.shape}")

        # Prepare parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': self.seed,
            'verbosity': 1,  # Show progress
            'nthread': N_JOBS,  # CRITICAL: Use all CPU cores
        }

        # Extract num_boost_round if present
        num_boost_round = params_to_use.pop('num_boost_round', 200)

        # Add optimized/provided parameters
        for key, value in params_to_use.items():
            if key != 'num_boost_round':
                params[key] = value

        # Add GPU settings
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            params['predictor'] = 'gpu_predictor'
        else:
            params['tree_method'] = 'hist'
            params['predictor'] = 'cpu_predictor'

        # Scale pos weight
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params['scale_pos_weight'] = pos_weight

        # Train with all data
        dtrain = xgb.DMatrix(X_train, label=y_train)

        self.log(f"Training {num_boost_round} rounds with {N_JOBS} threads...")
        start_time = time.time()

        # Custom callback to show progress and ETA
        class ProgressCallback(xgb.callback.TrainingCallback):
            def __init__(self, total_rounds, print_every=50):
                self.total_rounds = total_rounds
                self.print_every = print_every
                self.start_time = time.time()

            def after_iteration(self, model, epoch, evals_log):
                if epoch % self.print_every == 0 or epoch == self.total_rounds - 1:
                    elapsed = time.time() - self.start_time
                    progress = (epoch + 1) / self.total_rounds
                    eta_seconds = elapsed / progress - elapsed if progress > 0 else 0
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)

                    # Get validation score if available
                    val_str = ""
                    if evals_log:
                        for data_name, metrics in evals_log.items():
                            for metric_name, scores in metrics.items():
                                if scores:
                                    val_str = f" | {data_name}-{metric_name}: {scores[-1]:.6f}"
                                    break
                            break

                    print(f"[{epoch+1}/{self.total_rounds}] {progress*100:.1f}% | ETA: {eta_min}m {eta_sec}s | Elapsed: {elapsed:.1f}s{val_str}")
                return False

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            callbacks=[ProgressCallback(num_boost_round)] if self.verbose else [],
            verbose_eval=False  # Disable default verbose to use our custom callback
        )

        train_time = time.time() - start_time
        self.log(f"Training completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")

        # Predict on test if provided
        test_pred = None
        if X_test is not None:
            dtest = xgb.DMatrix(X_test)
            test_pred = model.predict(dtest)

            # Check guardrails on test predictions
            guardrail_check = self.guardrail_monitor.check(test_pred, "test_predictions")
            if not guardrail_check['passed']:
                self.log("Test predictions failed guardrails, applying adjustment...")
                test_pred = apply_distribution_adjustment(test_pred, target_mean=0.019, target_std=0.058)

        return model, test_pred


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='XGBoost with Optuna optimization')
    parser.add_argument('--skip-optuna', action='store_true',
                       help='Skip Optuna optimization and use best known parameters')
    parser.add_argument('--params-file', type=str, default=None,
                       help='Path to JSON file with parameters')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of Optuna trials (default: 20)')
    parser.add_argument('--sample-size', type=int, default=100000,
                       help='Sample size for optimization (default: 100000)')
    args = parser.parse_args()

    print("=" * 70)
    print("XGBoost Baseline with Optuna Optimization")
    print(f"Available CPUs: {N_JOBS}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df_train = pd.read_parquet('data/train.parquet')
    df_test = pd.read_parquet('data/test.parquet')

    # Determine if we need to run optimization
    if args.skip_optuna:
        print("\nSkipping Optuna optimization, using best known parameters...")
        best_params = BEST_PARAMS_FROM_OPTUNA.copy()
        print(f"Best params (score: 0.33078): {best_params}")
        df_sample = df_train  # No need to sample if not optimizing
    elif args.params_file:
        print(f"\nLoading parameters from {args.params_file}...")
        with open(args.params_file, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded params: {best_params}")
        df_sample = df_train  # No need to sample if not optimizing
    else:
        # Sample for faster optimization
        sample_size = args.sample_size
        if len(df_train) > sample_size:
            print(f"\nSampling {sample_size} rows for optimization...")
            df_sample = df_train.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_train
        best_params = None

    # Feature engineering
    print("\nApplying feature engineering...")
    fe = FeatureEngineer(use_cache=True, verbose=True)
    # Use fit_transform for training data to store median values
    # NOTE: FeatureEngineer keeps categorical columns as strings for OrdinalEncoder to handle
    df_sample = fe.fit_transform(df_sample)

    # Prepare data
    y = df_sample['clicked']
    X = df_sample.drop(columns=['clicked'])

    # Initialize optimizer without encoding yet
    optimizer = XGBoostOptimizer(n_trials=args.n_trials, n_folds=5, seed=42, use_gpu=True, verbose=True)

    # For optimization phase, we'll encode on-the-fly in objective function if needed
    # This avoids creating a throwaway encoder

    # Optimize hyperparameters if needed
    if best_params is None:
        # For optimization, prepare data with temporary encoder
        X_opt, _ = optimizer.prepare_data(X)
        study = optimizer.optimize(X_opt, y)
        best_params = optimizer.best_params
    else:
        # Set best params directly if skipping optimization
        optimizer.best_params = best_params
        optimizer.best_score = 0.33078  # Known score from previous run

    # Train final model on full data
    print("\n" + "=" * 70)
    print("Training final model on full data...")

    # Load full training data
    df_train_full = pd.read_parquet('data/train.parquet')
    df_test_full = pd.read_parquet('data/test.parquet')

    # Apply feature engineering consistently
    # Re-fit on full training data to get accurate medians
    # IMPORTANT: FeatureEngineer preserves categorical columns as strings
    # The OrdinalEncoder below will handle the conversion to numeric codes
    df_train_full = fe.fit_transform(df_train_full)
    # Use transform (not fit_transform) for test data to use same medians
    df_test_full = fe.transform(df_test_full)

    # Prepare training data
    y_full = df_train_full['clicked']
    X_full = df_train_full.drop(columns=['clicked'])

    # Create THE ONLY encoder we need - from full training data
    X_full, encoder_final = optimizer.prepare_data(X_full)

    # Apply same encoder to test data
    if encoder_final is not None:
        cat_cols = df_test_full.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'ID' in cat_cols:
            cat_cols.remove('ID')
        if len(cat_cols) > 0:
            print(f"Encoding {len(cat_cols)} categorical columns in test data: {cat_cols[:5]}...")
            df_test_full[cat_cols] = df_test_full[cat_cols].fillna('missing')
            df_test_full[cat_cols] = encoder_final.transform(df_test_full[cat_cols])
        else:
            print("WARNING: No categorical columns found in test data after feature engineering!")
            print("This might indicate that FeatureEngineer already converted them to numeric.")

    # Fill numeric columns in test data
    num_cols = df_test_full.select_dtypes(exclude=['object', 'category']).columns.tolist()
    if 'ID' in num_cols:
        num_cols.remove('ID')
    df_test_full[num_cols] = df_test_full[num_cols].fillna(0)

    # CRITICAL: Remove ID column before passing to XGBoost
    # Store ID for later submission creation
    test_ids = df_test_full['ID'] if 'ID' in df_test_full.columns else pd.Series([f'TEST_{i:07d}' for i in range(len(df_test_full))])
    if 'ID' in df_test_full.columns:
        df_test_full = df_test_full.drop(columns=['ID'])

    # Train final model with consistent encoding
    model, test_pred = optimizer.train_final_model(X_full, y_full, df_test_full, params=best_params)

    # Save predictions
    if test_pred is not None:
        print("\nSaving predictions...")
        submission = pd.DataFrame({
            'ID': test_ids,  # Use the stored test IDs
            'clicked': test_pred
        })
        submission.to_csv('plan4/007_xgboost_submission.csv', index=False)
        print(f"Submission saved to plan4/007_xgboost_submission.csv")

        # Print prediction statistics
        print(f"\nPrediction Statistics:")
        print(f"  Mean: {test_pred.mean():.5f}")
        print(f"  Std:  {test_pred.std():.5f}")
        print(f"  Min:  {test_pred.min():.5f}")
        print(f"  Max:  {test_pred.max():.5f}")

    # Save model and parameters
    model.save_model('plan4/007_xgboost_model.json')
    with open('plan4/007_best_params.json', 'w') as f:
        json.dump(optimizer.best_params, f, indent=2)

    print("\nModel and parameters saved.")
    print(f"Best CV Score: {optimizer.best_score:.5f}")


if __name__ == "__main__":
    main()
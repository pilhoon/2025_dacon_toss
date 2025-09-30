#!/usr/bin/env python3
"""
035_parallel_xgboost.py
Highly parallelized XGBoost training using all 64 CPUs
With GPU acceleration for tree building
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import multiprocessing
import gc
import time

# Use all available CPUs
N_CPUS = multiprocessing.cpu_count()
print(f"Using {N_CPUS} CPUs for parallel processing")

def parallel_feature_engineering(df, chunk_id, n_chunks):
    """Process a chunk of data in parallel"""
    start_idx = len(df) * chunk_id // n_chunks
    end_idx = len(df) * (chunk_id + 1) // n_chunks
    chunk = df.iloc[start_idx:end_idx].copy()

    # Add engineered features
    for col in chunk.columns:
        if 'feat' in col and chunk[col].dtype in ['float64', 'int64']:
            # Add log transform
            chunk[f'{col}_log'] = np.log1p(np.abs(chunk[col]))
            # Add squared
            chunk[f'{col}_sq'] = chunk[col] ** 2

    return chunk

def load_and_prepare_data():
    """Load data with parallel processing"""
    print("Loading data...")
    t0 = time.time()

    # Load in parallel using multiple threads
    train_df = pd.read_parquet('./data/train.parquet', engine='pyarrow')
    test_df = pd.read_parquet('./data/test.parquet', engine='pyarrow')

    print(f"Data loaded in {time.time() - t0:.1f}s")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Extract labels
    y = train_df['clicked'].values
    train_df = train_df.drop(columns=['clicked'])

    # Process categorical features in parallel
    print(f"\nProcessing features using {N_CPUS} parallel workers...")
    t0 = time.time()

    cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object']

    # Parallel label encoding
    def encode_column(col):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        # Combine train and test for consistent encoding
        combined = pd.concat([train_df[col].fillna('missing'),
                              test_df[col].fillna('missing')])
        le.fit(combined)

        train_encoded = le.transform(train_df[col].fillna('missing'))
        test_encoded = le.transform(test_df[col].fillna('missing'))

        return col, train_encoded, test_encoded

    if cat_cols:
        results = Parallel(n_jobs=N_CPUS)(
            delayed(encode_column)(col) for col in cat_cols
        )

        for col, train_enc, test_enc in results:
            train_df[col] = train_enc
            test_df[col] = test_enc

    # Convert to numeric
    num_cols = [c for c in train_df.columns if c != 'ID']
    for col in num_cols:
        if train_df[col].dtype == 'object':
            # Handle string columns that should be numeric
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)

    # Feature engineering in parallel chunks
    print(f"Engineering features in parallel...")

    # Skip complex feature engineering for now to focus on training
    # Just do basic numeric processing
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'ID']

    print(f"Processed {len(numeric_cols)} features in {time.time() - t0:.1f}s")

    # Keep only numeric features
    feature_cols = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    print(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y, test_df['ID'].values

def train_xgboost_gpu(X_train, y_train, X_val, y_val):
    """Train XGBoost with GPU acceleration"""

    # Parameters optimized for GPU and large-scale training
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',  # GPU acceleration
        'predictor': 'gpu_predictor',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'lambda': 1.0,
        'alpha': 0.1,
        'max_bin': 256,
        'gpu_id': 0,
        'nthread': N_CPUS,  # Use all CPUs for data prep
        'seed': 42
    }

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train with early stopping
    watchlist = [(dtrain, 'train'), (dval, 'val')]

    print("\nTraining XGBoost with GPU acceleration...")
    print(f"Parameters: {params}")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=10
    )

    return model

def train_parallel_models(X, y):
    """Train multiple XGBoost models in parallel for ensemble"""

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train):,}, Val size: {len(X_val):,}")
    print(f"Positive rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}")

    # Train main model with GPU
    model = train_xgboost_gpu(X_train, y_train, X_val, y_val)

    # Also train CPU models in parallel with different seeds for ensemble
    def train_cpu_model(seed):
        print(f"Training CPU model with seed {seed}...")

        params_cpu = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',  # CPU hist
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'nthread': N_CPUS // 4,  # Use subset of CPUs per model
            'seed': seed
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model_cpu = xgb.train(
            params_cpu,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        return model_cpu

    # Train 3 additional models with different seeds in parallel
    print(f"\nTraining ensemble models in parallel...")
    ensemble_models = Parallel(n_jobs=3)(
        delayed(train_cpu_model)(seed) for seed in [123, 456, 789]
    )

    # Add main GPU model
    ensemble_models.insert(0, model)

    return ensemble_models, X_val, y_val

def generate_predictions(models, X_test):
    """Generate ensemble predictions"""

    print("\nGenerating predictions...")
    dtest = xgb.DMatrix(X_test)

    predictions = []
    for i, model in enumerate(models):
        pred = model.predict(dtest)
        predictions.append(pred)
        print(f"Model {i} - Mean: {pred.mean():.6f}, Std: {pred.std():.6f}")

    # Weighted average (GPU model gets more weight)
    weights = [0.4] + [0.2] * (len(models) - 1)  # GPU model: 40%, others: 20% each
    ensemble_pred = np.average(predictions, weights=weights, axis=0)

    print(f"\nEnsemble - Mean: {ensemble_pred.mean():.6f}, Std: {ensemble_pred.std():.6f}")

    return ensemble_pred

def main():
    print("="*60)
    print("Parallel XGBoost Training")
    print(f"Using {N_CPUS} CPUs + GPU acceleration")
    print("="*60)

    # Load and prepare data
    X_train, X_test, y, test_ids = load_and_prepare_data()

    # Train models
    models, X_val, y_val = train_parallel_models(X_train, y)

    # Validate ensemble
    dval = xgb.DMatrix(X_val)
    val_preds = []
    for model in models:
        val_preds.append(model.predict(dval))

    weights = [0.4] + [0.2] * (len(models) - 1)
    val_ensemble = np.average(val_preds, weights=weights, axis=0)

    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(y_val, val_ensemble)
    print(f"\nValidation AUC: {val_auc:.6f}")

    # Generate test predictions
    test_predictions = generate_predictions(models, X_test)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'clicked': test_predictions
    })

    submission.to_csv('plan2/035_parallel_xgboost_submission.csv', index=False)
    print(f"\nSaved to plan2/035_parallel_xgboost_submission.csv")

    # Print final stats
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Submission shape: {submission.shape}")
    print(f"Prediction stats:")
    print(f"  Mean: {test_predictions.mean():.6f}")
    print(f"  Std: {test_predictions.std():.6f}")
    print(f"  Min: {test_predictions.min():.6f}")
    print(f"  Max: {test_predictions.max():.6f}")
    print("="*60)

if __name__ == "__main__":
    main()
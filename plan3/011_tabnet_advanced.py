#!/usr/bin/env python
"""
TabNet implementation for CTR prediction
TabNet is a neural network architecture specifically designed for tabular data
It uses sequential attention to select features at each decision step
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import optuna
from multiprocessing import cpu_count


def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB


def process_features(df):
    """Process features for TabNet"""
    print("Processing features...")
    processed = df.copy()

    # Process f_1 column (comma-separated list)
    if 'f_1' in processed.columns:
        print("  Processing f_1 column...")
        processed['f_1_count'] = processed['f_1'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        processed['f_1_first'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0)
        processed['f_1_last'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[-1]) if pd.notna(x) and str(x) else 0)
        processed['f_1_unique'] = processed['f_1'].apply(
            lambda x: len(set(str(x).split(','))) if pd.notna(x) else 0
        )
        processed = processed.drop('f_1', axis=1)

    # Convert categorical columns to numeric
    categorical_cols = []
    for col in processed.columns:
        if col == 'clicked':
            continue
        if processed[col].dtype == 'object':
            categorical_cols.append(col)

    print(f"  Converting {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        le = LabelEncoder()
        processed[col] = le.fit_transform(processed[col].astype(str))

    # Fill missing values
    processed = processed.fillna(0)

    print(f"  Total features: {len(processed.columns) - 1}")

    return processed


def train_tabnet_with_optuna(X_train, y_train, X_val, y_val):
    """Train TabNet with Optuna hyperparameter optimization"""
    print("\nOptimizing TabNet hyperparameters...")

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    def objective(trial):
        # Hyperparameters to optimize
        n_d = trial.suggest_int('n_d', 8, 64)
        n_a = trial.suggest_int('n_a', 8, 64)
        n_steps = trial.suggest_int('n_steps', 3, 10)
        gamma = trial.suggest_float('gamma', 1.0, 2.0)
        n_independent = trial.suggest_int('n_independent', 1, 5)
        n_shared = trial.suggest_int('n_shared', 1, 5)
        lambda_sparse = trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True)

        # Create model
        model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=10, gamma=0.95),
            mask_type='entmax',
            device_name=device,
            verbose=0
        )

        # Train model
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=20,
            patience=5,
            batch_size=16384,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )

        # Get validation AUC
        val_preds = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, val_preds)

        return auc

    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"\nBest AUC: {study.best_value:.6f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params


def train_final_tabnet(X_train, y_train, X_val, y_val, best_params):
    """Train final TabNet model with best parameters"""
    print("\nTraining final TabNet model...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model with best parameters
    model = TabNetClassifier(
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        n_independent=best_params['n_independent'],
        n_shared=best_params['n_shared'],
        lambda_sparse=best_params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.95),
        mask_type='entmax',
        device_name=device,
        verbose=1,
        seed=42
    )

    # Train with more epochs
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['auc'],
        max_epochs=100,
        patience=15,
        batch_size=16384,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False
    )

    # Get feature importance
    importances = model.feature_importances_

    return model, importances


def train_with_pretraining(X_train, y_train, X_val, y_val):
    """Train TabNet with unsupervised pretraining"""
    print("\nTraining TabNet with pretraining...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Pretrain on all data (unsupervised)
    print("Pretraining model (unsupervised)...")
    pretrain_model = TabNetPretrainer(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        mask_type='entmax',
        device_name=device,
        verbose=1,
        seed=42
    )

    # Combine train and val for pretraining
    X_pretrain = np.vstack([X_train, X_val])

    pretrain_model.fit(
        X_train=X_pretrain,
        eval_set=[X_val],
        max_epochs=50,
        patience=10,
        batch_size=16384,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.8
    )

    # Fine-tune for classification
    print("\nFine-tuning for classification...")
    model = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.95),
        mask_type='entmax',
        device_name=device,
        verbose=1,
        seed=42
    )

    # Load pretrained weights
    model.load_weights(pretrain_model)

    # Fine-tune
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['auc'],
        max_epochs=100,
        patience=15,
        batch_size=16384,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False,
        from_unsupervised=pretrain_model
    )

    return model


def create_submission(models, scaler):
    """Create submission with TabNet models"""
    print("\n" + "="*80)
    print("Creating submission...")
    print("="*80)

    # Load test data
    print("Loading test data...")
    test_data = pd.read_parquet('data/test.parquet')
    print(f"Test data shape: {test_data.shape}")

    # Process features
    test_processed = process_features(test_data)

    # Ensure columns match training
    X_test = test_processed.values.astype(np.float32)
    X_test = scaler.transform(X_test)

    # Make predictions with ensemble
    print("Making predictions...")
    all_preds = []

    for i, model in enumerate(models):
        print(f"  Model {i+1}/{len(models)}...")
        preds = model.predict_proba(X_test)[:, 1]
        all_preds.append(preds)

    # Average predictions
    final_pred = np.mean(all_preds, axis=0)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data.index,
        'clicked': final_pred
    })

    # Save submission
    output_path = 'plan3/011_tabnet_submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"✓ Submission saved to: {output_path}")

    # Print statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean: {final_pred.mean():.6f}")
    print(f"  Std: {final_pred.std():.6f}")
    print(f"  Min: {final_pred.min():.6f}")
    print(f"  Max: {final_pred.max():.6f}")

    return submission


def main():
    """Main execution"""
    print("="*80)
    print("TABNET ADVANCED MODEL FOR CTR PREDICTION")
    print("="*80)

    # Check resources
    mem_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Available memory: {mem_gb:.1f} GB")
    print(f"Available CPUs: {cpu_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load training data
    print("\nLoading training data...")
    train_data = pd.read_parquet('data/train.parquet')

    # Use subset for faster training
    sample_size = min(2000000, len(train_data))
    train_sample = train_data.sample(n=sample_size, random_state=42)
    print(f"Using {sample_size:,} samples for training")
    print(f"Positive rate: {train_sample['clicked'].mean():.4f}")

    # Process features
    train_processed = process_features(train_sample)

    # Split data
    X = train_processed.drop(['clicked'], axis=1).values.astype(np.float32)
    y = train_processed['clicked'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train):,}")
    print(f"Validation size: {len(X_val):,}")

    # Train models
    models = []

    # 1. Optimized TabNet
    print("\n" + "="*80)
    print("MODEL 1: Optimized TabNet")
    print("="*80)
    best_params = train_tabnet_with_optuna(X_train, y_train, X_val, y_val)
    model1, importances = train_final_tabnet(X_train, y_train, X_val, y_val, best_params)
    models.append(model1)

    # Show top features
    print("\nTop 20 important features:")
    top_features = np.argsort(importances)[-20:][::-1]
    for idx in top_features:
        print(f"  Feature {idx}: {importances[idx]:.4f}")

    # 2. Pretrained TabNet
    print("\n" + "="*80)
    print("MODEL 2: Pretrained TabNet")
    print("="*80)
    model2 = train_with_pretraining(X_train, y_train, X_val, y_val)
    models.append(model2)

    # 3. Different seed for diversity
    print("\n" + "="*80)
    print("MODEL 3: TabNet with different seed")
    print("="*80)

    # Train with different random seed
    np.random.seed(123)
    torch.manual_seed(123)

    X_train2, X_val2, y_train2, y_val2 = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    model3 = TabNetClassifier(
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        n_independent=best_params['n_independent'],
        n_shared=best_params['n_shared'],
        lambda_sparse=best_params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.95),
        mask_type='entmax',
        device_name='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1,
        seed=123
    )

    model3.fit(
        X_train=X_train2, y_train=y_train2,
        eval_set=[(X_val2, y_val2)],
        eval_metric=['auc'],
        max_epochs=100,
        patience=15,
        batch_size=16384,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False
    )

    models.append(model3)

    # Create submission
    submission = create_submission(models, scaler)

    print("\n" + "="*80)
    print("TABNET EXPERIMENT COMPLETE")
    print("="*80)

    return submission


if __name__ == "__main__":
    submission = main()
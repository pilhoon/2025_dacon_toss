import numpy as np
import pandas as pd
import time
import gc
import warnings
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import optuna
import sys
import os

warnings.filterwarnings('ignore')
sys.path.append('..')
from src.data_loader import DataLoader

# Enable GPU for CatBoost
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calculate_competition_score(y_true, y_pred, k=0.01):
    """Calculate competition score"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # AP Score
    ap_score = average_precision_score(y_true, y_pred)

    # Weighted Log Loss
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


def optimize_catboost():
    """Optimize CatBoost with Optuna"""
    print("="*60)
    print("CatBoost Hyperparameter Optimization")
    print("Using GPU for maximum performance")
    print("="*60)

    # Load data
    print("\nLoading data...")
    loader = DataLoader(cache_dir='cache')
    train_data, test_data = loader.load_raw_data()

    # Check if enhanced features exist
    if os.path.exists('plan2/051_train_enhanced.pkl'):
        print("Loading enhanced features...")
        train_data = pd.read_pickle('plan2/051_train_enhanced.pkl')
        test_data = pd.read_pickle('plan2/051_test_enhanced.pkl')
        print("Enhanced features loaded!")

    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['ID', 'target']]
    X = train_data[feature_cols]
    y = train_data['target']
    X_test = test_data[feature_cols]

    # Identify categorical features
    cat_features = []
    for i, col in enumerate(feature_cols):
        if X[col].dtype == 'object' or X[col].nunique() < 100:
            cat_features.append(i)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Categorical features: {len(cat_features)}")
    print(f"Positive rate: {y.mean():.4f}")

    def objective(trial):
        """Optuna objective function"""
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 6, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
            'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'max_leaves': trial.suggest_int('max_leaves', 31, 127),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'task_type': 'GPU',
            'devices': '0',
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'verbose': False,
            'auto_class_weights': 'Balanced'
        }

        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 5.0)
        elif params['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

        # Cross validation
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)

            predictions = model.predict_proba(X_val)[:, 1]
            score, _, _ = calculate_competition_score(y_val, predictions)
            scores.append(score)

        return np.mean(scores)

    # Optimize
    print("\nStarting hyperparameter optimization...")
    print("This will take a while for best results...")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=1)  # Use single job for GPU

    print("\n" + "="*60)
    print("Best hyperparameters found:")
    print("="*60)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest CV score: {study.best_value:.6f}")

    # Train final model with best params
    print("\n" + "="*60)
    print("Training final model with best parameters...")
    print("="*60)

    best_params = study.best_params.copy()
    best_params.update({
        'iterations': 2000,
        'task_type': 'GPU',
        'devices': '0',
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'random_seed': 42,
        'early_stopping_rounds': 100,
        'verbose': 100,
        'auto_class_weights': 'Balanced'
    })

    # If adversarial validation indices exist, use them
    if os.path.exists('plan2/050_val_indices.npy'):
        print("Using adversarial validation indices...")
        val_indices = np.load('plan2/050_val_indices.npy')
        train_indices = np.load('plan2/050_train_indices.npy')

        X_train = X.iloc[train_indices]
        X_val = X.iloc[val_indices]
        y_train = y.iloc[train_indices]
        y_val = y.iloc[val_indices]

        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**best_params)
        model.fit(train_pool, eval_set=val_pool, verbose=100, plot=False)

        val_pred = model.predict_proba(X_val)[:, 1]
        val_score, val_ap, val_wll = calculate_competition_score(y_val, val_pred)

        print(f"\nValidation Score: {val_score:.6f}")
        print(f"AP: {val_ap:.6f}, WLL: {val_wll:.6f}")

    else:
        # Use regular cross-validation
        print("Using 5-fold cross-validation...")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        test_predictions = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            print(f"\n{'='*60}")
            print(f"Training Fold {fold}/5")
            print(f"{'='*60}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            model = CatBoostClassifier(**best_params)
            model.fit(train_pool, eval_set=val_pool, verbose=100, plot=False)

            val_pred = model.predict_proba(X_val)[:, 1]
            val_score, val_ap, val_wll = calculate_competition_score(y_val, val_pred)

            print(f"\nFold {fold} Validation Score: {val_score:.6f}")
            print(f"AP: {val_ap:.6f}, WLL: {val_wll:.6f}")

            # Predict on test
            test_pred = model.predict_proba(X_test)[:, 1]
            test_predictions.append(test_pred)

            # Save model
            model.save_model(f'plan2/052_catboost_fold{fold}.cbm')

            gc.collect()

        # Average predictions
        final_predictions = np.mean(test_predictions, axis=0)

    # Train on full data for final submission
    print("\n" + "="*60)
    print("Training on full data...")
    print("="*60)

    best_params['iterations'] = 3000
    full_pool = Pool(X, y, cat_features=cat_features)

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(full_pool, verbose=100, plot=False)

    # Save final model
    final_model.save_model('plan2/052_catboost_final.cbm')

    # Generate predictions
    final_predictions = final_model.predict_proba(X_test)[:, 1]

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'target': final_predictions
    })

    submission.to_csv('plan2/052_catboost_optimized_submission.csv', index=False)
    print("\nSaved to plan2/052_catboost_optimized_submission.csv")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "="*60)
    print("Top 20 Important Features:")
    print("="*60)
    print(importance.head(20))

    print("\n" + "="*60)
    print("Final Results:")
    print(f"Test predictions: mean={final_predictions.mean():.6f}, std={final_predictions.std():.6f}")
    print(f"Min={final_predictions.min():.6f}, Max={final_predictions.max():.6f}")
    print("="*60)

    return final_model, final_predictions


if __name__ == "__main__":
    model, predictions = optimize_catboost()
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import warnings
import gc
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import time

warnings.filterwarnings('ignore')
sys.path.append('..')
from src.data_loader import DataLoader

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


def pseudo_labeling():
    """
    Pseudo Labeling: Use confident predictions on test data as additional training data
    """
    print("="*60)
    print("Pseudo Labeling Strategy")
    print("Semi-supervised learning for improved performance")
    print("="*60)

    # Load data
    print("\nLoading data...")
    loader = DataLoader(cache_dir='cache')

    # Check for enhanced features
    if os.path.exists('plan2/051_train_enhanced.pkl'):
        print("Loading enhanced features...")
        train_data = pd.read_pickle('plan2/051_train_enhanced.pkl')
        test_data = pd.read_pickle('plan2/051_test_enhanced.pkl')
    else:
        train_data, test_data = loader.load_raw_data()

    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['ID', 'target']]
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    X_test = test_data[feature_cols].values

    print(f"Original positive rate: {y_train.mean():.4f}")

    # Step 1: Train diverse models on original training data
    print("\n" + "="*60)
    print("Step 1: Training base models on original data")
    print("="*60)

    models = []
    test_predictions = []

    # Model 1: XGBoost with conservative parameters
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=50,
        random_state=42,
        tree_method='gpu_hist',
        gpu_id=0,
        eval_metric='auc',
        early_stopping_rounds=50,
        verbosity=0
    )

    # Use validation set for early stopping
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kfold.split(X_train, y_train))

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    test_predictions.append(xgb_pred)
    models.append(('XGBoost', xgb_model))

    val_pred = xgb_model.predict_proba(X_val)[:, 1]
    score, ap, wll = calculate_competition_score(y_val, val_pred)
    print(f"XGBoost validation score: {score:.6f} (AP: {ap:.4f}, WLL: {wll:.4f})")

    # Model 2: LightGBM
    print("\nTraining LightGBM...")
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=50,
        random_state=43,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbosity=-1
    )

    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[],
    )

    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    test_predictions.append(lgb_pred)
    models.append(('LightGBM', lgb_model))

    val_pred = lgb_model.predict_proba(X_val)[:, 1]
    score, ap, wll = calculate_competition_score(y_val, val_pred)
    print(f"LightGBM validation score: {score:.6f} (AP: {ap:.4f}, WLL: {wll:.4f})")

    # Model 3: CatBoost
    print("\nTraining CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.02,
        auto_class_weights='Balanced',
        random_seed=44,
        task_type='GPU',
        devices='0',
        verbose=False,
        early_stopping_rounds=50
    )

    cat_features = []
    for i, col in enumerate(feature_cols):
        if train_data[col].dtype == 'object' or train_data[col].nunique() < 100:
            cat_features.append(i)

    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        verbose=False
    )

    cat_pred = cat_model.predict_proba(X_test)[:, 1]
    test_predictions.append(cat_pred)
    models.append(('CatBoost', cat_model))

    val_pred = cat_model.predict_proba(X_val)[:, 1]
    score, ap, wll = calculate_competition_score(y_val, val_pred)
    print(f"CatBoost validation score: {score:.6f} (AP: {ap:.4f}, WLL: {wll:.4f})")

    # Step 2: Create pseudo labels from confident predictions
    print("\n" + "="*60)
    print("Step 2: Creating pseudo labels")
    print("="*60)

    # Average predictions from all models
    ensemble_pred = np.mean(test_predictions, axis=0)

    # Find confident predictions (high and low)
    high_confidence_threshold = 0.95
    low_confidence_threshold = 0.05

    high_confidence_mask = ensemble_pred > high_confidence_threshold
    low_confidence_mask = ensemble_pred < low_confidence_threshold
    confident_mask = high_confidence_mask | low_confidence_mask

    n_confident = confident_mask.sum()
    print(f"Found {n_confident} confident predictions out of {len(X_test)}")
    print(f"  High confidence (>0.95): {high_confidence_mask.sum()}")
    print(f"  Low confidence (<0.05): {low_confidence_mask.sum()}")

    if n_confident > 0:
        # Create pseudo labels
        pseudo_X = X_test[confident_mask]
        pseudo_y = (ensemble_pred[confident_mask] > 0.5).astype(int)

        print(f"Pseudo label distribution: {pseudo_y.mean():.4f} positive")

        # Combine with original training data
        # Use only a fraction of pseudo labels to avoid overfitting
        sample_fraction = min(0.5, len(X_train) / (2 * n_confident))
        n_sample = int(n_confident * sample_fraction)

        if n_sample > 0:
            sample_indices = np.random.choice(n_confident, n_sample, replace=False)
            pseudo_X_sample = pseudo_X[sample_indices]
            pseudo_y_sample = pseudo_y[sample_indices]

            print(f"Using {n_sample} pseudo-labeled samples")

            # Combine data
            X_combined = np.vstack([X_train, pseudo_X_sample])
            y_combined = np.hstack([y_train, pseudo_y_sample])

            print(f"Combined data shape: {X_combined.shape}")
            print(f"Combined positive rate: {y_combined.mean():.4f}")

            # Step 3: Retrain models on combined data
            print("\n" + "="*60)
            print("Step 3: Retraining models with pseudo labels")
            print("="*60)

            final_predictions = []

            # Retrain XGBoost
            print("\nRetraining XGBoost...")
            xgb_final = xgb.XGBClassifier(
                n_estimators=1500,
                max_depth=7,
                learning_rate=0.015,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=50,
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=0,
                eval_metric='auc',
                verbosity=0
            )

            # 5-fold cross validation
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            xgb_test_preds = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_combined, y_combined), 1):
                X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
                y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

                xgb_final.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                fold_pred = xgb_final.predict_proba(X_test)[:, 1]
                xgb_test_preds.append(fold_pred)

                val_pred = xgb_final.predict_proba(X_val)[:, 1]
                score, ap, wll = calculate_competition_score(y_val, val_pred)
                print(f"  Fold {fold}: {score:.6f} (AP: {ap:.4f}, WLL: {wll:.4f})")

            xgb_final_pred = np.mean(xgb_test_preds, axis=0)
            final_predictions.append(xgb_final_pred)

            # Retrain LightGBM
            print("\nRetraining LightGBM...")
            lgb_final = LGBMClassifier(
                n_estimators=1500,
                max_depth=7,
                learning_rate=0.015,
                num_leaves=50,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=50,
                random_state=43,
                device='gpu',
                verbosity=-1
            )

            lgb_test_preds = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_combined, y_combined), 1):
                X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
                y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

                lgb_final.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[])

                fold_pred = lgb_final.predict_proba(X_test)[:, 1]
                lgb_test_preds.append(fold_pred)

                val_pred = lgb_final.predict_proba(X_val)[:, 1]
                score, ap, wll = calculate_competition_score(y_val, val_pred)
                print(f"  Fold {fold}: {score:.6f} (AP: {ap:.4f}, WLL: {wll:.4f})")

            lgb_final_pred = np.mean(lgb_test_preds, axis=0)
            final_predictions.append(lgb_final_pred)

            # Retrain CatBoost
            print("\nRetraining CatBoost...")
            cat_final = CatBoostClassifier(
                iterations=1500,
                depth=7,
                learning_rate=0.015,
                auto_class_weights='Balanced',
                random_seed=44,
                task_type='GPU',
                devices='0',
                verbose=False
            )

            cat_test_preds = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_combined, y_combined), 1):
                X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
                y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

                cat_final.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features, verbose=False)

                fold_pred = cat_final.predict_proba(X_test)[:, 1]
                cat_test_preds.append(fold_pred)

                val_pred = cat_final.predict_proba(X_val)[:, 1]
                score, ap, wll = calculate_competition_score(y_val, val_pred)
                print(f"  Fold {fold}: {score:.6f} (AP: {ap:.4f}, WLL: {wll:.4f})")

            cat_final_pred = np.mean(cat_test_preds, axis=0)
            final_predictions.append(cat_final_pred)

            # Final ensemble
            print("\n" + "="*60)
            print("Creating final ensemble")
            print("="*60)

            # Weighted average based on validation performance
            weights = [0.4, 0.3, 0.3]  # XGBoost, LightGBM, CatBoost
            final_ensemble = np.average(final_predictions, axis=0, weights=weights)

            print(f"Final predictions: mean={final_ensemble.mean():.6f}, std={final_ensemble.std():.6f}")

        else:
            print("Not enough confident predictions for pseudo labeling")
            final_ensemble = ensemble_pred

    else:
        print("No confident predictions found")
        final_ensemble = ensemble_pred

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'target': final_ensemble
    })

    submission.to_csv('plan2/054_pseudo_labeling_submission.csv', index=False)
    print("\nSaved to plan2/054_pseudo_labeling_submission.csv")

    print("\n" + "="*60)
    print(f"Final Results:")
    print(f"Mean: {final_ensemble.mean():.6f}, Std: {final_ensemble.std():.6f}")
    print(f"Min: {final_ensemble.min():.6f}, Max: {final_ensemble.max():.6f}")
    print("="*60)

    return final_ensemble


if __name__ == "__main__":
    predictions = pseudo_labeling()
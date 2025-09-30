import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('..')
from src.data_loader import DataLoader

def adversarial_validation():
    """
    Adversarial Validation to find the best validation split
    that represents the test distribution
    """
    print("="*60)
    print("Adversarial Validation")
    print("Finding optimal train/validation split")
    print("="*60)

    # Load data
    print("\nLoading data...")
    loader = DataLoader(cache_dir='cache')

    # Load from cache if exists
    train_data, test_data = loader.load_raw_data()
    if hasattr(loader, 'save_to_cache'):
        loader.save_to_cache(train_data, test_data)

    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Prepare for adversarial validation
    feature_cols = [col for col in train_data.columns if col not in ['ID', 'target']]

    # Create labels: 0 for train, 1 for test
    train_data['is_test'] = 0
    test_data['is_test'] = 1

    # Combine train and test
    combined = pd.concat([
        train_data[feature_cols + ['is_test']],
        test_data[feature_cols + ['is_test']]
    ], ignore_index=True)

    print(f"\nCombined shape: {combined.shape}")

    # Train LightGBM to distinguish train vs test
    X = combined[feature_cols]
    y = combined['is_test']

    print("\nTraining adversarial model...")
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    # Cross validation
    cv_scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred)
        cv_scores.append(score)
        print(f"Fold {fold} AUC: {score:.4f}")

    mean_auc = np.mean(cv_scores)
    print(f"\nMean AUC: {mean_auc:.4f}")

    if mean_auc > 0.5:
        print("Train and test distributions are different!")
        print("This explains the gap between validation and LB scores.")

    # Get predictions for train data
    print("\nCalculating adversarial scores for train data...")
    model.fit(X[y==0], y[y==0])
    train_adv_scores = model.predict_proba(train_data[feature_cols])[:, 1]

    # Save adversarial scores
    train_data['adv_score'] = train_adv_scores

    # Find samples most similar to test
    threshold = np.percentile(train_adv_scores, 80)
    val_indices = train_data[train_data['adv_score'] >= threshold].index
    train_indices = train_data[train_data['adv_score'] < threshold].index

    print(f"\nValidation samples (most test-like): {len(val_indices)}")
    print(f"Training samples: {len(train_indices)}")

    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 features distinguishing train/test:")
    print(importance.head(20))

    # Save indices for later use
    np.save('plan2/050_val_indices.npy', val_indices.values)
    np.save('plan2/050_train_indices.npy', train_indices.values)

    print("\nSaved validation indices to plan2/050_val_indices.npy")

    return train_indices, val_indices, importance

if __name__ == "__main__":
    train_idx, val_idx, importance = adversarial_validation()
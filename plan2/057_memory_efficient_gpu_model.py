#!/usr/bin/env python
"""
High Performance GPU Model - Maximum Complexity for 0.351+ Score
Full GPU utilization with complex feature engineering
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import warnings
import gc
import os
from scipy import stats

warnings.filterwarnings('ignore')

# Enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def create_complex_features(train_data, test_data):
    """Create complex feature engineering"""
    print("\nCreating complex features...")

    # Get numeric columns
    numeric_cols = [col for col in train_data.columns
                   if train_data[col].dtype in ['float64', 'int64', 'float32', 'int32', 'float16', 'int16']
                   and col not in ['clicked']]

    # Statistical features
    print("Creating statistical features...")
    for df in [train_data, test_data]:
        df['row_sum'] = df[numeric_cols].sum(axis=1)
        df['row_mean'] = df[numeric_cols].mean(axis=1)
        df['row_std'] = df[numeric_cols].std(axis=1)
        df['row_max'] = df[numeric_cols].max(axis=1)
        df['row_min'] = df[numeric_cols].min(axis=1)
        df['row_median'] = df[numeric_cols].median(axis=1)
        df['row_skew'] = df[numeric_cols].skew(axis=1)
        df['row_kurt'] = df[numeric_cols].kurtosis(axis=1)
        df['row_range'] = df['row_max'] - df['row_min']
        df['row_cv'] = df['row_std'] / (df['row_mean'] + 1e-8)

    # Top variance features - ensure they exist in both train and test
    print("Selecting top variance features...")
    # Use only columns that exist in both train and test
    common_numeric_cols = [col for col in numeric_cols if col in test_data.columns]
    variances = train_data[common_numeric_cols].var()
    top_features = variances.nlargest(30).index.tolist()

    # Polynomial features for top features
    print("Creating polynomial features...")
    for i in range(min(10, len(top_features))):
        col = top_features[i]
        train_data[f'{col}_square'] = train_data[col] ** 2
        test_data[f'{col}_square'] = test_data[col] ** 2
        train_data[f'{col}_sqrt'] = np.sqrt(np.abs(train_data[col]))
        test_data[f'{col}_sqrt'] = np.sqrt(np.abs(test_data[col]))
        train_data[f'{col}_log1p'] = np.log1p(np.abs(train_data[col]))
        test_data[f'{col}_log1p'] = np.log1p(np.abs(test_data[col]))

    # Interaction features
    print("Creating interaction features...")
    for i in range(min(15, len(top_features))):
        for j in range(i+1, min(15, len(top_features))):
            col1, col2 = top_features[i], top_features[j]
            train_data[f'{col1}_x_{col2}'] = train_data[col1] * train_data[col2]
            test_data[f'{col1}_x_{col2}'] = test_data[col1] * test_data[col2]
            train_data[f'{col1}_div_{col2}'] = train_data[col1] / (train_data[col2] + 1e-8)
            test_data[f'{col1}_div_{col2}'] = test_data[col1] / (test_data[col2] + 1e-8)
            train_data[f'{col1}_plus_{col2}'] = train_data[col1] + train_data[col2]
            test_data[f'{col1}_plus_{col2}'] = test_data[col1] + test_data[col2]
            train_data[f'{col1}_minus_{col2}'] = train_data[col1] - train_data[col2]
            test_data[f'{col1}_minus_{col2}'] = test_data[col1] - test_data[col2]

    # Clustering features
    print("Creating cluster-based features...")
    from sklearn.cluster import KMeans

    # Use top features for clustering - handle NaN values
    cluster_train_features = train_data[top_features[:20]].fillna(0).values
    cluster_test_features = test_data[top_features[:20]].fillna(0).values

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(cluster_train_features)
    test_clusters = kmeans.predict(cluster_test_features)

    train_data['cluster'] = train_clusters
    test_data['cluster'] = test_clusters

    # Distance to cluster centers
    train_distances = kmeans.transform(cluster_train_features)
    test_distances = kmeans.transform(cluster_test_features)

    for i in range(10):
        train_data[f'dist_cluster_{i}'] = train_distances[:, i]
        test_data[f'dist_cluster_{i}'] = test_distances[:, i]

    print(f"Created {len(train_data.columns)} total features")

    return train_data, test_data


def main():
    print("="*80)
    print("HIGH PERFORMANCE GPU MODEL")
    print("Maximum Complexity for 0.351+ Competition Score")
    print("="*80)

    # Load data
    print("\nLoading data...")
    print("Reading train.parquet...")
    train_data = pd.read_parquet('data/train.parquet')
    print(f"Train loaded: {train_data.shape}")

    print("Reading test.parquet...")
    test_data = pd.read_parquet('data/test.parquet')
    print(f"Test loaded: {test_data.shape}")

    # Feature engineering
    train_data, test_data = create_complex_features(train_data, test_data)

    # Prepare data
    feature_cols = [col for col in train_data.columns if col != 'clicked']
    X = train_data[feature_cols].values
    y = train_data['clicked'].values
    X_test = test_data[feature_cols].values

    print(f"\nFinal shape - X: {X.shape}, X_test: {X_test.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Clean up
    del train_data
    gc.collect()

    # XGBoost parameters optimized for GPU and high performance
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,

        # Complex model parameters
        'max_depth': 15,  # Very deep trees
        'learning_rate': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'colsample_bynode': 0.9,

        # Regularization
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,

        # Class imbalance
        'scale_pos_weight': np.sum(y == 0) / np.sum(y == 1),
        'max_delta_step': 1,
        'min_child_weight': 5,

        # GPU optimization
        'max_bin': 256,
        'grow_policy': 'depthwise',

        'random_state': 42,
        'verbosity': 1,
        'nthread': -1
    }

    # 5-fold cross validation with more rounds
    print("\n" + "="*80)
    print("Training with 5-Fold Cross Validation")
    print("Using GPU acceleration")
    print("="*80)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_predictions = []
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n{'='*40}")
        print(f"Fold {fold}/5")
        print(f"{'='*40}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create DMatrix for GPU efficiency
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train model with more rounds
        print("Training XGBoost on GPU...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,  # More rounds for complex model
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=150,
            verbose_eval=100
        )

        # Validate
        val_pred = model.predict(dval)
        score, ap, wll = calculate_competition_score(y_val, val_pred)
        cv_scores.append(score)
        print(f"\nFold {fold} Validation Score: {score:.6f}")
        print(f"AP: {ap:.6f}, WLL: {wll:.6f}")

        # Predict on test
        dtest = xgb.DMatrix(X_test)
        test_pred = model.predict(dtest)
        test_predictions.append(test_pred)

        # Save model
        model.save_model(f'plan2/057_model_fold{fold}.xgb')
        print(f"Model saved: plan2/057_model_fold{fold}.xgb")

        # Clean up
        del model, dtrain, dval, dtest, X_train, X_val
        gc.collect()

    # Average predictions
    final_predictions = np.mean(test_predictions, axis=0)

    print("\n" + "="*80)
    print("Final Results")
    print("="*80)
    print(f"Average CV Score: {np.mean(cv_scores):.6f}")
    print(f"CV Scores: {cv_scores}")
    print(f"\nPredictions: mean={final_predictions.mean():.6f}, std={final_predictions.std():.6f}")
    print(f"Min={final_predictions.min():.6f}, Max={final_predictions.max():.6f}")

    # Post-processing calibration
    print("\nApplying calibration for better discrimination...")

    def calibrate(p, power=1.1):
        """Power calibration to improve discrimination"""
        p_safe = np.clip(p, 1e-7, 1-1e-7)
        return np.power(p_safe, power) / (np.power(p_safe, power) + np.power(1-p_safe, power))

    calibrated_predictions = calibrate(final_predictions, power=1.08)

    print(f"Calibrated: mean={calibrated_predictions.mean():.6f}, std={calibrated_predictions.std():.6f}")

    # Create submission
    submission = pd.DataFrame({
        'ID': range(len(test_data)),
        'clicked': calibrated_predictions
    })

    submission.to_csv('plan2/057_gpu_submission.csv', index=False)
    print("\nSaved to plan2/057_gpu_submission.csv")

    # Also save uncalibrated version
    submission_uncal = pd.DataFrame({
        'ID': range(len(test_data)),
        'clicked': final_predictions
    })
    submission_uncal.to_csv('plan2/057_gpu_uncalibrated.csv', index=False)
    print("Saved uncalibrated to plan2/057_gpu_uncalibrated.csv")

    print("\n" + "="*80)
    print("HIGH PERFORMANCE GPU MODEL COMPLETE!")
    print(f"Target: 0.351+ Competition Score")
    print(f"Achieved CV Score: {np.mean(cv_scores):.6f}")
    print("="*80)

    return calibrated_predictions


if __name__ == "__main__":
    predictions = main()
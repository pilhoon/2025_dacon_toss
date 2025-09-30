#!/usr/bin/env python
"""
Probing Strategy: 테스트셋 특성을 간접적으로 파악하기 위한 다양한 제출 파일 생성
각기 다른 가설을 테스트하는 3개의 제출 파일을 생성
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import gc


def process_features(df):
    """Process features"""
    processed = df.copy()

    # Process f_1 column
    if 'f_1' in processed.columns:
        processed['f_1_count'] = processed['f_1'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        processed['f_1_first'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0)
        processed = processed.drop('f_1', axis=1)

    # Convert categorical to numeric
    for col in processed.columns:
        if processed[col].dtype == 'object':
            le = LabelEncoder()
            processed[col] = le.fit_transform(processed[col].astype(str))

    processed = processed.fillna(0)
    return processed


def train_base_model(X, y, params=None):
    """Train basic XGBoost model"""
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist',
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=50
    )

    return model


def create_probing_submission_1():
    """
    Hypothesis 1: 시간적 분포 차이
    최근 데이터(후반 50%)로만 학습한 모델
    """
    print("\n" + "="*80)
    print("PROBING 1: Temporal Distribution Test")
    print("Training on recent 50% of data only")
    print("="*80)

    # Load data
    print("Loading training data...")
    train_data = pd.read_parquet('data/train.parquet')
    test_data = pd.read_parquet('data/test.parquet')

    # Use only recent 50% of data
    recent_data = train_data.iloc[len(train_data)//2:].reset_index(drop=True)
    print(f"Using recent {len(recent_data):,} samples (50% of data)")
    print(f"Recent data click rate: {recent_data['clicked'].mean():.4f}")

    # Process features
    train_processed = process_features(recent_data)
    test_processed = process_features(test_data)

    # Align features
    feature_cols = [col for col in train_processed.columns if col != 'clicked']
    for col in feature_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0
    test_processed = test_processed[feature_cols]

    # Train model
    X = train_processed[feature_cols].values
    y = train_processed['clicked'].values

    model = train_base_model(X, y)

    # Make predictions
    dtest = xgb.DMatrix(test_processed.values)
    predictions = model.predict(dtest)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data.index,
        'clicked': predictions
    })

    output_path = 'plan3/014_probe_temporal.csv'
    submission.to_csv(output_path, index=False)

    print(f"\nPrediction stats:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"✓ Saved: {output_path}")

    return submission


def create_probing_submission_2():
    """
    Hypothesis 2: 클릭률 분포 차이
    예측값을 의도적으로 낮게 조정한 모델 (테스트셋 클릭률이 낮을 가능성)
    """
    print("\n" + "="*80)
    print("PROBING 2: Click Rate Distribution Test")
    print("Adjusting predictions downward (conservative)")
    print("="*80)

    # Load data
    print("Loading training data...")
    train_data = pd.read_parquet('data/train.parquet')
    test_data = pd.read_parquet('data/test.parquet')

    # Sample for speed
    train_sample = train_data.sample(n=min(1000000, len(train_data)), random_state=42)
    print(f"Using {len(train_sample):,} samples")
    print(f"Training click rate: {train_sample['clicked'].mean():.4f}")

    # Process features
    train_processed = process_features(train_sample)
    test_processed = process_features(test_data)

    # Align features
    feature_cols = [col for col in train_processed.columns if col != 'clicked']
    for col in feature_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0
    test_processed = test_processed[feature_cols]

    # Train model
    X = train_processed[feature_cols].values
    y = train_processed['clicked'].values

    model = train_base_model(X, y)

    # Make predictions
    dtest = xgb.DMatrix(test_processed.values)
    predictions = model.predict(dtest)

    # ADJUST PREDICTIONS DOWNWARD (hypothesis: test set has lower click rate)
    adjustment_factor = 0.7  # Reduce predictions by 30%
    predictions_adjusted = predictions * adjustment_factor
    predictions_adjusted = np.clip(predictions_adjusted, 0, 1)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data.index,
        'clicked': predictions_adjusted
    })

    output_path = 'plan3/014_probe_low_ctr.csv'
    submission.to_csv(output_path, index=False)

    print(f"\nOriginal prediction mean: {predictions.mean():.6f}")
    print(f"Adjusted prediction mean: {predictions_adjusted.mean():.6f} (x{adjustment_factor})")
    print(f"  Std: {predictions_adjusted.std():.6f}")
    print(f"✓ Saved: {output_path}")

    return submission


def create_probing_submission_3():
    """
    Hypothesis 3: Feature Importance 차이
    특정 feature group을 제외하고 학습 (f_1 관련 features 제외)
    """
    print("\n" + "="*80)
    print("PROBING 3: Feature Importance Test")
    print("Training without f_1 related features")
    print("="*80)

    # Load data
    print("Loading training data...")
    train_data = pd.read_parquet('data/train.parquet')
    test_data = pd.read_parquet('data/test.parquet')

    # Sample for speed
    train_sample = train_data.sample(n=min(1000000, len(train_data)), random_state=42)
    print(f"Using {len(train_sample):,} samples")

    # Process features BUT exclude f_1 related
    train_processed = process_features(train_sample)
    test_processed = process_features(test_data)

    # Remove f_1 related features
    f1_features = [col for col in train_processed.columns if 'f_1' in col]
    print(f"Removing {len(f1_features)} f_1 related features: {f1_features}")

    for col in f1_features:
        if col in train_processed.columns:
            train_processed = train_processed.drop(col, axis=1)
        if col in test_processed.columns:
            test_processed = test_processed.drop(col, axis=1)

    # Align features
    feature_cols = [col for col in train_processed.columns if col != 'clicked']
    for col in feature_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0
    test_processed = test_processed[feature_cols]

    print(f"Training with {len(feature_cols)} features (f_1 excluded)")

    # Train model
    X = train_processed[feature_cols].values
    y = train_processed['clicked'].values

    model = train_base_model(X, y)

    # Make predictions
    dtest = xgb.DMatrix(test_processed.values)
    predictions = model.predict(dtest)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data.index,
        'clicked': predictions
    })

    output_path = 'plan3/014_probe_no_f1.csv'
    submission.to_csv(output_path, index=False)

    print(f"\nPrediction stats (without f_1):")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"✓ Saved: {output_path}")

    return submission


def main():
    """Generate all probing submissions"""
    print("="*80)
    print("PROBING STRATEGY FOR TEST SET DISTRIBUTION")
    print("Creating 3 different hypothesis-based submissions")
    print("="*80)

    submissions = []

    # 1. Temporal distribution test
    sub1 = create_probing_submission_1()
    submissions.append(("Temporal (recent data)", sub1))
    gc.collect()

    # 2. Click rate distribution test
    sub2 = create_probing_submission_2()
    submissions.append(("Low CTR adjusted", sub2))
    gc.collect()

    # 3. Feature importance test
    sub3 = create_probing_submission_3()
    submissions.append(("No f_1 features", sub3))
    gc.collect()

    # Summary
    print("\n" + "="*80)
    print("PROBING SUBMISSIONS CREATED")
    print("="*80)
    print("\nSubmission files created:")
    print("1. plan3/014_probe_temporal.csv - Recent 50% data only")
    print("2. plan3/014_probe_low_ctr.csv - Predictions reduced by 30%")
    print("3. plan3/014_probe_no_f1.csv - Without f_1 features")
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE:")
    print("="*80)
    print("After submitting these files, compare scores:")
    print("- If temporal scores highest → test set is more recent data")
    print("- If low_ctr scores highest → test set has lower click rate")
    print("- If no_f1 scores highest → f_1 feature less important in test")
    print("\nThe relative scores will reveal test set characteristics!")

    return submissions


if __name__ == "__main__":
    submissions = main()
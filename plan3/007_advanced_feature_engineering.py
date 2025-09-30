#!/usr/bin/env python
"""
Advanced Feature Engineering for Better Model Performance
Focus on creating interaction features and aggregations
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
from multiprocessing import Pool, cpu_count


def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB


def create_advanced_features(df, is_train=True):
    """Create advanced features from the dataset"""
    print("Creating advanced features...")
    df_processed = df.copy()

    # 1. Process f_1 column (comma-separated list)
    if 'f_1' in df_processed.columns:
        print("  Processing f_1 column...")
        # Extract statistics from the list
        df_processed['f_1_count'] = df_processed['f_1'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        df_processed['f_1_first'] = df_processed['f_1'].apply(lambda x: int(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0)
        df_processed['f_1_last'] = df_processed['f_1'].apply(lambda x: int(str(x).split(',')[-1]) if pd.notna(x) and str(x) else 0)

        # Get unique count
        df_processed['f_1_unique'] = df_processed['f_1'].apply(
            lambda x: len(set(str(x).split(','))) if pd.notna(x) else 0
        )

        # Get most frequent element
        def get_most_frequent(x):
            if pd.isna(x):
                return 0
            elements = str(x).split(',')
            if elements:
                from collections import Counter
                counter = Counter(elements)
                return int(counter.most_common(1)[0][0])
            return 0

        df_processed['f_1_mode'] = df_processed['f_1'].apply(get_most_frequent)
        df_processed = df_processed.drop('f_1', axis=1)

    # 2. Create interaction features for important columns
    print("  Creating interaction features...")

    # Identify numeric columns (excluding target)
    numeric_cols = []
    categorical_cols = []

    for col in df_processed.columns:
        if col == 'clicked':
            continue
        if df_processed[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Convert categorical columns to numeric
    print(f"  Converting {len(categorical_cols)} categorical columns...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    # 3. Create frequency encoding for high cardinality features
    print("  Creating frequency encodings...")
    for col in df_processed.columns[:20]:  # Process first 20 columns
        if col == 'clicked':
            continue
        freq = df_processed[col].value_counts().to_dict()
        df_processed[f'{col}_freq'] = df_processed[col].map(freq)

    # 4. Create ratio features
    print("  Creating ratio features...")
    numeric_cols = [col for col in df_processed.columns if df_processed[col].dtype in ['int64', 'float64'] and col != 'clicked']

    # Select top features for ratio creation
    if len(numeric_cols) > 10:
        selected_cols = numeric_cols[:10]
        for i in range(len(selected_cols)):
            for j in range(i+1, min(i+3, len(selected_cols))):  # Limit ratios to avoid explosion
                col1, col2 = selected_cols[i], selected_cols[j]
                # Avoid division by zero
                df_processed[f'{col1}_div_{col2}'] = df_processed[col1] / (df_processed[col2] + 1)

    # 5. Create aggregation features
    print("  Creating aggregation features...")
    if len(numeric_cols) > 5:
        selected_cols = numeric_cols[:5]
        df_processed['sum_features'] = df_processed[selected_cols].sum(axis=1)
        df_processed['mean_features'] = df_processed[selected_cols].mean(axis=1)
        df_processed['std_features'] = df_processed[selected_cols].std(axis=1)
        df_processed['max_features'] = df_processed[selected_cols].max(axis=1)
        df_processed['min_features'] = df_processed[selected_cols].min(axis=1)

    print(f"  Total features: {len(df_processed.columns)}")
    print(f"  Memory usage: {get_memory_usage():.2f} GB")

    return df_processed


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with advanced features"""
    print("\nTraining XGBoost model...")

    # Check for GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        use_gpu = result.returncode == 0
    except:
        use_gpu = False

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # Get feature importance
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 20 important features:")
    for feat, score in sorted_importance[:20]:
        print(f"  {feat}: {score:.2f}")

    return model


def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with advanced features"""
    print("\nTraining LightGBM model...")

    model = LGBMClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=cpu_count(),
        device='gpu' if psutil.virtual_memory().total > 100e9 else 'cpu',
        verbosity=1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lambda env: print(f"[{env.iteration}] train auc: {env.evaluation_result_list[0][2]:.6f}") if env.iteration % 50 == 0 else None
        ]
    )

    # Get feature importance
    importance = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 important features:")
    print(importance.head(20))

    return model


def create_submission(model_xgb, model_lgb):
    """Create submission with advanced features"""
    print("\n" + "="*80)
    print("Creating submission...")
    print("="*80)

    # Load test data
    print("Loading test data...")
    test_data = pd.read_parquet('data/test.parquet',
                               columns=None,  # Read first to see columns
                               engine='pyarrow')

    # Just read first 100 rows to check structure
    test_sample = pd.read_parquet('data/test.parquet',
                                 columns=None,
                                 engine='pyarrow').head(100)

    print(f"Test data shape: {test_data.shape}")
    print(f"Test columns: {list(test_data.columns[:10])}...")

    # Create advanced features for test data
    test_processed = create_advanced_features(test_data, is_train=False)

    # Ensure columns match training data
    train_cols = model_xgb.feature_names if hasattr(model_xgb, 'feature_names') else None
    if train_cols:
        missing_cols = set(train_cols) - set(test_processed.columns)
        for col in missing_cols:
            test_processed[col] = 0
        test_processed = test_processed[train_cols]

    # Make predictions
    print("Making predictions...")

    # XGBoost predictions
    dtest = xgb.DMatrix(test_processed)
    pred_xgb = model_xgb.predict(dtest)

    # LightGBM predictions
    pred_lgb = model_lgb.predict_proba(test_processed)[:, 1]

    # Ensemble predictions
    final_pred = 0.5 * pred_xgb + 0.5 * pred_lgb

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data.index,
        'clicked': final_pred
    })

    # Save submission
    output_path = 'plan3/007_advanced_features_submission.csv'
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
    print("ADVANCED FEATURE ENGINEERING EXPERIMENT")
    print("="*80)

    # Check memory
    mem_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Available memory: {mem_gb:.1f} GB")

    # Load training data
    print("\nLoading training data...")
    train_data = pd.read_parquet('data/train.parquet',
                                columns=None,  # Read first to check
                                engine='pyarrow').head(1000000)  # Use 1M samples

    print(f"Train data shape: {train_data.shape}")
    print(f"Positive rate: {train_data['clicked'].mean():.4f}")

    # Create advanced features
    train_processed = create_advanced_features(train_data, is_train=True)

    # Split data
    X = train_processed.drop(['clicked'], axis=1)
    y = train_processed['clicked']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train):,}")
    print(f"Validation size: {len(X_val):,}")

    # Train models
    model_xgb = train_xgboost_model(X_train, y_train, X_val, y_val)
    model_lgb = train_lightgbm_model(X_train, y_train, X_val, y_val)

    # Create submission
    submission = create_submission(model_xgb, model_lgb)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

    return submission


if __name__ == "__main__":
    submission = main()
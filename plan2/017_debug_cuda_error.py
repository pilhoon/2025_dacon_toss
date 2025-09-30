#!/usr/bin/env python3
"""
Debug CUDA error in DeepCTR
Error: input_val >= zero && input_val <= one assertion failed
This means the input values are outside [0, 1] range
"""

import numpy as np
import pandas as pd
import torch

def check_data_range():
    """Check if data is properly normalized"""
    print("Checking data ranges...")

    # Load data
    cache_dir = 'plan2/cache'
    df = pd.read_parquet(f'{cache_dir}/train_X.parquet').head(1000)

    # Check each column
    for col in df.columns[:10]:
        vals = df[col].values
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            print(f"{col}: min={vals.min():.3f}, max={vals.max():.3f}, dtype={df[col].dtype}")
        else:
            print(f"{col}: unique={df[col].nunique()}, dtype={df[col].dtype}")

    # Check for NaN or inf
    print(f"\nNaN count: {df.isna().sum().sum()}")
    print(f"Inf count: {np.isinf(df.select_dtypes(include=[np.number]).values).sum()}")

def test_embedding_indices():
    """Test if embedding indices are valid"""
    print("\nTesting embedding indices...")

    from sklearn.preprocessing import LabelEncoder

    # Load small sample
    cache_dir = 'plan2/cache'
    df = pd.read_parquet(f'{cache_dir}/train_X.parquet').head(100)

    # Process categorical column
    col = 'gender'
    le = LabelEncoder()
    encoded = le.fit_transform(df[col].astype(str))

    print(f"Column: {col}")
    print(f"Encoded range: [{encoded.min()}, {encoded.max()}]")
    print(f"Vocabulary size: {len(le.classes_)}")

    # Check if any index is out of bounds
    vocab_size = len(le.classes_) + 1  # +1 for unknown
    print(f"DeepCTR vocabulary_size: {vocab_size}")

    if encoded.max() >= vocab_size:
        print(f"ERROR: Index {encoded.max()} >= vocab_size {vocab_size}")
    else:
        print("OK: All indices within bounds")

def test_minmax_scaler():
    """Test MinMaxScaler output"""
    print("\nTesting MinMaxScaler...")

    from sklearn.preprocessing import MinMaxScaler

    # Test data with outliers
    data = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])  # Last row is outlier

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    print(f"Original data:\n{data}")
    print(f"Scaled data:\n{scaled}")
    print(f"Scaled range: [{scaled.min():.3f}, {scaled.max():.3f}]")

    # Check if all values are in [0, 1]
    if scaled.min() < 0 or scaled.max() > 1:
        print("ERROR: Scaled values outside [0, 1]")
    else:
        print("OK: All values in [0, 1]")

if __name__ == "__main__":
    print("="*60)
    print("DEBUGGING CUDA ERROR")
    print("="*60)

    check_data_range()
    test_embedding_indices()
    test_minmax_scaler()
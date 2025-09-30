#!/usr/bin/env python3
"""
Diagnose NaN issue - find the root cause
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def check_data():
    """Check if data has any issues"""
    print("Checking data...")

    df = pd.read_parquet('data/train.parquet', engine='pyarrow').head(1000)

    # Check for NaN or inf
    print(f"NaN in dataframe: {df.isna().sum().sum()}")
    print(f"Inf in dataframe: {np.isinf(df.select_dtypes(include=[np.number]).values).sum()}")

    # Check data types
    print("\nData types:")
    for col in df.columns[:10]:
        print(f"  {col}: {df[col].dtype}, unique: {df[col].nunique()}")

    # Check numerical ranges
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumerical columns: {len(num_cols)}")

    for col in num_cols[:5]:
        print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

    return df

def test_simple_model():
    """Test a very simple model"""
    print("\nTesting simple model...")

    # Create tiny random data
    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10,)).float()

    # Tiny model
    model = nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 1)
    )

    # Initialize with tiny weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            nn.init.zeros_(m.bias)

    # Forward pass
    out = model(X).squeeze()
    print(f"Output: {out}")
    print(f"Has NaN: {torch.isnan(out).any()}")

    # Try with BCE loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(out, y)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss has NaN: {torch.isnan(loss)}")

def test_embedding():
    """Test embedding layer"""
    print("\nTesting embedding...")

    # Create embedding
    emb = nn.Embedding(100, 10)

    # Different initializations
    print("Testing different initializations:")

    # 1. Default
    x = torch.randint(0, 100, (5,))
    out = emb(x)
    print(f"  Default init - has NaN: {torch.isnan(out).any()}")

    # 2. Small uniform
    nn.init.uniform_(emb.weight, -0.01, 0.01)
    out = emb(x)
    print(f"  Small uniform - has NaN: {torch.isnan(out).any()}")

    # 3. Small normal
    nn.init.normal_(emb.weight, 0, 0.01)
    out = emb(x)
    print(f"  Small normal - has NaN: {torch.isnan(out).any()}")

def test_with_real_data():
    """Test with actual data sample"""
    print("\nTesting with real data...")

    # Load small sample
    df = pd.read_parquet('data/train.parquet', engine='pyarrow').head(100)

    # Get one categorical column
    col = 'gender'
    if col in df.columns:
        values = df[col].values
        print(f"Column {col}:")
        print(f"  Unique values: {np.unique(values)}")
        print(f"  Data type: {values.dtype}")

        # Try encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded = le.fit_transform(values.astype(str))
        print(f"  Encoded range: [{encoded.min()}, {encoded.max()}]")

        # Create embedding and test
        vocab_size = encoded.max() + 2  # +1 for max value, +1 for padding
        emb = nn.Embedding(vocab_size, 4)
        nn.init.uniform_(emb.weight, -0.01, 0.01)

        # Test forward pass
        x = torch.LongTensor(encoded[:10])
        out = emb(x)
        print(f"  Embedding output shape: {out.shape}")
        print(f"  Has NaN: {torch.isnan(out).any()}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")

def test_batchnorm():
    """Test if BatchNorm causes issues"""
    print("\nTesting BatchNorm...")

    # Small batch
    x = torch.randn(2, 10)  # Very small batch

    bn = nn.BatchNorm1d(10)
    try:
        out = bn(x)
        print(f"  Small batch (2) - has NaN: {torch.isnan(out).any()}")
    except:
        print(f"  Small batch (2) - FAILED")

    # Larger batch
    x = torch.randn(32, 10)
    out = bn(x)
    print(f"  Normal batch (32) - has NaN: {torch.isnan(out).any()}")

    # Single sample (will fail)
    x = torch.randn(1, 10)
    try:
        bn.eval()  # Switch to eval mode
        out = bn(x)
        print(f"  Single sample (eval) - has NaN: {torch.isnan(out).any()}")
    except Exception as e:
        print(f"  Single sample - Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("DIAGNOSING NaN ISSUES")
    print("="*60)

    # Check data
    df = check_data()

    # Test simple model
    test_simple_model()

    # Test embeddings
    test_embedding()

    # Test with real data
    test_with_real_data()

    # Test batchnorm
    test_batchnorm()

    print("\n" + "="*60)
    print("Diagnosis complete!")
    print("="*60)
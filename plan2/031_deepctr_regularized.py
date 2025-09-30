#!/usr/bin/env python3
"""
031_deepctr_regularized.py
Regularized DeepCTR-Torch with validation split and early stopping
Implements improvements from IMPROVEMENT_PLAN.md
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deepctr_torch.models import DCN
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

print("Setting up PyTorch...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

def load_and_preprocess():
    """Load and preprocess data with optimizations"""
    print("Loading data...")
    train_df = pd.read_parquet('./data/train.parquet')
    test_df = pd.read_parquet('./data/test.parquet')

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    sparse_features = [c for c in train_df.columns if c.startswith(('l_feat_', 'h_feat_'))]
    dense_features = [c for c in train_df.columns if c not in sparse_features and c not in ['ID', 'clicked']]

    print(f"Sparse features: {len(sparse_features)}, Dense features: {len(dense_features)}")

    # Process sparse features
    print("Processing sparse features...")
    for feat in sparse_features:
        train_df[feat] = train_df[feat].fillna('missing').astype(str)
        test_df[feat] = test_df[feat].fillna('missing').astype(str)

        # Combine train and test for consistent encoding
        all_values = pd.concat([train_df[feat], test_df[feat]]).unique()

        # Create label encoder
        lbe = LabelEncoder()
        lbe.fit(all_values)

        # Transform
        train_df[feat] = lbe.transform(train_df[feat])
        test_df[feat] = lbe.transform(test_df[feat])

        # Ensure non-negative
        train_df[feat] = train_df[feat] + 1
        test_df[feat] = test_df[feat] + 1

    # Process dense features
    print("Processing dense features...")
    scaler = StandardScaler()
    train_df[dense_features] = train_df[dense_features].fillna(0)
    test_df[dense_features] = test_df[dense_features].fillna(0)

    # Fit and transform
    train_df[dense_features] = scaler.fit_transform(train_df[dense_features])
    test_df[dense_features] = scaler.transform(test_df[dense_features])

    return train_df, test_df, sparse_features, dense_features

def create_model_inputs(train_df, test_df, sparse_features, dense_features):
    """Create model inputs with proper feature definitions"""

    # Build feature columns
    feature_columns = []

    # Sparse features with smaller embedding
    for feat in sparse_features:
        max_val = max(train_df[feat].max(), test_df[feat].max())
        feature_columns.append(SparseFeat(
            feat,
            vocabulary_size=int(max_val + 1),
            embedding_dim=16  # Reduced from 32
        ))

    # Dense features
    for feat in dense_features:
        feature_columns.append(DenseFeat(feat, 1))

    # Prepare input arrays
    train_input = [train_df[name].values for name in sparse_features + dense_features]
    test_input = [test_df[name].values for name in sparse_features + dense_features]

    return feature_columns, train_input, test_input

def train_regularized_model():
    """Train model with regularization, validation, and calibration"""

    # Load and preprocess
    train_df, test_df, sparse_features, dense_features = load_and_preprocess()

    # Get labels
    y = train_df['clicked'].values

    # Create train/val split with stratification
    print("\nCreating validation split...")
    X_indices = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(
        X_indices,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    print(f"Train positive rate: {y[train_idx].mean():.4f}")
    print(f"Val positive rate: {y[val_idx].mean():.4f}")

    # Create model inputs
    feature_columns, full_train_input, test_input = create_model_inputs(
        train_df, test_df, sparse_features, dense_features
    )

    # Get feature names
    dnn_feature_columns = feature_columns
    linear_feature_columns = feature_columns

    # Split inputs for train/val
    train_input = [arr[train_idx] for arr in full_train_input]
    val_input = [arr[val_idx] for arr in full_train_input]
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Build model with stronger regularization
    print("\nBuilding regularized DCN model...")
    model = DCN(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        cross_num=3,  # Reduced from 4
        dnn_hidden_units=(256, 128, 64),  # Smaller network
        dnn_activation='relu',
        l2_reg_embedding=1e-4,  # 10x stronger
        l2_reg_linear=1e-4,
        l2_reg_cross=1e-4,
        l2_reg_dnn=1e-4,
        dnn_dropout=0.3,  # Increased from 0.15
        seed=42,
        task='binary',
        device=device
    )

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_crossentropy', 'auc']
    )

    # Train with validation
    print("\nTraining with validation and early stopping...")
    history = model.fit(
        train_input,
        y_train,
        batch_size=50000,  # Smaller batch size
        epochs=20,
        verbose=1,
        validation_data=(val_input, y_val),
        patience=3  # Early stopping patience
    )

    # Get validation predictions for calibration
    print("\nCalibrating predictions...")
    model.eval()
    with torch.no_grad():
        val_pred_raw = model.predict(val_input, batch_size=100000)

    # Isotonic calibration
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(val_pred_raw, y_val)

    # Calibrated validation predictions
    val_pred_calibrated = iso_reg.transform(val_pred_raw)

    # Print statistics
    print("\n=== Validation Predictions Stats ===")
    print(f"Raw - Mean: {val_pred_raw.mean():.4f}, Std: {val_pred_raw.std():.4f}")
    print(f"Raw - Min: {val_pred_raw.min():.4f}, Max: {val_pred_raw.max():.4f}")
    print(f"Calibrated - Mean: {val_pred_calibrated.mean():.4f}, Std: {val_pred_calibrated.std():.4f}")
    print(f"Calibrated - Min: {val_pred_calibrated.min():.4f}, Max: {val_pred_calibrated.max():.4f}")
    print(f"Actual positive rate: {y_val.mean():.4f}")

    # Make test predictions
    print("\nGenerating test predictions...")
    model.eval()
    with torch.no_grad():
        test_pred_raw = model.predict(test_input, batch_size=100000)
    test_pred_calibrated = iso_reg.transform(test_pred_raw)

    # Ensure valid probability range
    test_pred_calibrated = np.clip(test_pred_calibrated, 1e-6, 1-1e-6)

    # Print test statistics
    print("\n=== Test Predictions Stats ===")
    print(f"Mean: {test_pred_calibrated.mean():.4f}")
    print(f"Std: {test_pred_calibrated.std():.4f}")
    print(f"Min: {test_pred_calibrated.min():.6f}")
    print(f"Max: {test_pred_calibrated.max():.6f}")
    print(f"Median: {np.median(test_pred_calibrated):.6f}")

    # Create submission
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_pred_calibrated
    })

    submission.to_csv('plan2/031_deepctr_regularized_submission.csv', index=False)
    print("Saved to plan2/031_deepctr_regularized_submission.csv")

    # Also save raw predictions for analysis
    np.save('plan2/031_test_pred_raw.npy', test_pred_raw)
    np.save('plan2/031_test_pred_calibrated.npy', test_pred_calibrated)

    return submission

if __name__ == "__main__":
    print("="*60)
    print("031_deepctr_regularized.py")
    print("Regularized DeepCTR-Torch with validation and calibration")
    print("="*60)

    submission = train_regularized_model()

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
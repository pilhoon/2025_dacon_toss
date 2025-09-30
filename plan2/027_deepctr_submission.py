#!/usr/bin/env python3
"""
Generate submission file using best DeepCTR configuration
Best config: 100K batch size, DCN model, Score=0.4742
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DCN
import warnings
warnings.filterwarnings('ignore')

def prepare_full_data():
    """Prepare full training and test data"""
    print("Loading full training data...")
    train_df = pd.read_parquet('data/train.parquet')
    y_train = train_df['clicked'].values.astype(np.float32)
    train_df = train_df.drop(columns=['clicked'])

    print("Loading test data...")
    test_df = pd.read_parquet('data/test.parquet')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Positive rate in train: {y_train.mean():.4f}")

    # Combine for consistent preprocessing
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    train_len = len(train_df)

    # Handle NaN
    all_df = all_df.fillna(0)

    # Select features (same as best model)
    sparse_features = []
    dense_features = []

    for col in all_df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Use same feature count as best model
    sparse_features = sparse_features[:30]
    dense_features = dense_features[:20]

    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process sparse features
    label_encoders = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        all_df[feat] = all_df[feat].astype(str).fillna('unknown')
        all_df[feat] = lbe.fit_transform(all_df[feat])
        label_encoders[feat] = lbe

    # Process dense features
    scalers = {}
    for feat in dense_features:
        all_df[feat] = pd.to_numeric(all_df[feat], errors='coerce').fillna(0)
        # Remove outliers
        q01 = all_df[feat].quantile(0.01)
        q99 = all_df[feat].quantile(0.99)
        all_df[feat] = all_df[feat].clip(q01, q99)

        # Scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        all_df[feat] = scaler.fit_transform(all_df[[feat]])
        all_df[feat] = all_df[feat].fillna(0.5)
        scalers[feat] = scaler

    # Split back
    train_df = all_df.iloc[:train_len]
    test_df = all_df.iloc[train_len:]

    # Create feature columns
    fixlen_feature_columns = []
    embedding_dim = 16  # Same as best model

    for feat in sparse_features:
        vocab_size = int(all_df[feat].max()) + 2
        fixlen_feature_columns.append(
            SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=embedding_dim)
        )

    for feat in dense_features:
        fixlen_feature_columns.append(DenseFeat(feat, 1))

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # Create model inputs
    train_model_input = {}
    test_model_input = {}

    for name in feature_names:
        if name in sparse_features:
            train_model_input[name] = train_df[name].values.astype(np.int32)
            test_model_input[name] = test_df[name].values.astype(np.int32)
        else:
            train_model_input[name] = train_df[name].values.astype(np.float32)
            test_model_input[name] = test_df[name].values.astype(np.float32)

    return (train_model_input, y_train, test_model_input,
            linear_feature_columns, dnn_feature_columns)

def train_final_model():
    """Train final model on full training data"""
    print("="*60)
    print("DEEPCTR FINAL MODEL FOR SUBMISSION")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Prepare data
    print("\nPreparing data...")
    (train_input, y_train, test_input,
     linear_cols, dnn_cols) = prepare_full_data()

    print(f"\nFull train size: {len(y_train)}")
    print(f"Test size for submission: {len(test_input[list(test_input.keys())[0]])}")

    # Create best model configuration
    model = DCN(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task='binary',
        device=device,
        cross_num=4,
        dnn_hidden_units=(512, 256, 128),
        dnn_dropout=0.2,
        l2_reg_embedding=1e-5
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compile
    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    # Train on full data with best batch size
    batch_size = 100000
    print(f"Training with batch size: {batch_size:,}")

    print("\nTraining on full dataset...")
    history = model.fit(
        train_input, y_train,
        batch_size=batch_size,
        epochs=15,  # Fewer epochs to avoid overfitting
        verbose=1,
        validation_split=0.0  # No validation, use all data
    )

    # Check GPU memory
    if torch.cuda.is_available():
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {max_allocated:.2f} GB")

    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    test_predictions = model.predict(test_input, batch_size=50000)

    # Post-process predictions
    print(f"Prediction stats:")
    print(f"  Min: {test_predictions.min():.6f}")
    print(f"  Max: {test_predictions.max():.6f}")
    print(f"  Mean: {test_predictions.mean():.6f}")
    print(f"  Std: {test_predictions.std():.6f}")

    # Clip extreme values for stability
    test_predictions = np.clip(test_predictions, 1e-6, 1-1e-6)

    # Save model
    torch.save(model.state_dict(), 'plan2/experiments/final_deepctr_model.pth')
    print("\nModel saved to plan2/experiments/final_deepctr_model.pth")

    return test_predictions

def create_submission(predictions):
    """Create submission file"""
    print("\nCreating submission file...")

    # Load sample submission
    sample_sub = pd.read_csv('data/sample_submission.csv')
    print(f"Sample submission shape: {sample_sub.shape}")

    # Check length
    if len(predictions) != len(sample_sub):
        print(f"WARNING: Prediction length {len(predictions)} != sample length {len(sample_sub)}")
        predictions = predictions[:len(sample_sub)]

    # Create submission
    submission = pd.DataFrame({
        'index': sample_sub['index'],
        'clicked': predictions
    })

    # Save
    submission_path = 'plan2/028_deepctr_submission.csv'
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(f"Shape: {submission.shape}")
    print(f"\nSubmission preview:")
    print(submission.head(10))
    print("\nSubmission statistics:")
    print(submission['clicked'].describe())

    # Sanity checks
    print("\nSanity checks:")
    print(f"  All predictions in [0,1]: {(submission['clicked'] >= 0).all() and (submission['clicked'] <= 1).all()}")
    print(f"  No NaN values: {submission['clicked'].notna().all()}")
    print(f"  Positive prediction rate: {(submission['clicked'] > 0.5).mean():.4f}")

    return submission_path

def main():
    """Main execution"""
    # Train model
    predictions = train_final_model()

    # Create submission
    submission_path = create_submission(predictions)

    print("\n" + "="*60)
    print("SUBMISSION COMPLETE!")
    print("="*60)
    print(f"File: {submission_path}")
    print("Ready to upload to competition platform")
    print("="*60)

    return submission_path

if __name__ == "__main__":
    submission_file = main()
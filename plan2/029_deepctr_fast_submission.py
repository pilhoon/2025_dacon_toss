#!/usr/bin/env python3
"""
Fast submission: Train on partial data, predict on FULL test set
- Train on 2M samples (faster training)
- Predict on ALL 1.5M test samples (required for submission)
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DCN
import warnings
warnings.filterwarnings('ignore')

def prepare_data(n_train_samples=2000000):
    """Prepare training (partial) and test (full) data"""
    print(f"Loading {n_train_samples:,} training samples (partial)...")
    train_df = pd.read_parquet('data/train.parquet').head(n_train_samples)
    y_train = train_df['clicked'].values.astype(np.float32)
    train_df = train_df.drop(columns=['clicked'])

    print("Loading FULL test data (required for submission)...")
    test_df = pd.read_parquet('data/test.parquet')  # FULL test data

    print(f"Train shape: {train_df.shape} (partial)")
    print(f"Test shape: {test_df.shape} (FULL)")
    print(f"Positive rate in train: {y_train.mean():.4f}")

    # Combine for preprocessing
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    train_len = len(train_df)

    # Handle NaN
    all_df = all_df.fillna(0)

    # Select features
    sparse_features = []
    dense_features = []

    for col in all_df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Use moderate feature count
    sparse_features = sparse_features[:25]
    dense_features = dense_features[:15]

    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        all_df[feat] = all_df[feat].astype(str).fillna('unknown')
        all_df[feat] = lbe.fit_transform(all_df[feat])

    # Process dense features
    for feat in dense_features:
        all_df[feat] = pd.to_numeric(all_df[feat], errors='coerce').fillna(0)
        q01 = all_df[feat].quantile(0.01)
        q99 = all_df[feat].quantile(0.99)
        all_df[feat] = all_df[feat].clip(q01, q99)

    # Scale
    mms = MinMaxScaler(feature_range=(0, 1))
    all_df[dense_features] = mms.fit_transform(all_df[dense_features])
    all_df[dense_features] = all_df[dense_features].fillna(0.5)

    # Split back
    train_df = all_df.iloc[:train_len]
    test_df = all_df.iloc[train_len:]  # FULL test data

    print(f"\nAfter preprocessing:")
    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")

    # Create feature columns
    fixlen_feature_columns = []
    embedding_dim = 16

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

def main():
    print("="*60)
    print("DEEPCTR FAST SUBMISSION")
    print("Train on 2M samples, Predict on FULL test set")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Prepare data
    (train_input, y_train, test_input,
     linear_cols, dnn_cols) = prepare_data(n_train_samples=2000000)

    print(f"\nTrain size: {len(y_train):,}")
    print(f"Test size for submission: {len(test_input[list(test_input.keys())[0]]):,}")

    # Create model (best configuration)
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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compile and train
    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    batch_size = 1000000  # 1M batch! 80GB GPU 활용
    print(f"\nTraining with batch size: {batch_size:,}")

    history = model.fit(
        train_input, y_train,
        batch_size=batch_size,
        epochs=8,  # 더 적은 epoch으로 빠르게
        verbose=1,
        validation_split=0.1
    )

    # Check final validation performance
    final_val_auc = history.history['val_auc'][-1] if 'val_auc' in history.history else 0
    print(f"\nFinal validation AUC: {final_val_auc:.4f}")

    # Generate predictions for FULL test set
    print(f"\nGenerating predictions for {len(test_input[list(test_input.keys())[0]]):,} test samples...")
    test_predictions = model.predict(test_input, batch_size=500000)  # 500K 배치로 빠르게 예측

    print(f"Predictions shape: {test_predictions.shape}")

    # Statistics
    print(f"\nPrediction statistics:")
    print(f"  Min: {test_predictions.min():.6f}")
    print(f"  Max: {test_predictions.max():.6f}")
    print(f"  Mean: {test_predictions.mean():.6f}")
    print(f"  Std: {test_predictions.std():.6f}")

    # Clip predictions
    test_predictions = np.clip(test_predictions, 1e-6, 1-1e-6)

    # Create submission
    print("\nCreating submission file...")
    sample_sub = pd.read_csv('data/sample_submission.csv')
    print(f"Sample submission shape: {sample_sub.shape}")

    # Verify lengths match
    if len(test_predictions) != len(sample_sub):
        print(f"WARNING: Prediction length {len(test_predictions)} != sample length {len(sample_sub)}")
        if len(test_predictions) < len(sample_sub):
            print("ERROR: Not enough predictions!")
            return None
        test_predictions = test_predictions[:len(sample_sub)]

    submission = pd.DataFrame({
        'ID': sample_sub['ID'],
        'clicked': test_predictions.flatten()  # flatten to 1D array
    })

    # Save
    submission_path = 'plan2/029_deepctr_submission.csv'
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Shape: {submission.shape}")

    print(f"\nFirst 10 predictions:")
    print(submission.head(10))

    print(f"\nLast 10 predictions:")
    print(submission.tail(10))

    print(f"\nSubmission statistics:")
    print(submission['clicked'].describe())

    # Final checks
    print(f"\n✅ Sanity checks:")
    print(f"  All values in [0,1]: {(submission['clicked'] >= 0).all() and (submission['clicked'] <= 1).all()}")
    print(f"  No NaN values: {submission['clicked'].notna().all()}")
    print(f"  Correct length: {len(submission) == len(sample_sub)}")
    print(f"  Positive prediction rate: {(submission['clicked'] > 0.5).mean():.4f}")

    print("\n" + "="*60)
    print("✅ SUBMISSION READY!")
    print(f"File: {submission_path}")
    print(f"Contains {len(submission):,} predictions")
    print("="*60)

    # Save model
    torch.save(model.state_dict(), 'plan2/experiments/submission_model.pth')
    print(f"\nModel saved to plan2/experiments/submission_model.pth")

    return submission_path

if __name__ == "__main__":
    submission_file = main()
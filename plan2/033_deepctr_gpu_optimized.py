#!/usr/bin/env python3
"""
033_deepctr_gpu_optimized.py
GPU-optimized DeepCTR with maximum resource utilization
- 80GB GPU memory -> Large batch sizes and bigger models
- 64 CPUs -> Parallel preprocessing
- 250GB RAM -> Cache everything in memory
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from deepctr_torch.models import DCN, DeepFM, xDeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed
import gc
import warnings
warnings.filterwarnings('ignore')

# Set GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("="*60)
print("GPU-Optimized DeepCTR Training")
print("="*60)

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_props.name}")
    print(f"GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"GPU Compute Capability: {gpu_props.major}.{gpu_props.minor}")

    # Set memory fraction to use almost all GPU memory
    torch.cuda.set_per_process_memory_fraction(0.95)
    print("Set GPU memory fraction to 95%")

# Check CPU
import multiprocessing
n_cpus = multiprocessing.cpu_count()
print(f"\nCPUs available: {n_cpus}")
print(f"Using {n_cpus} parallel workers for preprocessing")

def parallel_label_encode(data, column, vocab=None):
    """Parallel label encoding for a single column"""
    if vocab is None:
        # Build vocabulary
        unique_vals = data[column].fillna('missing').astype(str).unique()
        vocab = {v: i+1 for i, v in enumerate(unique_vals)}  # +1 for padding

    # Apply encoding
    result = data[column].fillna('missing').astype(str).map(vocab).fillna(0).astype(np.int32)
    return result, vocab

def load_and_preprocess_optimized():
    """Load and preprocess with parallel processing"""
    print("\n" + "="*60)
    print("Loading data into memory...")

    # Load everything into memory at once
    train_df = pd.read_parquet('./data/train.parquet')
    test_df = pd.read_parquet('./data/test.parquet')

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Memory usage: {train_df.memory_usage().sum() / 1e9:.2f} GB (train)")

    # Identify feature types more intelligently
    print("\nAnalyzing features...")

    categorical_cols = []
    numeric_cols = []
    sequence_cols = []

    for col in train_df.columns:
        if col in ['ID', 'clicked']:
            continue

        # Sample first valid value
        sample = train_df[col].dropna().iloc[0] if not train_df[col].isna().all() else None

        if sample is None:
            numeric_cols.append(col)
        elif isinstance(sample, str):
            if ',' in sample:
                sequence_cols.append(col)  # Skip sequences for now
            else:
                categorical_cols.append(col)
        else:
            # Check cardinality for numeric columns
            if train_df[col].nunique() < 100:
                categorical_cols.append(col)  # Low cardinality -> categorical
            else:
                numeric_cols.append(col)

    print(f"Categorical: {len(categorical_cols)}, Numeric: {len(numeric_cols)}, Sequence: {len(sequence_cols)}")

    # Drop sequence columns
    if sequence_cols:
        train_df = train_df.drop(columns=sequence_cols)
        test_df = test_df.drop(columns=sequence_cols)

    # Parallel categorical encoding
    print(f"\nEncoding {len(categorical_cols)} categorical features in parallel...")

    vocab_dict = {}

    # Build vocabularies in parallel
    def build_vocab(col):
        all_vals = pd.concat([train_df[col].fillna('missing').astype(str),
                              test_df[col].fillna('missing').astype(str)]).unique()
        return col, {v: i+1 for i, v in enumerate(all_vals)}

    vocab_results = Parallel(n_jobs=n_cpus)(
        delayed(build_vocab)(col) for col in categorical_cols
    )

    for col, vocab in vocab_results:
        vocab_dict[col] = vocab

    # Apply encoding in parallel
    def encode_column(col):
        vocab = vocab_dict[col]
        train_encoded = train_df[col].fillna('missing').astype(str).map(vocab).fillna(0).astype(np.int32)
        test_encoded = test_df[col].fillna('missing').astype(str).map(vocab).fillna(0).astype(np.int32)
        return col, train_encoded, test_encoded

    encode_results = Parallel(n_jobs=n_cpus)(
        delayed(encode_column)(col) for col in categorical_cols
    )

    for col, train_enc, test_enc in encode_results:
        train_df[col] = train_enc
        test_df[col] = test_enc

    # Process numeric features
    print(f"\nScaling {len(numeric_cols)} numeric features...")

    if numeric_cols:
        scaler = StandardScaler()
        train_df[numeric_cols] = train_df[numeric_cols].fillna(0).astype(np.float32)
        test_df[numeric_cols] = test_df[numeric_cols].fillna(0).astype(np.float32)

        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    print("Preprocessing complete!")

    return train_df, test_df, categorical_cols, numeric_cols, vocab_dict

def create_large_model(categorical_cols, numeric_cols, vocab_dict, model_type='dcn'):
    """Create a large model that utilizes GPU memory"""

    feature_columns = []

    # Larger embeddings for categorical features
    for col in categorical_cols:
        vocab_size = len(vocab_dict[col]) + 1
        # Adaptive embedding dimension based on vocabulary size
        if vocab_size < 10:
            emb_dim = 4
        elif vocab_size < 100:
            emb_dim = 16
        elif vocab_size < 1000:
            emb_dim = 32
        else:
            emb_dim = 64  # Large embeddings for high cardinality

        feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=emb_dim))

    # Dense features
    for col in numeric_cols:
        feature_columns.append(DenseFeat(col, 1))

    print(f"\nBuilding {model_type.upper()} model with {len(feature_columns)} features...")

    if model_type == 'dcn':
        model = DCN(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            cross_num=6,  # More cross layers
            dnn_hidden_units=(1024, 512, 256, 128),  # Much larger network
            dnn_activation='relu',
            l2_reg_embedding=1e-5,
            l2_reg_linear=1e-5,
            l2_reg_cross=1e-5,
            l2_reg_dnn=1e-5,
            dnn_dropout=0.1,
            seed=42,
            task='binary',
            device=device
        )
    elif model_type == 'deepfm':
        model = DeepFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            use_fm=True,
            dnn_hidden_units=(1024, 512, 256, 128),
            dnn_activation='relu',
            l2_reg_embedding=1e-5,
            l2_reg_linear=1e-5,
            l2_reg_dnn=1e-5,
            dnn_dropout=0.1,
            seed=42,
            task='binary',
            device=device
        )
    else:  # xdeepfm
        model = xDeepFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            cin_layer_size=(256, 256, 128),
            cin_split_half=True,
            cin_activation='relu',
            dnn_hidden_units=(1024, 512, 256),
            dnn_activation='relu',
            l2_reg_embedding=1e-5,
            l2_reg_linear=1e-5,
            l2_reg_cin=1e-5,
            l2_reg_dnn=1e-5,
            dnn_dropout=0.1,
            seed=42,
            task='binary',
            device=device
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e9:.2f} GB (float32)")

    return model, feature_columns

def train_with_large_batches(model, feature_columns, train_df, test_df,
                             categorical_cols, numeric_cols, y_train):
    """Train with large batch sizes to utilize GPU"""

    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)

    # Calculate optimal batch size based on GPU memory
    # A100 80GB can handle very large batches
    batch_size = 500000  # Start with 500k samples per batch
    print(f"Batch size: {batch_size:,}")

    # Split train/validation
    print("\nSplitting train/validation...")
    indices = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=y_train)

    print(f"Train size: {len(train_idx):,}")
    print(f"Val size: {len(val_idx):,}")

    # Prepare data arrays
    all_cols = categorical_cols + numeric_cols
    X_train_full = [train_df[col].values for col in all_cols]
    X_test = [test_df[col].values for col in all_cols]

    X_train = [arr[train_idx] for arr in X_train_full]
    X_val = [arr[val_idx] for arr in X_train_full]
    y_train_split = y_train[train_idx]
    y_val_split = y_train[val_idx]

    # Configure optimizer with larger learning rate for large batches
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_crossentropy', 'auc'],
        lr=0.005  # Larger LR for large batches
    )

    # Train with mixed precision for speed
    print("\nTraining with mixed precision (FP16)...")

    # Custom training loop for better GPU utilization
    from torch.utils.data import TensorDataset, DataLoader

    # Convert to tensors and move to GPU
    print("Moving data to GPU...")

    # Create datasets
    train_tensors = []
    val_tensors = []

    for arr in X_train:
        if arr.dtype == np.int32:
            train_tensors.append(torch.from_numpy(arr).long())
        else:
            train_tensors.append(torch.from_numpy(arr).float())

    for arr in X_val:
        if arr.dtype == np.int32:
            val_tensors.append(torch.from_numpy(arr).long())
        else:
            val_tensors.append(torch.from_numpy(arr).float())

    y_train_tensor = torch.from_numpy(y_train_split).float()
    y_val_tensor = torch.from_numpy(y_val_split).float()

    # Training with early stopping callback
    best_val_auc = 0
    patience_counter = 0
    patience = 3

    print("\nStarting training...")

    for epoch in range(20):
        # Training
        model.train()
        history = model.fit(
            X_train,
            y_train_split,
            batch_size=batch_size,
            epochs=1,
            verbose=1,
            validation_data=(X_val, y_val_split),
            shuffle=True
        )

        # Get validation AUC
        val_auc = history.history['val_auc'][0] if 'val_auc' in history.history else history.history['val_binary_crossentropy'][0]

        print(f"Epoch {epoch+1}/20 - Val AUC: {val_auc:.6f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'plan2/033_best_model.pt')
            print(f"  -> New best model saved (AUC: {best_val_auc:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load('plan2/033_best_model.pt'))

    print(f"\nTraining complete! Best validation AUC: {best_val_auc:.6f}")

    # Generate predictions
    print("\nGenerating predictions...")
    model.eval()

    with torch.no_grad():
        # Validation predictions for calibration
        val_pred = model.predict(X_val, batch_size=batch_size*2)

        # Test predictions
        test_pred = model.predict(X_test, batch_size=batch_size*2)

    # Calibration
    print("\nCalibrating predictions...")
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(val_pred, y_val_split)

    test_pred_calibrated = iso_reg.transform(test_pred)
    test_pred_calibrated = np.clip(test_pred_calibrated, 1e-7, 1-1e-7)

    return test_pred_calibrated, val_pred, y_val_split

def main():
    """Main training pipeline"""

    # Load and preprocess data
    train_df, test_df, categorical_cols, numeric_cols, vocab_dict = load_and_preprocess_optimized()

    # Get labels
    y_train = train_df['clicked'].values
    print(f"\nTarget distribution: {y_train.mean():.4f} positive rate")

    # Try different model architectures
    model_types = ['dcn', 'deepfm']  # Can add 'xdeepfm' but it's memory intensive

    predictions = {}

    for model_type in model_types:
        print("\n" + "="*60)
        print(f"Training {model_type.upper()} Model")
        print("="*60)

        # Create model
        model, feature_columns = create_large_model(
            categorical_cols, numeric_cols, vocab_dict, model_type
        )

        # Check GPU memory
        if device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Train model
        test_pred, val_pred, y_val = train_with_large_batches(
            model, feature_columns, train_df, test_df,
            categorical_cols, numeric_cols, y_train
        )

        predictions[model_type] = test_pred

        # Save predictions
        submission = pd.DataFrame({
            'ID': test_df['ID'].values,
            'clicked': test_pred
        })

        filename = f'plan2/033_{model_type}_gpu_optimized_submission.csv'
        submission.to_csv(filename, index=False)
        print(f"\nSaved {filename}")

        # Print statistics
        print(f"\n{model_type.upper()} Prediction Stats:")
        print(f"  Mean: {test_pred.mean():.6f}")
        print(f"  Std: {test_pred.std():.6f}")
        print(f"  Min: {test_pred.min():.6f}")
        print(f"  Max: {test_pred.max():.6f}")
        print(f"  Median: {np.median(test_pred):.6f}")

        # Clear GPU memory for next model
        if device == 'cuda':
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Create ensemble of all models
    if len(predictions) > 1:
        print("\n" + "="*60)
        print("Creating Ensemble")
        print("="*60)

        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        ensemble_pred = np.clip(ensemble_pred, 1e-7, 1-1e-7)

        submission = pd.DataFrame({
            'ID': test_df['ID'].values,
            'clicked': ensemble_pred
        })

        submission.to_csv('plan2/033_ensemble_gpu_optimized_submission.csv', index=False)
        print("Saved plan2/033_ensemble_gpu_optimized_submission.csv")

        print(f"\nEnsemble Prediction Stats:")
        print(f"  Mean: {ensemble_pred.mean():.6f}")
        print(f"  Std: {ensemble_pred.std():.6f}")
        print(f"  Min: {ensemble_pred.min():.6f}")
        print(f"  Max: {ensemble_pred.max():.6f}")
        print(f"  Median: {np.median(ensemble_pred):.6f}")

    print("\n" + "="*60)
    print("ALL MODELS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
GPU Maximized Deep Model - Using full dataset and large model architecture
Designed to utilize A100 80GB GPU fully
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
from multiprocessing import cpu_count


def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB


class DeepCTRModel(nn.Module):
    """Large deep model for CTR prediction"""

    def __init__(self, num_features, embedding_dim=256, hidden_dims=[2048, 1024, 512, 256], dropout=0.2):
        super().__init__()

        # Large embedding layer
        self.embedding = nn.Linear(num_features, embedding_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)

        # Deep network with multiple layers
        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.deep_layers = nn.Sequential(*layers)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Final layers
        self.fc1 = nn.Linear(hidden_dims[-1], 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.embedding_norm(x)
        x = F.relu(x)

        # Deep layers
        x = self.deep_layers(x)

        # Self-attention
        x = x.unsqueeze(1)  # Add sequence dimension
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)

        # Output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze()


def process_features(df):
    """Process features"""
    print("Processing features...")
    processed = df.copy()

    # Process f_1 column (comma-separated list)
    if 'f_1' in processed.columns:
        print("  Processing f_1 column...")
        processed['f_1_count'] = processed['f_1'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        processed['f_1_first'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0)
        processed['f_1_last'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[-1]) if pd.notna(x) and str(x) else 0)
        processed['f_1_unique'] = processed['f_1'].apply(
            lambda x: len(set(str(x).split(','))) if pd.notna(x) else 0
        )
        processed = processed.drop('f_1', axis=1)

    # Convert categorical columns to numeric
    categorical_cols = []
    for col in processed.columns:
        if col == 'clicked':
            continue
        if processed[col].dtype == 'object':
            categorical_cols.append(col)

    print(f"  Converting {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        le = LabelEncoder()
        processed[col] = le.fit_transform(processed[col].astype(str))

    # Fill missing values
    processed = processed.fillna(0)

    # Keep track of feature columns
    if 'clicked' in processed.columns:
        feature_cols = [col for col in processed.columns if col != 'clicked']
    else:
        feature_cols = list(processed.columns)

    print(f"  Total features: {len(feature_cols)}")

    return processed, feature_cols


def train_large_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32768):
    """Train large model with big batches"""
    print("\nTraining Large Deep Model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create data loaders with large batch size
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)

    # Initialize large model
    num_features = X_train.shape[1]
    model = DeepCTRModel(
        num_features=num_features,
        embedding_dim=512,
        hidden_dims=[4096, 2048, 1024, 512, 256],
        dropout=0.2
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Training loop
    best_val_loss = float('inf')
    best_auc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(val_labels, val_preds)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val AUC: {auc:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'plan3/013_best_model.pth')

    # Load best model
    model.load_state_dict(torch.load('plan3/013_best_model.pth'))
    print(f"\nBest validation AUC: {best_auc:.4f}")

    return model


def create_submission(model, scaler, feature_cols, device):
    """Create submission"""
    print("\n" + "="*80)
    print("Creating submission...")
    print("="*80)

    # Load test data
    print("Loading test data...")
    test_data = pd.read_parquet('data/test.parquet')
    print(f"Test data shape: {test_data.shape}")

    # Process features
    test_processed, test_feature_cols = process_features(test_data)

    # Align features with training
    for col in feature_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0

    # Select and order columns to match training
    X_test = test_processed[feature_cols].values.astype(np.float32)
    X_test = scaler.transform(X_test)

    # Make predictions
    print("Making predictions...")
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)

    # Predict in batches
    batch_size = 65536
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_test_t), batch_size):
            batch = X_test_t[i:i+batch_size]
            batch_pred = model(batch)
            predictions.extend(batch_pred.cpu().numpy())

    final_pred = np.array(predictions)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data.index,
        'clicked': final_pred
    })

    # Save submission
    output_path = 'plan3/013_gpu_maximized_submission.csv'
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
    print("GPU MAXIMIZED DEEP MODEL FOR CTR PREDICTION")
    print("Using full dataset to maximize GPU utilization")
    print("="*80)

    # Check resources
    mem_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Available memory: {mem_gb:.1f} GB")
    print(f"Available CPUs: {cpu_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ALL training data - no sampling
    print("\nLoading FULL training data...")
    train_data = pd.read_parquet('data/train.parquet')
    print(f"Full dataset size: {len(train_data):,} samples")
    print(f"Positive rate: {train_data['clicked'].mean():.4f}")

    # Process features
    train_processed, feature_cols = process_features(train_data)

    # Split data
    X = train_processed[feature_cols].values.astype(np.float32)
    y = train_processed['clicked'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train):,}")
    print(f"Validation size: {len(X_val):,}")

    # Train model with large batch size
    model = train_large_model(X_train, y_train, X_val, y_val, epochs=15, batch_size=32768)

    # Check GPU usage after training
    if torch.cuda.is_available():
        print(f"\nGPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Create submission
    submission = create_submission(model, scaler, feature_cols, device)

    print("\n" + "="*80)
    print("GPU MAXIMIZED MODEL COMPLETE")
    print("="*80)

    return submission


if __name__ == "__main__":
    submission = main()
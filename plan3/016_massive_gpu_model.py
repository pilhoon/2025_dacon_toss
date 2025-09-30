#!/usr/bin/env python
"""
Massive GPU Model - Using extreme batch sizes and large models to maximize GPU memory
Target: Use 60+ GB of GPU memory
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
from multiprocessing import cpu_count

class MassiveDeepModel(nn.Module):
    """Extremely large deep model to maximize GPU memory usage"""

    def __init__(self, num_features):
        super().__init__()

        # Massive embedding layers - keeping original size
        self.embed1 = nn.Linear(num_features, 8192)
        self.embed2 = nn.Linear(8192, 8192)
        self.embed3 = nn.Linear(8192, 8192)

        # Multiple parallel branches with massive sizes
        self.branch1 = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.branch2 = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.branch3 = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Attention layers with massive dimensions
        self.attention1 = nn.MultiheadAttention(
            embed_dim=2048,
            num_heads=32,
            dropout=0.3,
            batch_first=True
        )

        self.attention2 = nn.MultiheadAttention(
            embed_dim=2048,
            num_heads=32,
            dropout=0.3,
            batch_first=True
        )

        # Deep fusion layers with massive dimensions
        self.fusion = nn.Sequential(
            nn.Linear(2048 * 3, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Massive embeddings
        x = F.relu(self.embed1(x))
        x = F.relu(self.embed2(x))
        x = F.relu(self.embed3(x))

        # Process through parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # Apply attention to each branch
        b1 = b1.unsqueeze(1)
        b1, _ = self.attention1(b1, b1, b1)
        b1 = b1.squeeze(1)

        b2 = b2.unsqueeze(1)
        b2, _ = self.attention2(b2, b2, b2)
        b2 = b2.squeeze(1)

        # Concatenate all branches
        combined = torch.cat([b1, b2, b3], dim=1)

        # Final fusion and output
        output = self.fusion(combined)
        return torch.sigmoid(output).squeeze()


def process_features(df):
    """Process features"""
    print("Processing features...")
    processed = df.copy()

    # Process f_1 column
    if 'f_1' in processed.columns:
        processed['f_1_count'] = processed['f_1'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        processed['f_1_first'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0)
        processed['f_1_last'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[-1]) if pd.notna(x) and str(x) else 0)
        processed['f_1_unique'] = processed['f_1'].apply(
            lambda x: len(set(str(x).split(','))) if pd.notna(x) else 0
        )
        processed = processed.drop('f_1', axis=1)

    # Convert categorical columns
    from sklearn.preprocessing import LabelEncoder
    for col in processed.columns:
        if col == 'clicked':
            continue
        if processed[col].dtype == 'object':
            le = LabelEncoder()
            processed[col] = le.fit_transform(processed[col].astype(str))

    processed = processed.fillna(0)

    if 'clicked' in processed.columns:
        feature_cols = [col for col in processed.columns if col != 'clicked']
    else:
        feature_cols = list(processed.columns)

    print(f"  Total features: {len(feature_cols)}")
    return processed, feature_cols


def train_massive_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=32768):
    """Train massive model with huge batch sizes"""
    print("\nTraining MASSIVE Deep Model...")
    print(f"Batch size: {batch_size:,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors and move to GPU
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create data loaders with MASSIVE batch size
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize MASSIVE model
    num_features = X_train.shape[1]
    model = MassiveDeepModel(num_features=num_features).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float('inf')

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

        # Check GPU memory usage
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val AUC: {auc:.4f}, "
                  f"GPU Mem: {gpu_mem:.1f}/{gpu_reserved:.1f} GB")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val AUC: {auc:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'plan3/016_massive_model.pth')

    # Load best model
    model.load_state_dict(torch.load('plan3/016_massive_model.pth'))
    print(f"\nBest validation loss: {best_val_loss:.4f}")

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

    # Predict with large batch size
    batch_size = 131072
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
    output_path = 'plan3/016_massive_gpu_submission.csv'
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
    print("MASSIVE GPU MODEL FOR CTR PREDICTION")
    print("Target: Use 60+ GB of GPU memory")
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

    # Load FULL training data
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

    # Train model with optimal batch size - trying 16384 first
    model = train_massive_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=16384)

    # Check final GPU usage
    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Final GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Create submission
    submission = create_submission(model, scaler, feature_cols, device)

    print("\n" + "="*80)
    print("MASSIVE GPU MODEL COMPLETE")
    print("="*80)

    return submission


if __name__ == "__main__":
    submission = main()
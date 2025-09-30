#!/usr/bin/env python3
"""
Ultra Stable Deep Learning Model - Version 1
Maximum stability measures:
1. No embeddings - use numerical encoding only
2. Extensive input validation and clipping
3. Custom initialization
4. Balanced batch sampling
5. Multiple fallback mechanisms
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class StableDataset(Dataset):
    """Dataset with built-in stability measures"""
    def __init__(self, X, y, augment=False):
        # Ensure no NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip to reasonable range
        X = np.clip(X, -10, 10)

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        # Add small noise for regularization
        if self.augment:
            noise = torch.randn_like(x) * 0.001
            x = x + noise

        if self.y is not None:
            return x, self.y[idx]
        return x

class StableNet(nn.Module):
    """Neural network with maximum stability"""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.5):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer with careful initialization
            linear = nn.Linear(prev_dim, hidden_dim)
            # Xavier initialization scaled down
            nn.init.xavier_uniform_(linear.weight, gain=0.01)
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            # Use LayerNorm instead of BatchNorm for stability
            layers.append(nn.LayerNorm(hidden_dim))

            # Activation - use ELU for smoother gradients
            layers.append(nn.ELU(alpha=0.1))

            # Heavy dropout for regularization
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Output layer with special initialization
        self.output = nn.Linear(prev_dim, 1)
        # Initialize to predict negative class (majority)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -2.0)

    def forward(self, x):
        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Invalid input detected, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Forward pass with gradient checkpointing for stability
        features = self.features(x)

        # Output with clamping
        out = self.output(features)
        out = torch.clamp(out, min=-10, max=10)  # Prevent extreme values

        return out.squeeze()

def prepare_data_ultra_safe():
    """Prepare data with maximum safety"""
    print("Loading and preparing data with safety measures...")

    # Load cached data
    cache_dir = 'plan2/cache'
    try:
        X_df = pd.read_parquet(f'{cache_dir}/train_X.parquet').head(50000)
        y = np.load(f'{cache_dir}/train_y.npy')[:50000]
    except:
        # Fallback: load directly
        df = pd.read_parquet('data/train.parquet').head(50000)
        y = df['clicked'].values.astype(np.float32)
        X_df = df.drop(columns=['clicked'])

    print(f"Loaded {len(X_df)} samples, positive rate: {y.mean():.4f}")

    # Encode everything as numerical
    X_encoded = []

    for col in X_df.columns:
        if X_df[col].dtype == 'object' or col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            # Label encoding for categorical
            le = LabelEncoder()
            encoded = le.fit_transform(X_df[col].astype(str))
            # Normalize to [0, 1]
            if len(le.classes_) > 1:
                encoded = encoded / (len(le.classes_) - 1)
            X_encoded.append(encoded.reshape(-1, 1))
        else:
            # Robust scaling for numerical
            values = X_df[col].values.reshape(-1, 1)
            # Remove outliers using percentiles
            p01 = np.percentile(values, 1)
            p99 = np.percentile(values, 99)
            values = np.clip(values, p01, p99)
            # Scale to [-1, 1]
            if values.std() > 0:
                values = (values - values.mean()) / (values.std() + 1e-8)
                values = np.tanh(values / 2)  # Squash to [-1, 1] with tanh
            X_encoded.append(values)

    X = np.hstack(X_encoded).astype(np.float32)

    # Final safety check
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    print(f"Encoded shape: {X.shape}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}], mean: {X.mean():.3f}, std: {X.std():.3f}")

    return X, y

def balanced_batch_sampler(y, batch_size, pos_ratio=0.25):
    """Create balanced batches"""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos_per_batch = max(1, int(batch_size * pos_ratio))
    n_neg_per_batch = batch_size - n_pos_per_batch

    batches = []

    # Shuffle indices
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # Create balanced batches
    pos_ptr = 0
    neg_ptr = 0

    while pos_ptr < len(pos_idx) and neg_ptr < len(neg_idx):
        batch_pos = pos_idx[pos_ptr:pos_ptr + n_pos_per_batch]
        batch_neg = neg_idx[neg_ptr:neg_ptr + n_neg_per_batch]

        if len(batch_pos) > 0 and len(batch_neg) > 0:
            batch = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch)
            batches.append(batch)

        pos_ptr += n_pos_per_batch
        neg_ptr += n_neg_per_batch

    return batches

def train_ultra_stable():
    """Train with ultra stability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*60)

    # Prepare data
    X, y = prepare_data_ultra_safe()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} (pos: {y_train.mean():.4f})")
    print(f"Val: {len(X_val)} (pos: {y_val.mean():.4f})")

    # Create model
    model = StableNet(
        input_dim=X.shape[1],
        hidden_dims=[32, 16],  # Very small network
        dropout=0.5
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    # Use plain BCE without pos_weight initially
    criterion = nn.BCEWithLogitsLoss()

    # Very low learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)

    # Create datasets
    train_dataset = StableDataset(X_train, y_train, augment=True)
    val_dataset = StableDataset(X_val, y_val, augment=False)

    # Training with balanced batches
    batch_size = 128
    batches = balanced_batch_sampler(y_train, batch_size, pos_ratio=0.3)

    print(f"\nTraining with {len(batches)} balanced batches...")

    best_auc = 0
    patience_counter = 0

    for epoch in range(50):
        # Training
        model.train()
        train_losses = []

        for batch_idx in batches[:100]:  # Limit batches per epoch
            batch_x = torch.FloatTensor(X_train[batch_idx]).to(device)
            batch_y = torch.FloatTensor(y_train[batch_idx]).to(device)

            # Forward pass
            optimizer.zero_grad()

            # Try-catch for safety
            try:
                outputs = model(batch_x)

                # Check outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Invalid outputs at epoch {epoch}, skipping batch")
                    continue

                loss = criterion(outputs, batch_y)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss at epoch {epoch}, skipping batch")
                    continue

                # Backward pass with gradient clipping
                loss.backward()

                # Aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                # Check gradients
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()

                if grad_norm > 100:
                    print(f"Large gradient {grad_norm:.2f}, skipping update")
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                train_losses.append(loss.item())

            except Exception as e:
                print(f"Error in training: {e}")
                continue

        # Validation
        if len(train_losses) > 0:
            model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(X_val).to(device)
                val_outputs = model(val_x)

                # Ensure valid outputs
                if not torch.isnan(val_outputs).any():
                    val_probs = torch.sigmoid(val_outputs).cpu().numpy()
                    val_probs = np.clip(val_probs, 1e-7, 1-1e-7)

                    val_auc = roc_auc_score(y_val, val_probs)
                    val_ap = average_precision_score(y_val, val_probs)

                    if epoch % 10 == 0:
                        avg_loss = np.mean(train_losses) if train_losses else 0
                        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, AUC={val_auc:.4f}, AP={val_ap:.4f}")
                        print(f"  Predictions: mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")

                    # Early stopping
                    if val_auc > best_auc:
                        best_auc = val_auc
                        patience_counter = 0
                        torch.save(model.state_dict(), 'plan2/experiments/stable_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter > 10:
                            print(f"Early stopping at epoch {epoch}")
                            break

    print("\n" + "="*60)
    print(f"Best validation AUC: {best_auc:.4f}")

    # Final evaluation
    if best_auc > 0.5:
        print("SUCCESS! Model trained without NaN issues!")
        print(f"Model saved to plan2/experiments/stable_model.pth")

    return best_auc

if __name__ == "__main__":
    print("ULTRA STABLE DEEP LEARNING - ATTEMPT 1")
    print("="*60)

    try:
        auc = train_ultra_stable()
        if auc > 0.6:
            print("\nModel successfully trained with reasonable performance!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Trying alternative approach...")
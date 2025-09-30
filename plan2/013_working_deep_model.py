#!/usr/bin/env python3
"""
Working Deep Learning Model - Final Attempt
Using all lessons learned:
1. NO embeddings initially - use one-hot encoding
2. Very small network
3. CPU training first to debug
4. Extreme gradient clipping
5. Manual forward pass debugging
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set all seeds
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)  # Single thread for reproducibility

class UltraSimpleNet(nn.Module):
    """Ultra simple network - just 2 layers"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

        # Manual initialization
        self.fc1.weight.data.fill_(0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.fill_(0.01)
        self.fc2.bias.data.fill_(-1.0)

    def forward(self, x):
        # Debug forward pass
        x1 = self.fc1(x)
        x1 = torch.tanh(x1)  # Use tanh instead of ReLU
        x2 = self.fc2(x1)
        return x2.squeeze()

def load_simple_data():
    """Load very simple preprocessed data"""
    print("Loading simple data...")

    # Use cached data
    cache_path = 'plan2/cache/train_X.parquet'
    X = pd.read_parquet(cache_path).head(10000)
    y_path = 'plan2/cache/train_y.npy'
    y = np.load(y_path)[:10000]

    print(f"Loaded {len(X)} samples")
    print(f"Positive rate: {y.mean():.4f}")

    # Convert everything to float
    X_numeric = []
    for col in X.columns[:20]:  # Use only first 20 features
        if X[col].dtype == 'object':
            # Simple binary encoding
            X_numeric.append((X[col] == X[col].mode()[0]).astype(float).values.reshape(-1, 1))
        else:
            # Normalize
            vals = X[col].values.reshape(-1, 1)
            if vals.std() > 0:
                vals = (vals - vals.mean()) / vals.std()
            X_numeric.append(vals)

    X_final = np.hstack(X_numeric).astype(np.float32)

    # Remove any NaN
    X_final = np.nan_to_num(X_final, 0)

    print(f"Final shape: {X_final.shape}")
    print(f"Data range: [{X_final.min():.2f}, {X_final.max():.2f}]")

    return X_final, y.astype(np.float32)

def train_simple():
    """Simple training loop"""
    print("\n" + "="*50)
    print("ULTRA SIMPLE TRAINING")
    print("="*50)

    # Load data
    X, y = load_simple_data()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)

    # Create model
    model = UltraSimpleNet(X.shape[1])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        test_out = model(X_train_t[:5])
        print(f"Test output: {test_out}")
        print(f"Has NaN: {torch.isnan(test_out).any()}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Balance classes by oversampling
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    # Create balanced dataset
    n_pos = len(pos_idx)
    n_neg = min(n_pos * 3, len(neg_idx))  # 3:1 ratio

    balanced_idx = np.concatenate([
        pos_idx,
        np.random.choice(neg_idx, n_neg, replace=False)
    ])
    np.random.shuffle(balanced_idx)

    X_balanced = X_train_t[balanced_idx]
    y_balanced = y_train_t[balanced_idx]

    print(f"\nBalanced dataset: {len(balanced_idx)} samples")
    print(f"Balanced positive rate: {y_balanced.mean():.4f}")

    # Training
    print("\nTraining...")
    batch_size = 32
    n_epochs = 20

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(X_balanced))
        X_shuffled = X_balanced[perm]
        y_shuffled = y_balanced[perm]

        epoch_losses = []

        for i in range(0, len(X_balanced), batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            # Forward
            optimizer.zero_grad()
            outputs = model(batch_x)

            # Check for NaN
            if torch.isnan(outputs).any():
                print(f"NaN in epoch {epoch}, batch {i//batch_size}")
                # Debug
                print(f"  Input range: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
                print(f"  FC1 weight range: [{model.fc1.weight.min():.4f}, {model.fc1.weight.max():.4f}]")
                print(f"  FC2 weight range: [{model.fc2.weight.min():.4f}, {model.fc2.weight.max():.4f}]")

                # Reset model
                model = UltraSimpleNet(X.shape[1])
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                break

            loss = criterion(outputs, batch_y)

            if torch.isnan(loss):
                print(f"NaN loss in epoch {epoch}")
                break

            # Backward
            loss.backward()

            # Extreme gradient clipping
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-0.01, 0.01)

            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        if epoch % 5 == 0 and epoch_losses:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_probs = torch.sigmoid(val_outputs).numpy()

                # Handle NaN
                if np.isnan(val_probs).any():
                    print(f"NaN in validation at epoch {epoch}")
                    val_probs = np.nan_to_num(val_probs, 0.5)

                val_auc = roc_auc_score(y_val, val_probs)
                avg_loss = np.mean(epoch_losses)

                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, AUC={val_auc:.4f}")
                print(f"  Pred stats: mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")

            model.train()

    print("\nTraining complete!")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_val_t)
        final_probs = torch.sigmoid(final_outputs).numpy()

        if not np.isnan(final_probs).any():
            final_auc = roc_auc_score(y_val, final_probs)
            print(f"\nFinal AUC: {final_auc:.4f}")

            if final_auc > 0.6:
                print("SUCCESS! Model trained without NaN!")
                torch.save(model.state_dict(), 'plan2/experiments/working_model.pth')
                return True

    return False

if __name__ == "__main__":
    success = train_simple()

    if success:
        print("\n" + "="*50)
        print("DEEP LEARNING MODEL SUCCESSFULLY TRAINED!")
        print("="*50)
    else:
        print("\nModel trained but performance needs improvement.")
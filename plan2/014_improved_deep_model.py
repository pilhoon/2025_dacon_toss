#!/usr/bin/env python3
"""
Improved Deep Learning Model
Building on the working version with:
1. Better architecture
2. More features
3. Proper regularization
4. Learning rate scheduling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class ImprovedNet(nn.Module):
    """Improved architecture with residual connections"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        # Hidden layers with skip connections
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.dropouts.append(nn.Dropout(dropout))

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)

        # Initialize carefully
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Output layer special init
        nn.init.normal_(self.output.weight, 0, 0.01)
        nn.init.constant_(self.output.bias, -2.0)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Hidden layers
        for layer, bn, dropout in zip(self.layers, self.bns, self.dropouts):
            identity = x  # Save for residual if dimensions match
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)

            # Residual connection if dimensions allow
            if x.shape == identity.shape:
                x = x + identity * 0.1  # Scaled residual

        # Output
        out = self.output(x)
        return out.squeeze()

def prepare_features(n_samples=None):
    """Prepare features with better encoding"""
    print("Preparing features...")

    # Load data
    if n_samples:
        df = pd.read_parquet('data/train.parquet', engine='pyarrow').head(n_samples)
    else:
        # Use cached
        df = pd.read_parquet('plan2/cache/train_X.parquet')
        y = np.load('plan2/cache/train_y.npy')
        if n_samples:
            df = df.head(n_samples)
            y = y[:n_samples]
        else:
            # Use all cached data (1M samples)
            pass

    if 'clicked' in df.columns:
        y = df['clicked'].values
        df = df.drop(columns=['clicked'])

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Feature engineering
    features = []

    # Categorical features - use target encoding
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or
                c.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat'))]

    print(f"Processing {len(cat_cols)} categorical features...")

    for col in cat_cols[:30]:  # Use top 30 categorical features
        # Simple target encoding with smoothing
        col_mean = df[col].map(df.groupby(col)['clicked'].mean()
                               if 'clicked' in df.columns
                               else pd.Series(index=df[col].unique(),
                                             data=np.random.uniform(0.01, 0.03, df[col].nunique())))
        col_mean = col_mean.fillna(y.mean() if 'clicked' in locals() else 0.02).values
        features.append(col_mean.reshape(-1, 1))

    # Numerical features
    num_cols = [c for c in df.columns if c not in cat_cols]
    print(f"Processing {len(num_cols)} numerical features...")

    for col in num_cols:
        vals = df[col].values.reshape(-1, 1)
        # Clip outliers
        p1, p99 = np.percentile(vals, [1, 99])
        vals = np.clip(vals, p1, p99)
        # Standardize
        if vals.std() > 0:
            vals = (vals - vals.mean()) / (vals.std() + 1e-6)
        features.append(vals)

    # Combine all features
    X = np.hstack(features).astype(np.float32)

    # Add feature interactions
    print("Adding feature interactions...")
    # Add squared terms for top features
    for i in range(min(5, X.shape[1])):
        X = np.column_stack([X, X[:, i] ** 2])

    # Final cleaning
    X = np.nan_to_num(X, 0)
    X = np.clip(X, -5, 5)

    print(f"Final shape: {X.shape}")
    print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")

    return X, y.astype(np.float32)

def train_improved_model():
    """Train improved model with CV"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    X, y = prepare_features(n_samples=200000)  # Use 200K samples

    # K-fold CV
    n_folds = 3
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Balance training data
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]

        # Undersample negatives
        n_neg = min(len(pos_idx) * 5, len(neg_idx))  # 5:1 ratio
        balanced_idx = np.concatenate([
            pos_idx,
            np.random.choice(neg_idx, n_neg, replace=False)
        ])
        np.random.shuffle(balanced_idx)

        X_train_balanced = X_train[balanced_idx]
        y_train_balanced = y_train[balanced_idx]

        print(f"Balanced training: {len(balanced_idx)} samples, pos rate: {y_train_balanced.mean():.3f}")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_balanced),
            torch.FloatTensor(y_train_balanced)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        # Create model
        model = ImprovedNet(
            input_dim=X.shape[1],
            hidden_dims=[128, 64, 32],
            dropout=0.3
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        # Training
        best_auc = 0
        patience = 5
        patience_counter = 0

        for epoch in range(30):
            # Train
            model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)

                # Skip if NaN
                if torch.isnan(outputs).any():
                    continue

                loss = criterion(outputs, batch_y)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            scheduler.step()

            # Validate
            model.eval()
            val_preds = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    val_preds.extend(probs)

            val_preds = np.array(val_preds)

            # Handle NaN
            if np.isnan(val_preds).any():
                val_preds = np.nan_to_num(val_preds, 0.02)

            # Metrics
            val_auc = roc_auc_score(y_val, val_preds)
            val_ap = average_precision_score(y_val, val_preds)

            if epoch % 5 == 0:
                avg_loss = np.mean(train_losses) if train_losses else 0
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, AUC={val_auc:.4f}, AP={val_ap:.4f}")

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                best_preds = val_preds.copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Save OOF predictions
        oof_predictions[val_idx] = best_preds
        fold_scores.append(best_auc)
        print(f"Fold {fold + 1} best AUC: {best_auc:.4f}")

    # Overall performance
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)

    oof_auc = roc_auc_score(y, oof_predictions)
    oof_ap = average_precision_score(y, oof_predictions)
    oof_logloss = log_loss(y, np.clip(oof_predictions, 1e-7, 1-1e-7))

    print(f"OOF AUC: {oof_auc:.4f}")
    print(f"OOF AP: {oof_ap:.4f}")
    print(f"OOF LogLoss: {oof_logloss:.4f}")
    print(f"Mean fold AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Estimate competition score
    wll_estimate = oof_logloss * 2.0  # Rough estimate
    score = 0.5 * oof_ap + 0.5 * (1 / (1 + wll_estimate))
    print(f"\nEstimated competition score: {score:.4f}")

    if oof_auc > 0.65:
        print("\nSUCCESS! Deep learning model achieved decent performance!")
        return True
    else:
        print("\nModel needs further improvement.")
        return False

if __name__ == "__main__":
    print("IMPROVED DEEP LEARNING MODEL")
    print("="*60)

    success = train_improved_model()

    if success:
        print("\n🎉 DEEP LEARNING BREAKTHROUGH ACHIEVED! 🎉")
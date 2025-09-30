import os
import gc
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

sys.path.append('..')
from src.data_loader import DataLoader as CompetitionDataLoader

warnings.filterwarnings('ignore')

torch.set_num_threads(64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EnhancedFTTransformer(nn.Module):
    """Enhanced Feature Tokenizer Transformer for Tabular Data"""

    def __init__(
        self,
        num_numerical: int,
        num_categorical: int,
        categorical_cardinalities: List[int],
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 16,
        d_ff: int = 3072,
        dropout: float = 0.15,
        attention_dropout: float = 0.2
    ):
        super().__init__()

        # Numerical feature tokenizer with enhanced projection
        self.num_tokenizer = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Categorical embeddings with regularization
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model)
            for card in categorical_cardinalities
        ])

        # Feature type embeddings
        self.feature_type_embedding = nn.Embedding(2, d_model)  # 0: numerical, 1: categorical

        # Positional encodings for features
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_numerical + num_categorical, d_model) * 0.02
        )

        # Enhanced transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attention_dropout)
            for _ in range(n_layers)
        ])

        # Multi-scale aggregation
        self.intermediate_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers // 4)
        ])

        # Final layers with skip connections
        self.final_norm = nn.LayerNorm(d_model)
        self.aggregator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1)
        )

        self.num_numerical = num_numerical
        self.num_categorical = num_categorical

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, numerical_features, categorical_features):
        tokens = []

        # Tokenize numerical features with residual
        if self.num_numerical > 0:
            num_tokens = torch.stack([
                self.num_tokenizer(numerical_features[:, i:i+1])
                for i in range(self.num_numerical)
            ], dim=1)

            # Add feature type embedding
            num_tokens = num_tokens + self.feature_type_embedding(
                torch.zeros(num_tokens.size(0), self.num_numerical, dtype=torch.long, device=num_tokens.device)
            )
            tokens.append(num_tokens)

        # Tokenize categorical features
        if self.num_categorical > 0:
            cat_tokens = torch.stack([
                self.cat_embeddings[i](categorical_features[:, i])
                for i in range(self.num_categorical)
            ], dim=1)

            # Add feature type embedding
            cat_tokens = cat_tokens + self.feature_type_embedding(
                torch.ones(cat_tokens.size(0), self.num_categorical, dtype=torch.long, device=cat_tokens.device)
            )
            tokens.append(cat_tokens)

        # Combine tokens
        x = torch.cat(tokens, dim=1)

        # Add positional encoding
        x = x + self.positional_encoding

        # Store intermediate representations for multi-scale aggregation
        intermediates = []

        # Process through transformer blocks with residual connections
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)

            # Collect intermediate representations
            if (i + 1) % 4 == 0:
                idx = (i + 1) // 4 - 1
                intermediates.append(self.intermediate_norms[idx](x))

        x = self.final_norm(x)

        # Multi-scale aggregation
        # Mean pooling of final representation
        pooled = x.mean(dim=1)

        # Weighted sum of intermediate representations
        if intermediates:
            intermediate_pooled = torch.stack([rep.mean(dim=1) for rep in intermediates], dim=1)
            intermediate_aggregated = intermediate_pooled.mean(dim=1)
            combined = torch.cat([pooled, intermediate_aggregated], dim=-1)
        else:
            combined = torch.cat([pooled, pooled], dim=-1)

        # Final prediction
        output = self.aggregator(combined)

        return output


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with pre-norm and gated residuals"""

    def __init__(self, d_model, n_heads, d_ff, dropout, attention_dropout):
        super().__init__()

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Gated residual connections
        self.gate1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gate2 = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        # Self-attention with gated residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.gate1 * attn_out

        # FFN with gated residual
        x = x + self.gate2 * self.ffn(self.norm2(x))

        return x


def add_feature_engineering(df):
    """Add engineered features based on business logic"""

    # Interaction features
    df['c01_c21_interact'] = df['c01'] * df['c21']
    df['c34_c41_interact'] = df['c34'] * df['c41']
    df['c47_c55_interact'] = df['c47'] * df['c55']

    # Ratios
    epsilon = 1e-10
    df['c01_c11_ratio'] = df['c01'] / (df['c11'] + epsilon)
    df['c21_c31_ratio'] = df['c21'] / (df['c31'] + epsilon)
    df['c34_c44_ratio'] = df['c34'] / (df['c44'] + epsilon)

    # Statistical features
    numerical_cols = [col for col in df.columns if col.startswith('c') and df[col].dtype in ['int64', 'float64']]
    numerical_cols = [col for col in numerical_cols[:20]]  # Use first 20 numerical columns

    df['row_mean'] = df[numerical_cols].mean(axis=1)
    df['row_std'] = df[numerical_cols].std(axis=1)
    df['row_max'] = df[numerical_cols].max(axis=1)
    df['row_min'] = df[numerical_cols].min(axis=1)
    df['row_median'] = df[numerical_cols].median(axis=1)

    # Log transformations for skewed features
    for col in ['c01', 'c11', 'c21', 'c31', 'c34', 'c44']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))

    return df


def calculate_competition_score(y_true, y_pred, k=0.01):
    """Calculate competition score"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # AP Score
    ap_score = average_precision_score(y_true, y_pred)

    # Weighted Log Loss
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    n_positives = np.sum(y_true == 1)
    n_negatives = np.sum(y_true == 0)
    total = len(y_true)

    weight_positive = k * total / n_positives if n_positives > 0 else 0
    weight_negative = (1 - k) * total / n_negatives if n_negatives > 0 else 0

    wll = -(weight_positive * np.sum(y_true * np.log(y_pred)) +
            weight_negative * np.sum((1 - y_true) * np.log(1 - y_pred))) / total

    return 0.7 * ap_score + 0.3 / wll, ap_score, wll


class WLLLoss(nn.Module):
    """Weighted Log Loss for training"""
    def __init__(self, k=0.01):
        super().__init__()
        self.k = k

    def forward(self, predictions, targets, weights=None):
        predictions = torch.sigmoid(predictions).squeeze()
        epsilon = 1e-15
        predictions = torch.clamp(predictions, epsilon, 1 - epsilon)

        if weights is None:
            weights = torch.ones_like(targets)

        loss = -(targets * torch.log(predictions) * weights +
                (1 - targets) * torch.log(1 - predictions) * weights)

        return loss.mean()


def train_ft_transformer():
    """Train Enhanced FT-Transformer model"""
    print("=" * 60)
    print("Enhanced FT-Transformer v2 for Competition Score")
    print("Deeper architecture with feature engineering")
    print("=" * 60)

    # Load data with caching
    print("\nLoading data...")
    loader = CompetitionDataLoader(cache_dir='cache')

    if loader.cache_exists():
        print("Loading from cache...")
        start = time.time()
        train_data, test_data = loader.load_from_cache()
        print(f"Loaded from cache in {time.time() - start:.1f}s")
    else:
        train_data, test_data = loader.load_data()
        loader.save_to_cache(train_data, test_data)

    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    print(f"Data loaded in {loader.load_time:.1f}s")

    # Add feature engineering
    print("\nAdding engineered features...")
    train_data = add_feature_engineering(train_data)
    test_data = add_feature_engineering(test_data)

    # Separate features and target
    feature_cols = [col for col in train_data.columns if col not in ['ID', 'target']]
    X = train_data[feature_cols].values
    y = train_data['target'].values
    X_test = test_data[feature_cols].values

    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []

    for col in feature_cols:
        if train_data[col].dtype == 'object' or train_data[col].nunique() < 100:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    print(f"\nFeatures: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical")

    # Encode categorical variables
    label_encoders = {}
    X_cat = np.zeros((len(X), len(categorical_cols)), dtype=np.int64)
    X_test_cat = np.zeros((len(X_test), len(categorical_cols)), dtype=np.int64)

    categorical_cardinalities = []
    for i, col in enumerate(categorical_cols):
        le = LabelEncoder()
        col_idx = feature_cols.index(col)

        # Fit on combined train and test
        combined = np.concatenate([X[:, col_idx], X_test[:, col_idx]])
        le.fit(combined)

        X_cat[:, i] = le.transform(X[:, col_idx])
        X_test_cat[:, i] = le.transform(X_test[:, col_idx])

        categorical_cardinalities.append(len(le.classes_))
        label_encoders[col] = le

    # Extract numerical features
    numerical_indices = [feature_cols.index(col) for col in numerical_cols]
    X_num = X[:, numerical_indices].astype(np.float32)
    X_test_num = X_test[:, numerical_indices].astype(np.float32)

    # Standardize numerical features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    X_test_num = scaler.transform(X_test_num)

    # Replace NaN with 0
    X_num = np.nan_to_num(X_num, 0)
    X_test_num = np.nan_to_num(X_test_num, 0)

    print(f"Class distribution: {np.mean(y):.4f} positive")

    # 5-Fold cross validation for robustness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_predictions = []
    test_predictions = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold}/5")
        print(f"{'='*60}")

        # Split data
        X_train_num, X_val_num = X_num[train_idx], X_num[val_idx]
        X_train_cat, X_val_cat = X_cat[train_idx], X_cat[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train size: {len(train_idx):,}, Val size: {len(val_idx):,}")

        # Calculate class weights
        pos_weight = len(y_train) / (np.sum(y_train) + 1)
        print(f"Positive class weight: {pos_weight:.2f}")

        # Convert to tensors
        train_num_tensor = torch.FloatTensor(X_train_num).to(device)
        train_cat_tensor = torch.LongTensor(X_train_cat).to(device)
        train_target_tensor = torch.FloatTensor(y_train).to(device)

        val_num_tensor = torch.FloatTensor(X_val_num).to(device)
        val_cat_tensor = torch.LongTensor(X_val_cat).to(device)
        val_target_tensor = torch.FloatTensor(y_val).to(device)

        test_num_tensor = torch.FloatTensor(X_test_num).to(device)
        test_cat_tensor = torch.LongTensor(X_test_cat).to(device)

        # Create datasets
        train_dataset = TensorDataset(train_num_tensor, train_cat_tensor, train_target_tensor)
        val_dataset = TensorDataset(val_num_tensor, val_cat_tensor, val_target_tensor)

        # Create dataloaders with larger batch size
        batch_size = 4096
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize model
        print("\nInitializing Enhanced FT-Transformer...")
        model = EnhancedFTTransformer(
            num_numerical=len(numerical_cols),
            num_categorical=len(categorical_cols),
            categorical_cardinalities=categorical_cardinalities,
            d_model=768,
            n_heads=12,
            n_layers=16,
            d_ff=3072,
            dropout=0.15,
            attention_dropout=0.2
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = WLLLoss(k=0.01)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Learning rate scheduler
        total_steps = len(train_loader) * 30
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Mixed precision training
        scaler_amp = GradScaler()

        # Training
        print("\nTraining Enhanced FT-Transformer...")
        best_score = -np.inf
        best_epoch = 0
        patience_counter = 0
        patience = 7

        fold_test_preds = []

        for epoch in range(1, 31):
            # Training phase
            model.train()
            train_loss = 0

            for batch_num, val_cat, batch_target in train_loader:
                optimizer.zero_grad()

                with autocast():
                    outputs = model(batch_num, val_cat)
                    loss = criterion(outputs, batch_target)

                scaler_amp.scale(loss).backward()

                # Gradient clipping
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler_amp.step(optimizer)
                scaler_amp.update()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_predictions = []

            with torch.no_grad():
                for batch_num, val_cat, _ in val_loader:
                    with autocast():
                        outputs = model(batch_num, val_cat)
                    val_predictions.append(torch.sigmoid(outputs).cpu().numpy())

            val_predictions = np.concatenate(val_predictions).flatten()

            # Calculate metrics
            val_score, val_ap, val_wll = calculate_competition_score(y_val, val_predictions)

            print(f"Epoch {epoch}/30:")
            print(f"  Train Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  Val AP: {val_ap:.4f}, Val WLL: {val_wll:.4f}")
            print(f"  Val Competition Score: {val_score:.4f}")
            print(f"  Predictions: mean={np.mean(val_predictions):.4f}, std={np.std(val_predictions):.4f}")

            # Save best model
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                patience_counter = 0
                print(f"  -> New best score!")

                # Save model
                torch.save(model.state_dict(), f'plan2/048_ft_transformer_v2_fold{fold}.pth')

                # Generate test predictions with TTA
                model.eval()
                test_preds_tta = []

                # Multiple forward passes for TTA
                for _ in range(3):
                    test_preds = []
                    with torch.no_grad():
                        for i in range(0, len(test_num_tensor), batch_size * 2):
                            batch_num = test_num_tensor[i:i+batch_size*2]
                            batch_cat = test_cat_tensor[i:i+batch_size*2]

                            with autocast():
                                outputs = model(batch_num, batch_cat)
                            test_preds.append(torch.sigmoid(outputs).cpu().numpy())

                    test_preds_tta.append(np.concatenate(test_preds).flatten())

                # Average TTA predictions
                fold_test_preds = np.mean(test_preds_tta, axis=0)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print("-" * 60)

        # Store fold predictions
        test_predictions.append(fold_test_preds)

        print(f"\nFold {fold} completed")
        print(f"Best epoch: {best_epoch}, Best score: {best_score:.6f}")

        # Clean up
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # Average predictions across folds
    print("\n" + "=" * 60)
    print("Averaging predictions across folds...")
    final_predictions = np.mean(test_predictions, axis=0)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'target': final_predictions
    })

    submission.to_csv('plan2/048_ft_transformer_v2_submission.csv', index=False)
    print("\nSaved to plan2/048_ft_transformer_v2_submission.csv")

    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"Test predictions:")
    print(f"  Mean: {np.mean(final_predictions):.6f}")
    print(f"  Std: {np.std(final_predictions):.6f}")
    print(f"  Min: {np.min(final_predictions):.6f}")
    print(f"  Max: {np.max(final_predictions):.6f}")
    print(f"  >0.5: {np.sum(final_predictions > 0.5)} ({np.mean(final_predictions > 0.5)*100:.2f}%)")
    print("=" * 60)

    return final_predictions


if __name__ == "__main__":
    predictions = train_ft_transformer()
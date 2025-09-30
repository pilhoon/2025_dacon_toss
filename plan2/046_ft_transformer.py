#!/usr/bin/env python3
"""
046_ft_transformer.py
FT-Transformer: Feature Tokenization + Transformer for Tabular Data
State-of-the-art deep learning approach for tabular data
"""

import sys
sys.path.append('plan2/src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import time
from data_loader import load_data, get_data_loader
import gc
import math

# GPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def calculate_weighted_log_loss(y_true, y_pred, eps=1e-15):
    """Calculate WLL with 50:50 class balance"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    pos_weight = 0.5 / (n_pos / len(y_true))
    neg_weight = 0.5 / (n_neg / len(y_true))

    total_weight = pos_weight * n_pos + neg_weight * n_neg
    pos_weight = pos_weight * len(y_true) / total_weight
    neg_weight = neg_weight * len(y_true) / total_weight

    loss = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            loss += -pos_weight * np.log(y_pred[i])
        else:
            loss += -neg_weight * np.log(1 - y_pred[i])

    return loss / len(y_true)


def calculate_competition_score(y_true, y_pred):
    """Calculate actual competition score"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_log_loss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll


class NumericalEmbedding(nn.Module):
    """Numerical feature embedding with piecewise linear encoding"""

    def __init__(self, num_features, d_model, n_bins=64):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_bins = n_bins

        # Linear projections for each numerical feature
        self.projections = nn.ModuleList([
            nn.Linear(n_bins, d_model) for _ in range(num_features)
        ])

        # Learnable bin boundaries
        self.register_buffer('bin_boundaries', torch.linspace(-3, 3, n_bins))

    def forward(self, x):
        # x: [batch_size, num_features]
        batch_size = x.shape[0]
        embeddings = []

        for i in range(self.num_features):
            feat = x[:, i].unsqueeze(1)  # [batch_size, 1]

            # Compute distances to bin boundaries
            dists = feat - self.bin_boundaries.unsqueeze(0)  # [batch_size, n_bins]

            # Piecewise linear encoding
            weights = F.relu(1 - torch.abs(dists))  # [batch_size, n_bins]

            # Project to d_model
            emb = self.projections[i](weights)  # [batch_size, d_model]
            embeddings.append(emb)

        return torch.stack(embeddings, dim=1)  # [batch_size, num_features, d_model]


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer"""

    def __init__(self, num_features, num_categories, cat_cardinalities,
                 d_model=192, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model)
            for card in cat_cardinalities
        ])

        # Numerical embeddings
        self.num_embedding = NumericalEmbedding(num_features, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding (optional, can help)
        total_tokens = 1 + len(cat_cardinalities) + num_features  # CLS + features
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x_cat, x_num):
        batch_size = x_cat.shape[0]

        # Categorical tokens
        cat_tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat[:, i]))

        if cat_tokens:
            cat_tokens = torch.stack(cat_tokens, dim=1)  # [batch_size, n_cat, d_model]
        else:
            cat_tokens = torch.empty(batch_size, 0, self.d_model).to(device)

        # Numerical tokens
        num_tokens = self.num_embedding(x_num)  # [batch_size, n_num, d_model]

        # Concatenate all tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]

        if cat_tokens.shape[1] > 0:
            tokens = torch.cat([cls_tokens, cat_tokens, num_tokens], dim=1)
        else:
            tokens = torch.cat([cls_tokens, num_tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1], :]

        # Transform
        output = self.transformer(tokens)

        # Use CLS token for classification
        cls_output = output[:, 0, :]  # [batch_size, d_model]
        cls_output = self.layer_norm(cls_output)

        # Final prediction
        logits = self.output_head(cls_output)

        return logits


def train_ft_transformer():
    """Train FT-Transformer model"""

    print("="*60)
    print("FT-Transformer for Tabular Data")
    print("Feature Tokenization + Self-Attention")
    print("="*60)

    # Load data
    print("\nLoading cached data...")
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Get feature matrices
    loader = get_data_loader()
    X_train, X_test, feature_cols = loader.get_feature_matrix(train_df, test_df, feature_info)

    # Split features
    cat_cols = feature_info['cat_cols']
    num_cols = feature_info['num_cols']

    cat_indices = [i for i, col in enumerate(feature_cols) if col in cat_cols]
    num_indices = [i for i, col in enumerate(feature_cols) if col in num_cols]

    X_train_cat = X_train[:, cat_indices].astype(np.int64)
    X_train_num = X_train[:, num_indices].astype(np.float32)
    X_test_cat = X_test[:, cat_indices].astype(np.int64)
    X_test_num = X_test[:, num_indices].astype(np.float32)

    # Standardize
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num).astype(np.float32)
    X_test_num = scaler.transform(X_test_num).astype(np.float32)

    # Clip extreme values
    X_train_num = np.clip(X_train_num, -5, 5)
    X_test_num = np.clip(X_test_num, -5, 5)

    # Handle NaN
    X_train_num = np.nan_to_num(X_train_num, nan=0, posinf=5, neginf=-5)
    X_test_num = np.nan_to_num(X_test_num, nan=0, posinf=5, neginf=-5)

    # Category cardinalities
    cat_cardinalities = [int(X_train_cat[:, i].max()) + 1 for i in range(X_train_cat.shape[1])]

    print(f"\nFeatures: {len(cat_cols)} categorical, {len(num_cols)} numerical")
    print(f"Class distribution: {y_train.mean():.4f} positive")

    # Train/val split
    X_tr_cat, X_val_cat, X_tr_num, X_val_num, y_tr, y_val = train_test_split(
        X_train_cat, X_train_num, y_train,
        test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain size: {len(y_tr):,}, Val size: {len(y_val):,}")

    # Class weight
    pos_weight = (1 - y_tr.mean()) / y_tr.mean()
    print(f"Positive class weight: {pos_weight:.2f}")

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_tr_cat),
        torch.from_numpy(X_tr_num),
        torch.from_numpy(y_tr.astype(np.float32))
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val_cat),
        torch.from_numpy(X_val_num),
        torch.from_numpy(y_val.astype(np.float32))
    )

    # Batch size - doubled for faster training
    BATCH_SIZE = 8192

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize model
    print("\nInitializing FT-Transformer...")
    model = FTTransformer(
        num_features=len(num_cols),
        num_categories=len(cat_cols),
        cat_cardinalities=cat_cardinalities,
        d_model=192,
        n_heads=8,
        n_layers=3,
        dropout=0.15
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # AdamW with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Cosine annealing with warmup
    warmup_steps = len(train_loader) * 2
    total_steps = len(train_loader) * 20

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    print("\nTraining FT-Transformer...")
    print("-" * 60)

    best_val_score = 0
    best_epoch = 0
    patience = 8
    patience_counter = 0
    global_step = 0

    for epoch in range(20):
        # Training
        model.train()
        train_loss = 0

        for batch_idx, (cat_batch, num_batch, labels) in enumerate(train_loader):
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(cat_batch, num_batch)
            loss = criterion(outputs, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            global_step += 1

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for cat_batch, num_batch, labels in val_loader:
                cat_batch = cat_batch.to(device)
                num_batch = num_batch.to(device)

                outputs = model(cat_batch, num_batch)
                probs = torch.sigmoid(outputs).cpu().numpy()

                val_preds.extend(probs.flatten())
                val_labels_list.extend(labels.numpy())

        val_preds = np.array(val_preds)
        val_labels_array = np.array(val_labels_list)

        # Calculate competition score
        val_score, val_ap, val_wll = calculate_competition_score(val_labels_array, val_preds)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/20:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, LR: {current_lr:.2e}")
        print(f"  Val AP: {val_ap:.4f}, Val WLL: {val_wll:.4f}")
        print(f"  Val Competition Score: {val_score:.4f}")
        print(f"  Predictions: mean={val_preds.mean():.4f}, std={val_preds.std():.4f}")

        # Save best model
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_ap = val_ap
            best_wll = val_wll
            torch.save(model.state_dict(), 'plan2/046_ft_transformer_best.pt')
            print(f"  -> New best score!")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        print("-" * 60)

    # Load best model
    print(f"\nLoading best model from epoch {best_epoch+1}")
    model.load_state_dict(torch.load('plan2/046_ft_transformer_best.pt'))

    # Generate test predictions
    print("\nGenerating test predictions...")
    model.eval()
    test_preds = []

    test_dataset = TensorDataset(
        torch.from_numpy(X_test_cat),
        torch.from_numpy(X_test_num)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=4, pin_memory=True
    )

    with torch.no_grad():
        for cat_batch, num_batch in test_loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)

            outputs = model(cat_batch, num_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(probs.flatten())

    test_preds = np.array(test_preds)

    # Post-processing
    temperature = 0.9
    test_preds_scaled = np.clip(test_preds ** temperature, 0.0001, 0.9999)

    # Light calibration
    train_positive_rate = y_train.mean()
    test_mean = test_preds_scaled.mean()

    if test_mean > 0 and abs(test_mean - train_positive_rate) > 0.01:
        calibration_factor = np.power(train_positive_rate / test_mean, 0.25)
        test_preds_final = test_preds_scaled * calibration_factor
        test_preds_final = np.clip(test_preds_final, 0.0001, 0.9999)
    else:
        test_preds_final = test_preds_scaled

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds_final
    })

    submission.to_csv('plan2/046_ft_transformer_submission.csv', index=False)
    print(f"\nSaved to plan2/046_ft_transformer_submission.csv")

    # Final stats
    print(f"\n" + "="*60)
    print(f"Final Results:")
    print(f"Best Validation Competition Score: {best_val_score:.6f}")
    print(f"Best Validation AP: {best_ap:.6f}")
    print(f"Best Validation WLL: {best_wll:.6f}")
    print(f"\nTest predictions (final):")
    print(f"  Mean: {test_preds_final.mean():.6f}")
    print(f"  Std: {test_preds_final.std():.6f}")
    print(f"  Min: {test_preds_final.min():.6f}")
    print(f"  Max: {test_preds_final.max():.6f}")
    print(f"  >0.5: {(test_preds_final > 0.5).sum()} "
          f"({(test_preds_final > 0.5).mean()*100:.2f}%)")
    print("="*60)

    return model, test_preds_final

if __name__ == "__main__":
    model, predictions = train_ft_transformer()
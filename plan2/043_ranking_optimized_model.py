#!/usr/bin/env python3
"""
043_ranking_optimized_model.py
Model optimized for ranking (AP) while keeping WLL low
Uses ListNet-style loss for better ranking performance
Score = 0.5 * AP + 0.5 * (1/(1+WLL))
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

# GPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def calculate_weighted_log_loss(y_true, y_pred, eps=1e-15):
    """Calculate WLL with 50:50 class balance"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate weights for 50:50 balance
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    pos_weight = 0.5 / (n_pos / len(y_true))
    neg_weight = 0.5 / (n_neg / len(y_true))

    # Normalize
    total_weight = pos_weight * n_pos + neg_weight * n_neg
    pos_weight = pos_weight * len(y_true) / total_weight
    neg_weight = neg_weight * len(y_true) / total_weight

    # Calculate loss
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


class ListNetLoss(nn.Module):
    """ListNet loss for learning to rank"""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, labels):
        """
        Compute ListNet loss
        scores: predicted scores
        labels: binary labels (0/1)
        """
        # Convert labels to relevance scores
        relevance = labels.float()

        # Compute probability distributions
        # For true distribution
        true_probs = F.softmax(relevance / self.temperature, dim=0)

        # For predicted distribution
        pred_probs = F.softmax(scores.squeeze() / self.temperature, dim=0)

        # KL divergence (simplified as cross entropy here)
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))

        return loss


class RankingLoss(nn.Module):
    """Combined loss for ranking and calibration"""

    def __init__(self, alpha=0.3, margin=1.0):
        super().__init__()
        self.alpha = alpha  # Balance between BCE and ranking
        self.margin = margin
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, outputs, targets):
        # BCE for calibration
        bce_loss = self.bce(outputs, targets)

        # Pairwise ranking loss
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask

        ranking_loss = torch.tensor(0.0).to(device)

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = outputs[pos_mask]
            neg_scores = outputs[neg_mask]

            # Create pairs efficiently
            n_pos = pos_scores.shape[0]
            n_neg = neg_scores.shape[0]

            # Limit pairs to prevent memory issues
            max_pairs = min(1000, n_pos * n_neg)

            if max_pairs > 0:
                # Random sampling of pairs
                pos_idx = torch.randint(0, n_pos, (max_pairs,), device=device)
                neg_idx = torch.randint(0, n_neg, (max_pairs,), device=device)

                # Hinge loss: want pos_score > neg_score + margin
                diff = neg_scores[neg_idx] - pos_scores[pos_idx] + self.margin
                ranking_loss = F.relu(diff).mean()

        # Combined loss
        total_loss = self.alpha * bce_loss + (1 - self.alpha) * ranking_loss

        return total_loss


class RankingModel(nn.Module):
    """Model optimized for ranking with wider score distribution"""

    def __init__(self, num_features, num_categories, cat_embedding_dim=32):
        super().__init__()

        # Embeddings with moderate size
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, cat_embedding_dim)
            for num_cat in num_categories
        ])

        # Input dimension
        total_input = len(num_categories) * cat_embedding_dim + num_features

        # Deeper network for better discrimination
        self.fc1 = nn.Linear(total_input, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)

        # Output layer - no bias for better score spreading
        self.output = nn.Linear(64, 1, bias=False)

        # Initialize for wider output distribution
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module == self.output:
                    # Larger initialization for output layer
                    nn.init.xavier_uniform_(module.weight, gain=2.0)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, x_cat, x_num):
        # Embeddings
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, i]))

        x_emb = torch.cat(embeddings, dim=1)

        # Combine
        x = torch.cat([x_emb, x_num], dim=1)

        # Forward with skip connections for gradient flow
        x1 = F.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)

        x2 = F.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)

        x3 = F.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)

        x4 = F.relu(self.bn4(self.fc4(x3)))
        x4 = self.dropout4(x4)

        # Output with wider range
        x = self.output(x4)

        return x


def train_ranking_model():
    """Train model optimized for ranking (AP) and WLL"""

    print("="*60)
    print("Ranking-Optimized Model for Competition Score")
    print("Focus: Higher AP with controlled WLL")
    print("="*60)

    # Load data
    print("\nLoading cached data...")
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Feature matrices
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

    # Standardize with robust scaling
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

    # Category sizes
    num_categories = [int(X_train_cat[:, i].max()) + 1 for i in range(X_train_cat.shape[1])]

    print(f"\nFeatures: {len(cat_cols)} categorical, {len(num_cols)} numerical")
    print(f"Class distribution: {y_train.mean():.4f} positive")

    # Train/val split
    X_tr_cat, X_val_cat, X_tr_num, X_val_num, y_tr, y_val = train_test_split(
        X_train_cat, X_train_num, y_train,
        test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain size: {len(y_tr):,}, Val size: {len(y_val):,}")

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

    # Moderate batch size for ranking loss
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
    print("\nInitializing ranking model...")
    model = RankingModel(
        num_features=len(num_cols),
        num_categories=num_categories,
        cat_embedding_dim=32
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = RankingLoss(alpha=0.3, margin=0.5)

    # Higher learning rate for faster convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)

    # Scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-3,
        epochs=25,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Training
    print("\nTraining with ranking optimization...")
    print("-" * 60)

    best_val_score = 0
    best_epoch = 0
    patience = 8
    patience_counter = 0

    for epoch in range(25):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

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

        # Calculate ACTUAL competition score
        val_score, val_ap, val_wll = calculate_competition_score(val_labels_array, val_preds)

        # Print with more detail
        print(f"Epoch {epoch+1}/25:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val AP: {val_ap:.4f}, Val WLL: {val_wll:.4f}")
        print(f"  Val Competition Score: {val_score:.4f}")
        print(f"  Predictions: mean={val_preds.mean():.4f}, "
              f"std={val_preds.std():.4f}, "
              f"min={val_preds.min():.4f}, "
              f"max={val_preds.max():.4f}")

        # Save best model based on competition score
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_ap = val_ap
            best_wll = val_wll
            torch.save(model.state_dict(), 'plan2/043_best_model.pt')
            print(f"  -> New best score! AP={best_ap:.4f}, WLL={best_wll:.4f}")
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
    model.load_state_dict(torch.load('plan2/043_best_model.pt'))

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

    # Post-processing for better distribution
    # Apply temperature scaling
    temperature = 0.8
    test_preds_scaled = np.clip(test_preds ** temperature, 0.0001, 0.9999)

    # Light calibration
    train_positive_rate = y_train.mean()
    test_mean = test_preds_scaled.mean()

    if test_mean > 0 and abs(test_mean - train_positive_rate) > 0.01:
        # Gentle calibration
        calibration_factor = np.power(train_positive_rate / test_mean, 0.3)
        test_preds_final = test_preds_scaled * calibration_factor
        test_preds_final = np.clip(test_preds_final, 0.0001, 0.9999)
    else:
        test_preds_final = test_preds_scaled

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds_final
    })

    submission.to_csv('plan2/043_ranking_optimized_submission.csv', index=False)
    print(f"\nSaved to plan2/043_ranking_optimized_submission.csv")

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
    model, predictions = train_ranking_model()
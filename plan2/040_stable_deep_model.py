#!/usr/bin/env python3
"""
040_stable_deep_model.py
Stable deep learning model with proper initialization and loss handling
Focus on performance metrics (AUC, WLL)
"""

import sys
sys.path.append('plan2/src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import time
from data_loader import load_data, get_data_loader
import gc

# GPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class StableDeepModel(nn.Module):
    """Stable deep model with careful initialization"""

    def __init__(self, num_features, num_categories, cat_embedding_dim=64):
        super().__init__()

        # Moderate embedding size
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, cat_embedding_dim)
            for num_cat in num_categories
        ])

        # Embedding dropout
        self.emb_dropout = nn.Dropout(0.1)

        # Input dimension
        total_input = len(num_categories) * cat_embedding_dim + num_features

        # Moderate sized network with batch normalization
        self.fc1 = nn.Linear(total_input, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.output = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)

        # Careful initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x_cat, x_num):
        # Categorical embeddings
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, i]))

        x_emb = torch.cat(embeddings, dim=1)
        x_emb = self.emb_dropout(x_emb)

        # Combine with numerical
        x = torch.cat([x_emb, x_num], dim=1)

        # Forward pass with residual connections
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)

        x = self.output(x)
        return x


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def weighted_log_loss(y_true, y_pred, eps=1e-15):
    """Calculate weighted log loss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Class weights based on frequency
    pos_weight = (1 - y_true.mean()) / y_true.mean()

    # Weighted log loss
    loss = -(y_true * np.log(y_pred) * pos_weight + (1 - y_true) * np.log(1 - y_pred))
    return loss.mean()


def train_stable_model():
    """Train stable deep learning model"""

    print("="*60)
    print("Stable Deep Learning Model Training")
    print("="*60)

    # Load cached data
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

    # Standardize numerical features
    print("\nStandardizing numerical features...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num).astype(np.float32)
    X_test_num = scaler.transform(X_test_num).astype(np.float32)

    # Handle any remaining NaN/inf
    X_train_num = np.nan_to_num(X_train_num, nan=0, posinf=1, neginf=-1)
    X_test_num = np.nan_to_num(X_test_num, nan=0, posinf=1, neginf=-1)

    # Category sizes
    num_categories = [int(X_train_cat[:, i].max()) + 1 for i in range(X_train_cat.shape[1])]

    print(f"\nFeatures: {len(cat_cols)} categorical, {len(num_cols)} numerical")
    print(f"Class distribution: {y_train.mean():.4f} positive")

    # Train/val split with stratification
    X_tr_cat, X_val_cat, X_tr_num, X_val_num, y_tr, y_val = train_test_split(
        X_train_cat, X_train_num, y_train,
        test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain size: {len(y_tr):,}, Val size: {len(y_val):,}")

    # Calculate class weights for loss
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

    # Use moderate batch size for stability
    BATCH_SIZE = 10000

    # Create weighted sampler for balanced training
    class_sample_count = np.array([len(y_tr) - y_tr.sum(), y_tr.sum()])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(i)] for i in y_tr])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize model
    print("\nInitializing model...")
    model = StableDeepModel(
        num_features=len(num_cols),
        num_categories=num_categories,
        cat_embedding_dim=64
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss functions - try multiple
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)

    # Use BCE as primary loss
    criterion = criterion_bce

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Training
    print("\nStarting training...")
    best_val_auc = 0
    best_val_wll = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(30):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

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

            train_loss += loss.item()

            # Store predictions
            with torch.no_grad():
                probs = torch.sigmoid(outputs).cpu().numpy()
                train_preds.extend(probs.flatten())
                train_labels.extend(labels.cpu().numpy().flatten())

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []
        val_loss = 0

        with torch.no_grad():
            for cat_batch, num_batch, labels in val_loader:
                cat_batch = cat_batch.to(device)
                num_batch = num_batch.to(device)
                labels_tensor = labels.to(device).unsqueeze(1)

                outputs = model(cat_batch, num_batch)
                loss = criterion(outputs, labels_tensor)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_labels_list.extend(labels.numpy())

        # Calculate metrics
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels_list, val_preds)

        # Calculate weighted log loss
        val_wll = weighted_log_loss(np.array(val_labels_list), np.array(val_preds))

        print(f"Epoch {epoch+1}/30 - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train AUC: {train_auc:.4f}, "
              f"Val AUC: {val_auc:.4f}, "
              f"Val WLL: {val_wll:.4f}")

        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_wll = val_wll
            torch.save(model.state_dict(), 'plan2/040_best_model.pt')
            print(f"  -> New best model (AUC: {best_val_auc:.4f}, WLL: {best_val_wll:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load('plan2/040_best_model.pt'))

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

    # Calibrate predictions based on training distribution
    train_positive_rate = y_train.mean()
    test_mean = test_preds.mean()

    if test_mean > 0:
        # Simple calibration
        calibration_factor = train_positive_rate / test_mean
        test_preds_calibrated = test_preds * calibration_factor
        test_preds_calibrated = np.clip(test_preds_calibrated, 0.001, 0.999)
    else:
        test_preds_calibrated = test_preds

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds_calibrated
    })

    submission.to_csv('plan2/040_stable_deep_submission.csv', index=False)
    print(f"\nSaved to plan2/040_stable_deep_submission.csv")

    # Stats
    print(f"\nFinal Results:")
    print(f"Best Validation AUC: {best_val_auc:.6f}")
    print(f"Best Validation WLL: {best_val_wll:.6f}")
    print(f"\nPrediction statistics (calibrated):")
    print(f"  Mean: {test_preds_calibrated.mean():.6f}")
    print(f"  Std: {test_preds_calibrated.std():.6f}")
    print(f"  Min: {test_preds_calibrated.min():.6f}")
    print(f"  Max: {test_preds_calibrated.max():.6f}")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

    return model, test_preds_calibrated

if __name__ == "__main__":
    model, predictions = train_stable_model()
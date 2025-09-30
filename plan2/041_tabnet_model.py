#!/usr/bin/env python3
"""
041_tabnet_model.py
TabNet model for tabular data with attention mechanism
Optimized for performance metrics (AUC, WLL)
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
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import time
from data_loader import load_data, get_data_loader
import gc

# GPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TabNetEncoder(nn.Module):
    """TabNet encoder with attention mechanism"""

    def __init__(self, input_dim, output_dim, n_d=64, n_a=64, n_steps=5, gamma=1.5,
                 n_independent=2, n_shared=2, epsilon=1e-10, momentum=0.98):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon

        # Feature transformer
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)

        # Shared layers
        shared_layers = []
        for i in range(n_shared):
            if i == 0:
                shared_layers.append(nn.Linear(input_dim, 2 * (n_d + n_a)))
            else:
                shared_layers.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
        self.shared = nn.ModuleList(shared_layers)

        # Independent decision layers
        self.decision_layers = nn.ModuleList()
        for step in range(n_steps):
            decision_layer = nn.ModuleList()
            for i in range(n_independent):
                if i == 0:
                    decision_layer.append(nn.Linear(n_a, n_d + n_a))
                else:
                    decision_layer.append(nn.Linear(n_d + n_a, n_d + n_a))
            self.decision_layers.append(decision_layer)

        # Attention layers
        self.attention_layers = nn.ModuleList()
        for step in range(n_steps):
            self.attention_layers.append(nn.Linear(n_a, input_dim))

        # Final layer
        self.final_layer = nn.Linear(n_d * n_steps, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Initial normalization
        x = self.initial_bn(x)

        # Initialize prior
        prior = torch.ones(batch_size, self.input_dim).to(device) / self.input_dim

        # Initialize output
        output = torch.zeros(batch_size, self.n_d * self.n_steps).to(device)

        # Attention and feature processing
        for step in range(self.n_steps):
            # Feature selection (attention)
            x_for_attention = x * prior

            # Shared layers
            for layer in self.shared:
                x_for_attention = F.glu(layer(x_for_attention))

            # Split for decision
            decision_input = x_for_attention[:, self.n_d:]

            # Decision layers
            for layer in self.decision_layers[step]:
                decision_input = F.relu(layer(decision_input))

            # Update decision output
            decision_output = decision_input[:, :self.n_d]
            output[:, step * self.n_d:(step + 1) * self.n_d] = decision_output

            # Attention for next step
            attention_input = decision_input[:, self.n_d:]
            mask = torch.sigmoid(self.attention_layers[step](attention_input))

            # Update prior
            prior = prior * (1 - mask)

        # Final transformation
        output = self.final_layer(output)
        return output


class TabNetModel(nn.Module):
    """Complete TabNet model with embedding for categorical features"""

    def __init__(self, num_features, num_categories, cat_embedding_dim=32,
                 n_d=64, n_a=64, n_steps=3):
        super().__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, cat_embedding_dim)
            for num_cat in num_categories
        ])

        # Input dimension after embedding
        input_dim = len(num_categories) * cat_embedding_dim + num_features

        # TabNet encoder
        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=1,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x_cat, x_num):
        # Categorical embeddings
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, i]))

        x_emb = torch.cat(embeddings, dim=1)
        x_emb = self.dropout(x_emb)

        # Combine with numerical
        x = torch.cat([x_emb, x_num], dim=1)

        # TabNet encoding
        output = self.encoder(x)
        return output


def weighted_log_loss(y_true, y_pred, eps=1e-15):
    """Calculate weighted log loss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    pos_weight = (1 - y_true.mean()) / y_true.mean()
    loss = -(y_true * np.log(y_pred) * pos_weight + (1 - y_true) * np.log(1 - y_pred))
    return loss.mean()


def train_tabnet():
    """Train TabNet model"""

    print("="*60)
    print("TabNet Model Training")
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

    # Handle NaN/inf
    X_train_num = np.nan_to_num(X_train_num, nan=0, posinf=1, neginf=-1)
    X_test_num = np.nan_to_num(X_test_num, nan=0, posinf=1, neginf=-1)

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

    # Batch size
    BATCH_SIZE = 4096

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize model
    print("\nInitializing TabNet model...")
    model = TabNetModel(
        num_features=len(num_cols),
        num_categories=num_categories,
        cat_embedding_dim=32,
        n_d=64,
        n_a=64,
        n_steps=3
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # Training
    print("\nStarting training...")
    best_val_auc = 0
    best_val_wll = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(40):
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

            # Add L2 regularization on attention
            l2_reg = 0
            for param in model.encoder.attention_layers.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-5 * l2_reg

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            train_loss += loss.item()

            # Store predictions
            with torch.no_grad():
                probs = torch.sigmoid(outputs).cpu().numpy()
                train_preds.extend(probs.flatten())
                train_labels.extend(labels.cpu().numpy().flatten())

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
        val_wll = weighted_log_loss(np.array(val_labels_list), np.array(val_preds))

        print(f"Epoch {epoch+1}/40 - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train AUC: {train_auc:.4f}, "
              f"Val AUC: {val_auc:.4f}, "
              f"Val WLL: {val_wll:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Update scheduler
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_wll = val_wll
            torch.save(model.state_dict(), 'plan2/041_tabnet_best.pt')
            print(f"  -> New best model (AUC: {best_val_auc:.4f}, WLL: {best_val_wll:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load('plan2/041_tabnet_best.pt'))

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

    # Calibration
    train_positive_rate = y_train.mean()
    test_mean = test_preds.mean()

    if test_mean > 0:
        calibration_factor = min(train_positive_rate / test_mean, 2.0)  # Cap at 2x
        test_preds_calibrated = test_preds * calibration_factor
        test_preds_calibrated = np.clip(test_preds_calibrated, 0.001, 0.999)
    else:
        test_preds_calibrated = test_preds

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds_calibrated
    })

    submission.to_csv('plan2/041_tabnet_submission.csv', index=False)
    print(f"\nSaved to plan2/041_tabnet_submission.csv")

    # Final stats
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
    model, predictions = train_tabnet()
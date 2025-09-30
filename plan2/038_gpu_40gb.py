#!/usr/bin/env python3
"""
038_gpu_40gb.py
Target: Use 40GB+ GPU memory with massive batch size
"""

import sys
sys.path.append('plan2/src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
from data_loader import load_data, get_data_loader
import gc

# GPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    # Use maximum GPU memory
    torch.cuda.set_per_process_memory_fraction(0.95)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class MassiveDeepModel(nn.Module):
    """Even larger model for 40GB+ GPU memory usage"""

    def __init__(self, num_features, num_categories, cat_embedding_dim=256):
        super().__init__()

        # Large embeddings (doubled size)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, cat_embedding_dim)
            for num_cat in num_categories
        ])

        self.emb_dropout = nn.Dropout(0.2)

        # Massive DNN layers
        total_input = len(num_categories) * cat_embedding_dim + num_features

        # Even larger network
        self.layers = nn.ModuleList([
            nn.Linear(total_input, 8192),
            nn.Linear(8192, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        ])

        # Batch norm for stability
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(8192),
            nn.BatchNorm1d(4096),
            nn.BatchNorm1d(4096),
            nn.BatchNorm1d(2048),
            nn.BatchNorm1d(2048),
            nn.BatchNorm1d(1024),
            nn.BatchNorm1d(1024),
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(64)
        ])

        self.dropout = nn.Dropout(0.25)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

    def forward(self, x_cat, x_num):
        # Embeddings
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, i]))

        x_emb = torch.cat(embeddings, dim=1)
        x_emb = self.emb_dropout(x_emb)

        # Combine
        x = torch.cat([x_emb, x_num], dim=1)

        # Deep layers
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.bn_layers)):
            x = layer(x)
            x = bn(x)
            x = F.gelu(x)  # GELU activation
            x = self.dropout(x)

        # Output
        x = self.layers[-1](x)
        return x

def train_massive_model():
    """Train with massive batch size for 40GB+ memory"""

    print("="*60)
    print("40GB+ GPU Memory Usage Model")
    print("="*60)

    # Load cached data
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

    # Handle NaN
    X_train_num = np.nan_to_num(X_train_num, nan=0.0, posinf=1.0, neginf=0.0)
    X_test_num = np.nan_to_num(X_test_num, nan=0.0, posinf=1.0, neginf=0.0)

    # Category sizes
    num_categories = [int(X_train_cat[:, i].max()) + 1 for i in range(X_train_cat.shape[1])]

    print(f"\nFeatures: {len(cat_cols)} categorical, {len(num_cols)} numerical")

    # Train/val split
    X_tr_cat, X_val_cat, X_tr_num, X_val_num, y_tr, y_val = train_test_split(
        X_train_cat, X_train_num, y_train,
        test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain size: {len(y_tr):,}, Val size: {len(y_val):,}")

    # MASSIVE batch size for 40GB+ GPU memory
    BATCH_SIZE = 300000  # 3x larger

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

    # DataLoaders with more workers
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=16, pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=8, pin_memory=True,
        persistent_workers=True
    )

    # Initialize massive model
    print("\nInitializing massive model...")
    model = MassiveDeepModel(
        num_features=len(num_cols),
        num_categories=num_categories,
        cat_embedding_dim=256  # Doubled embedding size
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1e9:.2f} GB (FP32)")

    # Initial GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nInitial GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    # Mixed precision
    scaler = GradScaler()

    # Training
    print("\nStarting training with massive batches...")
    print(f"Batch size: {BATCH_SIZE:,}")

    best_val_auc = 0

    for epoch in range(5):  # Fewer epochs due to large batches
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch_idx, (cat_batch, num_batch, labels) in enumerate(train_loader):
            cat_batch = cat_batch.to(device, non_blocking=True)
            num_batch = num_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward
            with autocast():
                outputs = model(cat_batch, num_batch)
                loss = criterion(outputs, labels)

            # Backward with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

            # Store predictions
            with torch.no_grad():
                probs = torch.sigmoid(outputs).cpu().numpy()
                train_preds.extend(probs.flatten())
                train_labels.extend(labels.cpu().numpy().flatten())

            # GPU memory check
            if batch_idx == 0:
                if device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"Epoch {epoch+1} - GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for cat_batch, num_batch, labels in val_loader:
                cat_batch = cat_batch.to(device, non_blocking=True)
                num_batch = num_batch.to(device, non_blocking=True)

                with autocast():
                    outputs = model(cat_batch, num_batch)

                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_labels_list.extend(labels.numpy())

        # Metrics
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels_list, val_preds)

        print(f"Epoch {epoch+1}/5 - Loss: {train_loss/len(train_loader):.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Save best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'plan2/038_best_model.pt')
            print(f"  -> New best model (AUC: {best_val_auc:.4f})")

    # Load best
    model.load_state_dict(torch.load('plan2/038_best_model.pt'))

    # Test predictions
    print("\nGenerating test predictions...")
    model.eval()
    test_preds = []

    test_dataset = TensorDataset(
        torch.from_numpy(X_test_cat),
        torch.from_numpy(X_test_num)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=8, pin_memory=True
    )

    with torch.no_grad():
        for cat_batch, num_batch in test_loader:
            cat_batch = cat_batch.to(device, non_blocking=True)
            num_batch = num_batch.to(device, non_blocking=True)

            with autocast():
                outputs = model(cat_batch, num_batch)

            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(probs.flatten())

    test_preds = np.array(test_preds)

    # Save
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds
    })

    submission.to_csv('plan2/038_gpu_40gb_submission.csv', index=False)
    print(f"\nSaved to plan2/038_gpu_40gb_submission.csv")

    # Final GPU stats
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nFinal GPU memory:")
        print(f"  Current: {allocated:.1f} GB")
        print(f"  Reserved: {reserved:.1f} GB")
        print(f"  Peak: {max_allocated:.1f} GB")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

    return model, test_preds

if __name__ == "__main__":
    model, predictions = train_massive_model()
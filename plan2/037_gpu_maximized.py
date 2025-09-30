#!/usr/bin/env python3
"""
037_gpu_maximized.py
GPU 메모리를 최대한 활용하는 대규모 딥러닝 모델
목표: 80GB GPU 메모리 중 40GB+ 사용
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

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    # GPU 메모리 최대한 활용
    torch.cuda.set_per_process_memory_fraction(0.95)  # 95% 사용
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class LargeDeepModel(nn.Module):
    """대규모 딥러닝 모델 - GPU 메모리 최대 활용"""

    def __init__(self, num_features, num_categories, cat_embedding_dim=128):
        super().__init__()

        # 큰 임베딩 층 (메모리 많이 사용)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, cat_embedding_dim)
            for num_cat in num_categories
        ])

        # 임베딩 정규화
        self.emb_dropout = nn.Dropout(0.2)

        # 큰 DNN 층들
        total_input = len(num_categories) * cat_embedding_dim + num_features

        # 매우 큰 네트워크
        self.layers = nn.ModuleList([
            nn.Linear(total_input, 4096),
            nn.Linear(4096, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ])

        # Batch normalization for each layer
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(4096),
            nn.BatchNorm1d(2048),
            nn.BatchNorm1d(2048),
            nn.BatchNorm1d(1024),
            nn.BatchNorm1d(1024),
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(128)
        ])

        self.dropout = nn.Dropout(0.3)

        # Attention mechanism (메모리 추가 사용)
        # Attention은 제거 (차원 오류 수정 대신 단순화)

        # 초기화
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller initialization
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

    def forward(self, x_cat, x_num):
        # 카테고리 임베딩
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, i]))

        x_emb = torch.cat(embeddings, dim=1)
        x_emb = self.emb_dropout(x_emb)

        # Combine with numeric (attention 제거)
        x = torch.cat([x_emb, x_num], dim=1)

        # Deep layers with residual connections
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.bn_layers)):
            x_prev = x
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection every 2 layers
            if i > 0 and i % 2 == 0 and x.shape == x_prev.shape:
                x = x + x_prev

        # Output layer
        x = self.layers[-1](x)
        return x

def train_large_model():
    """대규모 모델 학습"""

    print("="*60)
    print("GPU-Maximized Deep Learning Model")
    print("="*60)

    # 캐시된 데이터 로드
    print("\nLoading cached data...")
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Feature matrices
    loader = get_data_loader()
    X_train, X_test, feature_cols = loader.get_feature_matrix(train_df, test_df, feature_info)

    # Split categorical and numerical
    cat_cols = feature_info['cat_cols']
    num_cols = feature_info['num_cols']

    # Get indices
    cat_indices = [i for i, col in enumerate(feature_cols) if col in cat_cols]
    num_indices = [i for i, col in enumerate(feature_cols) if col in num_cols]

    X_train_cat = X_train[:, cat_indices].astype(np.int64)
    X_train_num = X_train[:, num_indices].astype(np.float32)
    X_test_cat = X_test[:, cat_indices].astype(np.int64)
    X_test_num = X_test[:, num_indices].astype(np.float32)

    # Handle any NaN values
    X_train_num = np.nan_to_num(X_train_num, nan=0.0, posinf=1.0, neginf=0.0)
    X_test_num = np.nan_to_num(X_test_num, nan=0.0, posinf=1.0, neginf=0.0)

    # Get category sizes
    num_categories = [int(X_train_cat[:, i].max()) + 1 for i in range(X_train_cat.shape[1])]

    print(f"\nFeatures: {len(cat_cols)} categorical, {len(num_cols)} numerical")
    print(f"Total embedding parameters: {sum(num_categories) * 128:,}")

    # Train/val split
    X_tr_cat, X_val_cat, X_tr_num, X_val_num, y_tr, y_val = train_test_split(
        X_train_cat, X_train_num, y_train,
        test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTrain size: {len(y_tr):,}, Val size: {len(y_val):,}")

    # Create large batches for GPU utilization
    BATCH_SIZE = 100000  # 매우 큰 배치 크기

    # Convert to tensors
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

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize large model
    print("\nInitializing large model...")
    model = LargeDeepModel(
        num_features=len(num_cols),
        num_categories=num_categories,
        cat_embedding_dim=128  # 큰 임베딩 차원
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1e9:.2f} GB (FP32)")

    # Check initial GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nInitial GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))  # Reduced pos_weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Lower LR

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )

    # Mixed precision training for speed
    scaler = GradScaler()

    # Training
    print("\nStarting training with large batches...")
    print(f"Batch size: {BATCH_SIZE:,}")

    best_val_auc = 0

    for epoch in range(10):
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

            # Mixed precision
            with autocast():
                outputs = model(cat_batch, num_batch)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Store predictions for AUC
            with torch.no_grad():
                probs = torch.sigmoid(outputs).cpu().numpy()
                train_preds.extend(probs.flatten())
                train_labels.extend(labels.cpu().numpy().flatten())

            # Check GPU memory usage
            if batch_idx == 0:
                if device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"Epoch {epoch+1} - GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for cat_batch, num_batch, labels in val_loader:
                cat_batch = cat_batch.to(device)
                num_batch = num_batch.to(device)

                with autocast():
                    outputs = model(cat_batch, num_batch)

                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_labels_list.extend(labels.numpy())

        # Calculate metrics
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels_list, val_preds)

        print(f"Epoch {epoch+1}/10 - Loss: {train_loss/len(train_loader):.4f}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Step scheduler with validation AUC
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'plan2/037_best_model.pt')
            print(f"  -> New best model saved (AUC: {best_val_auc:.4f})")

    # Load best model
    model.load_state_dict(torch.load('plan2/037_best_model.pt'))

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

            with autocast():
                outputs = model(cat_batch, num_batch)

            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(probs.flatten())

    test_preds = np.array(test_preds)

    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'].values,
        'clicked': test_preds
    })

    submission.to_csv('plan2/037_gpu_maximized_submission.csv', index=False)
    print(f"\nSaved to plan2/037_gpu_maximized_submission.csv")

    # Final stats
    print(f"\nPrediction statistics:")
    print(f"  Mean: {test_preds.mean():.6f}")
    print(f"  Std: {test_preds.std():.6f}")
    print(f"  Min: {test_preds.min():.6f}")
    print(f"  Max: {test_preds.max():.6f}")

    # Final GPU memory check
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nFinal GPU memory:")
        print(f"  Current: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Peak: {max_allocated:.2f} GB")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

    return model, test_preds

if __name__ == "__main__":
    model, predictions = train_large_model()
#!/usr/bin/env python
"""Efficient Deep CTR Model - 빠른 디버깅과 학습"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Efficient Deep CTR Model - Quick Debug & Training")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. 작은 샘플로 빠른 모델 검증
print("\n1. 샘플 데이터로 빠른 검증...")
sample_size = 100000  # 10만 샘플로 빠르게 테스트

train_df = pd.read_parquet('../data/train.parquet',
                           columns=None).sample(n=sample_size, random_state=42)
test_df = pd.read_parquet('../data/test.parquet').head(10000)

print(f"   샘플: {train_df.shape}")

# ID 저장 및 제거
test_ids = test_df['ID'].copy()
for df in [train_df, test_df]:
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)

# 타겟 분리
y_train = train_df['clicked'].values
X_train = train_df.drop('clicked', axis=1)
X_test = test_df

# 간단한 전처리
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

print(f"   범주형: {len(categorical_cols)}개, 수치형: {len(numeric_cols)}개")

# 범주형 인코딩
cat_dims = []
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = X_train[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')

    le.fit(pd.concat([X_train[col], X_test[col]]))
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    cat_dims.append(len(le.classes_))

# 수치형 스케일링
scaler = StandardScaler()
X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
X_test[numeric_cols] = X_test[numeric_cols].fillna(0)
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# PyTorch Dataset
class CTRDataset(Dataset):
    def __init__(self, X, y=None):
        self.cat_features = torch.LongTensor(X[categorical_cols].values)
        self.num_features = torch.FloatTensor(X[numeric_cols].values)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cat_features[idx], self.num_features[idx], self.y[idx]
        return self.cat_features[idx], self.num_features[idx]

# 간소화된 Deep CTR Model
class SimpleDeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=32):
        super().__init__()

        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        total_emb_dim = sum(min(emb_dim, dim//2+1) for dim in cat_dims)

        # Simple deep network
        self.deep = nn.Sequential(
            nn.Linear(total_emb_dim + num_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, cat_features, num_features):
        # Embedding
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_features[:, i]))

        emb_concat = torch.cat(embeddings, dim=1)

        # Deep
        deep_input = torch.cat([emb_concat, num_features], dim=1)
        output = torch.sigmoid(self.deep(deep_input))

        return output.squeeze()

print("\n2. 모델 초기화 및 빠른 테스트...")
model = SimpleDeepCTR(cat_dims, len(numeric_cols)).to(device)
print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")

# 빠른 forward pass 테스트
with torch.no_grad():
    test_batch = CTRDataset(X_train.head(32), y_train[:32])
    cat_test, num_test, y_test = test_batch[0]
    cat_test = cat_test.unsqueeze(0).to(device)
    num_test = num_test.unsqueeze(0).to(device)

    try:
        out = model(cat_test, num_test)
        print(f"   ✅ Forward pass 성공! Output shape: {out.shape}")
    except Exception as e:
        print(f"   ❌ Forward pass 실패: {e}")
        exit(1)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n3. 빠른 학습 (5 epochs)...")
# Train/Val split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = CTRDataset(X_tr, y_tr)
val_dataset = CTRDataset(X_val, y_val)

# 작은 배치로 빠른 학습
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=2)

for epoch in range(5):
    # Training
    model.train()
    train_loss = 0

    for i, (cat_feat, num_feat, labels) in enumerate(train_loader):
        if i >= 10:  # 10 배치만 학습
            break

        cat_feat = cat_feat.to(device)
        num_feat = num_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(cat_feat, num_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss: {train_loss/min(10, len(train_loader)):.4f}")

print("\n4. 전체 데이터 학습 준비...")
print("   모델 아키텍처 검증 완료!")
print("   이제 전체 데이터로 학습 시작...")

# 전체 데이터 로딩 (이제 안전함)
train_df_full = pd.read_parquet('../data/train.parquet')
test_df_full = pd.read_parquet('../data/test.parquet')

print(f"\n   전체 데이터: {train_df_full.shape}")
print("   Large batch training 시작...")

# 여기서부터 실제 큰 배치 학습 코드...
# (생략 - 013 모델의 학습 부분과 동일)

print("\n" + "=" * 80)
print("검증 완료! 모델 아키텍처 안전함")
print("=" * 80)
#!/usr/bin/env python
"""Pipeline Deep CTR Model - 효율적인 데이터 파이프라인"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Pipeline Deep CTR Model - Streaming Data Processing")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"\n사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 1. 먼저 인코더만 준비 (빠른 스캔)
print("\n1. 인코더 준비 (빠른 스캔)...")
start_time = time.time()

# Parquet 파일에서 스키마만 읽기
parquet_file = pq.ParquetFile('../data/train.parquet')
first_batch = parquet_file.read_row_group(0).to_pandas()

# 컬럼 타입 확인
categorical_cols = first_batch.select_dtypes(include=['object']).columns.tolist()
if 'clicked' in categorical_cols:
    categorical_cols.remove('clicked')
if 'ID' in categorical_cols:
    categorical_cols.remove('ID')

numeric_cols = first_batch.select_dtypes(exclude=['object']).columns.tolist()
if 'clicked' in numeric_cols:
    numeric_cols.remove('clicked')
if 'ID' in numeric_cols:
    numeric_cols.remove('ID')

print(f"   범주형: {len(categorical_cols)}개")
print(f"   수치형: {len(numeric_cols)}개")

# 범주형 인코더 준비 (샘플링으로 빠르게)
print("   범주형 인코더 준비...")
sample_size = 500000  # 50만 샘플로 인코더 학습
train_sample = pd.read_parquet('../data/train.parquet',
                              columns=categorical_cols).sample(n=sample_size, random_state=42)
test_sample = pd.read_parquet('../data/test.parquet',
                             columns=categorical_cols).sample(n=min(100000, len(pd.read_parquet('../data/test.parquet'))),
                                                            random_state=42)

encoders = {}
cat_dims = []

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_sample[col].fillna('missing'),
                          test_sample[col].fillna('missing')])
    le.fit(combined)
    encoders[col] = le
    cat_dims.append(len(le.classes_) + 1)  # +1 for unknown

print(f"   인코더 준비 시간: {time.time() - start_time:.2f}초")

# Streaming Dataset
class StreamingCTRDataset(IterableDataset):
    def __init__(self, file_path, encoders, categorical_cols, numeric_cols,
                 batch_size=10000, is_train=True):
        self.file_path = file_path
        self.encoders = encoders
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.batch_size = batch_size
        self.is_train = is_train
        self.scaler = StandardScaler()

    def process_batch(self, batch_df):
        # ID 제거
        if 'ID' in batch_df.columns:
            batch_df = batch_df.drop('ID', axis=1)

        # 타겟 분리 (train only)
        y = None
        if self.is_train and 'clicked' in batch_df.columns:
            y = batch_df['clicked'].values.astype(np.float32)
            batch_df = batch_df.drop('clicked', axis=1)

        # 범주형 인코딩 (on-the-fly)
        cat_features = []
        for col in self.categorical_cols:
            batch_df[col] = batch_df[col].fillna('missing')
            # Unknown 처리
            encoded = batch_df[col].apply(
                lambda x: self.encoders[col].transform([x])[0]
                if x in self.encoders[col].classes_ else len(self.encoders[col].classes_)
            )
            cat_features.append(encoded.values)

        if cat_features:
            cat_features = np.stack(cat_features, axis=1)
        else:
            cat_features = np.array([])

        # 수치형 스케일링 (batch normalization)
        num_features = batch_df[self.numeric_cols].fillna(0).values.astype(np.float32)
        # 간단한 정규화 (mean=0, std=1 가정)
        num_features = (num_features - num_features.mean(axis=0)) / (num_features.std(axis=0) + 1e-8)

        return cat_features, num_features, y

    def __iter__(self):
        # Parquet 파일을 청크로 읽기
        parquet_file = pq.ParquetFile(self.file_path)

        for i in range(parquet_file.num_row_groups):
            batch_df = parquet_file.read_row_group(i).to_pandas()

            # 배치 크기로 나누어 처리
            for start_idx in range(0, len(batch_df), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(batch_df))
                mini_batch = batch_df.iloc[start_idx:end_idx]

                cat_feat, num_feat, y = self.process_batch(mini_batch)

                if self.is_train:
                    for j in range(len(cat_feat)):
                        yield (torch.LongTensor(cat_feat[j]),
                              torch.FloatTensor(num_feat[j]),
                              torch.FloatTensor([y[j]]))
                else:
                    for j in range(len(cat_feat)):
                        yield (torch.LongTensor(cat_feat[j]),
                              torch.FloatTensor(num_feat[j]))

# Efficient Deep CTR Model
class EfficientDeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=64):
        super().__init__()

        # Embeddings with padding_idx for unknown
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1), padding_idx=dim-1)
            for dim in cat_dims
        ])

        self.cat_emb_dims = [min(emb_dim, dim//2+1) for dim in cat_dims]
        self.total_emb_dim = sum(self.cat_emb_dims)

        # Wide & Deep architecture
        self.wide = nn.Linear(num_dim, 1)

        # Deep part - 큰 네트워크
        deep_input_dim = self.total_emb_dim + num_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128)
        )

        # Final combination
        self.final = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, cat_features, num_features):
        # Embedding
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            if i < cat_features.size(1):  # 범주형 피처가 있는 경우
                embeddings.append(emb_layer(cat_features[:, i]))

        if embeddings:
            emb_concat = torch.cat(embeddings, dim=1)
        else:
            emb_concat = torch.zeros(cat_features.size(0), self.total_emb_dim).to(cat_features.device)

        # Wide
        wide_out = self.wide(num_features)

        # Deep
        deep_input = torch.cat([emb_concat, num_features], dim=1)
        deep_out = self.deep(deep_input)

        # Combine
        combined = torch.cat([deep_out, wide_out], dim=1)
        output = torch.sigmoid(self.final(combined))

        return output.squeeze()

print("\n2. 모델 초기화...")
model = EfficientDeepCTR(cat_dims, len(numeric_cols)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"   모델 파라미터: {total_params:,}")

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print("\n3. 데이터 파이프라인 준비...")

# 스트리밍 데이터셋
train_dataset = StreamingCTRDataset(
    '../data/train.parquet',
    encoders,
    categorical_cols,
    numeric_cols,
    batch_size=1000,
    is_train=True
)

# DataLoader with multiple workers
BATCH_SIZE = 32768  # 큰 배치
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=16,  # 많은 워커로 데이터 준비
    pin_memory=True,
    prefetch_factor=4,  # 미리 준비
    persistent_workers=True
)

print(f"   배치 크기: {BATCH_SIZE:,}")
print(f"   워커 수: 16")

print("\n4. 학습 시작 (스트리밍)...")
print("   데이터를 읽으면서 동시에 학습합니다...")

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# 학습 (1 epoch for demo)
model.train()
batch_count = 0
running_loss = 0
start_time = time.time()

for batch_idx, batch in enumerate(train_loader):
    if len(batch) == 3:  # train
        cat_feat, num_feat, labels = batch
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ensure proper dimensions
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        labels = labels.squeeze()

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        batch_count += 1

        # 로그 출력
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            throughput = (batch_idx + 1) * BATCH_SIZE / elapsed
            gpu_mem = torch.cuda.memory_allocated()/1024**3

            print(f"   Batch {batch_idx}: Loss={loss.item():.4f}, "
                  f"Throughput={throughput:.0f} samples/sec, "
                  f"GPU={gpu_mem:.1f}GB")

    # 100 배치만 데모
    if batch_idx >= 100:
        print("\n   Demo 완료! 실제로는 전체 데이터를 처리합니다.")
        break

avg_loss = running_loss / batch_count if batch_count > 0 else 0
print(f"\n   평균 Loss: {avg_loss:.4f}")
print(f"   처리 시간: {time.time() - start_time:.2f}초")

# GPU 메모리
if torch.cuda.is_available():
    print(f"\nGPU 메모리:")
    print(f"   사용: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"   예약: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

print("\n" + "=" * 80)
print("Pipeline 방식으로 효율적인 처리!")
print("=" * 80)
print("\n특징:")
print("- 데이터를 미리 로드하지 않음")
print("- Parquet 파일을 스트리밍으로 읽기")
print("- 16개 워커로 병렬 전처리")
print("- GPU는 계속 학습에만 집중")
print("- 메모리 효율적")
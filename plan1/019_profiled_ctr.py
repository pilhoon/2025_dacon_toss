#!/usr/bin/env python
"""Profiled CTR Model - 병목 지점 찾기"""

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Profiled CTR Model - Finding Bottlenecks")
print("=" * 80)

def timer(name):
    """시간 측정 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"   [{name}] 시간: {time.time() - start:.2f}초")
            return result
        return wrapper
    return decorator

# GPU 체크
start_total = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n=== 프로파일링 시작 ===\n")

# 1. 샘플링으로 컬럼 타입 확인
@timer("1. 컬럼 타입 확인 (샘플 1000개)")
def check_columns():
    sample = pd.read_parquet('../data/train.parquet', engine='pyarrow',
                            columns=None).head(1000)

    categorical_cols = sample.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = sample.select_dtypes(exclude=['object']).columns.tolist()

    for col in ['clicked', 'ID']:
        if col in categorical_cols:
            categorical_cols.remove(col)
        if col in numeric_cols:
            numeric_cols.remove(col)

    print(f"      범주형: {len(categorical_cols)}개, 수치형: {len(numeric_cols)}개")
    return categorical_cols, numeric_cols, sample

categorical_cols, numeric_cols, sample = check_columns()

# 2. 인코더 준비 (작은 샘플)
@timer("2. 인코더 준비 (10만 샘플)")
def prepare_encoders(categorical_cols):
    # 10만 샘플로 인코더 학습
    train_sample = pd.read_parquet('../data/train.parquet',
                                  columns=categorical_cols).sample(n=100000, random_state=42)
    test_sample = pd.read_parquet('../data/test.parquet',
                                 columns=categorical_cols).head(10000)

    encoders = {}
    cat_dims = []

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([
            train_sample[col].fillna('missing'),
            test_sample[col].fillna('missing')
        ])
        le.fit(combined)
        encoders[col] = le
        cat_dims.append(len(le.classes_) + 1)

    del train_sample, test_sample
    gc.collect()

    return encoders, cat_dims

encoders, cat_dims = prepare_encoders(categorical_cols)

# 3. 전체 데이터 로딩 (가장 느린 부분)
print("\n3. 데이터 로딩 (전체)")

@timer("   3.1 Train 데이터 로딩")
def load_train():
    return pd.read_parquet('../data/train.parquet', engine='pyarrow')

@timer("   3.2 Test 데이터 로딩")
def load_test():
    return pd.read_parquet('../data/test.parquet', engine='pyarrow')

train_df = load_train()
test_df = load_test()

print(f"      Train shape: {train_df.shape}")
print(f"      Test shape: {test_df.shape}")

# 4. 데이터 전처리
@timer("4. 전체 데이터 전처리")
def preprocess_data(df, encoders, categorical_cols, numeric_cols, is_train=True):
    # ID 처리
    ids = df['ID'].values if 'ID' in df.columns else None
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # 타겟 처리
    y = None
    if is_train and 'clicked' in df.columns:
        y = df['clicked'].values.astype(np.float32)
        df = df.drop('clicked', axis=1)

    # 범주형 인코딩
    print("      범주형 인코딩 중...")
    for i, col in enumerate(categorical_cols):
        df[col] = df[col].fillna('missing')
        # 벡터화된 변환
        df[col] = df[col].apply(
            lambda x: encoders[col].transform([x])[0]
            if x in encoders[col].classes_ else len(encoders[col].classes_)
        )
        if (i+1) % 2 == 0:
            print(f"         {i+1}/{len(categorical_cols)} 완료")

    # 수치형 처리
    print("      수치형 스케일링 중...")
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(np.float32)
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std() + 1e-8)

    return df, y, ids

train_processed, y_train, _ = preprocess_data(
    train_df, encoders, categorical_cols, numeric_cols, is_train=True
)

test_processed, _, test_ids = preprocess_data(
    test_df, encoders, categorical_cols, numeric_cols, is_train=False
)

del train_df, test_df
gc.collect()

# 5. PyTorch Dataset 생성
@timer("5. PyTorch Dataset 생성")
def create_datasets(train_processed, y_train, test_processed):
    class FastDataset(Dataset):
        def __init__(self, df, y=None):
            self.cat_features = df[categorical_cols].values.astype(np.int32)
            self.num_features = df[numeric_cols].values.astype(np.float32)
            self.y = y.astype(np.float32) if y is not None else None

        def __len__(self):
            return len(self.cat_features)

        def __getitem__(self, idx):
            if self.y is not None:
                return (torch.LongTensor(self.cat_features[idx]),
                       torch.FloatTensor(self.num_features[idx]),
                       torch.FloatTensor([self.y[idx]]))
            return (torch.LongTensor(self.cat_features[idx]),
                   torch.FloatTensor(self.num_features[idx]))

    # Train/Val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_processed, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    train_dataset = FastDataset(X_tr, y_tr)
    val_dataset = FastDataset(X_val, y_val)
    test_dataset = FastDataset(test_processed)

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = create_datasets(
    train_processed, y_train, test_processed
)

# 6. 모델 생성 (간단한 버전)
@timer("6. 모델 생성 및 GPU 전송")
def create_model(cat_dims, num_dim):
    class SimpleModel(nn.Module):
        def __init__(self, cat_dims, num_dim):
            super().__init__()

            # Embeddings
            self.embeddings = nn.ModuleList([
                nn.Embedding(dim, min(32, dim//2+1))
                for dim in cat_dims
            ])

            total_emb_dim = sum(min(32, dim//2+1) for dim in cat_dims)

            # Simple network
            self.fc = nn.Sequential(
                nn.Linear(total_emb_dim + num_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        def forward(self, cat_feat, num_feat):
            embeddings = [emb(cat_feat[:, i]) for i, emb in enumerate(self.embeddings)]
            emb_concat = torch.cat(embeddings, dim=1)
            combined = torch.cat([emb_concat, num_feat], dim=1)
            return torch.sigmoid(self.fc(combined)).squeeze()

    model = SimpleModel(cat_dims, len(numeric_cols)).to(device)
    return model

model = create_model(cat_dims, len(numeric_cols))

# 7. DataLoader 생성
@timer("7. DataLoader 생성")
def create_dataloaders(train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8192,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

# 8. 첫 배치 테스트
@timer("8. 첫 GPU 배치 처리")
def test_first_batch(model, train_loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    # 첫 배치만
    for cat_feat, num_feat, labels in train_loader:
        cat_feat = cat_feat.to(device)
        num_feat = num_feat.to(device)
        labels = labels.to(device).squeeze()

        optimizer.zero_grad()
        outputs = model(cat_feat, num_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"      Loss: {loss.item():.4f}")
        break

test_first_batch(model, train_loader)

# 총 시간
total_time = time.time() - start_total
print("\n" + "=" * 80)
print(f"총 시간: {total_time:.2f}초")
print("=" * 80)

print("\n=== 병목 분석 ===")
print("1. 데이터 로딩 (3.1, 3.2)이 가장 오래 걸림")
print("2. 데이터 전처리 (4)도 상당한 시간 소요")
print("3. GPU 사용 시작 (8)까지 대부분의 시간이 데이터 준비에 소요")
print("\n해결 방안:")
print("- Lazy loading: 데이터를 미리 다 로드하지 않고 필요할 때 읽기")
print("- 청크 처리: 데이터를 작은 단위로 나누어 처리")
print("- 병렬 전처리: 데이터 로딩과 전처리를 병렬화")
print("- 캐싱: 전처리된 데이터를 디스크에 저장")
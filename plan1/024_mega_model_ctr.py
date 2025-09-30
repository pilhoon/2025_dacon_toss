#!/usr/bin/env python
"""Mega Model CTR - 모델 크기 극대화로 GPU 메모리 최대 활용"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MEGA MODEL Deep CTR - Maximize Model Size for GPU Memory")
print("=" * 80)

device = torch.device('cuda')
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 1. 최소한의 데이터 준비
print("\n1. 빠른 데이터 준비...")
start_time = time.time()

# 샘플로 컬럼 타입만 확인
sample = pd.read_parquet('../data/train.parquet', columns=None).head(1000)
categorical_cols = [c for c in sample.select_dtypes(['object']).columns if c not in ['clicked', 'ID']]
numeric_cols = [c for c in sample.select_dtypes(['number']).columns if c not in ['clicked', 'ID']]
del sample

print(f"   범주형: {len(categorical_cols)}개, 수치형: {len(numeric_cols)}개")

# 최소 샘플로 인코더
encoders = {}
cat_dims = []
train_sample = pd.read_parquet('../data/train.parquet').sample(30000, random_state=42)
test_sample = pd.read_parquet('../data/test.parquet').head(3000)

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_sample[col].fillna('NA'), test_sample[col].fillna('NA')])
    le.fit(combined)
    encoders[col] = le
    cat_dims.append(len(le.classes_) + 1)

del train_sample, test_sample
gc.collect()
print(f"   시간: {time.time() - start_time:.1f}초")

# 2. 데이터 처리 (빠르게)
print("\n2. 데이터 로딩...")
start_time = time.time()

def fast_process(file_path, encoders, is_train=True):
    df = pd.read_parquet(file_path)

    ids = df.pop('ID').values if 'ID' in df else None
    y = df.pop('clicked').values.astype(np.float32) if is_train and 'clicked' in df else None

    # 범주형 - 빠른 변환
    for col in categorical_cols:
        df[col] = df[col].fillna('NA')
        mapping = {v: i for i, v in enumerate(encoders[col].classes_)}
        df[col] = df[col].map(mapping).fillna(len(encoders[col].classes_)).astype(np.int32)

    # 수치형 - 간단 정규화
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(np.float32)
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std() + 1e-8)

    return df, y, ids

train_df, y_train, _ = fast_process('../data/train.parquet', encoders, True)
test_df, _, test_ids = fast_process('../data/test.parquet', encoders, False)

print(f"   시간: {time.time() - start_time:.1f}초")

# 3. 초대형 모델 (GPU 메모리 최대 활용)
class MegaModel(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=512):  # 더 큰 임베딩
        super().__init__()

        # 거대한 임베딩
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        total_emb = sum(min(emb_dim, dim//2+1) for dim in cat_dims)

        # 초거대 네트워크 (GPU 메모리 채우기)
        print(f"\n   모델 구조:")
        print(f"   - 임베딩: {emb_dim}")
        print(f"   - 총 임베딩: {total_emb}")

        # Transformer-style blocks (메모리 집약적)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=8192,  # 4096 -> 8192 (2배)
                dropout=0.1,
                batch_first=True
            ) for _ in range(12)  # 6 -> 12개 레이어 (2배)
        ])

        # 거대한 Deep Network
        deep_input = total_emb + num_dim
        layers = []

        # 매우 큰 히든 레이어들 (2배 증가)
        hidden_sizes = [32768, 16384, 8192, 4096, 2048, 1024, 512, 256]
        prev_size = deep_input

        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden

        self.deep = nn.Sequential(*layers)

        # Wide Network (2배 크게)
        self.wide = nn.Sequential(
            nn.Linear(num_dim, 8192),
            nn.LayerNorm(8192),
            nn.GELU(),
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 1)
        )

        # Cross Network (2배 크게)
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_emb, 8192),
                nn.LayerNorm(8192),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(8192, 4096),
                nn.LayerNorm(4096),
                nn.GELU(),
                nn.Linear(4096, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Linear(2048, 1024)
            ) for _ in range(6)  # 3 -> 6개 (2배)
        ])

        # 거대한 Final (cross 6개로 수정)
        self.final = nn.Sequential(
            nn.Linear(256 + 6*1024 + 1, 16384),
            nn.LayerNorm(16384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(16384, 8192),
            nn.LayerNorm(8192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1)
        )

    def forward(self, cat_feat, num_feat):
        # Embeddings
        embs = [emb(cat_feat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_concat = torch.cat(embs, dim=1)

        # Transformer blocks (선택적 - 메모리 많이 사용)
        if emb_concat.size(1) == 512:  # 크기가 맞으면
            emb_transformed = emb_concat.unsqueeze(1)  # (B, 1, D)
            for transformer in self.transformer_blocks[:2]:  # 2개만 사용
                emb_transformed = transformer(emb_transformed)
            emb_concat = emb_transformed.squeeze(1)

        # Wide
        wide_out = self.wide(num_feat)

        # Deep
        deep_in = torch.cat([emb_concat, num_feat], dim=1)
        deep_out = self.deep(deep_in)

        # Cross
        cross_outs = [cross(emb_concat) for cross in self.cross_layers]
        cross_out = torch.cat(cross_outs, dim=1)

        # Final
        combined = torch.cat([deep_out, cross_out, wide_out], dim=1)
        return torch.sigmoid(self.final(combined)).squeeze()

# 4. 모델 생성
print("\n3. 초거대 모델 생성...")
model = MegaModel(cat_dims, len(numeric_cols)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"   총 파라미터: {total_params:,}")
print(f"   예상 GPU 메모리: ~{total_params * 4 / 1024**3:.1f} GB (FP32)")

# 5. Dataset
class FastDataset(Dataset):
    def __init__(self, df, y=None):
        self.cat_data = torch.LongTensor(df[categorical_cols].values)
        self.num_data = torch.FloatTensor(df[numeric_cols].values)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cat_data[idx], self.num_data[idx], self.y[idx]
        return self.cat_data[idx], self.num_data[idx]

# 6. 학습 준비
print("\n4. 학습 준비...")

X_tr, X_val, y_tr, y_val = train_test_split(
    train_df, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = FastDataset(X_tr, y_tr)
val_dataset = FastDataset(X_val, y_val)

# 큰 배치 크기 (GPU 메모리 더 많이 사용)
BATCH_SIZE = 32768  # 2배 증가
VAL_BATCH_SIZE = 65536  # 2배 증가

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
)

val_loader = DataLoader(
    val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True
)

print(f"   배치: 학습 {len(train_loader)}, 검증 {len(val_loader)}")

# 7. 학습
print("\n5. 초거대 모델 학습...")

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([52.0]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(5):  # 빠른 테스트
    model.train()
    epoch_start = time.time()

    for i, (cat_feat, num_feat, labels) in enumerate(train_loader):
        if i >= 50:  # 처음 50 배치만
            break

        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 10 == 0:
            gpu_mem = torch.cuda.memory_allocated()/1024**3
            gpu_reserved = torch.cuda.memory_reserved()/1024**3
            print(f"   Epoch {epoch+1}, Batch {i}: Loss={loss.item():.4f}, "
                  f"GPU={gpu_mem:.1f}/{gpu_reserved:.1f}GB")

    print(f"Epoch {epoch+1} 완료: {time.time()-epoch_start:.1f}초")
    print(f"최대 GPU 메모리: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

# 8. 예측 및 저장
print("\n6. 예측...")
model.eval()

test_dataset = FastDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, num_workers=8, pin_memory=True)

preds = []
with torch.no_grad():
    for cat_feat, num_feat in test_loader:
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)
        preds.extend(outputs.cpu().numpy())

preds = np.array(preds)

# 9. 저장
print("\n7. 저장...")
print(f"   예측 평균: {preds.mean():.4f}")
print(f"   예측 표준편차: {preds.std():.4f}")

submission = pd.DataFrame({'ID': test_ids, 'clicked': preds})
submission.to_csv('024_mega_model_submission.csv', index=False)

print("\n" + "=" * 80)
print(f"최종 GPU 메모리: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB / 80 GB")
print("제출 파일: 024_mega_model_submission.csv")
print("=" * 80)
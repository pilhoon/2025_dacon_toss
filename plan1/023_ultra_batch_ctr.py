#!/usr/bin/env python
"""Ultra Batch CTR - 초대형 배치로 GPU 메모리 최대 활용"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ULTRA BATCH Deep CTR - Maximum Memory via Batch Size")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 1. 빠른 인코더 준비
print("\n1. 인코더 준비...")
start_time = time.time()

sample = pd.read_parquet('../data/train.parquet').head(5000)
categorical_cols = sample.select_dtypes(include=['object']).columns.tolist()
numeric_cols = sample.select_dtypes(exclude=['object']).columns.tolist()

for col in ['clicked', 'ID']:
    if col in categorical_cols: categorical_cols.remove(col)
    if col in numeric_cols: numeric_cols.remove(col)

print(f"   범주형: {len(categorical_cols)}개, 수치형: {len(numeric_cols)}개")

# 인코더 (최소 샘플)
train_sample = pd.read_parquet('../data/train.parquet').sample(n=50000, random_state=42)
test_sample = pd.read_parquet('../data/test.parquet').head(5000)

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

del sample, train_sample, test_sample
gc.collect()
print(f"   시간: {time.time() - start_time:.1f}초")

# 2. 데이터 처리 (간소화)
print("\n2. 데이터 로딩 및 처리...")
start_time = time.time()

def process_data(file_path, encoders, is_train=True):
    df = pd.read_parquet(file_path)

    ids = df['ID'].values if 'ID' in df.columns else None
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    y = None
    if is_train and 'clicked' in df.columns:
        y = df['clicked'].values.astype(np.float32)
        df = df.drop('clicked', axis=1)

    for col in categorical_cols:
        df[col] = df[col].fillna('missing')
        mapping = {v: i for i, v in enumerate(encoders[col].classes_)}
        default = len(encoders[col].classes_)
        df[col] = df[col].map(mapping).fillna(default).astype(np.int32)

    df[numeric_cols] = df[numeric_cols].fillna(0).astype(np.float32)

    # 간단한 정규화
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std() + 1e-8
        df[col] = (df[col] - mean_val) / std_val

    return df, y, ids

train_df, y_train, _ = process_data('../data/train.parquet', encoders, True)
test_df, _, test_ids = process_data('../data/test.parquet', encoders, False)

print(f"   시간: {time.time() - start_time:.1f}초")
print(f"   클릭률: {y_train.mean():.4f}")

# 3. 초대형 배치를 위한 모델 (큰 모델 유지)
class UltraBatchCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=256):
        super().__init__()

        # 큰 Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        total_emb_dim = sum(min(emb_dim, dim//2+1) for dim in cat_dims)
        print(f"\n   모델 구조:")
        print(f"   - 임베딩 차원: {emb_dim}")
        print(f"   - 총 임베딩 크기: {total_emb_dim}")

        # Wide Network
        self.wide = nn.Sequential(
            nn.Linear(num_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )

        # Deep Network (큰 네트워크 유지)
        deep_input_dim = total_emb_dim + num_dim
        hidden_dims = [4096, 2048, 1024, 512, 256]

        print(f"   - Deep layers: {hidden_dims}")

        layers = []
        prev_dim = deep_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.deep = nn.Sequential(*layers)

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(total_emb_dim, total_emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_emb_dim * 2, total_emb_dim),
            nn.Sigmoid()
        )

        # Cross Network
        self.cross = nn.Sequential(
            nn.Linear(total_emb_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        )

        # Final
        final_input = 256 + 256 + 1
        self.final = nn.Sequential(
            nn.Linear(final_input, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, cat_feat, num_feat):
        # Embeddings
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(cat_feat[:, i]))
        emb_concat = torch.cat(embeddings, dim=1)

        # Attention
        weights = self.attention(emb_concat)
        emb_attended = emb_concat * weights

        # Wide
        wide_out = self.wide(num_feat)

        # Deep
        deep_input = torch.cat([emb_attended, num_feat], dim=1)
        deep_out = self.deep(deep_input)

        # Cross
        cross_out = self.cross(emb_attended)

        # Final
        combined = torch.cat([deep_out, cross_out, wide_out], dim=1)
        return torch.sigmoid(self.final(combined)).squeeze()

# 4. 모델 생성
print("\n3. 모델 생성...")
model = UltraBatchCTR(cat_dims, len(numeric_cols)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"   총 파라미터: {total_params:,}")
print(f"   모델 메모리: ~{total_params * 4 / 1024**3:.1f} GB")

# 5. 데이터셋 (메모리 효율적)
class FastDataset(Dataset):
    def __init__(self, df, y=None):
        # float16으로 저장하여 메모리 절약
        self.cat_data = torch.LongTensor(df[categorical_cols].values)
        self.num_data = torch.HalfTensor(df[numeric_cols].values)  # float16
        self.y = torch.HalfTensor(y) if y is not None else None  # float16

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cat_data[idx], self.num_data[idx], self.y[idx]
        return self.cat_data[idx], self.num_data[idx]

# 6. 학습 준비
print("\n4. 초대형 배치 준비...")

X_tr, X_val, y_tr, y_val = train_test_split(
    train_df, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = FastDataset(X_tr, y_tr)
val_dataset = FastDataset(X_val, y_val)

# 초대형 배치 (GPU 메모리 최대 활용)
BATCH_SIZE = 262144  # 256K - 4배 증가!
VAL_BATCH_SIZE = 524288  # 512K - 4배 증가!

print(f"   초대형 배치 크기:")
print(f"   - 학습: {BATCH_SIZE:,} ({BATCH_SIZE/1024:.0f}K)")
print(f"   - 검증: {VAL_BATCH_SIZE:,} ({VAL_BATCH_SIZE/1024:.0f}K)")

# Gradient Accumulation 설정
ACCUMULATION_STEPS = 4  # 그래디언트 누적
EFFECTIVE_BATCH = BATCH_SIZE * ACCUMULATION_STEPS
print(f"   - 유효 배치 크기: {EFFECTIVE_BATCH:,} ({EFFECTIVE_BATCH/1024:.0f}K)")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=12,  # 더 많은 워커
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True
)

print(f"   배치 수: 학습 {len(train_loader)}, 검증 {len(val_loader)}")

# 7. Optimizer
pos_weight = torch.tensor([(1-y_train.mean())/y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 큰 배치에 맞는 학습률
base_lr = 0.001
lr = base_lr * np.sqrt(EFFECTIVE_BATCH / 32768)  # Linear scaling
print(f"   조정된 학습률: {lr:.5f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr*3, epochs=10, steps_per_epoch=len(train_loader)//ACCUMULATION_STEPS
)

# 8. 학습
print("\n5. 초대형 배치 학습...")
print("   GPU 메모리를 최대한 활용합니다...")

scaler = torch.cuda.amp.GradScaler()

for epoch in range(10):
    model.train()
    train_loss = 0
    epoch_start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (cat_feat, num_feat, labels) in enumerate(train_loader):
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True).float()  # float32로 변환
        labels = labels.to(device, non_blocking=True).float()  # float32로 변환

        # Mixed precision으로 메모리 절약
        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)
            loss = criterion(outputs, labels) / ACCUMULATION_STEPS  # 누적을 위한 스케일링

        scaler.scale(loss).backward()

        # Gradient Accumulation
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        train_loss += loss.item() * ACCUMULATION_STEPS

        if batch_idx % 2 == 0:
            gpu_mem_alloc = torch.cuda.memory_allocated()/1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved()/1024**3
            print(f"   Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={loss.item()*ACCUMULATION_STEPS:.4f}, "
                  f"GPU={gpu_mem_alloc:.1f}/{gpu_mem_reserved:.1f}GB")

    # Validation (매 2 에포크)
    if epoch % 2 == 0:
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for cat_feat, num_feat, labels in val_loader:
                cat_feat = cat_feat.to(device, non_blocking=True)
                num_feat = num_feat.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast():
                    outputs = model(cat_feat, num_feat)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                # 메모리 절약을 위해 배치로 처리
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels_list, val_preds)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}: Val Loss={val_loss/len(val_loader):.4f}, "
              f"Val AUC={val_auc:.4f}, Time={time.time()-epoch_start:.1f}s")
        print(f"Max GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
        print(f"Reserved GPU Memory: {torch.cuda.memory_reserved()/1024**3:.1f} GB")
        print(f"{'='*60}\n")

        torch.save(model.state_dict(), '023_ultra_batch_model.pth')
        print("✅ Model saved!")

# 9. 예측
print("\n6. 최종 예측...")
model.eval()

test_dataset = FastDataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=12,
    pin_memory=True
)

predictions = []
with torch.no_grad():
    for cat_feat, num_feat in test_loader:
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)

        predictions.extend(outputs.cpu().numpy())

predictions = np.array(predictions)

# 10. 저장
print("\n7. 결과 저장...")
print(f"   예측 평균: {predictions.mean():.4f}")
print(f"   예측 표준편차: {predictions.std():.4f}")

submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': predictions
})

submission.to_csv('023_ultra_batch_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print(f"최대 GPU 메모리 사용: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
print(f"예약 GPU 메모리: {torch.cuda.memory_reserved()/1024**3:.1f} GB")
print("제출 파일: 023_ultra_batch_submission.csv")
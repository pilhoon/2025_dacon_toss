#!/usr/bin/env python
"""Optimized Deep CTR - 빠른 시작, 큰 모델"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Optimized Deep CTR Model - Fast Start, Big Model")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. 빠른 데이터 준비 (샘플링)
print("\n1. 빠른 전처리 준비...")
start_time = time.time()

# 작은 샘플로 인코더 준비
train_sample = pd.read_parquet('../data/train.parquet').sample(n=50000, random_state=42)
test_sample = pd.read_parquet('../data/test.parquet').head(5000)

# 컬럼 타입 확인
categorical_cols = train_sample.select_dtypes(include=['object']).columns.tolist()
numeric_cols = train_sample.select_dtypes(exclude=['object']).columns.tolist()

for col in ['clicked', 'ID']:
    if col in categorical_cols:
        categorical_cols.remove(col)
    if col in numeric_cols:
        numeric_cols.remove(col)

print(f"   범주형: {len(categorical_cols)}개, 수치형: {len(numeric_cols)}개")

# 인코더 준비 (빠르게)
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

print(f"   준비 시간: {time.time() - start_time:.2f}초")

# 2. 전체 데이터 처리 (최적화)
print("\n2. 전체 데이터 처리...")
start_time = time.time()

# 청크로 처리
def process_data_fast(file_path, encoders, categorical_cols, numeric_cols, is_train=True):
    """빠른 데이터 처리"""
    df = pd.read_parquet(file_path)

    # ID/타겟 처리
    test_ids = None
    if 'ID' in df.columns:
        test_ids = df['ID'].values
        df = df.drop('ID', axis=1)

    y = None
    if is_train and 'clicked' in df.columns:
        y = df['clicked'].values.astype(np.float32)
        df = df.drop('clicked', axis=1)

    # 범주형 빠른 인코딩
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')
        # 더 빠른 map 사용
        mapping = {v: i for i, v in enumerate(encoders[col].classes_)}
        default = len(encoders[col].classes_)
        df[col] = df[col].map(mapping).fillna(default).astype(np.int32)

    # 수치형 처리
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(np.float32)

    return df, y, test_ids

# 처리
train_df, y_train, _ = process_data_fast(
    '../data/train.parquet', encoders, categorical_cols, numeric_cols, is_train=True
)

test_df, _, test_ids = process_data_fast(
    '../data/test.parquet', encoders, categorical_cols, numeric_cols, is_train=False
)

print(f"   처리 시간: {time.time() - start_time:.2f}초")
print(f"   클릭률: {y_train.mean():.4f}")

# 수치형 스케일링 (간단하게)
for col in numeric_cols:
    mean_val = train_df[col].mean()
    std_val = train_df[col].std() + 1e-8
    train_df[col] = (train_df[col] - mean_val) / std_val
    test_df[col] = (test_df[col] - mean_val) / std_val

# 3. 효율적인 Dataset
class FastDataset(Dataset):
    def __init__(self, df, y=None, categorical_cols=None, numeric_cols=None):
        self.cat_data = torch.LongTensor(df[categorical_cols].values)
        self.num_data = torch.FloatTensor(df[numeric_cols].values)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cat_data[idx], self.num_data[idx], self.y[idx]
        return self.cat_data[idx], self.num_data[idx]

# 4. 큰 모델 (성능 중시)
class BigDeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=64):
        super().__init__()

        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        total_emb_dim = sum(min(emb_dim, dim//2+1) for dim in cat_dims)

        # Wide & Deep
        self.wide = nn.Linear(num_dim, 1)

        # Deep part - 큰 네트워크
        self.deep = nn.Sequential(
            nn.Linear(total_emb_dim + num_dim, 2048),
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

        # Final
        self.final = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, cat_features, num_features):
        # Embeddings
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_features[:, i]))
        emb_concat = torch.cat(embeddings, dim=1)

        # Wide & Deep
        wide_out = self.wide(num_features)
        deep_input = torch.cat([emb_concat, num_features], dim=1)
        deep_out = self.deep(deep_input)

        # Combine
        combined = torch.cat([deep_out, wide_out], dim=1)
        return torch.sigmoid(self.final(combined)).squeeze()

print("\n3. 모델 초기화...")
model = BigDeepCTR(cat_dims, len(numeric_cols)).to(device)
print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")

# 5. 학습 준비
print("\n4. 학습 준비...")

# Train/Val split
X_tr, X_val, y_tr, y_val = train_test_split(
    train_df, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = FastDataset(X_tr, y_tr, categorical_cols, numeric_cols)
val_dataset = FastDataset(X_val, y_val, categorical_cols, numeric_cols)

# 큰 배치
BATCH_SIZE = 32768
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE*2,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"   배치 수: 학습 {len(train_loader)}, 검증 {len(val_loader)}")

# Loss and Optimizer
pos_weight = torch.tensor([(1-y_train.mean())/y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.003,
    epochs=15,
    steps_per_epoch=len(train_loader)
)

# 6. 학습
print("\n5. 모델 학습...")
best_val_loss = float('inf')
patience_counter = 0

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

for epoch in range(15):
    # Training
    model.train()
    train_loss = 0
    epoch_start = time.time()

    for batch_idx, (cat_feat, num_feat, labels) in enumerate(train_loader):
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            gpu_mem = torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0
            print(f"   Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, GPU={gpu_mem:.1f}GB")

    # Validation
    if epoch % 2 == 0:
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for cat_feat, num_feat, labels in val_loader:
                cat_feat = cat_feat.to(device, non_blocking=True)
                num_feat = num_feat.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(cat_feat, num_feat)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels_list, val_preds)

        print(f"\nEpoch {epoch+1}: Val Loss={avg_val_loss:.4f}, "
              f"Val AUC={val_auc:.4f}, Time={time.time()-epoch_start:.1f}s")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '021_best_model.pth')
            patience_counter = 0
            print(f"✅ Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping!")
                break

# 7. 예측
print("\n6. 최종 예측...")
model.load_state_dict(torch.load('021_best_model.pth'))
model.eval()

test_dataset = FastDataset(test_df, None, categorical_cols, numeric_cols)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE*2,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        cat_feat, num_feat = batch
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)

        predictions.extend(outputs.cpu().numpy())

predictions = np.array(predictions)

# 8. 결과 저장
print("\n7. 결과 분석...")
print(f"   예측 평균: {predictions.mean():.4f}")
print(f"   예측 표준편차: {predictions.std():.4f}")
print(f"   예측 범위: [{predictions.min():.6f}, {predictions.max():.6f}]")

submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': predictions
})

submission.to_csv('021_optimized_deep_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print("\n제출 파일: 021_optimized_deep_submission.csv")

if torch.cuda.is_available():
    print(f"최종 GPU 메모리: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
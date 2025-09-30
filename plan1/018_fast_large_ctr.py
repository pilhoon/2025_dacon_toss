#!/usr/bin/env python
"""Fast Large CTR Model - 큰 모델, 효율적인 데이터 처리"""
#   주요 특징:
#   1. 큰 모델 유지:
#     - Deep layers: [4096, 2048, 1024, 512, 256]
#     - 임베딩 크기: 128
#     - Attention + Cross features
#   2. 효율적인 데이터 처리:
#     - 샘플링으로 빠른 인코더 준비
#     - numpy arrays 사용 (메모리 효율)
#     - 벡터화된 전처리
#     - gc.collect()로 메모리 관리
#   3. 최적화:
#     - Mixed precision training
#     - Gradient clipping
#     - 큰 배치 크기 (32768/65536)
#     - 8 workers로 데이터 로딩
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import gc
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Fast Large CTR Model - Big Model, Efficient Processing")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"\n사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 1. 빠른 인코더 준비 (샘플링)
print("\n1. 인코더 준비 (샘플링)...")
start_time = time.time()

# 작은 샘플로 컬럼 타입과 인코더 준비
sample_df = pd.read_parquet('../data/train.parquet', engine='pyarrow').sample(n=100000, random_state=42)
test_sample = pd.read_parquet('../data/test.parquet', engine='pyarrow').head(10000)

# 컬럼 분리
categorical_cols = sample_df.select_dtypes(include=['object']).columns.tolist()
if 'clicked' in categorical_cols:
    categorical_cols.remove('clicked')
if 'ID' in categorical_cols:
    categorical_cols.remove('ID')

numeric_cols = sample_df.select_dtypes(exclude=['object']).columns.tolist()
if 'clicked' in numeric_cols:
    numeric_cols.remove('clicked')
if 'ID' in numeric_cols:
    numeric_cols.remove('ID')

print(f"   범주형: {len(categorical_cols)}개, 수치형: {len(numeric_cols)}개")

# 인코더 준비
encoders = {}
cat_dims = []
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([sample_df[col].fillna('missing'), test_sample[col].fillna('missing')])
    le.fit(combined)
    encoders[col] = le
    cat_dims.append(len(le.classes_) + 1)  # +1 for unknown

# 샘플 메모리 해제
del sample_df, test_sample
gc.collect()

print(f"   준비 시간: {time.time() - start_time:.2f}초")

# 2. 데이터 로딩 및 빠른 전처리
print("\n2. 데이터 로딩 및 전처리...")
start_time = time.time()

# 청크로 읽고 바로 변환
def process_data(file_path, encoders, is_train=True):
    df = pd.read_parquet(file_path, engine='pyarrow')

    # ID 저장 및 제거
    ids = None
    if 'ID' in df.columns:
        ids = df['ID'].values
        df = df.drop('ID', axis=1)

    # 타겟 분리
    y = None
    if is_train and 'clicked' in df.columns:
        y = df['clicked'].values.astype(np.float32)
        df = df.drop('clicked', axis=1)

    # 범주형 인코딩 (벡터화)
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')
        # Unknown 처리
        df[col] = df[col].apply(lambda x: encoders[col].transform([x])[0]
                                if x in encoders[col].classes_ else len(encoders[col].classes_))

    # 수치형 처리
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(np.float32)

    # 간단한 스케일링
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std() + 1e-8)

    return df, y, ids

# 학습 데이터 처리
train_df, y_train, _ = process_data('../data/train.parquet', encoders, is_train=True)
test_df, _, test_ids = process_data('../data/test.parquet', encoders, is_train=False)

print(f"   처리 시간: {time.time() - start_time:.2f}초")
print(f"   학습: {train_df.shape}, 테스트: {test_df.shape}")

# PyTorch Dataset (메모리 효율적)
class FastCTRDataset(Dataset):
    def __init__(self, df, y=None):
        # numpy arrays로 저장 (메모리 효율)
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

# LARGE Deep CTR Model (성능 중심)
class LargeDeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=128):
        super().__init__()

        # Large embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        self.cat_emb_dims = [min(emb_dim, dim//2+1) for dim in cat_dims]
        self.total_emb_dim = sum(self.cat_emb_dims)

        print(f"\n   모델 구조:")
        print(f"   - 총 임베딩 차원: {self.total_emb_dim}")

        # Large Wide part
        self.wide = nn.Sequential(
            nn.Linear(num_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        # Very Large Deep part
        deep_input_dim = self.total_emb_dim + num_dim
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

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.total_emb_dim, self.total_emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.total_emb_dim * 2, self.total_emb_dim),
            nn.Sigmoid()
        )

        # Cross features
        self.cross = nn.Sequential(
            nn.Linear(self.total_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # Final ensemble
        self.final = nn.Sequential(
            nn.Linear(256 + 256 + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, cat_features, num_features):
        # Embeddings
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_features[:, i]))

        emb_concat = torch.cat(embeddings, dim=1)

        # Attention
        attention_weights = self.attention(emb_concat)
        emb_attended = emb_concat * attention_weights

        # Wide
        wide_out = self.wide(num_features)

        # Deep
        deep_input = torch.cat([emb_attended, num_features], dim=1)
        deep_out = self.deep(deep_input)

        # Cross
        cross_out = self.cross(emb_attended)

        # Final
        combined = torch.cat([deep_out, cross_out, wide_out], dim=1)
        output = torch.sigmoid(self.final(combined))

        return output.squeeze()

print("\n3. 모델 초기화...")
model = LargeDeepCTR(cat_dims, len(numeric_cols)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"   총 파라미터: {total_params:,}")
print(f"   예상 GPU 메모리: ~{total_params * 4 / 1024**3:.1f} GB")

# Loss and Optimizer
pos_weight = torch.tensor([(1-y_train.mean())/y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

print("\n4. 학습 준비...")

# Train/Val split
X_tr, X_val, y_tr, y_val = train_test_split(
    train_df, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = FastCTRDataset(X_tr, y_tr)
val_dataset = FastCTRDataset(X_val, y_val)

# 큰 배치 크기
BATCH_SIZE = 32768
VAL_BATCH_SIZE = 65536

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

print(f"   학습 배치: {len(train_loader)} (배치 크기: {BATCH_SIZE:,})")
print(f"   검증 배치: {len(val_loader)} (배치 크기: {VAL_BATCH_SIZE:,})")

print("\n5. 모델 학습...")
best_val_auc = 0
patience_counter = 0
max_patience = 10

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

for epoch in range(30):
    # Training
    model.train()
    train_loss = 0
    train_batches = 0

    epoch_start = time.time()

    for batch_idx, (cat_feat, num_feat, labels) in enumerate(train_loader):
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).squeeze()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        train_batches += 1

        # Progress
        if batch_idx % 10 == 0:
            gpu_mem = torch.cuda.memory_allocated()/1024**3
            print(f"   Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, GPU={gpu_mem:.1f}GB")

    # Validation (매 2 에포크)
    if epoch % 2 == 0:
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for cat_feat, num_feat, labels in val_loader:
                cat_feat = cat_feat.to(device, non_blocking=True)
                num_feat = num_feat.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).squeeze()

                with torch.cuda.amp.autocast():
                    outputs = model(cat_feat, num_feat)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds)

        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val AUC={val_auc:.4f}")
        print(f"Time: {time.time() - epoch_start:.1f}s")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), '018_best_model.pth')
            patience_counter = 0
            print(f"✅ Best model saved! AUC={val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping! Best AUC={best_val_auc:.4f}")
                break
    else:
        avg_train_loss = train_loss / train_batches
        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Time={time.time() - epoch_start:.1f}s")

print("\n6. 최종 예측...")

# Load best model
model.load_state_dict(torch.load('018_best_model.pth'))
model.eval()

test_dataset = FastCTRDataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        if len(batch) == 2:
            cat_feat, num_feat = batch
            cat_feat = cat_feat.to(device, non_blocking=True)
            num_feat = num_feat.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(cat_feat, num_feat)

            predictions.extend(torch.sigmoid(outputs).cpu().numpy())

predictions = np.array(predictions)

print("\n7. 결과 분석...")
print(f"\n예측 통계:")
print(f"   평균: {predictions.mean():.4f}")
print(f"   표준편차: {predictions.std():.4f}")
print(f"   최소: {predictions.min():.6f}")
print(f"   최대: {predictions.max():.6f}")
print(f"   >0.5: {(predictions > 0.5).sum():,}개 ({100*(predictions > 0.5).mean():.2f}%)")

# GPU 메모리
if torch.cuda.is_available():
    print(f"\nGPU 메모리:")
    print(f"   사용: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"   최대: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# 제출 파일
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': predictions
})

submission.to_csv('018_fast_large_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print(f"\n제출 파일: 018_fast_large_submission.csv")
print(f"Best Val AUC: {best_val_auc:.4f}")

if predictions.std() > 0.1 and 0.01 < predictions.mean() < 0.05:
    print("\n✅ 균형잡힌 예측! 0.349 돌파 기대")
else:
    print(f"\n⚠️  예측 분포 확인 필요")
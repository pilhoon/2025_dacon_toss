#!/usr/bin/env python
"""Huge Deep CTR Model - A100 80GB 최대 활용"""

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
print("HUGE Deep CTR Model - Maximum GPU Utilization")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"\n사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 데이터 로딩
print("\n1. 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
test_df = pd.read_parquet('../data/test.parquet')
print(f"   학습: {train_df.shape}, 테스트: {test_df.shape}")

# ID 저장
test_ids = test_df['ID'].copy()

# ID 제거
for df in [train_df, test_df]:
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)

# 타겟 분리
y_train = train_df['clicked'].values
X_train = train_df.drop('clicked', axis=1)
X_test = test_df

print(f"   클릭률: {y_train.mean():.4f}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

print("\n2. 피처 엔지니어링...")

# 범주형과 수치형 분리
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

print(f"   범주형: {len(categorical_cols)}개")
print(f"   수치형: {len(numeric_cols)}개")

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

print(f"   최종 피처 수: {X_train.shape[1]}")

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

# HUGE Deep CTR Model
class HugeDeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=128):  # 임베딩 크기 4배 증가
        super().__init__()

        # Larger embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        self.cat_emb_dims = [min(emb_dim, dim//2+1) for dim in cat_dims]
        self.total_emb_dim = sum(self.cat_emb_dims)

        print(f"   각 임베딩 차원: {self.cat_emb_dims}")
        print(f"   총 임베딩 차원: {self.total_emb_dim}")

        # Huge Wide part
        self.wide = nn.Sequential(
            nn.Linear(num_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Huge Deep part - 훨씬 큰 네트워크
        deep_input_dim = self.total_emb_dim + num_dim
        hidden_dims = [4096, 2048, 1024, 512, 256, 128]  # 매우 큰 레이어들

        print(f"   Deep 입력 차원: {deep_input_dim}")
        print(f"   Deep 히든 레이어: {hidden_dims}")

        layers = []
        prev_dim = deep_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.deep = nn.Sequential(*layers)
        self.deep_output_dim = prev_dim

        # Transformer-like attention layers (여러 개)
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_emb_dim, self.total_emb_dim * 2),
                nn.ReLU(),
                nn.Linear(self.total_emb_dim * 2, self.total_emb_dim),
                nn.Sigmoid()
            ) for _ in range(3)  # 3개의 attention layers
        ])

        # Multiple Cross features
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_emb_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256)
            ),
            nn.Sequential(
                nn.Linear(self.total_emb_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256)
            )
        ])

        # Huge Final ensemble
        # deep_output_dim(128) + cross_outputs(256+256) + wide_output(1)
        final_input_dim = self.deep_output_dim + 512 + 1
        print(f"   Final 입력 차원: {final_input_dim}")

        self.final = nn.Sequential(
            nn.Linear(final_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, cat_features, num_features):
        batch_size = cat_features.size(0)

        # Embedding layer
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_features[:, i]))

        emb_concat = torch.cat(embeddings, dim=1)

        # Multiple attention layers
        emb_attended = emb_concat
        for attention_layer in self.attention_layers:
            attention_weights = attention_layer(emb_attended)
            emb_attended = emb_attended * attention_weights

        # Wide part
        wide_out = self.wide(num_features)

        # Deep part
        deep_input = torch.cat([emb_attended, num_features], dim=1)
        deep_out = self.deep(deep_input)

        # Multiple Cross features
        cross_outs = []
        for cross_layer in self.cross_layers:
            cross_outs.append(cross_layer(emb_attended))
        cross_out = torch.cat(cross_outs, dim=1)

        # Final ensemble
        combined = torch.cat([deep_out, cross_out, wide_out], dim=1)
        output = torch.sigmoid(self.final(combined))

        return output.squeeze()

print("\n3. 모델 초기화...")
model = HugeDeepCTR(cat_dims, len(numeric_cols)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"   모델 파라미터: {total_params:,}")
print(f"   예상 메모리: ~{total_params * 4 / 1024**3:.1f} GB (FP32)")

# Loss and Optimizer
pos_weight = torch.tensor([(1-y_train.mean())/y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 더 강한 정규화
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=20,
    steps_per_epoch=100,  # 대략적인 배치 수
    pct_start=0.3
)

print("\n4. 학습 준비...")
# Train/Validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = CTRDataset(X_tr, y_tr)
val_dataset = CTRDataset(X_val, y_val)

# 매우 큰 배치 크기로 GPU 최대 활용
BATCH_SIZE = 65536  # 4배 증가
VAL_BATCH_SIZE = 131072  # 4배 증가

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=12,  # 더 많은 워커
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2  # 프리페칭
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

print(f"   학습 배치: {len(train_loader)} (배치 크기: {BATCH_SIZE:,})")
print(f"   검증 배치: {len(val_loader)} (배치 크기: {VAL_BATCH_SIZE:,})")

print("\n5. 모델 학습...")
best_val_auc = 0
patience_counter = 0
max_patience = 5  # 더 짧게

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

for epoch in range(20):
    # Training
    model.train()
    train_loss = 0
    train_batches = 0

    for batch_idx, (cat_feat, num_feat, labels) in enumerate(train_loader):
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # 메모리 효율적

        # Mixed precision training
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

        # 5 배치마다 출력
        if batch_idx % 5 == 0:
            current_lr = scheduler.get_last_lr()[0]
            gpu_mem = torch.cuda.memory_allocated()/1024**3
            print(f"   Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}, "
                  f"GPU: {gpu_mem:.1f}GB")

    # Validation (매 에포크)
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for cat_feat, num_feat, labels in val_loader:
            cat_feat = cat_feat.to(device, non_blocking=True)
            num_feat = num_feat.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(cat_feat, num_feat)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_train_loss = train_loss / train_batches
    avg_val_loss = val_loss / len(val_loader)

    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(val_labels, val_preds)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB / "
          f"{torch.cuda.memory_reserved()/1024**3:.1f}GB")
    print(f"{'='*60}")

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), '016_best_huge_model.pth')
        patience_counter = 0
        print(f"   ✅ Best model saved! AUC: {val_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping! Best AUC: {best_val_auc:.4f}")
            break

print("\n6. 최종 예측...")
# Load best model
model.load_state_dict(torch.load('016_best_huge_model.pth'))
model.eval()

test_dataset = CTRDataset(X_test)
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
        num_feat = num_feat.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)

        predictions.extend(torch.sigmoid(outputs).cpu().numpy())

predictions = np.array(predictions)

print("\n7. 결과 분석...")
print(f"\n예측 확률 통계:")
print(f"   평균: {predictions.mean():.4f}")
print(f"   표준편차: {predictions.std():.4f}")
print(f"   최소: {predictions.min():.6f}")
print(f"   최대: {predictions.max():.6f}")
print(f"   >0.5: {(predictions > 0.5).sum():,}개 ({100*(predictions > 0.5).mean():.2f}%)")

# 최종 GPU 메모리 사용량
if torch.cuda.is_available():
    print(f"\n최종 GPU 메모리:")
    print(f"   사용: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"   예약: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"   최대: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': predictions
})

submission.to_csv('016_huge_deep_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print("\n제출 파일: 016_huge_deep_submission.csv")
print(f"Best Validation AUC: {best_val_auc:.4f}")

if predictions.std() > 0.1 and 0.01 < predictions.mean() < 0.05:
    print("\n✅ 균형잡힌 대규모 딥러닝 예측! 0.349 돌파 기대")
else:
    print(f"\n⚠️  예측 분포 확인 필요")
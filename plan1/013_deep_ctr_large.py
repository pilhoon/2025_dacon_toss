#!/usr/bin/env python
"""Large Deep CTR Model - GPU 최대 활용"""

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
print("Large Deep CTR Model - A100 80GB 최대 활용")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True  # GPU 최적화

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

# Large Deep CTR Model
class LargeDeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=64, hidden_dims=[2048, 1024, 512, 256, 128]):
        super().__init__()

        # Larger embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        self.total_emb_dim = sum(min(emb_dim, dim//2+1) for dim in cat_dims)

        # Multi-head attention for embeddings
        self.attention_heads = 4
        self.attention_dim = (self.total_emb_dim // self.attention_heads) * self.attention_heads

        # Wide part
        self.wide = nn.Sequential(
            nn.Linear(num_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Deep part - Much larger network
        deep_input_dim = self.attention_dim + num_dim
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

        # Attention projection layer
        self.attention_proj = nn.Linear(self.total_emb_dim, self.attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads,
            batch_first=True
        )

        # Cross features
        self.cross = nn.Sequential(
            nn.Linear(self.attention_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # Final ensemble
        self.final = nn.Sequential(
            nn.Linear(prev_dim + 256 + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
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

        # Apply multi-head attention
        emb_proj = self.attention_proj(emb_concat)  # Project to attention_dim
        emb_reshaped = emb_proj.unsqueeze(1)  # (batch, 1, attention_dim)
        attended, _ = self.attention(emb_reshaped, emb_reshaped, emb_reshaped)
        emb_attended = attended.squeeze(1)

        # Wide part
        wide_out = self.wide(num_features)

        # Deep part - use emb_proj + num_features for correct dimension
        deep_input = torch.cat([emb_proj, num_features], dim=1)
        deep_out = self.deep(deep_input)

        # Cross features - use attention output
        cross_out = self.cross(emb_attended)

        # Final ensemble
        combined = torch.cat([deep_out, cross_out, wide_out], dim=1)
        output = torch.sigmoid(self.final(combined))

        return output.squeeze()

print("\n3. 모델 초기화...")
model = LargeDeepCTR(cat_dims, len(numeric_cols)).to(device)
print(f"   모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

# Loss and Optimizer
# Weighted loss for class imbalance
pos_weight = torch.tensor([(1-y_train.mean())/y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 더 정교한 optimizer 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

print("\n4. 학습 준비...")
# Train/Validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = CTRDataset(X_tr, y_tr)
val_dataset = CTRDataset(X_val, y_val)

# 대폭 늘린 배치 크기 - GPU 메모리 최대 활용
BATCH_SIZE = 32768  # 4096 -> 32768 (8배)
VAL_BATCH_SIZE = 65536  # 검증은 더 크게

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,  # 더 많은 워커
    pin_memory=True,  # GPU 전송 최적화
    persistent_workers=True  # 워커 재사용
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

# Mixed precision training for faster computation
scaler = torch.cuda.amp.GradScaler()

for epoch in range(50):  # 더 많은 에포크
    # Training
    model.train()
    train_loss = 0
    train_batches = 0

    for cat_feat, num_feat, labels in train_loader:
        cat_feat = cat_feat.to(device, non_blocking=True)
        num_feat = num_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(cat_feat, num_feat)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_batches += 1

    scheduler.step()

    # Validation
    if epoch % 2 == 0:  # 매 2 에포크마다 검증
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

        print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), '013_best_large_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping! Best AUC: {best_val_auc:.4f}")
                break
    else:
        avg_train_loss = train_loss / train_batches
        print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}")

print("\n6. 최종 예측...")
# Load best model
model.load_state_dict(torch.load('013_best_large_model.pth'))
model.eval()

test_dataset = CTRDataset(X_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=8,
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

# GPU 메모리 사용량
if torch.cuda.is_available():
    print(f"\nGPU 메모리 사용: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"GPU 메모리 예약: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': predictions
})

submission.to_csv('013_large_deep_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print("\n제출 파일: 013_large_deep_submission.csv")
print(f"Best Validation AUC: {best_val_auc:.4f}")

if predictions.std() > 0.08 and 0.01 < predictions.mean() < 0.05:
    print("\n✅ 균형잡힌 대규모 딥러닝 예측! 0.349 돌파 기대")
else:
    print(f"\n⚠️  예측 분포 확인 필요")
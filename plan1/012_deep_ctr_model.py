#!/usr/bin/env python
"""Deep CTR Model with GPU - Wide & Deep + Attention"""

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
print("Deep CTR Model with A100 GPU")
print("=" * 80)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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
cat_encoders = {}
cat_dims = []

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = X_train[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')

    # Fit on both train and test
    le.fit(pd.concat([X_train[col], X_test[col]]))
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

    cat_encoders[col] = le
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

# Deep CTR Model
class DeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=16, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, dim//2+1))
            for dim in cat_dims
        ])

        # Calculate total embedding dimension
        self.total_emb_dim = sum(min(emb_dim, dim//2+1) for dim in cat_dims)

        # Wide part - linear for numeric features
        self.wide = nn.Linear(num_dim, 1)

        # Deep part
        deep_input_dim = self.total_emb_dim + num_dim
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

        # Attention mechanism for embeddings
        self.attention = nn.Sequential(
            nn.Linear(self.total_emb_dim, self.total_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.total_emb_dim // 2, self.total_emb_dim),
            nn.Sigmoid()
        )

        # Final layer
        self.final = nn.Linear(prev_dim + 1, 1)

    def forward(self, cat_features, num_features):
        # Embedding layer
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_features[:, i]))

        # Concatenate embeddings
        emb_concat = torch.cat(embeddings, dim=1)

        # Apply attention
        attention_weights = self.attention(emb_concat)
        emb_attended = emb_concat * attention_weights

        # Wide part
        wide_out = self.wide(num_features)

        # Deep part
        deep_input = torch.cat([emb_attended, num_features], dim=1)
        deep_out = self.deep(deep_input)

        # Combine wide and deep
        combined = torch.cat([deep_out, wide_out], dim=1)
        output = torch.sigmoid(self.final(combined))

        return output.squeeze()

print("\n3. 모델 초기화...")
model = DeepCTR(cat_dims, len(numeric_cols)).to(device)
print(f"   모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

print("\n4. 학습 준비...")
# Train/Validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

train_dataset = CTRDataset(X_tr, y_tr)
val_dataset = CTRDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8192, shuffle=False, num_workers=4)

print(f"   학습 배치: {len(train_loader)}, 검증 배치: {len(val_loader)}")

print("\n5. 모델 학습...")
best_val_loss = float('inf')
patience_counter = 0
max_patience = 10

for epoch in range(30):
    # Training
    model.train()
    train_loss = 0
    train_batches = 0

    for cat_feat, num_feat, labels in train_loader:
        cat_feat = cat_feat.to(device)
        num_feat = num_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(cat_feat, num_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for cat_feat, num_feat, labels in val_loader:
            cat_feat = cat_feat.to(device)
            num_feat = num_feat.to(device)
            labels = labels.to(device)

            outputs = model(cat_feat, num_feat)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_train_loss = train_loss / train_batches
    avg_val_loss = val_loss / len(val_loader)

    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(val_labels, val_preds)

    print(f"Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_deep_ctr_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print("Early stopping!")
            break

print("\n6. 최종 예측...")
# Load best model
model.load_state_dict(torch.load('best_deep_ctr_model.pth'))
model.eval()

test_dataset = CTRDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False, num_workers=4)

predictions = []
with torch.no_grad():
    for cat_feat, num_feat in test_loader:
        cat_feat = cat_feat.to(device)
        num_feat = num_feat.to(device)

        outputs = model(cat_feat, num_feat)
        predictions.extend(outputs.cpu().numpy())

predictions = np.array(predictions)

print("\n7. 결과 분석...")
print(f"\n예측 확률 통계:")
print(f"   평균: {predictions.mean():.4f}")
print(f"   표준편차: {predictions.std():.4f}")
print(f"   최소: {predictions.min():.6f}")
print(f"   최대: {predictions.max():.6f}")
print(f"   >0.5: {(predictions > 0.5).sum():,}개")

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': predictions
})

submission.to_csv('012_deep_ctr_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print("\n제출 파일: 012_deep_ctr_submission.csv")

if predictions.std() > 0.05 and 0.01 < predictions.mean() < 0.1:
    print("\n✅ 균형잡힌 딥러닝 예측! 0.349 돌파 기대")
else:
    print(f"\n⚠️  추가 조정 필요")
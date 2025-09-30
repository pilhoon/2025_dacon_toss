#!/usr/bin/env python
"""간단한 베이스라인 모델 - 최소 피처로 빠른 테스트"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
import time

# 최소 컬럼만 로드
print("데이터 로딩 중...")
start_time = time.time()

# 필수 컬럼만 선택
cols_to_read = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour', 'clicked']
train_df = pd.read_parquet('../data/train.parquet', columns=cols_to_read)

# 샘플링
train_df = train_df.sample(n=50000, random_state=42)
print(f"데이터 로딩 완료: {time.time() - start_time:.2f}초")
print(f"Train shape: {train_df.shape}")

# 전처리
X = train_df.drop('clicked', axis=1)
y = train_df['clicked']

# 범주형 변수 인코딩
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = encoder.fit_transform(X)

# CV 설정
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = []

print("\nCross Validation 시작...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_encoded, y), 1):
    X_train, X_val = X_encoded[train_idx], X_encoded[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 모델 학습
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)

    scores.append(auc)
    print(f"Fold {fold}: AUC = {auc:.4f}, LogLoss = {logloss:.4f}")

print(f"\n평균 AUC: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# 게이트 평가
mean_auc = np.mean(scores)
if mean_auc >= 0.70:
    print("✅ 베이스라인 성능 게이트 통과! (AUC >= 0.70)")
else:
    print(f"❌ 베이스라인 성능 게이트 미달 (AUC = {mean_auc:.4f} < 0.70)")
    print("   → 데이터/전처리 문제 점검 필요")
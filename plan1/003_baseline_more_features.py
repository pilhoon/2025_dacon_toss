#!/usr/bin/env python
"""베이스라인 모델 - 더 많은 피처 사용"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
import time

# 데이터 로딩
print("데이터 로딩 중...")
start_time = time.time()

# 더 많은 컬럼 포함
train_df = pd.read_parquet('../data/train.parquet')
print(f"전체 데이터 shape: {train_df.shape}")

# 샘플링
train_df = train_df.sample(n=100000, random_state=42)
print(f"샘플링 후 shape: {train_df.shape}")
print(f"데이터 로딩 완료: {time.time() - start_time:.2f}초")

# 컬럼 정보
print(f"\n컬럼 수: {len(train_df.columns)}")
print(f"타겟 분포: {train_df['clicked'].value_counts(normalize=True).to_dict()}")

# ID 컬럼 제거
if 'ID' in train_df.columns:
    train_df = train_df.drop('ID', axis=1)

# 피처와 타겟 분리
X = train_df.drop('clicked', axis=1)
y = train_df['clicked']

# 데이터 타입 확인 및 범주형 변수 처리
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"\n범주형 컬럼 수: {len(categorical_cols)}")
print(f"수치형 컬럼 수: {len(numeric_cols)}")

# 범주형 변수 인코딩
if categorical_cols:
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 결측치 처리
X = X.fillna(-999)

# CV 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_auc = []
scores_logloss = []

print("\nCross Validation 시작...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 모델 학습
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=None,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=0
    )

    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)

    scores_auc.append(auc)
    scores_logloss.append(logloss)
    print(f"Fold {fold}: AUC = {auc:.4f}, LogLoss = {logloss:.4f}")

print(f"\n평균 AUC: {np.mean(scores_auc):.4f} (+/- {np.std(scores_auc):.4f})")
print(f"평균 LogLoss: {np.mean(scores_logloss):.4f} (+/- {np.std(scores_logloss):.4f})")

# 게이트 평가
mean_auc = np.mean(scores_auc)
if mean_auc >= 0.70:
    print(f"\n✅ 베이스라인 성능 게이트 통과! (AUC = {mean_auc:.4f} >= 0.70)")
    print("   → Feature v1 엔지니어링으로 진행 가능")
else:
    print(f"\n❌ 베이스라인 성능 게이트 미달 (AUC = {mean_auc:.4f} < 0.70)")
    print("   → 데이터/전처리 문제 점검 필요")
    print("\n개선 방안:")
    print("  1. 고카디널리티 컬럼 처리 (inventory_id, l_feat_14 등)")
    print("  2. 결측치 처리 개선")
    print("  3. 피처 선택/중요도 분석")

# 피처 중요도 출력
feature_importance = model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-20:][::-1]
print("\nTop 20 중요 피처:")
for idx in top_features_idx:
    print(f"  {X.columns[idx]}: {feature_importance[idx]:.4f}")
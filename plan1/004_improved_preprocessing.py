#!/usr/bin/env python
"""개선된 전처리와 피처 엔지니어링"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("개선된 전처리 실험")
print("=" * 80)

# 데이터 로딩
print("\n1. 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
print(f"   전체 데이터: {train_df.shape}")

# 더 큰 샘플로 실험
SAMPLE_SIZE = 200000
train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)
print(f"   샘플링 후: {train_df.shape}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

# 타겟 분포
target_ratio = train_df['clicked'].value_counts(normalize=True)
print(f"\n   클릭률: {target_ratio[1]:.4f}")

# ID 제거
if 'ID' in train_df.columns:
    train_df = train_df.drop('ID', axis=1)

# 피처와 타겟 분리
y = train_df['clicked']
X = train_df.drop('clicked', axis=1)

print("\n2. 데이터 탐색...")
# 컬럼 타입별 분류
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"   범주형: {len(categorical_cols)}개")
print(f"   수치형: {len(numeric_cols)}개")

# 결측치 확인
missing_ratio = X.isnull().sum() / len(X)
high_missing = missing_ratio[missing_ratio > 0.5]
if len(high_missing) > 0:
    print(f"   결측 50% 이상: {len(high_missing)}개 컬럼")

print("\n3. 피처 엔지니어링...")

# 3.1 고카디널리티 컬럼 처리
print("   고카디널리티 컬럼 처리...")
high_card_cols = []
for col in categorical_cols:
    n_unique = X[col].nunique()
    if n_unique > 100:
        high_card_cols.append(col)
        print(f"     - {col}: {n_unique} unique values")

# Frequency Encoding for high cardinality columns
for col in high_card_cols:
    freq_map = X[col].value_counts(normalize=True).to_dict()
    X[f'{col}_freq'] = X[col].map(freq_map)
    # 원본 컬럼은 유지 (나중에 Target Encoding 적용)

# 3.2 시간 피처 추출
print("   시간 피처 생성...")
if 'hour' in X.columns:
    # hour가 문자열인 경우 숫자로 변환
    if X['hour'].dtype == 'object':
        X['hour'] = pd.to_numeric(X['hour'], errors='coerce')
    X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
    X['is_morning'] = (X['hour'] >= 6) & (X['hour'] < 12)
    X['is_afternoon'] = (X['hour'] >= 12) & (X['hour'] < 18)
    X['is_evening'] = (X['hour'] >= 18) & (X['hour'] < 24)
    X['is_night'] = (X['hour'] >= 0) & (X['hour'] < 6)

if 'day_of_week' in X.columns:
    X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)

# 3.3 상호작용 피처
print("   상호작용 피처 생성...")
if 'gender' in X.columns and 'age_group' in X.columns:
    X['gender_age'] = X['gender'].astype(str) + '_' + X['age_group'].astype(str)
    categorical_cols.append('gender_age')

if 'inventory_id' in X.columns and 'hour' in X.columns:
    X['inventory_hour'] = X['inventory_id'].astype(str) + '_' + (X['hour'] // 6).astype(str)
    categorical_cols.append('inventory_hour')

# 3.4 History 피처 집계
print("   History 피처 집계...")
history_cols = [col for col in X.columns if col.startswith('history_')]
if history_cols:
    X['history_mean'] = X[history_cols].mean(axis=1)
    X['history_std'] = X[history_cols].std(axis=1)
    X['history_max'] = X[history_cols].max(axis=1)
    X['history_min'] = X[history_cols].min(axis=1)
    X['history_range'] = X['history_max'] - X['history_min']

# 3.5 l_feat 피처 집계
print("   l_feat 피처 집계...")
l_feat_cols = [col for col in X.columns if col.startswith('l_feat_')]
if l_feat_cols:
    # 수치형 l_feat만 선택
    l_feat_numeric = []
    for col in l_feat_cols:
        if col not in categorical_cols:
            l_feat_numeric.append(col)

    if l_feat_numeric:
        X['l_feat_sum'] = X[l_feat_numeric].sum(axis=1)
        X['l_feat_mean'] = X[l_feat_numeric].mean(axis=1)
        X['l_feat_nonzero_count'] = (X[l_feat_numeric] != 0).sum(axis=1)

print("\n4. 전처리...")

# 4.1 결측치 처리
print("   결측치 처리...")
# 수치형: median
for col in numeric_cols:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

# 범주형: 'missing' 카테고리
for col in categorical_cols:
    if X[col].isnull().any():
        X[col] = X[col].fillna('missing')

# 4.2 범주형 인코딩
print("   범주형 인코딩...")
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 4.3 inf 값 처리
print("   이상치 처리...")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"\n최종 피처 수: {X.shape[1]}개")

print("\n5. Cross Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_auc = []
scores_logloss = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Target Encoding for high cardinality columns (CV 안에서 수행)
    for col in high_card_cols:
        if col in X.columns:
            # 스무딩 적용한 Target Encoding
            target_mean = y_train.mean()
            smooth_factor = 100

            target_map = {}
            for val in X_train[col].unique():
                mask = X_train[col] == val
                n = mask.sum()
                if n > 0:
                    target_sum = y_train[mask].sum()
                    target_map[val] = (target_sum + smooth_factor * target_mean) / (n + smooth_factor)
                else:
                    target_map[val] = target_mean

            X_train[f'{col}_target'] = X_train[col].map(target_map).fillna(target_mean)
            X_val[f'{col}_target'] = X_val[col].map(target_map).fillna(target_mean)

    # 모델 학습
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=8,
        learning_rate=0.05,
        max_bins=255,
        l2_regularization=0.1,
        min_samples_leaf=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=0
    )

    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict_proba(X_val)[:, 1]

    # 평가
    auc = roc_auc_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred)

    scores_auc.append(auc)
    scores_logloss.append(logloss)

    print(f"   Fold {fold}: AUC = {auc:.4f}, LogLoss = {logloss:.4f}")

print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)
print(f"평균 AUC: {np.mean(scores_auc):.4f} (+/- {np.std(scores_auc):.4f})")
print(f"평균 LogLoss: {np.mean(scores_logloss):.4f} (+/- {np.std(scores_logloss):.4f})")

# 게이트 체크
mean_auc = np.mean(scores_auc)
if mean_auc >= 0.70:
    print(f"\n✅ 베이스라인 성능 게이트 통과! (AUC = {mean_auc:.4f} >= 0.70)")
    print("   → Feature v1으로 진행 가능")
else:
    print(f"\n⚠️  베이스라인 성능 게이트 근접 (AUC = {mean_auc:.4f}, 목표 0.70)")
    print("   추가 개선 사항:")
    print("   - Target Encoding 스무딩 파라미터 조정")
    print("   - 더 많은 상호작용 피처 생성")
    print("   - 피처 선택 적용")

# 실험 결과 저장
import json
result = {
    'sample_size': SAMPLE_SIZE,
    'n_features': X.shape[1],
    'mean_auc': float(np.mean(scores_auc)),
    'std_auc': float(np.std(scores_auc)),
    'mean_logloss': float(np.mean(scores_logloss)),
    'std_logloss': float(np.std(scores_logloss)),
    'fold_aucs': [float(s) for s in scores_auc],
    'fold_loglosses': [float(s) for s in scores_logloss]
}

os.makedirs('experiments/exp_004', exist_ok=True)
with open('experiments/exp_004/results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n결과가 experiments/exp_004/results.json에 저장되었습니다.")
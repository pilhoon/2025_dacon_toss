#!/usr/bin/env python
"""제출 파일 생성 파이프라인"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("제출 파일 생성")
print("=" * 80)

# 최적 파라미터 로드 (있는 경우)
try:
    with open('experiments/exp_006/best_params.json', 'r') as f:
        best_params = json.load(f)
    print("\n최적 파라미터를 로드했습니다.")
except:
    print("\n기본 파라미터를 사용합니다.")
    best_params = {
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'max_depth': 8
    }

print("\n1. 학습 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
print(f"   학습 데이터: {train_df.shape}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

# ID 제거
if 'ID' in train_df.columns:
    train_df = train_df.drop('ID', axis=1)

# 피처와 타겟 분리
y_train = train_df['clicked']
X_train = train_df.drop('clicked', axis=1)

print("\n2. 테스트 데이터 로딩...")
test_df = pd.read_parquet('../data/test.parquet')
print(f"   테스트 데이터: {test_df.shape}")

# ID 저장
test_ids = test_df['ID'].copy()
X_test = test_df.drop('ID', axis=1)

print("\n3. 피처 엔지니어링...")

def feature_engineering(X, is_train=True):
    """피처 엔지니어링 함수"""
    X = X.copy()

    # 시간 피처
    if 'hour' in X.columns:
        if X['hour'].dtype == 'object':
            X['hour'] = pd.to_numeric(X['hour'], errors='coerce')
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['is_morning'] = ((X['hour'] >= 6) & (X['hour'] < 12)).astype(int)
        X['is_afternoon'] = ((X['hour'] >= 12) & (X['hour'] < 18)).astype(int)
        X['is_evening'] = ((X['hour'] >= 18) & (X['hour'] < 24)).astype(int)
        X['is_night'] = ((X['hour'] >= 0) & (X['hour'] < 6)).astype(int)

    if 'day_of_week' in X.columns:
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)

    # History 피처 집계
    history_cols = [col for col in X.columns if col.startswith('history_')]
    if history_cols:
        X['history_mean'] = X[history_cols].mean(axis=1)
        X['history_std'] = X[history_cols].std(axis=1)
        X['history_max'] = X[history_cols].max(axis=1)
        X['history_min'] = X[history_cols].min(axis=1)

    # l_feat 피처 집계
    l_feat_cols = [col for col in X.columns if col.startswith('l_feat_')]
    numeric_l_feat = [col for col in l_feat_cols if X[col].dtype != 'object']
    if numeric_l_feat:
        X['l_feat_sum'] = X[numeric_l_feat].sum(axis=1)
        X['l_feat_mean'] = X[numeric_l_feat].mean(axis=1)
        X['l_feat_nonzero_count'] = (X[numeric_l_feat] != 0).sum(axis=1)

    return X

print("   학습 데이터 피처 엔지니어링...")
X_train = feature_engineering(X_train, is_train=True)

print("   테스트 데이터 피처 엔지니어링...")
X_test = feature_engineering(X_test, is_train=False)

print(f"   최종 피처 수: {X_train.shape[1]}개")

print("\n4. 전처리...")

# 범주형 컬럼 확인
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# 범주형 인코딩
if categorical_cols:
    print(f"   범주형 변수 {len(categorical_cols)}개 인코딩...")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

# 결측치 처리
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# inf 값 처리
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print("\n5. 모델 학습...")

# LightGBM 파라미터
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'seed': 42,
    'verbose': 1,
    'n_jobs': -1,  # 모든 CPU 코어 사용
    **best_params
}

# 전체 데이터로 학습
print("   LightGBM 학습 중...")
lgb_train = lgb.Dataset(X_train, label=y_train)

model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train],
    callbacks=[lgb.log_evaluation(50)]
)

print("\n6. 예측...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 예측 확률 분포 확인
print(f"\n예측 확률 통계:")
print(f"   평균: {y_pred.mean():.4f}")
print(f"   최소: {y_pred.min():.4f}")
print(f"   최대: {y_pred.max():.4f}")
print(f"   표준편차: {y_pred.std():.4f}")

# 학습 데이터 클릭률과 비교
train_click_rate = y_train.mean()
print(f"\n학습 데이터 클릭률: {train_click_rate:.4f}")
print(f"예측 평균 클릭률: {y_pred.mean():.4f}")

print("\n7. 제출 파일 생성...")

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': y_pred
})

# 제출 파일 저장
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"   제출 파일 저장: {submission_path}")

# 피처 중요도 저장
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n8. 상위 20개 중요 피처:")
for idx, row in importance_df.head(20).iterrows():
    print(f"   {row['feature']}: {row['importance']:.2f}")

# 중요도 저장
os.makedirs('experiments/exp_008', exist_ok=True)
importance_df.to_csv('experiments/exp_008/feature_importance.csv', index=False)

# 모델 저장
model.save_model('experiments/exp_008/final_model.txt')

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print(f"\n제출 파일: {submission_path}")
print(f"모델: experiments/exp_008/final_model.txt")
print(f"피처 중요도: experiments/exp_008/feature_importance.csv")
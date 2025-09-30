#!/usr/bin/env python
"""Optuna를 사용한 하이퍼파라미터 튜닝"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Optuna 하이퍼파라미터 튜닝")
print("=" * 80)

# 데이터 로딩
print("\n1. 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
print(f"   전체 데이터: {train_df.shape}")

# 샘플링
SAMPLE_SIZE = 200000
train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)
print(f"   샘플링 후: {train_df.shape}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

# ID 제거
if 'ID' in train_df.columns:
    train_df = train_df.drop('ID', axis=1)

# 피처와 타겟 분리
y = train_df['clicked']
X = train_df.drop('clicked', axis=1)

# 전처리
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"\n2. 전처리...")
print(f"   범주형: {len(categorical_cols)}개")
print(f"   수치형: {len(numeric_cols)}개")

# 범주형 인코딩
if categorical_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 결측치 처리
X = X.fillna(-999)

# CV 설정
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 빠른 튜닝을 위해 3-fold

# Optuna objective function
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    return np.mean(scores)

print("\n3. Optuna 튜닝 시작...")
print("   (30 trials 진행)")

# Optuna study
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='lgbm_tuning'
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

print("\n" + "=" * 80)
print("튜닝 결과")
print("=" * 80)

best_params = study.best_params
best_value = study.best_value

print(f"\n최고 AUC: {best_value:.4f}")
print(f"\n최적 파라미터:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 최종 모델 학습 (전체 CV)
print("\n4. 최적 파라미터로 5-fold CV 실행...")
final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'seed': 42,
    'verbose': -1,
    'n_jobs': -1,
    **best_params
}

cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_scores = []

for fold, (train_idx, val_idx) in enumerate(cv_5fold.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    model = lgb.train(
        final_params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    final_scores.append(auc)
    print(f"   Fold {fold}: AUC = {auc:.4f}")

mean_auc = np.mean(final_scores)
std_auc = np.std(final_scores)

print(f"\n최종 평균 AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

# 게이트 체크
if mean_auc >= 0.72:
    print(f"\n✅ HPO 게이트 통과! (AUC = {mean_auc:.4f} >= 0.72)")
    print("   → 앙상블/스태킹으로 진행 가능")
elif mean_auc >= 0.70:
    print(f"\n⚠️  성능 개선 중 (AUC = {mean_auc:.4f})")
    print("   → 추가 피처 엔지니어링 또는 앙상블 필요")
else:
    print(f"\n❌ 성능 미달 (AUC = {mean_auc:.4f} < 0.70)")

# 결과 저장
result = {
    'sample_size': SAMPLE_SIZE,
    'n_trials': 30,
    'best_auc_3fold': float(best_value),
    'best_params': best_params,
    'final_mean_auc': float(mean_auc),
    'final_std_auc': float(std_auc),
    'final_fold_aucs': [float(s) for s in final_scores]
}

os.makedirs('experiments/exp_006', exist_ok=True)
with open('experiments/exp_006/results.json', 'w') as f:
    json.dump(result, f, indent=2)

# 최적 파라미터 저장
with open('experiments/exp_006/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\n결과가 experiments/exp_006/에 저장되었습니다.")
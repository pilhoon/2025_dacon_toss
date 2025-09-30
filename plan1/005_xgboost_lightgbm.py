#!/usr/bin/env python
"""XGBoost와 LightGBM 실험"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import lightgbm as lgb
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBoost & LightGBM 실험")
print("=" * 80)

# 데이터 로딩
print("\n1. 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
print(f"   전체 데이터: {train_df.shape}")

# 샘플링
SAMPLE_SIZE = 300000
train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)
print(f"   샘플링 후: {train_df.shape}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

# 타겟 분포
target_ratio = train_df['clicked'].value_counts(normalize=True)
print(f"   클릭률: {target_ratio[1]:.4f}")

# ID 제거
if 'ID' in train_df.columns:
    train_df = train_df.drop('ID', axis=1)

# 피처와 타겟 분리
y = train_df['clicked']
X = train_df.drop('clicked', axis=1)

print("\n2. 전처리...")
# 컬럼 타입별 분류
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"   범주형: {len(categorical_cols)}개")
print(f"   수치형: {len(numeric_cols)}개")

# 범주형 변수 인코딩 (XGBoost/LightGBM용)
print("   범주형 변수 인코딩...")
if categorical_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 결측치 처리
print("   결측치 처리...")
X = X.fillna(-999)  # GBDT는 결측치를 자체적으로 처리 가능

print("\n3. Cross Validation 시작...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {
    'xgboost': {'auc': [], 'logloss': []},
    'lightgbm': {'auc': [], 'logloss': []}
}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"\n   === Fold {fold} ===")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    print("   XGBoost 학습...")
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'n_jobs': -1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    y_pred_xgb = xgb_model.predict(dval)
    auc_xgb = roc_auc_score(y_val, y_pred_xgb)
    logloss_xgb = log_loss(y_val, y_pred_xgb)
    results['xgboost']['auc'].append(auc_xgb)
    results['xgboost']['logloss'].append(logloss_xgb)
    print(f"     XGBoost: AUC = {auc_xgb:.4f}, LogLoss = {logloss_xgb:.4f}")

    # LightGBM
    print("   LightGBM 학습...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    y_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    auc_lgb = roc_auc_score(y_val, y_pred_lgb)
    logloss_lgb = log_loss(y_val, y_pred_lgb)
    results['lightgbm']['auc'].append(auc_lgb)
    results['lightgbm']['logloss'].append(logloss_lgb)
    print(f"     LightGBM: AUC = {auc_lgb:.4f}, LogLoss = {logloss_lgb:.4f}")

print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)

for model_name, metrics in results.items():
    mean_auc = np.mean(metrics['auc'])
    std_auc = np.std(metrics['auc'])
    mean_logloss = np.mean(metrics['logloss'])
    std_logloss = np.std(metrics['logloss'])

    print(f"\n{model_name.upper()}:")
    print(f"  평균 AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"  평균 LogLoss: {mean_logloss:.4f} (+/- {std_logloss:.4f})")

# 최고 성능 모델 확인
xgb_mean_auc = np.mean(results['xgboost']['auc'])
lgb_mean_auc = np.mean(results['lightgbm']['auc'])
best_model = 'XGBoost' if xgb_mean_auc > lgb_mean_auc else 'LightGBM'
best_auc = max(xgb_mean_auc, lgb_mean_auc)

print(f"\n최고 성능 모델: {best_model} (AUC = {best_auc:.4f})")

# 게이트 체크
if best_auc >= 0.71:
    print(f"\n✅ GBDT 확대 게이트 통과! (AUC = {best_auc:.4f} >= 0.71)")
    print("   → HPO 및 앙상블로 진행 가능")
elif best_auc >= 0.70:
    print(f"\n⚠️  GBDT 성능 개선 중 (AUC = {best_auc:.4f})")
    print("   → 피처 엔지니어링 추가 필요")
else:
    print(f"\n❌ GBDT 성능 미달 (AUC = {best_auc:.4f} < 0.70)")
    print("   → 데이터 전처리 재검토 필요")

# 실험 결과 저장
experiment_result = {
    'sample_size': SAMPLE_SIZE,
    'xgboost': {
        'mean_auc': float(xgb_mean_auc),
        'std_auc': float(np.std(results['xgboost']['auc'])),
        'mean_logloss': float(np.mean(results['xgboost']['logloss'])),
        'std_logloss': float(np.std(results['xgboost']['logloss'])),
        'fold_aucs': [float(s) for s in results['xgboost']['auc']],
        'fold_loglosses': [float(s) for s in results['xgboost']['logloss']]
    },
    'lightgbm': {
        'mean_auc': float(lgb_mean_auc),
        'std_auc': float(np.std(results['lightgbm']['auc'])),
        'mean_logloss': float(np.mean(results['lightgbm']['logloss'])),
        'std_logloss': float(np.std(results['lightgbm']['logloss'])),
        'fold_aucs': [float(s) for s in results['lightgbm']['auc']],
        'fold_loglosses': [float(s) for s in results['lightgbm']['logloss']]
    },
    'best_model': best_model,
    'best_auc': float(best_auc)
}

os.makedirs('experiments/exp_005', exist_ok=True)
with open('experiments/exp_005/results.json', 'w') as f:
    json.dump(experiment_result, f, indent=2)

print(f"\n결과가 experiments/exp_005/results.json에 저장되었습니다.")
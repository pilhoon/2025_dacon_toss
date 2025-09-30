#!/usr/bin/env python
"""전체 데이터셋으로 최종 모델 학습"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("전체 데이터셋 최종 모델 학습")
print("=" * 80)

# 최적 파라미터 로드
with open('experiments/exp_006/best_params.json', 'r') as f:
    optuna_params = json.load(f)

print("\n최적화된 파라미터 로드 완료")

# 데이터 로딩
print("\n1. 전체 학습 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
print(f"   전체 데이터: {train_df.shape}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

# ID 제거
if 'ID' in train_df.columns:
    train_df = train_df.drop('ID', axis=1)

# 피처와 타겟 분리
y = train_df['clicked']
X = train_df.drop('clicked', axis=1)

print(f"\n   클릭률: {y.mean():.4f}")
print(f"   클릭 수: {y.sum():,}")

print("\n2. 전처리...")
# 범주형 컬럼 확인
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"   범주형: {len(categorical_cols)}개")
print(f"   수치형: {len(numeric_cols)}개")

# 범주형 인코딩
if categorical_cols:
    print("   범주형 인코딩...")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 결측치 처리
print("   결측치 처리...")
X = X.fillna(-999)

print("\n3. 3-Fold Cross Validation (전체 데이터)...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# LightGBM with Optuna params
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'seed': 42,
    'verbose': 0,
    'n_jobs': -1,  # 모든 CPU 사용
    **optuna_params
}

# XGBoost params (exp_005에서 좋았던 파라미터)
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

lgb_scores = []
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"\n   === Fold {fold}/3 ===")
    print(f"   학습: {len(train_idx):,} 샘플")
    print(f"   검증: {len(val_idx):,} 샘플")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    print("   LightGBM 학습...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    y_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    auc_lgb = roc_auc_score(y_val, y_pred_lgb)
    lgb_scores.append(auc_lgb)
    print(f"     LightGBM AUC: {auc_lgb:.4f}")

    # XGBoost
    print("   XGBoost 학습...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    y_pred_xgb = xgb_model.predict(dval)
    auc_xgb = roc_auc_score(y_val, y_pred_xgb)
    xgb_scores.append(auc_xgb)
    print(f"     XGBoost AUC: {auc_xgb:.4f}")

print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)

lgb_mean = np.mean(lgb_scores)
lgb_std = np.std(lgb_scores)
xgb_mean = np.mean(xgb_scores)
xgb_std = np.std(xgb_scores)

print(f"\nLightGBM (Optuna 튜닝):")
print(f"  평균 AUC: {lgb_mean:.4f} (+/- {lgb_std:.4f})")
print(f"  Fold AUCs: {lgb_scores}")

print(f"\nXGBoost:")
print(f"  평균 AUC: {xgb_mean:.4f} (+/- {xgb_std:.4f})")
print(f"  Fold AUCs: {xgb_scores}")

# 최종 모델 선택
if xgb_mean > lgb_mean:
    print(f"\n✅ 최종 모델: XGBoost (AUC {xgb_mean:.4f})")
    final_model_type = 'xgboost'
else:
    print(f"\n✅ 최종 모델: LightGBM (AUC {lgb_mean:.4f})")
    final_model_type = 'lightgbm'

print("\n4. 전체 데이터로 최종 모델 학습...")

if final_model_type == 'lightgbm':
    print("   LightGBM 최종 학습...")
    lgb_train_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        lgb_params,
        lgb_train_full,
        num_boost_round=500,
        valid_sets=[lgb_train_full],
        callbacks=[lgb.log_evaluation(50)]
    )

    # 모델 저장
    final_model.save_model('experiments/exp_009/final_model_lgb_full.txt')

    # 피처 중요도
    importance = final_model.feature_importance(importance_type='gain')
    feature_names = final_model.feature_name()

else:
    print("   XGBoost 최종 학습...")
    dtrain_full = xgb.DMatrix(X, label=y)
    final_model = xgb.train(
        xgb_params,
        dtrain_full,
        num_boost_round=500,
        evals=[(dtrain_full, 'train')],
        verbose_eval=50
    )

    # 모델 저장
    final_model.save_model('experiments/exp_009/final_model_xgb_full.json')

    # 피처 중요도
    importance = final_model.get_score(importance_type='gain')
    feature_names = list(importance.keys())
    importance = list(importance.values())

# 결과 저장
result = {
    'dataset_size': len(train_df),
    'lgb_cv_scores': [float(s) for s in lgb_scores],
    'lgb_mean_auc': float(lgb_mean),
    'lgb_std_auc': float(lgb_std),
    'xgb_cv_scores': [float(s) for s in xgb_scores],
    'xgb_mean_auc': float(xgb_mean),
    'xgb_std_auc': float(xgb_std),
    'final_model': final_model_type,
    'final_auc': float(max(lgb_mean, xgb_mean))
}

os.makedirs('experiments/exp_009', exist_ok=True)
with open('experiments/exp_009/results.json', 'w') as f:
    json.dump(result, f, indent=2)

# 피처 중요도 저장
importance_df = pd.DataFrame({
    'feature': feature_names[:50],  # 상위 50개
    'importance': importance[:50]
}).sort_values('importance', ascending=False)

importance_df.to_csv('experiments/exp_009/feature_importance_top50.csv', index=False)

print("\n상위 10개 중요 피처:")
for idx, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.2f}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print(f"\n최종 모델: experiments/exp_009/")
print(f"결과: experiments/exp_009/results.json")
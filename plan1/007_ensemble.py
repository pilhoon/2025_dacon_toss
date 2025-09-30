#!/usr/bin/env python
"""앙상블 및 블렌딩"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("앙상블 및 블렌딩 실험")
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
if categorical_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
X = X.fillna(-999)

print(f"\n2. 모델 정의...")

# 베이스 모델들
models = {
    'histgbm': HistGradientBoostingClassifier(
        max_iter=200, max_depth=8, learning_rate=0.05,
        min_samples_leaf=50, l2_regularization=0.1,
        random_state=42, verbose=0
    ),
    'xgboost': xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    ),
    'lightgbm': lgb.LGBMClassifier(
        n_estimators=200, num_leaves=127, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
}

print("\n3. Cross Validation 시작...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF predictions 저장
oof_predictions = {name: np.zeros(len(X)) for name in models.keys()}
oof_predictions['blend_avg'] = np.zeros(len(X))
oof_predictions['blend_weighted'] = np.zeros(len(X))

# 각 모델별 fold 성능
fold_scores = {name: [] for name in models.keys()}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"\n   === Fold {fold} ===")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_preds = {}

    # 각 모델 학습
    for name, model in models.items():
        print(f"   {name} 학습...")
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_val)[:, 1]
        fold_preds[name] = y_pred
        oof_predictions[name][val_idx] = y_pred

        auc = roc_auc_score(y_val, y_pred)
        fold_scores[name].append(auc)
        print(f"     {name}: AUC = {auc:.4f}")

    # 블렌딩 (단순 평균)
    blend_avg = np.mean([fold_preds[name] for name in models.keys()], axis=0)
    oof_predictions['blend_avg'][val_idx] = blend_avg
    auc_avg = roc_auc_score(y_val, blend_avg)
    print(f"     Blend (평균): AUC = {auc_avg:.4f}")

    # 블렌딩 (가중 평균 - AUC 기반)
    weights = np.array([fold_scores[name][-1] for name in models.keys()])
    weights = weights / weights.sum()
    blend_weighted = np.average([fold_preds[name] for name in models.keys()],
                                axis=0, weights=weights)
    oof_predictions['blend_weighted'][val_idx] = blend_weighted
    auc_weighted = roc_auc_score(y_val, blend_weighted)
    print(f"     Blend (가중): AUC = {auc_weighted:.4f}")

print("\n4. 스태킹 (메타 러너)...")

# 스태킹을 위한 메타 피처 준비
meta_features = np.column_stack([oof_predictions[name] for name in models.keys()])

# 메타 러너 학습 (Logistic Regression)
cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_scores = []

for train_idx, val_idx in cv_meta.split(meta_features, y):
    X_meta_train = meta_features[train_idx]
    X_meta_val = meta_features[val_idx]
    y_meta_train = y.iloc[train_idx]
    y_meta_val = y.iloc[val_idx]

    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(X_meta_train, y_meta_train)

    y_meta_pred = meta_model.predict_proba(X_meta_val)[:, 1]
    auc_meta = roc_auc_score(y_meta_val, y_meta_pred)
    meta_scores.append(auc_meta)

print(f"   메타 러너 평균 AUC: {np.mean(meta_scores):.4f}")

print("\n5. 칼리브레이션...")

# 가장 좋은 단일 모델 선택
best_single_model = max(fold_scores.keys(),
                       key=lambda x: np.mean(fold_scores[x]))
best_model_auc = np.mean(fold_scores[best_single_model])

print(f"   최고 단일 모델: {best_single_model} (AUC = {best_model_auc:.4f})")

# Platt Scaling
calibrated_clf = CalibratedClassifierCV(
    models[best_single_model], cv=3, method='sigmoid'
)
calibrated_clf.fit(X, y)

# 칼리브레이션 효과 테스트
test_idx = np.random.choice(len(X), size=int(len(X)*0.2), replace=False)
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

y_pred_uncalibrated = models[best_single_model].predict_proba(X_test)[:, 1]
y_pred_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]

logloss_uncalibrated = log_loss(y_test, y_pred_uncalibrated)
logloss_calibrated = log_loss(y_test, y_pred_calibrated)

print(f"   칼리브레이션 전 LogLoss: {logloss_uncalibrated:.4f}")
print(f"   칼리브레이션 후 LogLoss: {logloss_calibrated:.4f}")

print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)

# 모델별 평균 성능
print("\n개별 모델 성능:")
for name in models.keys():
    mean_auc = np.mean(fold_scores[name])
    std_auc = np.std(fold_scores[name])
    print(f"  {name}: {mean_auc:.4f} (+/- {std_auc:.4f})")

# 앙상블 성능
ensemble_results = {
    'blend_avg': roc_auc_score(y, oof_predictions['blend_avg']),
    'blend_weighted': roc_auc_score(y, oof_predictions['blend_weighted']),
    'stacking': np.mean(meta_scores)
}

print("\n앙상블 성능:")
for method, auc in ensemble_results.items():
    print(f"  {method}: {auc:.4f}")

# 최고 성능
best_method = max(ensemble_results.keys(), key=lambda x: ensemble_results[x])
best_auc = ensemble_results[best_method]

print(f"\n최고 앙상블 방법: {best_method} (AUC = {best_auc:.4f})")

# 게이트 체크
if best_auc >= 0.73:
    print(f"\n✅ 앙상블 게이트 통과! (AUC = {best_auc:.4f} >= 0.73)")
    print("   → 최종 모델 학습 및 제출 준비")
elif best_auc >= 0.71:
    print(f"\n⚠️  앙상블 성능 양호 (AUC = {best_auc:.4f})")
    print("   → 추가 모델 또는 피처 개선 권장")
else:
    print(f"\n❌ 앙상블 성능 개선 필요 (AUC = {best_auc:.4f})")

# 결과 저장
result = {
    'sample_size': SAMPLE_SIZE,
    'individual_models': {
        name: {
            'mean_auc': float(np.mean(fold_scores[name])),
            'std_auc': float(np.std(fold_scores[name])),
            'fold_aucs': [float(s) for s in fold_scores[name]]
        }
        for name in models.keys()
    },
    'ensemble_results': {k: float(v) for k, v in ensemble_results.items()},
    'best_method': best_method,
    'best_auc': float(best_auc),
    'calibration': {
        'logloss_before': float(logloss_uncalibrated),
        'logloss_after': float(logloss_calibrated)
    }
}

os.makedirs('experiments/exp_007', exist_ok=True)
with open('experiments/exp_007/results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n결과가 experiments/exp_007/results.json에 저장되었습니다.")
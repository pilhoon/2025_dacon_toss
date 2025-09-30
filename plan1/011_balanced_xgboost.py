#!/usr/bin/env python
"""균형잡힌 XGBoost - 0.349 목표"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("균형잡힌 XGBoost - 목표 0.349 돌파")
print("=" * 80)

# 데이터 로딩
print("\n1. 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
test_df = pd.read_parquet('../data/test.parquet')
print(f"   학습: {train_df.shape}, 테스트: {test_df.shape}")

# ID 저장
test_ids = test_df['ID'].copy()

# ID 제거
if 'ID' in train_df.columns:
    train_df = train_df.drop('ID', axis=1)
if 'ID' in test_df.columns:
    test_df = test_df.drop('ID', axis=1)

# 타겟 분리
y_train = train_df['clicked']
X_train = train_df.drop('clicked', axis=1)
X_test = test_df

print(f"   클릭률: {y_train.mean():.4f}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

print("\n2. 간단한 피처 엔지니어링...")

# 고중요도 피처 기반 (010 결과 활용)
important_features = ['l_feat_16', 'history_a_1', 'l_feat_2', 'inventory_id',
                     'l_feat_1', 'history_b_21', 'history_b_2', 'history_a_3',
                     'age_group', 'history_a_2']

# history 집계 피처
history_cols = [col for col in X_train.columns if col.startswith('history_')]
if history_cols:
    X_train['history_sum'] = X_train[history_cols].sum(axis=1)
    X_train['history_mean'] = X_train[history_cols].mean(axis=1)
    X_train['history_max'] = X_train[history_cols].max(axis=1)

    X_test['history_sum'] = X_test[history_cols].sum(axis=1)
    X_test['history_mean'] = X_test[history_cols].mean(axis=1)
    X_test['history_max'] = X_test[history_cols].max(axis=1)

print("   피처 수:", X_train.shape[1])

print("\n3. 전처리...")
# 범주형 컬럼 처리
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"   범주형 {len(categorical_cols)}개 인코딩...")

if categorical_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

# 결측치 처리
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

print("\n4. 5-Fold CV로 최적 scale_pos_weight 찾기...")

# 다양한 scale_pos_weight 테스트
scale_weights = [1, 5, 10, 20, 30]
cv_results = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for scale_weight in scale_weights:
    print(f"\n   Testing scale_pos_weight={scale_weight}")

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,  # 10 -> 8로 조정
        'learning_rate': 0.03,  # 0.05 -> 0.03
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,  # 5 -> 10
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0.1,  # 0 -> 0.1
        'scale_pos_weight': scale_weight,
        'seed': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }

    fold_scores = []
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=300,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=0
        )

        oof_preds[val_idx] = model.predict(dval)

    # 평가
    pred_mean = oof_preds.mean()
    pred_std = oof_preds.std()
    auc = roc_auc_score(y_train, oof_preds)
    logloss = log_loss(y_train, oof_preds)

    cv_results[scale_weight] = {
        'auc': auc,
        'logloss': logloss,
        'pred_mean': pred_mean,
        'pred_std': pred_std
    }

    print(f"     AUC: {auc:.4f}, LogLoss: {logloss:.4f}")
    print(f"     예측 평균: {pred_mean:.4f}, 표준편차: {pred_std:.4f}")

# 최적 scale_pos_weight 선택
best_scale = min(cv_results.keys(),
                 key=lambda x: abs(cv_results[x]['pred_mean'] - y_train.mean()))

print(f"\n선택된 scale_pos_weight: {best_scale}")
print(f"   예측 평균: {cv_results[best_scale]['pred_mean']:.4f}")
print(f"   실제 클릭률: {y_train.mean():.4f}")

print("\n5. 최종 모델 학습...")

final_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'scale_pos_weight': best_scale,
    'seed': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

dtrain_full = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

final_model = xgb.train(
    final_params,
    dtrain_full,
    num_boost_round=500,
    evals=[(dtrain_full, 'train')],
    verbose_eval=100
)

print("\n6. 예측 및 Calibration...")
y_pred_raw = final_model.predict(dtest)

# Isotonic Regression으로 calibration
print("   확률 보정 중...")
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(oof_preds, y_train)
y_pred_calibrated = iso_reg.transform(y_pred_raw)

print("\n7. 결과 분석...")
print(f"\nRaw 예측:")
print(f"   평균: {y_pred_raw.mean():.4f}")
print(f"   표준편차: {y_pred_raw.std():.4f}")

print(f"\nCalibrated 예측:")
print(f"   평균: {y_pred_calibrated.mean():.4f}")
print(f"   표준편차: {y_pred_calibrated.std():.4f}")
print(f"   >0.5: {(y_pred_calibrated > 0.5).sum():,}개")

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': y_pred_calibrated
})

submission.to_csv('011_balanced_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print("\n제출 파일: 011_balanced_submission.csv")

# 예상 점수
if cv_results[best_scale]['pred_std'] > 0.05 and abs(y_pred_calibrated.mean() - 0.0191) < 0.05:
    print("\n✅ 균형잡힌 예측! 0.349 돌파 기대")
else:
    print(f"\n⚠️  추가 조정 필요")
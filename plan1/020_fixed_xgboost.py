#!/usr/bin/env python
"""Fixed XGBoost - 예측값 0 문제 해결"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Fixed XGBoost - Balanced Predictions")
print("=" * 80)

# 1. 데이터 로딩
print("\n1. 데이터 로딩...")
train_df = pd.read_parquet('../data/train.parquet')
test_df = pd.read_parquet('../data/test.parquet')
print(f"   학습: {train_df.shape}, 테스트: {test_df.shape}")

# ID 저장 및 제거
test_ids = test_df['ID'].copy()
for df in [train_df, test_df]:
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)

# 타겟 분리
y_train = train_df['clicked'].values
X_train = train_df.drop('clicked', axis=1)
X_test = test_df

click_rate = y_train.mean()
print(f"   클릭률: {click_rate:.4f} ({y_train.sum():,} / {len(y_train):,})")

# 2. 피처 엔지니어링
print("\n2. 피처 엔지니어링...")

# 범주형과 수치형 분리
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

print(f"   범주형: {len(categorical_cols)}개, 수치형: {len(numeric_cols)}개")

# 범주형 인코딩
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = X_train[col].fillna('missing')
    X_test[col] = X_test[col].fillna('missing')

    le.fit(pd.concat([X_train[col], X_test[col]]))
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# 수치형 처리
scaler = StandardScaler()
X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
X_test[numeric_cols] = X_test[numeric_cols].fillna(0)
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 3. XGBoost 파라미터 설정
print("\n3. XGBoost 학습...")

# scale_pos_weight 계산 (불균형 데이터)
scale_pos_weight = (1 - click_rate) / click_rate
print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',  # auc 대신 logloss 사용
    'max_depth': 6,  # 덜 복잡한 모델
    'learning_rate': 0.05,  # 더 높은 학습률
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'scale_pos_weight': scale_pos_weight,  # 불균형 처리
    'seed': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

# 5-Fold CV로 안정적인 예측
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# OOF 예측 저장
oof_predictions = np.zeros(len(X_train))
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n   Fold {fold}/{n_folds}")

    # 데이터 분할
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # DMatrix 생성
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # 학습
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # 예측
    val_pred = model.predict(dval)
    test_pred = model.predict(dtest)

    # 예측값 확인
    print(f"      Val 예측 - 평균: {val_pred.mean():.4f}, 표준편차: {val_pred.std():.4f}")
    print(f"      Val 예측 - 최소: {val_pred.min():.6f}, 최대: {val_pred.max():.6f}")
    print(f"      Val 예측 - >0.5: {(val_pred > 0.5).sum()} ({100*(val_pred > 0.5).mean():.2f}%)")

    # 저장
    oof_predictions[val_idx] = val_pred
    test_predictions += test_pred / n_folds

print("\n4. 최종 예측 분석...")

# Isotonic Regression으로 캘리브레이션
print("\n   Isotonic Regression 캘리브레이션...")
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(oof_predictions, y_train)

# 테스트 예측 캘리브레이션
calibrated_predictions = iso_reg.predict(test_predictions)

# 최종 예측 분석
print(f"\n   원본 예측:")
print(f"      평균: {test_predictions.mean():.4f}")
print(f"      표준편차: {test_predictions.std():.4f}")
print(f"      최소: {test_predictions.min():.6f}")
print(f"      최대: {test_predictions.max():.6f}")
print(f"      >0.5: {(test_predictions > 0.5).sum()} ({100*(test_predictions > 0.5).mean():.2f}%)")

print(f"\n   캘리브레이션 후:")
print(f"      평균: {calibrated_predictions.mean():.4f}")
print(f"      표준편차: {calibrated_predictions.std():.4f}")
print(f"      최소: {calibrated_predictions.min():.6f}")
print(f"      최대: {calibrated_predictions.max():.6f}")
print(f"      >0.5: {(calibrated_predictions > 0.5).sum()} ({100*(calibrated_predictions > 0.5).mean():.2f}%)")

# 예측값이 모두 0인지 확인
if calibrated_predictions.max() < 0.001:
    print("\n   ⚠️ 경고: 모든 예측값이 거의 0입니다!")
    print("   원본 예측값 사용...")
    final_predictions = test_predictions
else:
    final_predictions = calibrated_predictions

# 최종 클리핑 (안전장치)
final_predictions = np.clip(final_predictions, 0.001, 0.999)

# 5. 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': final_predictions
})

submission.to_csv('020_fixed_xgboost_submission.csv', index=False)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print("\n제출 파일: 020_fixed_xgboost_submission.csv")
print(f"예측 평균: {final_predictions.mean():.4f}")
print(f"예측 표준편차: {final_predictions.std():.4f}")

if final_predictions.std() > 0.05:
    print("\n✅ 예측값 분포 정상!")
else:
    print("\n⚠️ 예측값 분산이 너무 낮음. 모델 재검토 필요.")
#!/usr/bin/env python
"""XGBoost로 개선된 제출 파일 생성"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBoost 제출 파일 생성")
print("=" * 80)

# 데이터 로딩
print("\n1. 데이터 로딩...")
start_time = time.time()

train_df = pd.read_parquet('../data/train.parquet')
test_df = pd.read_parquet('../data/test.parquet')
print(f"   학습: {train_df.shape}, 테스트: {test_df.shape}")
print(f"   로딩 시간: {time.time() - start_time:.2f}초")

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

print(f"\n   클릭률: {y_train.mean():.4f}")
print(f"   Positive: {y_train.sum():,}, Negative: {(1-y_train).sum():,}")

print("\n2. 전처리...")
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

print("\n3. XGBoost 학습...")

# 더 나은 기본 파라미터 (Optuna 파라미터 제거)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 10,  # 4 -> 10 증가
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.01,  # 0.1 -> 0.01 감소
    'reg_lambda': 1.0,
    'gamma': 0.0,  # min_gain_to_split 대신 gamma 사용
    'scale_pos_weight': (1-y_train.mean())/y_train.mean(),  # 클래스 불균형 처리
    'seed': 42,
    'n_jobs': -1,
    'tree_method': 'hist'  # 빠른 학습
}

print("   파라미터:")
for key, value in xgb_params.items():
    if key == 'scale_pos_weight':
        print(f"     {key}: {value:.2f}")
    else:
        print(f"     {key}: {value}")

# DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# 학습
print("\n   학습 중...")
model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train')],
    verbose_eval=50
)

print("\n4. 예측...")
y_pred = model.predict(dtest)

# 예측 통계
print(f"\n예측 확률 통계:")
print(f"   평균: {y_pred.mean():.4f}")
print(f"   표준편차: {y_pred.std():.4f} (목표: >0.05)")
print(f"   최소: {y_pred.min():.6f}")
print(f"   최대: {y_pred.max():.6f}")
print(f"   중앙값: {y_pred.median() if hasattr(y_pred, 'median') else np.median(y_pred):.4f}")

# 확률 분포 확인
high_conf = (y_pred > 0.5).sum()
low_conf = (y_pred < 0.01).sum()
print(f"\n확률 분포:")
print(f"   >0.5: {high_conf:,}개 ({100*high_conf/len(y_pred):.2f}%)")
print(f"   <0.01: {low_conf:,}개 ({100*low_conf/len(y_pred):.2f}%)")

# 학습 데이터와 비교
print(f"\n학습 클릭률: {y_train.mean():.4f}")
print(f"예측 평균: {y_pred.mean():.4f}")
print(f"차이: {abs(y_pred.mean() - y_train.mean()):.4f}")

print("\n5. 제출 파일 생성...")
submission = pd.DataFrame({
    'ID': test_ids,
    'clicked': y_pred
})

# 저장
submission_path = '010_xgboost_submission.csv'
submission.to_csv(submission_path, index=False)
print(f"   저장: {submission_path}")

# 피처 중요도 상위 20개
importance = model.get_score(importance_type='gain')
if importance:
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\n상위 20개 중요 피처:")
    for feat, score in sorted_importance:
        print(f"   {feat}: {score:.2f}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print(f"\n제출 파일: {submission_path}")

# 예상 점수 계산
pred_std = y_pred.std()
if pred_std > 0.05:
    print("\n✅ 예측 분산 개선됨! AP 향상 기대")
else:
    print("\n⚠️  예측 분산 여전히 낮음")
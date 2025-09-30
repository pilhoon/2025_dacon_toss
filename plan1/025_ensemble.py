#!/usr/bin/env python
"""Ensemble - XGBoost + Deep Learning 앙상블"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENSEMBLE - Combining XGBoost and Deep Learning")
print("=" * 80)

# 1. 예측 파일 로드
print("\n1. 예측 파일 로드...")

# XGBoost (010) - 리더보드 0.31631
xgb_df = pd.read_csv('010_xgboost_submission.csv')
print(f"   XGBoost (010): 리더보드 0.31631")
print(f"      평균: {xgb_df.clicked.mean():.4f}, 표준편차: {xgb_df.clicked.std():.4f}")

# Deep Learning (023) - 리더보드 0.1574
dl_df = pd.read_csv('023_ultra_batch_submission.csv')
print(f"   Deep Learning (023): 리더보드 0.1574")
print(f"      평균: {dl_df.clicked.mean():.4f}, 표준편차: {dl_df.clicked.std():.4f}")

# ID 확인
assert all(xgb_df.ID == dl_df.ID), "ID 순서가 다릅니다!"

# 2. 다양한 앙상블 방법 시도
print("\n2. 앙상블 방법 테스트...")

results = {}

# 2.1 Simple Average
simple_avg = (xgb_df.clicked + dl_df.clicked) / 2
results['simple_avg'] = simple_avg
print(f"\n   Simple Average:")
print(f"      평균: {simple_avg.mean():.4f}, 표준편차: {simple_avg.std():.4f}")

# 2.2 Weighted Average (XGBoost가 더 좋으므로 가중치 높게)
weights = [0.7, 0.3]  # XGBoost 70%, DL 30%
weighted_avg = xgb_df.clicked * weights[0] + dl_df.clicked * weights[1]
results['weighted_70_30'] = weighted_avg
print(f"\n   Weighted Average (70:30):")
print(f"      평균: {weighted_avg.mean():.4f}, 표준편차: {weighted_avg.std():.4f}")

# 2.3 Weighted Average (균형잡힌 가중치)
weights = [0.6, 0.4]  # XGBoost 60%, DL 40%
weighted_avg_60_40 = xgb_df.clicked * weights[0] + dl_df.clicked * weights[1]
results['weighted_60_40'] = weighted_avg_60_40
print(f"\n   Weighted Average (60:40):")
print(f"      평균: {weighted_avg_60_40.mean():.4f}, 표준편차: {weighted_avg_60_40.std():.4f}")

# 2.4 Geometric Mean (더 보수적인 예측)
geometric = np.sqrt(xgb_df.clicked * dl_df.clicked)
results['geometric'] = geometric
print(f"\n   Geometric Mean:")
print(f"      평균: {geometric.mean():.4f}, 표준편차: {geometric.std():.4f}")

# 2.5 Harmonic Mean (더욱 보수적)
# 0 값 처리
epsilon = 1e-8
harmonic = 2 / (1/(xgb_df.clicked + epsilon) + 1/(dl_df.clicked + epsilon))
results['harmonic'] = harmonic
print(f"\n   Harmonic Mean:")
print(f"      평균: {harmonic.mean():.4f}, 표준편차: {harmonic.std():.4f}")

# 2.6 Rank Average (순위 기반 앙상블)
rank_xgb = rankdata(xgb_df.clicked) / len(xgb_df)
rank_dl = rankdata(dl_df.clicked) / len(dl_df)
rank_avg = (rank_xgb + rank_dl) / 2
results['rank_avg'] = rank_avg
print(f"\n   Rank Average:")
print(f"      평균: {rank_avg.mean():.4f}, 표준편차: {rank_avg.std():.4f}")

# 2.7 Power Mean (조정 가능한 평균)
# p=2 (RMS), p=0.5 (더 보수적)
p = 0.5
power_mean = np.power((np.power(xgb_df.clicked, p) + np.power(dl_df.clicked + epsilon, p)) / 2, 1/p)
results['power_mean'] = power_mean
print(f"\n   Power Mean (p={p}):")
print(f"      평균: {power_mean.mean():.4f}, 표준편차: {power_mean.std():.4f}")

# 2.8 Min-Max Blending (극단값 조정)
min_pred = np.minimum(xgb_df.clicked, dl_df.clicked)
max_pred = np.maximum(xgb_df.clicked, dl_df.clicked)
minmax_blend = min_pred * 0.3 + max_pred * 0.7  # 최대값에 더 가중치
results['minmax_blend'] = minmax_blend
print(f"\n   Min-Max Blend (30:70):")
print(f"      평균: {minmax_blend.mean():.4f}, 표준편차: {minmax_blend.std():.4f}")

# 2.9 Calibrated Ensemble (확률 보정)
# XGBoost 예측을 기준으로 DL 예측 보정
iso_reg = IsotonicRegression(out_of_bounds='clip')
# 샘플링으로 학습 (메모리 절약)
sample_idx = np.random.choice(len(dl_df), size=min(100000, len(dl_df)), replace=False)
iso_reg.fit(dl_df.clicked.values[sample_idx], xgb_df.clicked.values[sample_idx])
calibrated_dl = iso_reg.predict(dl_df.clicked)
calibrated_ensemble = xgb_df.clicked * 0.6 + calibrated_dl * 0.4
results['calibrated'] = calibrated_ensemble
print(f"\n   Calibrated Ensemble:")
print(f"      평균: {calibrated_ensemble.mean():.4f}, 표준편차: {calibrated_ensemble.std():.4f}")

# 3. 최적 앙상블 선택
print("\n" + "=" * 80)
print("3. 최적 앙상블 선택")
print("=" * 80)

# 목표: 평균은 실제 클릭률(0.0191)에 가깝고, 표준편차는 크게
target_mean = 0.0191
best_score = float('inf')
best_method = None

for method, preds in results.items():
    mean_diff = abs(preds.mean() - target_mean)
    std_penalty = max(0, 0.15 - preds.std()) * 10  # 표준편차가 0.15보다 작으면 페널티
    score = mean_diff + std_penalty

    print(f"\n{method}:")
    print(f"   평균 차이: {mean_diff:.4f}")
    print(f"   표준편차: {preds.std():.4f}")
    print(f"   점수: {score:.4f} (낮을수록 좋음)")

    if score < best_score:
        best_score = score
        best_method = method

print(f"\n{'='*60}")
print(f"최적 방법: {best_method}")
print(f"{'='*60}")

# 4. 제출 파일 생성 (여러 개)
print("\n4. 제출 파일 생성...")

# 4.1 최적 방법
best_preds = results[best_method]
submission_best = pd.DataFrame({
    'ID': xgb_df.ID,
    'clicked': best_preds
})
submission_best.to_csv(f'025_ensemble_{best_method}.csv', index=False)
print(f"\n   최적: 025_ensemble_{best_method}.csv")
print(f"      평균: {best_preds.mean():.4f}, 표준편차: {best_preds.std():.4f}")

# 4.2 Weighted 60:40 (균형잡힌 버전)
submission_balanced = pd.DataFrame({
    'ID': xgb_df.ID,
    'clicked': results['weighted_60_40']
})
submission_balanced.to_csv('025_ensemble_balanced.csv', index=False)
print(f"\n   균형: 025_ensemble_balanced.csv")
print(f"      평균: {results['weighted_60_40'].mean():.4f}, 표준편차: {results['weighted_60_40'].std():.4f}")

# 4.3 Geometric (보수적 버전)
submission_conservative = pd.DataFrame({
    'ID': xgb_df.ID,
    'clicked': results['geometric']
})
submission_conservative.to_csv('025_ensemble_conservative.csv', index=False)
print(f"\n   보수적: 025_ensemble_conservative.csv")
print(f"      평균: {results['geometric'].mean():.4f}, 표준편차: {results['geometric'].std():.4f}")

# 4.4 Rank Average (순위 기반)
submission_rank = pd.DataFrame({
    'ID': xgb_df.ID,
    'clicked': results['rank_avg']
})
submission_rank.to_csv('025_ensemble_rank.csv', index=False)
print(f"\n   순위: 025_ensemble_rank.csv")
print(f"      평균: {results['rank_avg'].mean():.4f}, 표준편차: {results['rank_avg'].std():.4f}")

# 5. 추천
print("\n" + "=" * 80)
print("5. 제출 추천")
print("=" * 80)

print("\n제출 우선순위:")
print("1. 025_ensemble_conservative.csv - Geometric mean (평균을 낮추면서 분산 유지)")
print("2. 025_ensemble_balanced.csv - Weighted 60:40 (균형잡힌 앙상블)")
print(f"3. 025_ensemble_{best_method}.csv - 자동 선택된 최적 방법")
print("4. 025_ensemble_rank.csv - 순위 기반 (안정적)")

print("\n목표: 0.349 돌파!")
print("현재 최고: 0.31631 (XGBoost)")
print("\n앙상블로 상관관계 0.5442를 활용하여 더 나은 성능 기대!")

# 6. 분석
print("\n" + "=" * 80)
print("6. 상세 분석")
print("=" * 80)

# 예측 분포 비교
print("\n예측 분포 (>0.5인 비율):")
for method, preds in results.items():
    ratio = (preds > 0.5).mean() * 100
    print(f"   {method}: {ratio:.2f}%")

print(f"\n원본 비교:")
print(f"   XGBoost (010): {(xgb_df.clicked > 0.5).mean() * 100:.2f}%")
print(f"   Deep Learning (023): {(dl_df.clicked > 0.5).mean() * 100:.2f}%")

print("\n완료!")
print("=" * 80)
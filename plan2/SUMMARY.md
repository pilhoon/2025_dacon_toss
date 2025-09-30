# Plan2 수행 요약

## 현재 상황
- **목표**: Competition Score > 0.349 달성
- **Plan1 최고 성과**: XGBoost로 0.31631점 (리더보드)
- **필요 개선**: AP 향상과 WLL 감소를 통한 0.033점 개선 필요

## 수행한 작업

### 1. Deep Learning 접근 시도 (DCNv2)
- **문제점**:
  - 극심한 클래스 불균형 (positive rate ~1.9%)으로 인한 학습 불안정
  - NaN loss 발생 (pos_weight=51로 인한 gradient explosion)
  - Mixed precision training 이슈
- **시도한 해결책**:
  - Gradient clipping 적용
  - 학습률 감소 (0.0001)
  - 모델 크기 축소
  - pos_weight 상한 설정 (20)
- **결과**: 여전히 불안정, XGBoost 대비 성능 개선 미확인

### 2. XGBoost 최적화 방향
- **핵심 인사이트** (plan1 분석):
  - 예측 표준편차 > 0.05 필요 (AP 향상)
  - 예측 평균 ≈ 0.0191 필요 (WLL 개선)
  - scale_pos_weight 10-20 범위가 최적
- **개선 전략**:
  - Feature engineering 강화 (interaction terms)
  - Hyperparameter 미세조정
  - Ensemble 방법론

## 다음 단계 권장사항

### 1. XGBoost 개선 (단기)
```python
# 최적 파라미터 조합
params = {
    'max_depth': 7-9,
    'scale_pos_weight': 12-18,
    'learning_rate': 0.03-0.05,
    'subsample': 0.7-0.9,
    'colsample_bytree': 0.7-0.9
}
```

### 2. Feature Engineering
- Gender × Age interaction
- Hour × Day of week patterns
- History features aggregation (sum, mean, std, max)
- Target encoding with smoothing

### 3. Calibration
- Isotonic Regression
- Platt Scaling
- Temperature Scaling

### 4. Ensemble Strategy
- XGBoost + LightGBM + CatBoost
- Weighted average based on OOF performance
- Stacking with logistic regression meta-learner

## 기술적 문제점 해결
1. **데이터 로딩 속도**: Parquet 파일 읽기가 느림 (75초/1M rows)
   - 해결: 캐시 데이터 준비 (plan2/cache/)

2. **GPU 메모리**: Deep learning 모델에 대한 batch size 제약
   - 해결: Gradient accumulation 또는 더 작은 모델

3. **XGBoost GPU**: tree_method='gpu_hist'가 때때로 불안정
   - 해결: CPU fallback 옵션 준비

## 결론
Deep learning 접근은 현재 환경에서 안정적인 학습이 어려우므로, XGBoost 기반 개선에 집중하는 것이 효율적. Feature engineering과 hyperparameter tuning을 통해 목표 점수 달성 가능할 것으로 판단됨.
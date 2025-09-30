# Plan4 진행 상황 보고서

## 실험 결과 요약

### 제출 점수 기록
| 제출 파일 | 점수 | 설명 | 문제점 |
|----------|------|------|--------|
| 007_xgboost_submission.csv | **0.1166** | XGBoost with guardrails | 점수 매우 낮음 |

### 피드백 받은 버그들 (2025-09-29 02:20)
1. **Import 경로 문제** ✅ 수정됨
   - plan4/008, 009: `from src.prediction_guardrails` → `from plan4.src.prediction_guardrails`
   - check_predictions 함수 없음 → check() 사용

2. **WeightedLogLoss 공식 오류**
   - plan4/011: 잘못된 20:1 가중치 사용
   - 공식 대회 평가 지표와 불일치

3. **Guardrail 제거 실패**
   - plan4/012: 파일명과 달리 guardrail 로직 그대로 유지
   - 실제로는 guardrail이 제거되지 않음

### 주요 발견사항
1. **Guardrail 제약이 성능을 크게 해침**
   - 목표: mean ∈ [0.017, 0.021], std ≥ 0.055
   - 실제: mean = 0.0257, std = 0.0370
   - Guardrail 적용 후 점수가 급격히 하락

2. **Competition Score = 0.5×AP + 0.5×1/(1+WLL)**
   - 높을수록 좋은 점수
   - 현재 0.1166은 매우 낮은 수준

## 완료된 작업

### WS0: Metric & Data Audit ✅
- Competition score 구현
- Prediction guardrails 설정
- 7개 주요 버그 수정:
  1. Encoder mismatch 문제
  2. OrdinalEncoder fillna 누락
  3. Double encoding 버그
  4. Train/test median 불일치
  5. Categorical 이미 numeric 변환
  6. ID column XGBoost 에러
  7. Hour feature enumerate 인덱스 사용

### WS1: Calibrated Tree Baseline (진행 중)
- E1.1: Feature Cache Sync ✅
- E1.2: XGBoost 모든 버그 수정 ✅
- E1.2.1: 최종 모델 학습 ✅
- E1.3: Calibration Study (진행 중)
- E1.4: Submission 완료 (점수: 0.1166)

## 현재 문제점

1. **Guardrail이 모델 성능을 저해**
   - 인위적인 분포 조정이 예측 품질 훼손
   - 원래 분포가 더 적절할 가능성

2. **낮은 점수 원인 분석 필요**
   - Feature engineering 개선 필요
   - 모델 파라미터 최적화 필요
   - Guardrail 제거 후 재학습 필요

## 다음 단계

1. **Guardrail 없는 모델 재학습** (012_xgboost_no_guardrail.py)
2. **Feature engineering 개선**
3. **다른 모델 시도** (LightGBM, CatBoost)
4. **앙상블 전략 개발**

## 파일 구조
```
plan4/
├── 007_xgboost_optuna.py (완료, 점수 0.1166)
├── 008_calibrated_submission.csv (Calibration 시도)
├── 009_extreme_calibration.py (극단적 calibration)
├── 010_raw_predictions.csv (원본 예측)
├── 010_mild_scaled.csv (약한 스케일링)
├── 011_xgboost_no_guardrail.py (피클 파일 없어서 실패)
├── 012_xgboost_no_guardrail.py (수정 중)
├── src/
│   ├── feature_engineering.py
│   ├── prediction_guardrails.py
│   └── score.py
└── PROGRESS_REPORT.md (현재 파일)
```

## 시간 기록
- 시작: 2025-09-28 23:00
- XGBoost 학습 완료: 2025-09-29 01:29
- 첫 제출: 2025-09-29 02:00
- 점수 확인: 0.1166 (매우 낮음)
- LightGBM/CatBoost 시작: 2025-09-29 02:35
- 현재: 2025-09-29 02:38

## 현재 진행 상황 (2025-09-29 12:33)
- 피드백 버그 3개 모두 수정 완료 ✅
  1. Import 경로 문제 수정 (prediction_guardrails.py에 check_predictions() 래퍼 함수 추가)
  2. WeightedLogLoss 공식 수정 (011 파일: 공식 평가식으로 변경)
  3. Guardrail 완전 제거 (012 파일: 모든 guardrail 로직 제거)
- 018_xgboost_truly_no_guardrail.py object dtype 오류 수정 후 재실행
- 앙상블 파일 생성 완료 (017_ensemble_weighted.csv 추천)

## 리소스 사용
- CPU: 64 cores 활용
- System Memory: 200GB+ 중 ~87GB 사용
- GPU Memory: 80GB 사용 가능 (현재 XGBoost는 CPU 모드)
- GPU: 미사용 (XGBoost CPU 모드)
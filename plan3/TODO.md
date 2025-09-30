# Plan3 TODO List

## 진행 중
- [ ] XGBoost 완료 대기 (026_massive_gpu_xgboost.py 실행 중)

## 대기 중 - 우선순위 높음 (성능 향상 가능성 높음)
- [ ] FT-Transformer 구현 (plan2에서 0.3168 달성)
- [ ] Meta-learning 모델 수정 및 재실행 (020)
- [ ] GPU 메모리 활용률 높은 모델 생성 (현재 36% → 80% 목표)
- [ ] GPU 최적화 앙상블 모델 생성

## 대기 중 - 추가 실험
- [ ] CatBoost GPU 모델 구현
- [ ] SAINT 모델 구현
- [ ] LightGBM DART 모드 구현
- [ ] TabNet 모델 개선 (011 개선)
- [ ] DCN-V2 모델 구현
- [ ] DeepFM 모델 구현

## 완료된 실험
- [x] 018 Modern Transformer (batch_size=4000) → 0.2030
- [x] 019 Temporal Optimized Model → 0.1774
- [x] 026 XGBoost 재실행 (실행 중)
- [x] 015 GPU Maximized → 0.2206
- [x] 014 Probing Strategy → 0.1982-0.2175

## 성능 기록
- **Best**: FT Transformer (plan2) - 0.3168
- **Target**: 0.351+
- **Current Best (plan3)**: 0.2206

## Notes
- GPU 메모리 사용률이 핵심 (메모리를 많이 사용할수록 성능 향상 가능)
- 현재 29GB/80GB (36%) 사용 중 → 목표 64GB+ (80%+)
- Batch size 증가 시 OOM 주의 (transformer는 4000이 한계였음)
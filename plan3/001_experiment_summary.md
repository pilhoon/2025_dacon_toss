# DACON TOSS 클릭 예측 대회 실험 정리

## 목표
- **대회 스코어 0.351+ 달성**
- 평가 지표: 0.7 * Average Precision Score + 0.3 / Weighted Log Loss

## 데이터셋 개요
- Train: 10,704,179개 샘플, 119개 피처
- Test: 1,527,298개 샘플, 119개 피처
- 클래스 불균형: 양성 클래스 비율 ~1.9% (1:52)

---

## Plan 1: 초기 모델링 및 기본 실험 (001-025)

### 실험 목적
- 데이터셋 이해 및 베이스라인 구축
- 다양한 모델 아키텍처 실험
- 초기 성능 벤치마크 설정

### 주요 실험 (총 25개 파일)

#### 1. 기본 모델 (001-011)
- **001-003**: 데이터 로드 및 간단한 베이스라인 구축
- **004**: 개선된 전처리 파이프라인
- **005**: XGBoost, LightGBM 비교
- **006**: Optuna를 통한 하이퍼파라미터 튜닝
- **007**: 앙상블 방법론 실험
- **008-010**: 제출 파이프라인 구축 및 전체 데이터셋 훈련
- **011**: 클래스 불균형 처리 (Balanced XGBoost)

#### 2. 딥러닝 모델 (012-025)
- **012-019**: DeepCTR 계열 모델 실험
  - 초기 모델부터 대규모 모델까지 점진적 확장
  - 메모리 효율성과 성능 트레이드오프 탐색
- **020**: Fixed XGBoost
- **021-024**: 대규모 DeepCTR 변형 모델들
  - Optimized, Massive, Ultra Batch, Mega Model
- **025**: 앙상블 실험

### 주요 결과
- 초기 XGBoost 베이스라인: CV ~0.33
- DeepCTR 모델: 메모리 문제로 안정성 이슈
- 개선된 XGBoost: CV ~0.345
- 대부분의 딥러닝 모델이 메모리 이슈로 실패

---

## Plan 2: 고성능 모델링 및 복잡한 실험 (001-062)

### 실험 목적
- GPU/CPU 리소스 최대 활용
- 복잡한 피처 엔지니어링
- 다양한 최첨단 모델 아키텍처 실험
- 0.351+ 스코어 달성

### 주요 실험

#### 1. 기반 모델 재구축 (001-030)
- **001-003**: DCNv2 모델 테스트
- **004-005**: XGBoost 개선
- **006-014**: 안정적인 딥러닝 모델 (TabNet, DeepFM, Entity Embedding)
- **015-030**: DeepCTR 최적화 (배치 사이즈, GPU 활용)

#### 2. 고급 모델링 (031-050)
- **033**: GPU 최적화 DeepCTR
- **035-039**: 병렬 처리 XGBoost, GPU 가속
- **040**: 안정적인 딥 모델
- **041**: TabNet 모델
- **042-043**: WLL 및 Ranking 최적화
- **044-045**: CatBoost, LightGBM DART
- **046-048**: FT Transformer (Feature Tokenizer Transformer)
- **050**: Adversarial Validation

#### 3. 최종 고성능 모델 (051-060)
- **051**: 고급 피처 엔지니어링
- **052**: CatBoost 최적화
- **053**: SAINT (Self-Attention and Intersample Attention Transformer)
- **054**: Pseudo Labeling
- **055**: Ultimate XGBoost
- **056**: Stacking Ensemble
- **057**: 메모리 효율적인 GPU 모델 (**최고 성능**)
- **058-060**: 057 모델 제출 파일 생성

### 057 모델 상세 (최고 성능)
- **모델**: XGBoost with GPU acceleration
- **피처 엔지니어링**:
  - 통계적 피처 (mean, std, skew, kurtosis 등)
  - 다항식 피처 (square, sqrt, log1p)
  - 상호작용 피처 (곱셈, 나눗셈, 덧셈, 뺄셈)
  - 클러스터링 기반 피처 (KMeans 10개 클러스터)
  - 총 300+ 피처 생성
- **모델 파라미터**:
  - max_depth: 15
  - learning_rate: 0.01
  - num_boost_round: 3000
  - GPU 가속: tree_method='gpu_hist'
- **결과**:
  - Fold 1: 0.350558
  - Fold 2: 0.351149
  - Fold 3: 0.350519
  - Fold 4: 0.350899
  - Fold 5: 0.351223
  - **평균 CV Score: 0.350885** (목표 0.351에 매우 근접)

### 현재 진행 상황

#### 완료된 실험
- **057 모델**: CV 0.350885 달성 (목표 0.351에 0.00012 부족) ✅
- **060 제출 파일 생성**: plan2/060_gpu_submission.csv ✅
- **055 Ultimate XGBoost**: 에러 발생 (KeyError: 'target') ❌
- **056 Stacking Ensemble**: 에러 발생 (FileNotFoundError: train.csv) ❌

#### 백그라운드 실행 중 (40+ 프로세스)
주요 실행 중인 모델들:
- 033: DeepCTR GPU Optimized
- 035: Parallel XGBoost
- 036: XGBoost Cached (2개 인스턴스)
- 037: GPU Maximized (2개 인스턴스)
- 039: XGBoost GPU Large
- 040: Stable Deep Model
- 041: TabNet Model
- 042: WLL Optimized
- 043: Ranking Optimized
- 044: CatBoost (3개 인스턴스)
- 045: LightGBM DART (3개 인스턴스)
- 046: FT Transformer (2개 인스턴스)
- 048: FT Transformer V2 (2개 인스턴스)
- 050: Adversarial Validation (2개 인스턴스)
- 051: Advanced Features
- 052: CatBoost Optimized
- 053: SAINT Model
- 054: Pseudo Labeling

#### 핵심 성과
- 가장 높은 CV Score: **0.350885** (057 모델)
- 제출 가능 파일: plan2/060_gpu_submission.csv

---

## Plan 3: 새로운 실험 방향

### 분석 및 인사이트
1. **성공 요인**:
   - 복잡한 피처 엔지니어링이 성능 향상의 핵심
   - GPU 가속이 대규모 모델 훈련을 가능하게 함
   - 깊은 트리(max_depth=15)가 복잡한 패턴 포착

2. **한계점**:
   - 0.351 목표에 0.00012 부족 (0.350885)
   - 단일 모델로는 한계 도달
   - 딥러닝 모델들이 XGBoost보다 낮은 성능

### 제안하는 실험 방향

#### 1. 앙상블 전략 강화
- 057 모델을 베이스로 다른 고성능 모델들과 앙상블
- Stacking, Blending, Voting 등 다양한 앙상블 기법
- 모델 다양성 확보를 위한 서로 다른 시드값 사용

#### 2. 피처 엔지니어링 고도화
- Target encoding with regularization
- Frequency encoding for categorical features
- Time-based features (if applicable)
- Feature selection using importance scores

#### 3. 후처리 최적화
- Calibration 파라미터 튜닝 (현재 power=1.08)
- Threshold optimization
- Ensemble weight optimization using validation scores

#### 4. 준지도 학습 활용
- Test 데이터의 고신뢰도 예측을 pseudo label로 활용
- Self-training with confidence threshold

#### 5. 메타 학습
- 각 fold의 예측을 피처로 활용하는 2차 모델
- Out-of-fold predictions 활용

### 우선순위 실험 계획
1. **즉시 실행**: 057 모델 + 완성된 고성능 모델들의 앙상블
2. **단기 실행**: Calibration 최적화 및 후처리
3. **중기 실행**: 새로운 피처 엔지니어링
4. **장기 실행**: 메타 학습 및 준지도 학습

### 목표
- **단기 목표**: 0.351+ 달성
- **중기 목표**: 0.352+ 달성
- **장기 목표**: 0.353+ 달성
## 목표

- **문제**: 유저가 어떤 광고 소재를 클릭할 확률(CTR)을 빠르고 정확하게 예측.
- **제약**: 실시간 서빙 가능해야 하며, 추론 주기가 다른 복수 모델의 공존 허용.
- **데이터**: `data/train.parquet` (약 1,070만), `data/test.parquet` (약 152만), 총 119/118 컬럼. 타깃은 `clicked`.

## 데이터 개요 요약

- 공통 메타: `gender`, `age_group`, `inventory_id`, `day_of_week`, `hour`, `seq`
- 속성 피처: `l_feat_*` (특히 `l_feat_14`는 Ads set), `feat_[a~e]_*`
- 과거 인기도 피처: `history_a_*`
- 레이블: `clicked` (train만)
- 제출: `ID`, `clicked` (확률)

## 평가 지표 및 목표

- 내부 검증: ROC-AUC(주), logloss(보조), PR-AUC(보조)
- 대회 채점: 확률 제출 기반 → logloss 또는 ROC-AUC일 가능성. 두 지표 모두 최적화/모니터링.
- **목표**: 베이스라인 대비 +0.02 AUC 이상 향상, 칼리브레이션 개선으로 logloss 안정화.

## 검증 전략

- 기본: Stratified K-Fold (K=5, shuffle, seed 고정) on `clicked`.
- 시간 누수 점검: `seq`/`hour` 기준의 time-aware 홀드아웃(예: 마지막 구간)으로 최종 sanity check.
- 대안: GroupKFold(`inventory_id`)로 유사 분포 누수 점검.
- 리포팅: 각 fold AUC/Logloss, OOF 전체 AUC/Logloss, 홀드아웃 성능.

## 전처리/피처링 원칙

- 결측 처리: 수치는 median, 범주는 most_frequent/constant.
- 범주 인코딩: 1차는 OrdinalEncoding(안정/경량), 이후 Target/Freq/One-hot 혼합 실험.
- 고카디널리티: `inventory_id`, `l_feat_14` 등은 count/target encoding + 스무딩, CV 누수 방지.
- 상호작용: (`inventory_id`×시간대), (`age_group`×`gender`), 주요 `l_feat_*` 조합.
- 집계 피처: 최근 구간별 클릭율/노출 대비 클릭율 등 history 기반 집계(자료 허용 범위 내).
- 스케일링: 트리 계열은 불필요. 선형/신경망 계열에서만 적용.

## 모델 로드맵

1) 베이스라인 트리
- Scikit-learn HistGradientBoostingClassifier로 빠른 베이스라인 구축.
- 성능/메모리/속도 기준점 수립, 피처 중요도 점검.

2) Feature v1
- 카테고리 정제, frequency/target encoding, 기본 상호작용, count/ratio 집계 도입.

3) GBDT 계열 확대
- XGBoost/LightGBM/CatBoost 실험. CPU/GPU 리소스에 따라 선택.
- 조기중단, 하이퍼파라미터 튜닝(Optuna) 적용.

4) 신경망 계열(Wide & Deep)
- 범주 임베딩 + 수치 피처 결합. 작은 아키텍처로 속도/성능 타협.

5) Sequence-aware CTR (선택)
- `seq` 순서/`history_*` 활용. DIN/Transformers 기반 경량 모델 검토.

6) 앙상블/스태킹/칼리브레이션
- 단순 가중 블렌딩 → 메타 러너(로지스틱 회귀/작은 GBDT).
- 확률 칼리브레이션(Platt/Isotonic)으로 logloss 안정화.

7) 추론/제출/서빙
- 단일 명령으로 테스트 추론/제출 파일 생성.
- ONNX/TorchScript(해당 시)로 경량화, 실시간 SLA 검증.

## 실험 관리

- `plan1/experiments/`에 각 실험별 폴더 생성: 설정(config), 메트릭, OOF, 피처 중요도, 모델 아티팩트.
- 공통 설정은 YAML로 관리: 데이터 경로, 컬럼 선택 패턴, CV, 모델/튜닝 파라미터.

## 성능/자원 전략

- 데이터 I/O: PyArrow Parquet, 필요한 컬럼만 선택 로딩, `n_rows` 샘플링 옵션.
- 메모리: dtype downcast, 카테고리형 활용.
- 병렬: scikit-learn n_jobs, GBDT 백엔드의 스레드 활용.

## 리스크와 대응

- 누수: target encoding과 시간 축 혼합 시 fold 내 계산 철저.
- 분포 차이: 시간 홀드아웃 성능을 모니터링.
- 고카디널리티: 과적합 → 스무딩/정규화, 드롭/빈 합치기.

## 의사결정 게이트와 분기

1) 베이스라인 성능 게이트
- 조건: OOF AUC < 0.70 또는 logloss 개선 미미 → 데이터/전처리 문제 우선 점검.
- 액션: 결측/범주 처리 재검토, high-cardinality 컬럼 분포/희소도 점검, 단순 누수 탐지.
- 통과 시: Feature v1로 진행.

2) Feature v1 효과 게이트
- 조건: AUC +0.01 미만 향상 또는 logloss 개선 없음 → 인코딩/집계 전략 재설계.
- 액션: target encoding 스무딩/카테고리 합치기, 교차 카운트/시간대별 CTR 집계 강화.
- 통과 시: GBDT 계열 확대 및 HPO로 진행.

3) GBDT 확대/HPO 게이트
- 조건: XGB/LGBM/CatBoost 중 어느 하나가 OOF AUC 최고점 갱신 실패 → 모델 수 축소, 과적합 점검.
- 액션: 조기중단/정규화 강화, 중요 피처 50~200 범위로 제한 실험.
- 통과 시: Wide&Deep로 NN 도입 검토.

4) NN 도입 게이트
- 조건: Wide&Deep가 GBDT 대비 +0.005 AUC 이상 또는 logloss 유의 개선 시 채택.
- 미달 시: NN 라인 보류, GBDT 라인 튜닝 지속.

5) 시퀀스 모델 게이트(선택)
- 조건: `seq`/`history_*` 활용 시 +0.003 AUC 이상 혹은 특정 세그먼트(신규/재방문) 개선 명확.
- 미달 시: 시퀀스 라인은 보류하고 블렌딩으로 효과 흡수.

6) 앙상블/칼리브레이션 게이트
- 조건: 단일 최고 모델 대비 블렌딩이 CV에서 logloss 일관 개선.
- 미달 시: 단일 모델 + 칼리브레이션으로 단순화.

7) 서빙/속도 게이트
- 조건: p95 추론 지연 및 메모리 예산 내 충족. 미달 시 경량화(특징 축소/양자화/ONNX) 우선.

## 마일스톤 및 체크리스트

- [ ] 데이터 로더/스키마 요약 유틸
- [ ] 평가/스플리터 유틸(Stratified, time-aware)
- [ ] 베이스라인 GBDT 학습 스크립트 + OOF/리포트
- [ ] Feature v1 엔지니어링
- [ ] GBDT 계열 확장 및 튜닝
- [ ] Wide & Deep
- [ ] Sequence-aware CTR
- [ ] 블렌딩/스태킹/칼리브레이션
- [ ] 추론 파이프라인/제출 생성
- [ ] 서빙/경량화

## 다음 할 일(단기)

1) parquet 데이터 로더와 스키마 요약 유틸 추가
2) StratifiedKFold/시간 홀드아웃 스플리터 유틸 추가
3) 베이스라인 HistGBDT 학습 스크립트/설정/실험 디렉토리 준비



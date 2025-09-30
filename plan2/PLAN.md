## plan2: Deep CTR Master Plan

### Guiding Principles
- 성능 우선: 파라미터 규모, 학습 시간 증가 허용. 재현성은 seed로 확보.
- 분기형 실험: 각 단계에서 채택 임계치(게이트)를 만족하면 다음 단계로 진행.
- 공통 파이프라인: 동일 데이터 로더/전처리/메트릭으로 공정 비교.

### Compute Profile
- HW: NVIDIA A100 80GB (단일)
- Wall time: 제한 없음
- Precision: AMP 사용(bfloat16 우선), 필요 시 자동으로 float32 fallback

### Data & Features
- 데이터: plan1과 동일(train/test parquet). 타깃 `clicked`.
- 인코딩: 범주형은 Embedding(빈도 하한/rare bucket), 수치는 Standardize.
- 시퀀스: `seq`, `history_*`를 클릭 이력으로 사용. 길이 컷오프와 마스킹.

### Models (in order)
1) DCNv2 (CrossNetwork + Deep MLP)
2) xDeepFM (CIN + Deep)
3) FT-Transformer (tabular transformer)
4) DIN (attention on user behavior to target ad set: `l_feat_14`)
5) Two-Tower Retrieval (user/ad embeddings) + Reranker (FT-Transformer)

### Training Strategy
- Loss: Weighted LogLoss(WLL) + Focal loss 실험.
- Class imbalance: 기본값으로 pos_weight ≈ N_neg/N_pos 적용, focal은 대안으로 게이트 평가.
- Optimizer: AdamW, cosine decay with warmup.
- Regularization: dropout, L2, stochastic depth(FT-Transformer), mixout(실험).
- Calibration: temperature scaling / isotonic on val.
- Precision: AMP(bf16) 기본값, large-batch 우선. OOM 시 grad accumulation 사용.

### Additional Gates (post-plan1)
- Predicted distribution gate: val 예측 평균≈0.0191±0.005, 표준편차>0.05 미만 시 파라미터/정규화 조정.
- Calibration gate: 온도 스케일링/Isotonic 중 WLL 더 낮춘 방법 채택.
- Batch gate: 유효 배치가 256K↑에서 성능 저하 시 32K~131K로 축소.
- Evaluation: AP, WLL, ROC-AUC, PR-AUC. Early stopping on composite metric.

### Decision Gates
- Gate A (Baseline DL viability): DCNv2가 plan1 최고 GBDT 대비
  - AP +0.010 이상 또는 WLL 2% 이상 개선 → 채택, 다음 모델로.
  - 미달 → 특징 공학 개선(rare bucketing, interaction embeddings) 후 재시도.
- Gate B (xDeepFM vs DCNv2): xDeepFM가 DCNv2 대비 AP +0.005 이상 → 채택, FT-Transformer 진행.
- Gate C (FT-Transformer): FT-Transformer가 AP +0.005 또는 WLL 2% → 채택.
- Gate D (DIN): 시퀀스 사용 시 세그먼트(AP 신규/재방문) 이득 확인되면 채택.
- Gate E (Two-Tower + Rerank): 리콜@K 상승이 전체 AP 향상으로 연결되면 채택.

### Experiments (step-by-step, medium-sized)
1. Data pipeline v2
   - Embedding vocab build(빈도 하한), 수치 스케일링, train/val split 재현성.
   - 산출물: vocab.json, stats.json, fold indices.

2. DCNv2 v1
   - 임베딩 dim=16, cross_depth=3, mlp [256,128,64], dropout 0.1.
   - 5-fold OOF, AP/WLL 기록, 체크포인트 저장.

3. DCNv2 v2 ablation
   - 임베딩 dim sweep [16,32,64], cross_depth [2,3,4].
   - 게이트 A 평가.

4. xDeepFM v1
   - CIN layer sizes [128,128], deep [256,128,64].
   - 게이트 B 평가.

5. FT-Transformer v1
   - n_layers=4, d_model=256, n_heads=8, sd=0.1.
   - token-wise dropout, numerical Fourier features 실험.
   - 게이트 C 평가.

6. DIN v1 (sequence)
   - `history_*`를 user behavior로, target은 `l_feat_14`.
   - max_len=50, attention pooling, mask.
   - 세그먼트별(AP 신규/재방문) 평가, 게이트 D.

7. Two-Tower + Rerank
   - user/ad embedding으로 recall@K 향상 → 상위 K rerank는 FT-Transformer.
   - 게이트 E.

8. Calibration & Ensemble
   - temperature scaling/isotonic per fold.
   - 단순 가중 + stacking(LogReg)로 AP/WLL 개선 여부 확인.

9. Large-scale full training
   - best DL 모델 전체 데이터 재학습, OOF/홀드아웃, 제출 파이프라인.

### Artifacts & Logging
- 각 실험 폴더에: config, metrics.json, ckpt, tensorboard logs, oof.csv, preds_stats.json.

### Risks
- 시퀀스 구성 오류 → unit test로 마스킹/길이 검증.
- 메모리 초과 → gradient checkpointing/AMP/accumulation.



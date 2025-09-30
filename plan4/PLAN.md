## Plan4 Roadmap

### 1. Background Snapshot
- **Plan1** established tree-based baselines and surfaced the metric tension: public LB best 0.31631 (XGBoost, high mean prob) vs. over-regularised LightGBM at 0.21436. Key issues: poor AP when variance is low, poor WLL when mean drifts from the base rate (~1.9%).
- **Plan2** focused on DeepCTR variants. Stable DCN/DeepFM runs achieved offline score >0.45 when tuned, but leaderboard replication remains unverified because the competition metric definition diverged across notes.
- **Plan3** delivered heavy feature engineering (≈300 engineered fields) and GPU XGBoost with CV score 0.350885, yet public submissions are still capped at 0.316~0.317 due to calibration/format issues and missing metric parity.
- **Gap**: There is no unified offline pipeline reproducing the official competition score, and calibration + ensembling strategies are not yet consolidated.

### 2. Objectives & Success Criteria
| Horizon | Target | Success Signal |
|---------|--------|----------------|
| Short-term | Reproduce public LB 0.31631 with automated pipeline | Local run matches LB ±0.002 using held-out folds & metric script |
| Mid-term | Break 0.331 on public LB | Improved AP without harming WLL (Δscore ≥ +0.015 vs best) |
| Final | Achieve ≥0.351 public LB | Blend of calibrated GBDT + deep models beats leaderboard gate |

Risk tolerance: permit controlled exploration (≤20% wall-clock) while keeping a calibrated production track.

### 3. Guiding Principles
1. **Metric fidelity first** – fail-fast on any model that cannot meet the score gate on the replicated metric.
2. **Distribution alignment** – enforce prediction mean/variance checks to balance AP↔WLL trade-offs.
3. **Model diversity** – maintain at least one high-performing tree model and one neural model for ensemble gains.
4. **Documentation & versioning** – each experiment produces config, metrics.json, preds_stats.json, and if applicable submission CSV.

### 4. Workstreams & Gating
- **WS0: Metric & Data Audit**
  - Build `score.py` implementing official formula (needs confirmation from docs/discussions).
  - Validate folds (StratifiedKFold vs time-aware holdout) and data leakage checks.
  - Gate: offline metric vs public LB delta <0.005 for baseline model.

- **WS1: Calibrated Tree Baseline**
  - Re-run Plan3 `057` feature set with controlled hyperparameters, re-tune `scale_pos_weight`, learning rate, monotonic constraints.
  - Apply post-hoc calibration (isotonic, temperature, beta calibration) using score metric.
  - Gate: offline score ≥0.33, prediction mean within 0.018–0.022, std ≥0.05.

- **WS2: Neural Track Refresh**
  - Port Plan2 DCN/DeepFM runs into unified pipeline (PyTorch Lightning or DeepCTR with wrappers) using same feature set.
  - Introduce sequence-aware module (DIN-lite) focusing on `seq` + `l_feat_14`.
  - Gate: models must outperform calibrated tree on AP while keeping WLL within +5%.

- **WS3: Feature Expansion & Selection**
  - Systematically ablate Plan3 engineered features; add time-lag CTR stats, recency decay, conditional probabilities.
  - Run SHAP/feature importance to prune redundancy → reduce overfitting risk before ensemble.
  - Gate: offline score gain ≥+0.005 or maintain score with >20% feature reduction (efficiency win).

- **WS4: Ensemble & Meta-Modeling**
  - Blend top tree + neural models (logistic stacking, weighted average tuned on validation).
  - Explore rank averaging vs probability averaging; integrate calibration after blending.
  - Gate: ensemble must beat best single model by ≥0.01 score offline before submission.

- **WS5: Submission Operations**
  - Automate inference, score logging, and submission artifact packaging.
  - Track score drift between folds and public LB; trigger recalibration if drift >0.01.

### 5. Deliverables
- `plan4/EXPERIMENT_ROADMAP.md`: ordered experiments with configs, expected outputs, and stop criteria.
- `plan4/RESEARCH_TOPICS.md`: open questions needing external references (metric definition, feature semantics, advanced calibration).
- `plan4/CHECKLIST.md`: rolling status board for each workstream.
- Updated scripts: metric scorer, calibration utilities, ensemble runner (to be scoped separately).

### 6. Dependencies & Resources
- **Data**: `data/train.parquet`, `data/test.parquet`, engineered feature cache from Plan3 if available.
- **Compute**: 1× A100 80GB (neural track), CPU cluster for large XGBoost (16+ cores, ≥128GB RAM).
- **Tooling**: PyArrow, XGBoost GPU, LightGBM, CatBoost, PyTorch (Lightning), DeepCTR-Torch, Optuna, mlflow or simple logging.

### 7. Risk Register
| Risk | Impact | Mitigation |
|------|--------|------------|
| Metric mismatch | Invalid model ranking | Prioritize WS0; back-test with historical submissions |
| Prediction mean drift | WLL spike, LB drop | Hard checks in pipeline; scale_pos_weight sweep + calibration |
| GPU OOM | Training halts | Gradient accumulation, mixed precision, batch size tuning |
| Feature leakage | Inflated offline score | Strict fold separation, time-aware validation |
| Dev fragmentation | Repeated work | Shared configs + run logs per experiment |

### 8. Next Actions (immediate)
1. Implement official score computation and reconcile formula discrepancies (0.5/0.5 vs 0.7/0.3 weighting).
2. Reproduce Plan1 XGBoost submission with new scorer to set calibration targets.
3. Draft detailed experiment cards in `plan4/EXPERIMENT_ROADMAP.md` with owners, ETA, gating metrics.


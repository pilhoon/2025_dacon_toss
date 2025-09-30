# Plan4 Experiment Roadmap

## Phase 0 – Metric & Pipeline Alignment (WS0)
1. **E0.1 Score Reproduction**
   - Build `plan4/src/score.py` implementing official score. Compare both hypotheses:
     - `0.5*AP + 0.5*(1/(1+WLL))`
     - `0.7*AP + 0.3/(WLL)`
   - Backfill with historical submissions (`plan1/010_xgboost_submission.csv`, `plan3/046_ft_transformer.csv`).
   - Deliverable: `metrics_validation.md` documenting which formula matches leaderboard values.
   - Gate: |offline_score − public_score| ≤ 0.005 on at least two past submissions.

2. **E0.2 Validation Scheme Audit**
   - Re-run Plan1 5-fold Stratified CV and Plan3 time-aware split using score metric.
   - Investigate leak risk from `seq`/`history_*` by ensuring no future info enters folds.
   - Deliverable: `validation_report.json` with per-fold metrics + distribution stats.

3. **E0.3 Prediction Distribution Guardrails**
   - Add checks for mean, std, min/max, quantiles. Align to target mean 0.019±0.002 and std ≥0.05.
   - Deliverable: `pred_stats_template.json` and integration into training scripts.

## Phase 1 – Calibrated Tree Baseline (WS1)
4. **E1.1 Feature Cache Sync**
   - Port Plan3 engineered features into reproducible pipeline (document dataverse, caching rules).
   - Ensure dtype consistency and missing value handling alignment with Plan1.

5. **E1.2 XGBoost Baseline Refresh**
   - Start from Plan1 `010_xgboost_submission.py` settings. Sweep `scale_pos_weight`, `max_depth`, `eta`, `max_leaves` using Optuna but bound search space to avoid overfitting.
   - Evaluate on metric scorer. Stop early if score <0.30 after 3 folds.

6. **E1.3 Calibration Study**
   - Compare isotonic regression, Platt scaling, beta calibration, and `power` scaling used in Plan3.
   - Use nested CV to avoid leakage. Track AP, WLL, final score, and reliability plots.
   - Select method that maximizes score without increasing WLL by >5%.

7. **E1.4 Submission Dry Run**
   - Generate submission with new calibration; confirm offline/on-line parity.
   - Update `submission_log.md` with predictions stats + score comparison.

## Phase 2 – Neural Track Refresh (WS2)
8. **E2.1 Unified Data Loader**
   - Wrap DeepCTR input preparation to use same feature definitions as Phase 1 (shared vocab, scaling).
   - Add dynamic padding for `seq`, create mask tensors, confirm throughput on A100.

9. **E2.2 DCN Re-baseline**
   - Train DCN with moderate dimensions (embedding 24, cross 3) and ensure stable training (fp16 AMP, gradient clipping).
   - Evaluate offline score. Gate: +0.005 AP vs calibrated XGBoost.

10. **E2.3 DIN-lite Prototype**
    - Derive user behaviour sequence from `seq` tokens mapped onto target slot `l_feat_14`.
    - Implement additive attention with 50-step truncation, incorporate recency decay feature.
    - Track inference latency; keep under 50ms per 10k rows.

11. **E2.4 Neural Model Calibration**
    - Apply temperature scaling (per fold) and Dirichlet calibration.
    - Document WLL change to ensure neural models do not degrade second term of competition score.

## Phase 3 – Feature Expansion & Selection (WS3)
12. **E3.1 Target Encoding with Leakage Control**
    - Implement K-fold target encoding (Plan1 issue) with out-of-fold means and smoothing.
    - Features: `inventory_id`, `l_feat_14`, `age_group`, `gender`, combos.
    - Validate using metric scorer; drop if score decreases >0.005.

13. **E3.2 Time-window Aggregates**
    - Build rolling CTR stats by hour, day_of_week, inventory bins using only historical folds.
    - Evaluate contributions via SHAP to ensure they add unique signal.

14. **E3.3 Feature Pruning**
    - Run SHAP/feature importance from XGBoost & LightGBM; drop bottom 20% features and re-train.
    - Check for score drop; if stable, prefer reduced set for neural models to speed training.

## Phase 4 – Ensemble & Meta-modeling (WS4)
15. **E4.1 Blend Strategy Search**
    - Collect calibrated predictions from top 3 models (tree, DCN, DIN/FT-Transformer).
    - Optimize weights via Bayesian search on validation folds maximizing competition score.
    - Compare mean, rank-average, and logit-average blending.

16. **E4.2 Stacking Logistic Regressor**
    - Train meta-learner on OOF predictions + key features (e.g., base rate, inventory CTR).
    - Cross-validate to avoid leakage. Guard rails for overfitting (regularization, limited features).

17. **E4.3 Post-stack Calibration**
    - After stacking, re-run calibration methods; pick variant that stabilizes WLL and keeps AP gains.

18. **E4.4 Submission Ladder**
    - Prepare 3 candidate submissions (conservative, balanced, aggressive weightings).
    - Submit sequentially based on offline ranking; track LB results for future calibration.

## Phase 5 – Monitoring & Continuous Improvement (WS5)
19. **E5.1 Drift Monitoring**
    - After each submission, compute delta between offline folds and LB metric; update `drift_log.md`.
    - Investigate >0.01 deviations (check feature drift, calibration, time splits).

20. **E5.2 Incremental Learning**
    - Explore pseudo-labeling using high-confidence test predictions (prob >0.9 or <0.001) and re-train tree model.
    - Only proceed if drift monitoring suggests stable alignment.

21. **E5.3 Documentation & Roll-up**
    - Summarize outcomes per phase in `plan4/STATUS_REPORT.md` with lessons for next iteration.

## Experiment Governance
- Each experiment stores artifacts under `plan4/experiments/E##_{short_desc}/` with config, logs, metrics, preds.
- Use consistent random seeds (`seed=2025`) unless exploring variance.
- Halt criteria: if a phase fails gating twice, escalate review before additional runs.


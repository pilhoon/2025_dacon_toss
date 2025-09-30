# Plan4 Workstream Checklist

## WS0 Metric & Data Audit
- [ ] Implement official score calculator and validate vs public submissions
- [ ] Standardize validation splits (stratified + time-aware)
- [ ] Add prediction distribution guardrails to training outputs

## WS1 Calibrated Tree Baseline
- [ ] Port Plan3 engineered features into reproducible pipeline
- [ ] Re-run XGBoost baseline with controlled hyperparameters
- [ ] Compare calibration methods (isotonic / Platt / beta / power)
- [ ] Produce new calibrated submission and log metrics delta

## WS2 Neural Track Refresh
- [ ] Create shared data loader with sequence support
- [ ] Train stable DCN baseline with current feature set
- [ ] Prototype DIN-lite with recency-aware attention
- [ ] Calibrate neural outputs to maintain low WLL

## WS3 Feature Expansion & Selection
- [ ] Implement leakage-safe target encoding
- [ ] Generate time-window CTR aggregates and evaluate impact
- [ ] Perform feature pruning using importance/SHAP analysis

## WS4 Ensemble & Meta-modeling
- [ ] Collect calibrated predictions from top models
- [ ] Optimize blend weights for competition score
- [ ] Train stacking meta-learner and evaluate overfitting risk
- [ ] Calibrate final ensemble outputs

## WS5 Monitoring & Continuous Improvement
- [ ] Record offline vs leaderboard deltas per submission
- [ ] Evaluate pseudo-labeling feasibility with drift safeguards
- [ ] Compile status reports summarizing lessons per phase

Progress is reviewed after each submission cycle; unchecked items roll into the next iteration.

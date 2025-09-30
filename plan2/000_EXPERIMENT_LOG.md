# plan2 Experiment Log

- Format: timestamp | step | brief | key metrics/artifacts
- Keep entries concise; link to folders when relevant.

2025-09-17 00:00 | INIT | Initialized plan2 logging | created this file

2025-09-17 15:01:07 | 001 Data pipeline v2 | prepared (n_rows=200000) vocabs/stats/folds
  - train=(200000, 118), test=(200000, 117)
  - n_cat=78, n_num=39
  - pos_rate=0.01980
  - vocab_total=49,978 across 78 fields
  - folds=5 | counts={0: 40000, 1: 40000, 2: 40000, 3: 40000, 4: 40000}
  - out=plan2/experiments/001_data_v2

2025-09-17 16:00 | DCNv2 experiments | NaN loss issues encountered
  - Attempted multiple configurations
  - Issue: Gradient explosion with high class imbalance (pos_weight=51)
  - Tried: Lower LR, gradient clipping, smaller model, disabled AMP
  - Result: Training unstable, pivoting to GBDT baseline

2025-09-17 16:05 | Decision: Focus on XGBoost optimization
  - Plan1 best: XGBoost with score 0.31631
  - Deep learning models require more stable training setup
  - Priority: Improve XGBoost to reach target 0.349


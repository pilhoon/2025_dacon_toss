# Metrics Validation Report

## Summary

- Analyzed 24 submission files
- Official Formula: `Score = 0.5*AP + 0.5*(1/(1+WLL))`
- Alternative Formula: `Score = 0.7*AP + 0.3/WLL`

## Guardrail Requirements

- Mean: [0.017, 0.021]
- Std: >= 0.055 (updated from 0.05)

## Submission Analysis

| File | Samples | Mean | Std | Guardrails |
|------|---------|------|-----|------------|
| plan1/010_xgboost_submission.csv | 1,527,298 | 0.28854 | 0.18479 | ✗ |
| plan1/011_balanced_submission.csv | 1,527,298 | 0.00019 | 0.00124 | ✗ |
| plan1/023_ultra_batch_submission.csv | 1,527,298 | 0.14456 | 0.30543 | ✗ |
| plan1/submission.csv | 1,527,298 | 0.02046 | 0.02147 | ✗ |
| plan2/030_deepctr_best_submission.csv | 1,527,298 | 0.03232 | 0.07859 | ✗ |
| plan2/032_ensemble_conservative_submission.csv | 1,527,298 | 0.25010 | 0.16053 | ✗ |
| plan2/032_ensemble_rank_average_submission.csv | 1,527,298 | 0.27372 | 0.13634 | ✗ |
| plan2/032_ensemble_weighted_submission.csv | 1,527,298 | 0.21167 | 0.13736 | ✗ |
| plan2/036_xgboost_cached_submission.csv | 1,527,298 | 0.01906 | 0.02336 | ✗ |
| plan2/039_xgboost_gpu_large_submission.csv | 1,527,298 | 0.01874 | 0.02369 | ✗ |
| plan2/040_stable_deep_submission.csv | 1,527,298 | 0.01907 | 0.00080 | ✗ |
| plan2/041_tabnet_submission.csv | 1,527,298 | 0.01907 | 0.00811 | ✗ |
| plan2/042_wll_optimized_submission.csv | 1,527,298 | 0.03402 | 0.03330 | ✗ |
| plan2/043_ranking_optimized_submission.csv | 1,527,298 | 0.08021 | 0.03457 | ✗ |
| plan2/045_lightgbm_dart_submission.csv | 1,527,298 | 0.24624 | 0.10859 | ✗ |
| plan2/046_ft_transformer_submission.csv | 1,527,298 | 0.20405 | 0.08662 | ✗ |
| plan2/060_gpu_submission.csv | 1,527,298 | 0.00218 | 0.00151 | ✗ |
| plan3/005_aggressive_submission.csv | 1,527,298 | 0.00149 | 0.00110 | ✗ |
| plan3/005_best_calibrated_submission.csv | 1,527,298 | 0.00338 | 0.00216 | ✗ |
| plan3/005_conservative_submission.csv | 1,527,298 | 0.00338 | 0.00216 | ✗ |
| plan3/005_original_calibrated_submission.csv | 1,527,298 | 0.00218 | 0.00151 | ✗ |
| plan3/009_dacon_submission.csv | 1,527,298 | 0.00338 | 0.00216 | ✗ |
| plan3/018_transformer_submission.csv | 1,527,298 | 0.01904 | 0.00000 | ✗ |
| plan3/026_massive_gpu_xgboost_submission.csv | 1,527,298 | 0.01742 | 0.02086 | ✗ |

## Recommendations

1. Use official formula: `0.5*AP + 0.5*(1/(1+WLL))`
2. Ensure predictions have std >= 0.055 for better AP
3. Target mean prediction around 0.019 ± 0.002
4. Apply calibration methods to balance AP and WLL

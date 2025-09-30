# Plan3 Submission Scores

## 015_gpu_maximized_formatted.csv
- **Score: 0.2206231841**
- Date: 2025-09-21 00:34:51
- Model: GPU Maximized Deep Model (013)
- Validation AUC: 0.7339
- ID format: TEST_XXXXXXX (fixed format)

## 015_probe_low_ctr_formatted.csv
- **Score: 0.1981875607**
- Date: 2025-09-21 00:03:35
- Model: Probing Strategy - Low CTR hypothesis (014)
- Predictions reduced by 30%

## 015_probe_no_f1_formatted.csv
- **Score: 0.2086345778**
- Date: 2025-09-20 15:55:55
- Model: Probing Strategy - No f_1 feature (014)
- Feature f_1 excluded from training

## 015_probe_temporal_formatted.csv
- **Score: 0.2175164997**
- Date: 2025-09-20 15:40:17
- Model: Probing Strategy - Temporal hypothesis (014)
- Only recent 50% of data used for training

## 009_dacon_submission.csv
- **Score: 0.1350528416** ❌ (worse than baseline)
- Date: 2025-09-20
- Based on: 005_best_calibrated_submission.csv
- Calibration: power=1.30
- ID format: TEST_XXXXXXX
- Prediction stats:
  - Mean: 0.003373
  - Std: 0.002148
  - Min: 0.00000721
  - Max: 0.02340469

## 029_transformer_formatted.csv
- **Score: 0.2030194659**
- Date: 2025-09-23
- Model: Modern Transformer (batch_size=4000)

## 030_temporal_formatted.csv
- **Score: 0.1773401863**
- Date: 2025-09-23
- Model: Temporal Optimized Model (019)

## Comparison
- **046 FT Transformer: 0.3167889377** ✅ (best)
- **015 GPU Maximized: 0.2206231841**
- **029 Transformer: 0.2030194659**
- **030 Temporal: 0.1773401863**
- **009 submission: 0.1350528416** ❌ (worst)

## Notes
- Competition score is HIGHER is better (0.7 * AP + 0.3 / WLL)
- All plan3/015 files have been submitted with correct TEST_XXXXXXX format
- Waiting for clearer better model before new submissions per user instruction
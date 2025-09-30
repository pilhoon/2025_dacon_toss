# Validation Scheme Comparison Report

## Summary

- Data samples: 100,000
- Positive rate: 0.01936
- Tested schemes: 3

## OOF Performance Comparison

| Scheme | AP | WLL | Score | Pred Mean | Pred Std | Guardrails |
|--------|----|----- |-------|-----------|----------|------------|
| StratifiedKFold_5 | 0.04894 | 0.73236 | 0.31309 | 0.26879 | 0.18058 | ✗ |
| StratifiedGroupKFold_5 | 0.03831 | 0.86792 | 0.28683 | 0.20599 | 0.17549 | ✗ |
| TimeSeriesSplit_5 | 0.03962 | 3.64698 | 0.12741 | 0.17287 | 0.17647 | ✗ |

## Fold Stability Analysis


### StratifiedKFold_5

- Score Mean: 0.31421
- Score Std: 0.00417
- CV Stability: 1.33%

| Fold | N Train | N Val | Val Pos Rate | Score |
|------|---------|-------|--------------|-------|
| 1 | 80,000 | 20,000 | 0.01935 | 0.30842 |
| 2 | 80,000 | 20,000 | 0.01935 | 0.31874 |
| 3 | 80,000 | 20,000 | 0.01935 | 0.31799 |
| 4 | 80,000 | 20,000 | 0.01935 | 0.31572 |
| 5 | 80,000 | 20,000 | 0.01940 | 0.31018 |

### StratifiedGroupKFold_5

- Score Mean: 0.31035
- Score Std: 0.03481
- CV Stability: 11.22%

| Fold | N Train | N Val | Val Pos Rate | Score |
|------|---------|-------|--------------|-------|
| 1 | 88,230 | 11,770 | 0.02719 | 0.31275 |
| 2 | 99,594 | 406 | 0.02709 | 0.37152 |
| 3 | 86,625 | 13,375 | 0.02714 | 0.31178 |
| 4 | 35,989 | 64,011 | 0.01598 | 0.26789 |
| 5 | 89,562 | 10,438 | 0.02098 | 0.28782 |

### TimeSeriesSplit_5

- Score Mean: 0.28993
- Score Std: 0.02700
- CV Stability: 9.31%

| Fold | N Train | N Val | Val Pos Rate | Score |
|------|---------|-------|--------------|-------|
| 1 | 16,670 | 16,666 | 0.01938 | 0.24169 |
| 2 | 33,336 | 16,666 | 0.01752 | 0.27867 |
| 3 | 50,002 | 16,666 | 0.02064 | 0.30856 |
| 4 | 66,668 | 16,666 | 0.02058 | 0.30841 |
| 5 | 83,334 | 16,666 | 0.01848 | 0.31234 |

## Recommendations

1. **Best performing scheme**: StratifiedKFold_5 (Score: 0.31309)
2. Consider using StratifiedGroupKFold if group leakage is a concern
3. TimeSeriesSplit is valuable for temporal validation
4. Monitor fold stability - lower variance indicates more reliable CV

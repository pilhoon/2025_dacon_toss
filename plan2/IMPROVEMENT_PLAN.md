# Plan2 개선 계획

## 현재 상황
- **Plan1 XGBoost**: 0.3163 (최고 성능)
- **Plan2 DeepCTR**: 0.1384 (실망스러운 결과)
- **Gap**: -0.1779 (56% 하락)

## 문제 분석

### 1. Overfitting 심각
- Training AUC: 0.9973 (너무 높음)
- Test predictions: 극단적 분포 (median 0.001)
- Validation split 없이 학습한 것이 원인

### 2. 예측값 Calibration 문제
```
Prediction stats:
- Mean: 0.032
- Median: 0.001  # 너무 낮음
- Std: 0.079
- Positive rate: 0.48%  # 실제는 1.9%
```

## 즉시 개선 방안

### 1. Regularization 강화
```python
model = DCN(
    # ...
    dnn_dropout=0.3,  # 0.15 → 0.3
    l2_reg_embedding=1e-4,  # 1e-5 → 1e-4
    l2_reg_linear=1e-4,
    l2_reg_dnn=1e-4
)
```

### 2. Early Stopping 적용
```python
# Validation split 필수
history = model.fit(
    train_input, y_train,
    validation_split=0.2,  # 0.0 → 0.2
    epochs=12,
    patience=3  # Early stopping
)
```

### 3. Calibration 적용
```python
from sklearn.isotonic import IsotonicRegression

# Validation set으로 calibration
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(val_predictions, val_labels)
test_predictions_calibrated = iso_reg.transform(test_predictions)
```

### 4. Ensemble with XGBoost
```python
# XGBoost가 더 안정적이므로 높은 가중치
final_predictions = 0.7 * xgboost_pred + 0.3 * deepctr_pred
```

## 새로운 실험 코드

### 031_deepctr_regularized.py
```python
#!/usr/bin/env python3
"""
Regularized DeepCTR with validation and early stopping
"""

def train_regularized_model():
    # 1. Data split with validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels
    )

    # 2. Model with more regularization
    model = DCN(
        cross_num=3,  # Reduce complexity
        dnn_hidden_units=(256, 128, 64),  # Smaller
        dnn_dropout=0.3,  # More dropout
        l2_reg_embedding=1e-4,  # 10x stronger
        l2_reg_linear=1e-4,
        l2_reg_dnn=1e-4
    )

    # 3. Train with validation
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=50000,  # Smaller batch
        epochs=20,
        early_stopping=True,
        patience=3
    )

    # 4. Calibration
    val_pred = model.predict(X_val)
    iso_reg = IsotonicRegression()
    iso_reg.fit(val_pred, y_val)

    # 5. Calibrated predictions
    test_pred_raw = model.predict(test_data)
    test_pred_calibrated = iso_reg.transform(test_pred_raw)

    return test_pred_calibrated
```

### 032_xgb_deepctr_ensemble.py
```python
#!/usr/bin/env python3
"""
Ensemble XGBoost (stable) + DeepCTR (diverse)
"""

def ensemble_predictions():
    # Load predictions
    xgb_pred = pd.read_csv('plan1/010_xgboost_submission.csv')['clicked'].values
    dcn_pred = pd.read_csv('plan2/031_deepctr_regularized_submission.csv')['clicked'].values

    # Weighted average (XGBoost gets more weight)
    weights = {
        'xgboost': 0.7,
        'deepctr': 0.3
    }

    final_pred = (
        weights['xgboost'] * xgb_pred +
        weights['deepctr'] * dcn_pred
    )

    # Ensure proper range
    final_pred = np.clip(final_pred, 1e-6, 1-1e-6)

    return final_pred
```

## 실행 순서

1. **Regularized DeepCTR**
```bash
python plan2/031_deepctr_regularized.py
```

2. **Ensemble**
```bash
python plan2/032_xgb_deepctr_ensemble.py
```

## 예상 결과

| 모델 | 예상 Score | 근거 |
|------|-----------|------|
| Regularized DeepCTR | 0.25-0.28 | Overfitting 감소 |
| XGB + DeepCTR Ensemble | 0.33-0.35 | 다양성 활용 |

## 장기 개선 방안

### 1. Feature Engineering
- XGBoost에서 잘 작동한 feature 분석
- DeepCTR용 feature 재설계

### 2. Model Selection
- LightGBM, CatBoost 시도
- TabNet 재시도 (gradient 문제 해결 후)

### 3. Cross Validation
- 5-fold CV로 robust한 모델 선택
- Out-of-fold predictions로 stacking

### 4. Hyperparameter Optimization
- Optuna로 systematic search
- Validation score 기준 최적화

## 핵심 교훈

1. **Validation은 필수**: Training score만 보면 안됨
2. **Simple is better**: 복잡한 모델이 항상 좋은 건 아님
3. **Ensemble이 답**: 서로 다른 특성의 모델 조합
4. **Domain knowledge**: CTR 예측의 특성 이해 필요

## 다음 액션

1. ✅ Regularized DeepCTR 구현 및 테스트
2. ✅ XGBoost와 ensemble
3. ✅ 결과 제출 및 검증
4. ✅ 성능 개선 시 추가 최적화
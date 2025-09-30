# Deep Learning 접근 분석 보고서

## 문제점 진단

### 1. 데이터 특성
- **극심한 클래스 불균형**: Positive rate ~1.9% (52:1 비율)
- **고차원 희소 데이터**: 79개 범주형, 39개 수치형 특징
- **범주형 카디널리티**: 일부 특징이 매우 높은 cardinality (수만개)

### 2. NaN 발생 원인 분석
- **Gradient Explosion**: pos_weight가 높을 때 (>20) gradient가 폭발
- **Embedding 초기화**: 고차원 임베딩이 불안정한 초기값 생성
- **Batch Normalization**: 극소수 positive 샘플로 인한 통계 불안정
- **Numerical Overflow**: BCE loss에서 극단적 예측값 (0 또는 1에 가까운)

### 3. 시도한 해결책과 결과
1. **DCNv2**: Cross network에서 NaN 발생
2. **TabNet**: Attention mechanism에서 불안정
3. **DeepFM**: FM layer에서 수치 오버플로우
4. **Simple NN**: 첫 forward pass부터 NaN

## 성공적인 Deep Learning을 위한 권장사항

### 1. 데이터 전처리 개선
```python
# 로그 변환으로 수치 안정화
numerical_features = np.log1p(numerical_features)

# Target encoding with smoothing
for cat_col in categorical_cols:
    target_mean = train[cat_col].map(
        train.groupby(cat_col)['clicked'].agg(
            lambda x: (x.sum() + global_mean * 10) / (len(x) + 10)
        )
    )
```

### 2. 모델 아키텍처
```python
class StableCTR(nn.Module):
    def __init__(self):
        # 1. 작은 임베딩 차원 (4-8)
        # 2. Residual connections
        # 3. Layer normalization instead of batch norm
        # 4. Gradient checkpointing for memory
```

### 3. 학습 전략
```python
# 1. Curriculum Learning: 쉬운 샘플부터 학습
# 2. Progressive training: 작은 모델에서 시작
# 3. Ensemble with GBDT: XGBoost features를 NN input으로
```

### 4. 안정적 학습 설정
```python
# Loss function
class FocalLossWithClipping(nn.Module):
    def forward(self, input, target):
        input = torch.clamp(input, -10, 10)
        # Focal loss implementation

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,  # 매우 낮은 학습률
    weight_decay=0.01,
    eps=1e-4  # Numerical stability
)

# Gradient clipping
torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
```

## 대안 접근법

### 1. Two-Stage Approach
```
Stage 1: XGBoost/LightGBM으로 feature extraction
Stage 2: Neural network으로 refinement
```

### 2. Feature Engineering + Linear Model
```
- Polynomial features
- Feature interactions
- Frequency encoding
- Target encoding
→ Logistic Regression or Linear SVM
```

### 3. AutoML 도구 활용
- AutoGluon
- H2O.ai
- TPOT

## 실제 작동 가능한 솔루션

가장 현실적인 접근:
1. **XGBoost 최적화 계속**: 안정적이고 검증된 성능
2. **CatBoost 시도**: 범주형 처리에 특화
3. **Ensemble**: XGBoost + LightGBM + CatBoost

딥러닝을 반드시 사용해야 한다면:
1. **데이터 샘플링**: Positive 오버샘플링 + Negative 언더샘플링
2. **사전학습 모델**: TabNet, SAINT 등 검증된 구현체 사용
3. **하이브리드**: GBDT leaf indices를 NN input으로

## 결론

현재 데이터셋의 특성상 딥러닝보다는 **Gradient Boosting 기반 접근**이 더 적합합니다.
딥러닝을 성공시키려면:
1. 데이터 전처리 파이프라인 완전 재구성
2. 극도로 보수적인 초기화와 학습률
3. 앙상블의 일부로만 사용

목표 점수 0.349 달성을 위해서는 XGBoost/CatBoost 최적화에 집중하는 것을 권장합니다.
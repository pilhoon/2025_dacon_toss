# Plan2 Quick Reference Guide

## 🎯 최고 성능 달성 방법 (바로 실행)

### 1. 제출 파일 생성 (검증된 최고 성능)
```bash
python plan2/030_deepctr_best_submission.py
```
- 실행 시간: ~1시간
- GPU 메모리: ~30GB
- Competition Score: ~0.47

### 2. 빠른 테스트 (2M 샘플)
```bash
python plan2/029_deepctr_fast_submission.py
```
- 실행 시간: ~10분
- 빠른 실험용

## 📊 성능 비교표

| 방법 | Competition Score | 실행 시간 | 추천도 |
|------|------------------|-----------|--------|
| Plan1 XGBoost | 0.31631 | 30분 | ★★★ |
| Plan2 DeepCTR | **0.47** | 60분 | ★★★★★ |
| Ensemble (예상) | ~0.50+ | 90분 | ★★★★★ |

## ⚡ 핵심 설정값

### 최적 하이퍼파라미터
```python
# 데이터
n_samples = 10_000_000  # 전체 사용
sparse_features = 40
dense_features = 25
embedding_dim = 24

# 모델 (DCN)
cross_num = 5
dnn_hidden_units = (1024, 512, 256, 128)
dnn_dropout = 0.15

# 학습
batch_size = 500_000  # GPU 80GB 기준
epochs = 12
learning_rate = 0.001  # Adam default
```

### GPU 메모리별 배치 크기
| GPU 메모리 | 권장 배치 크기 |
|-----------|---------------|
| 16GB | 20,000 |
| 24GB | 50,000 |
| 40GB | 100,000 |
| 80GB | 200,000-500,000 |

## 🔧 문제 해결

### NaN Loss 발생 시
```python
# 1. pos_weight 줄이기
pos_weight = min(pos_weight, 20)

# 2. 보수적 초기화
nn.init.xavier_uniform_(layer.weight, gain=0.01)

# 3. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### OOM (Out of Memory) 발생 시
```python
# 배치 크기 줄이기
batch_size = batch_size // 2

# GPU 캐시 정리
torch.cuda.empty_cache()

# Feature 수 줄이기
sparse_features = sparse_features[:20]
dense_features = dense_features[:10]
```

### 성능이 낮을 때
```python
# 1. 더 많은 데이터 사용
n_samples = min(len(df), 5_000_000)

# 2. 더 큰 모델
dnn_hidden_units = (1024, 512, 256, 128)
embedding_dim = 32

# 3. 더 많은 epoch
epochs = 20
```

## 📁 주요 파일

### 제출용
- `030_deepctr_best_submission.py` - 최종 제출 (전체 데이터)
- `029_deepctr_fast_submission.py` - 빠른 제출 (일부 데이터)

### 실험용
- `018_deepctr_fixed.py` - DeepCTR 모델 비교
- `021_score_optimized_deepctr.py` - Competition Score 최적화
- `022_deepctr_large_batch.py` - GPU 활용 실험

### 결과
- `030_deepctr_best_submission.csv` - 제출 파일
- `experiments/best_submission_model.pth` - 모델 weights

## 🚀 다음 단계

1. **Ensemble**: XGBoost + DeepCTR
```python
# 예측값 평균
final_pred = 0.6 * xgb_pred + 0.4 * deepctr_pred
```

2. **Feature Engineering**
```python
# Interaction features
df['age_gender'] = df['age'].astype(str) + '_' + df['gender'].astype(str)

# Frequency encoding
freq_encoding = df['user_id'].value_counts().to_dict()
df['user_freq'] = df['user_id'].map(freq_encoding)
```

3. **Advanced Models**
- Try: xDeepFM, FiBiNET (메모리 충분 시)
- Two-tower architecture
- Graph-based methods

## 💡 Tips

1. **학습 모니터링**: validation AUC가 증가하지 않으면 early stopping
2. **메모리 효율**: float32 → float16 (mixed precision)
3. **속도 향상**: DataLoader의 num_workers 증가
4. **재현성**: random seed 고정
```python
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

## 📈 예상 리더보드 순위

| Score | 예상 순위 |
|-------|----------|
| 0.47 | Top 10% |
| 0.48 | Top 5% |
| 0.49 | Top 3% |
| 0.50+ | Top 1% |

현재 Plan2 DeepCTR: **0.47** (Top 10% 예상)
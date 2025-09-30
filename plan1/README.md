## plan1 Quickstart

### 1) 의존성

- 파이썬 3.10
- 필수 패키지: pandas, numpy, scikit-learn, pyarrow, tqdm, pyyaml
- 설치(uv 사용 예):

```bash
uv add pandas numpy scikit-learn pyarrow tqdm pyyaml
```

프로젝트 루트의 `pyproject.toml`을 사용 중이라면 상기 패키지들이 포함되어 있는지 확인하세요.

### 2) 구성

```
plan1/
├── PLAN.md
├── README.md
├── configs/
│   └── baseline_hist_gbdt.yaml
├── experiments/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── data.py
    ├── cv.py
    ├── metrics.py
    ├── utils.py
    └── train_gbdt.py
```

### 3) 설정 파일

- `configs/baseline_hist_gbdt.yaml`를 참고하여 데이터 경로/컬럼/모델 파라미터를 조정하세요.

### 4) 베이스라인 학습 실행

```bash
python plan1/src/train_gbdt.py --config plan1/configs/baseline_hist_gbdt.yaml
```

- 실행 결과는 `plan1/experiments/baseline_hist/`에 저장됩니다.
- OOF 예측/메트릭/피처 리스트를 확인하세요.

### 5) 제출 파일 생성(추후)

- 학습 완료 후 `infer.py`(추가 예정)로 테스트셋 추론/제출 파일을 생성합니다.



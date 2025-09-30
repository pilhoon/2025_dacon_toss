# 2025 DACON Toss Competition

## Competition Links
- **제출 페이지**: https://dacon.io/competitions/official/236575/mysubmission

## 제출 파일 형식

### 필수 요구사항
- **파일 형식**: CSV
- **인코딩**: UTF-8
- **행 수**: 1,527,299행 (헤더 포함)
- **컬럼**:
  - `ID`: TEST_0000000 ~ TEST_1527297 (7자리 0-padding)
  - `clicked`: 예측 확률값 (0.0 ~ 1.0)

### 예시
```csv
ID,clicked
TEST_0000000,0.015694018
TEST_0000001,0.018328346
TEST_0000002,0.022161013
...
TEST_1527297,0.031415926
```

## 환경 설정

### Prerequisites
- Python 3.10
- uv (Python package manager) 또는 pip

### Setup

#### 처음 설정
```bash
# uv로 프로젝트 초기화 (Python 3.10)
uv init --python 3.10

# 필요 패키지 설치
uv add pandas numpy scikit-learn tqdm torch pyarrow pip ipykernel
```

#### 기존 프로젝트 클론 후 설정
```bash
# pyproject.toml의 모든 패키지 자동 설치
uv sync
```

### 주요 패키지
- `pandas`: 데이터 처리
- `numpy`: 수치 연산
- `scikit-learn`: train/test split
- `torch`: 딥러닝 모델
- `pyarrow`: parquet 파일 읽기
- `tqdm`: 학습 진행률 표시

### 프로젝트 구조
```
.
├── baseline/           # 베이스라인 코드
│   └── baseline.ipynb
├── data/              # 데이터 디렉토리
├── main.py           # 메인 실행 파일
└── pyproject.toml    # 프로젝트 설정
```
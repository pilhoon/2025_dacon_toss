## plan2: Deep CTR Roadmap

목표: 리소스 제약보다 성능 극대화에 집중해 딥러닝 중심으로 점수 상향.

구성:
```
plan2/
├── 000_EXPERIMENT_LOG.md
├── 000_run_plan2.sh
├── PLAN.md
├── README.md
├── EVAL_NOTES.md
├── configs/
│   ├── dcnv2.yaml
│   ├── xdeepfm.yaml
│   ├── din.yaml
│   ├── ft_transformer.yaml
│   └── trainer.yaml
├── experiments/
│   └── 001_data_v2/
└── src/
    ├── __init__.py
    ├── 001_data_pipeline_v2.py
    ├── 002_dcnv2_v1.py
    ├── 003_dcnv2_ablation.py
    ├── 004_xdeepfm_v1.py
    ├── 005_ft_transformer_v1.py
    ├── 006_din_v1.py
    ├── 008_calibration_ensemble.py
    ├── log_utils.py
    ├── modules/
    │   ├── dcnv2.py
    │   ├── xdeepfm.py
    │   ├── din.py
    │   └── ft_transformer.py
    ├── dataset.py
    ├── train.py
    ├── infer.py
    ├── utils.py
    └── metrics.py
```

실행:
```bash
# 패키지 임포트를 위해 PYTHONPATH 설정 권장
export PYTHONPATH=.

# 001) 데이터 파이프라인 산출물 생성(부분 샘플로 빠른 검증)
python plan2/src/001_data_pipeline_v2.py --config plan2/configs/dcnv2.yaml --out plan2/experiments/001_data_v2 --folds 5 --n-rows 200000

# 002) DCNv2 v1 학습(시간/자원 소모 큼)
python plan2/src/002_dcnv2_v1.py

# 003) DCNv2 어블레이션 그리드(시간/자원 소모 큼)
python plan2/src/003_dcnv2_ablation.py

# 004) xDeepFM v1(시간/자원 소모 큼)
python plan2/src/004_xdeepfm_v1.py

# 005) FT-Transformer v1(시간/자원 소모 큼)
python plan2/src/005_ft_transformer_v1.py

# 006) DIN v1(시간/자원 소모 큼)
python plan2/src/006_din_v1.py

# 008) OOF 기반 캘리브레이션
python plan2/src/008_calibration_ensemble.py --oof plan2/experiments/002_dcnv2_v1/oof_probs.npy --labels plan2/experiments/002_dcnv2_v1/labels.npy --outdir plan2/experiments/008_calibration
```

노트:
- 모든 산출물은 plan2/experiments 하위에 생성되며, 파일/폴더 이름에 번호(001_, 002_, …)를 붙여 순서를 명확히 합니다.
- 중앙 로그: plan2/000_EXPERIMENT_LOG.md에 각 단계의 요약과 아티팩트 경로를 계속 추가합니다.

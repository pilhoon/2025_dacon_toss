## Plan4 Overview

Plan4 consolidates lessons from Plan1–Plan3 to push DACON Toss CTR performance past the 0.351 leaderboard threshold. It focuses on:

1. **Metric fidelity** – reproducing the official AP/WLL-based score locally.
2. **Calibrated modeling** – maintaining both high AP and controlled WLL.
3. **Model diversity & ensembling** – combining tree-based and neural models.
4. **Operational discipline** – consistent experiment logging, guardrails, and submission monitoring.

Core documents:
- `PLAN.md`: strategy, workstreams, gating criteria.
- `EXPERIMENT_ROADMAP.md`: ordered experiments with deliverables.
- `RESEARCH_TOPICS.md`: open questions requiring external references.
- `CHECKLIST.md`: progress tracker per workstream.
- `STATUS_REPORT.md`: template for periodic updates.

Create experiment outputs under `plan4/experiments/` using the naming convention `E##_{short_desc}`.


## 템플릿 간단 실행법
1. CV 학습 + OOF 점수 + 모델/인코더/캘리브레이터 저장
python plan4/train_cv_xgb.py --cfg plan4/configs/plan4_xgb.yaml

2. (위 출력 경로를 사용) 추론 + 제출 파일 생성
python plan4/infer_submit.py --exp_dir experiments/plan4/exp_YYYYmmdd_HHMMSS --cfg plan4/configs/plan4_xgb.y#aml --apply_calibration true

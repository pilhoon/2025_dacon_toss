#!/usr/bin/env bash
set -euo pipefail

# Simple orchestrator for plan2 experiments (runs sequentially)
# Usage: bash plan2/000_run_plan2.sh

export PYTHONPATH=.

echo "[001] Data pipeline (subset quick check)"
python plan2/src/001_data_pipeline_v2.py --config plan2/configs/dcnv2.yaml --out plan2/experiments/001_data_v2 --folds 5 --n-rows 200000

echo "[002] DCNv2 v1 (may be heavy)"
# Uncomment to run full training (requires GPU/time)
# python plan2/src/002_dcnv2_v1.py

echo "[003] DCNv2 ablation grid (heavy)"
# python plan2/src/003_dcnv2_ablation.py

echo "[004] xDeepFM v1 (heavy)"
# python plan2/src/004_xdeepfm_v1.py

echo "[005] FT-Transformer v1 (heavy)"
# python plan2/src/005_ft_transformer_v1.py

echo "[006] DIN v1 (heavy)"
# python plan2/src/006_din_v1.py

echo "[008] Calibration (requires OOF)"
# Example once OOF saved: python plan2/src/008_calibration_ensemble.py --oof plan2/experiments/002_dcnv2_v1/oof_probs.npy --labels plan2/experiments/002_dcnv2_v1/labels.npy --outdir plan2/experiments/008_calibration

echo "Done (scripts prepared; heavy steps commented)."



# CTR-Maximizer (offline accuracy only)

End-to-end pipeline to **train, evaluate, calibrate, and ensemble** strong CTR models with your official metric:

```
Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))
```
- **AP** = `sklearn.metrics.average_precision_score` (non-interpolated)
- **WLL** = (weighted) log loss via `sklearn.metrics.log_loss`

## Highlights
- Time-aware split (train/valid/test) when a time column is available.
- Handles ~10 anonymous features (unknown semantics OK). Strings → stable hash ints.
- Models: **XGBoost**, **CatBoost**, **Tabular Transformer (PyTorch)**.
- Per-model **isotonic calibration** (only if Score improves).
- **Weight-search ensemble** maximizing Score on validation.
- Large-scale friendly: Parquet + Polars; GPU support where available.
- Outputs: metrics JSON + **test** probabilities (per-model & ensemble).

> This repo is for **offline accuracy** only (no serving constraints).

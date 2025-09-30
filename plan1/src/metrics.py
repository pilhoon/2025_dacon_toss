from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics: dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        # clip for numerical stability
        y_prob_clip = np.clip(y_prob, 1e-7, 1 - 1e-7)
        metrics["logloss"] = float(log_loss(y_true, y_prob_clip))
    except Exception:
        metrics["logloss"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = float("nan")
    return metrics



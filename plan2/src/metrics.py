from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def compute_all(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    out = {}
    try:
        out["ap"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["ap"] = float("nan")
    try:
        out["wll"] = float(log_loss(y_true, y_prob))
    except Exception:
        out["wll"] = float("nan")
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    if np.isfinite(out.get("ap", np.nan)) and np.isfinite(out.get("wll", np.nan)):
        out["composite"] = 0.5 * out["ap"] + 0.5 * (1.0 / (1.0 + out["wll"]))
    else:
        out["composite"] = float("nan")
    return out



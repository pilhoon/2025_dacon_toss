from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold


def get_cv_splitter(method: str, n_splits: int, shuffle: bool, random_state: int | None):
    method = (method or "").lower()
    if method == "stratifiedkfold" or method == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if method == "groupkfold" or method == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unsupported CV method: {method}")


def time_holdout_mask(df: pd.DataFrame, time_col: str, holdout_ratio: float = 0.1) -> np.ndarray:
    # Simple time-based split: last quantile as holdout
    quantile = df[time_col].quantile(1 - holdout_ratio)
    return (df[time_col] >= quantile).to_numpy()



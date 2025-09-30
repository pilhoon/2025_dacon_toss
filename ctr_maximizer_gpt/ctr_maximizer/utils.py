
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict

def time_order_split(df: pd.DataFrame, time_col: Optional[str], label_col: str,
                     valid_frac: float = 0.1, test_frac: float = 0.1):
    assert 0 < valid_frac < 0.5 and 0 < test_frac < 0.5 and valid_frac + test_frac < 0.9
    if time_col and time_col in df.columns:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        n = len(df_sorted)
        n_test = int(n * test_frac)
        n_valid = int(n * valid_frac)
        idx_train = np.arange(0, n - n_valid - n_test)
        idx_valid = np.arange(n - n_valid - n_test, n - n_test)
        idx_test = np.arange(n - n_test, n)
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
        y = df[label_col].values
        idx_all = np.arange(len(df))
        train_valid_idx, test_idx = next(sss1.split(idx_all, y))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=valid_frac / (1 - test_frac), random_state=41)
        tr_idx, va_idx = next(sss2.split(train_valid_idx, y[train_valid_idx]))
        idx_train = train_valid_idx[tr_idx]
        idx_valid = train_valid_idx[va_idx]
        idx_test = test_idx
        df_sorted = df.reset_index(drop=True)
    return df_sorted, idx_train, idx_valid, idx_test

def hash_str_column(col, num_buckets=2**20):
    import hashlib
    def _h(x):
        s = str(x).encode('utf-8', errors='ignore')
        return int(hashlib.blake2b(s, digest_size=8).hexdigest(), 16) % num_buckets
    return col.map(_h).astype('int64')

def prepare_frame(df, label_col: str, time_col: Optional[str], weight_col: Optional[str]):
    X = df.copy()
    y = X.pop(label_col).astype('int8').values
    w = None
    if weight_col and weight_col in df.columns:
        w = df[weight_col].astype('float32').values
        X = X.drop(columns=[weight_col])
    if time_col and time_col in X.columns:
        X = X.drop(columns=[time_col])
    obj_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    for c in obj_cols:
        X[c] = hash_str_column(X[c])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y, w

def standardize_numeric(X_train, X_valid, X_test):
    num_cols = X_train.select_dtypes(include=['int64','int32','int16','float64','float32']).columns.tolist()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols].astype('float32'))
    def _tr(X):
        Y = X.copy()
        Y[num_cols] = scaler.transform(X[num_cols].astype('float32'))
        return Y
    return _tr(X_train), _tr(X_valid), _tr(X_test), num_cols, scaler

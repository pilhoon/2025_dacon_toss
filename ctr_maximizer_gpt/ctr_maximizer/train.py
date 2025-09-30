
import os, json, time, yaml
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from tqdm import tqdm

from .utils import time_order_split, prepare_frame, standardize_numeric
from .metrics import score_components
from .models_boost import train_xgboost, train_catboost
from .models_nn import train_tab_transformer

def load_df(path: str, use_polars: bool = True) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if use_polars:
        import polars as pl
        if ext in [".parquet", ".pq"]:
            return pl.read_parquet(path).to_pandas()
        else:
            return pl.read_csv(path).to_pandas()
    else:
        if ext in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)

def choose_device(prefer_gpu: bool=True):
    import torch
    if prefer_gpu and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def ensemble_weights_grid(preds: Dict[str, np.ndarray], y_valid: np.ndarray, w: Optional[np.ndarray]):
    keys = list(preds.keys())
    M = len(keys)
    best = (-1.0, None)
    grid = np.linspace(0,1,21)
    if M == 1:
        s = score_components(y_valid, preds[keys[0]], w)['Score']
        return {keys[0]: 1.0}, s
    elif M == 2:
        for a in grid:
            b = 1-a
            p = a*preds[keys[0]] + b*preds[keys[1]]
            s = score_components(y_valid, p, w)['Score']
            if s > best[0]: best = (s, {keys[0]: float(a), keys[1]: float(b)})
    else:
        for a in grid:
            for b in grid:
                if a+b <= 1.0:
                    c = 1.0 - a - b
                    p = a*preds[keys[0]] + b*preds[keys[1]] + c*preds[keys[2]]
                    s = score_components(y_valid, p, w)['Score']
                    if s > best[0]: best = (s, {keys[0]: float(a), keys[1]: float(b), keys[2]: float(c)})
    return best[1], best[0]

def isotonic_calibration(y_valid: np.ndarray, p_valid: np.ndarray, w: Optional[np.ndarray]):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip', y_min=1e-7, y_max=1-1e-7)
    ir.fit(p_valid, y_valid, sample_weight=w)
    return ir

def run(config: Dict[str, Any]):
    t0 = time.time()
    os.makedirs('artifacts', exist_ok=True)

    df = load_df(config['data_path'], use_polars=config.get('use_polars', True))
    label_col = config.get('label_col', 'label')
    time_col = config.get('time_col', 'time')
    weight_col = config.get('weight_col', None)

    # Split
    df_sorted, idx_tr, idx_va, idx_te = time_order_split(df, time_col, label_col,
                                                         valid_frac=config.get('valid_frac', 0.1),
                                                         test_frac=config.get('test_frac', 0.1))

    # Prepare features
    X, y, w = prepare_frame(df_sorted, label_col, time_col, weight_col)
    X_raw = X.copy()  # For CatBoost
    X_tr, X_va, X_te = X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
    w_tr = w[idx_tr] if w is not None else None
    w_va = w[idx_va] if w is not None else None
    w_te = w[idx_te] if w is not None else None

    # Standardize numerics for XGB/Transformer
    X_tr, X_va, X_te, num_cols, scaler = standardize_numeric(X_tr, X_va, X_te)

    # Models
    reports = {}
    valid_preds = {}
    test_preds = {}

    # 1) XGBoost
    if config.get('use_xgboost', True):
        xgb_params = config.get('xgboost_params', {})
        bst, xgb_predict = train_xgboost(X_tr, y_tr, X_va, y_va, sample_weight=w_tr, params=xgb_params)
        p_va = xgb_predict(X_va)
        p_te = xgb_predict(X_te)
        valid_preds['xgb'] = p_va
        test_preds['xgb'] = p_te
        reports['xgb'] = {
            'valid': score_components(y_va, p_va, w_va),
            'test':  score_components(y_te, p_te, w_te),
        }

    # 2) CatBoost (use RAW features, no standardization)
    if config.get('use_catboost', True):
        cb_params = config.get('catboost_params', {})
        cb, cb_predict = train_catboost(X_raw.iloc[idx_tr], y_tr, X_raw.iloc[idx_va], y_va, sample_weight=w_tr, params=cb_params)
        p_va = cb_predict(X_raw.iloc[idx_va])
        p_te = cb_predict(X_raw.iloc[idx_te])
        valid_preds['cat'] = p_va
        test_preds['cat'] = p_te
        reports['cat'] = {
            'valid': score_components(y_va, p_va, w_va),
            'test':  score_components(y_te, p_te, w_te),
        }

    # 3) Transformer
    if config.get('use_transformer', True):
        device = choose_device(config.get('prefer_gpu', True))
        nn_cfg = config.get('transformer_params', {})
        nn_model, nn_predict = train_tab_transformer(
            X_tr.values, y_tr, X_va.values, y_va,
            device=device,
            max_epochs=nn_cfg.get('max_epochs', 8),
            batch_size=nn_cfg.get('batch_size', 65536),
            lr=nn_cfg.get('lr', 2e-4),
            d_model=nn_cfg.get('d_model', 128),
            nhead=nn_cfg.get('nhead', 8),
            num_layers=nn_cfg.get('num_layers', 4),
            dim_feedforward=nn_cfg.get('dim_feedforward', 256),
            dropout=nn_cfg.get('dropout', 0.1),
            sample_weight=w_tr
        )
        p_va = nn_predict(X_va.values)
        p_te = nn_predict(X_te.values)
        valid_preds['tr'] = p_va
        test_preds['tr'] = p_te
        reports['transformer'] = {
            'valid': score_components(y_va, p_va, w_va),
            'test':  score_components(y_te, p_te, w_te),
        }

    # Optional isotonic calibration per model (apply only if it improves Score on valid)
    if config.get('calibrate', True):
        for k in list(valid_preds.keys()):
            ir = isotonic_calibration(y_va, valid_preds[k], w_va)
            p_va_c = ir.predict(valid_preds[k])
            before = reports['transformer' if k=='tr' else k]['valid']['Score']
            after  = score_components(y_va, p_va_c, w_va)['Score']
            if after > before:
                valid_preds[k] = p_va_c
                test_preds[k] = ir.predict(test_preds[k])
                target = 'transformer' if k=='tr' else k
                reports[target]['valid'] = score_components(y_va, valid_preds[k], w_va)
                reports[target]['test']  = score_components(y_te, test_preds[k], w_te)

    # Ensemble
    ens_w, ens_valid_score = ensemble_weights_grid(valid_preds, y_va, w_va)
    p_va_ens = np.zeros_like(y_va, dtype='float32')
    p_te_ens = np.zeros_like(y_te, dtype='float32')
    for k, a in ens_w.items():
        p_va_ens += a * valid_preds[k]
        p_te_ens += a * test_preds[k]
    reports['ensemble'] = {
        'weights': ens_w,
        'valid': score_components(y_va, p_va_ens, w_va),
        'test':  score_components(y_te, p_te_ens, w_te),
    }

    # Save artifacts
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(reports, f, indent=2)

    # Save TEST predictions only (avoid length mismatch)
    import pyarrow as pa, pyarrow.parquet as pq
    df_test = pd.DataFrame({f'test_{k}': v for k, v in test_preds.items()})
    df_test['test_ensemble'] = p_te_ens
    table = pa.Table.from_pandas(df_test)
    pq.write_table(table, 'artifacts/test_preds.parquet')

    with open('artifacts/run_summary.txt', 'w') as f:
        f.write(f"Split sizes: train={len(idx_tr)}, valid={len(idx_va)}, test={len(idx_te)}\n")
        f.write(json.dumps(reports, indent=2))
    print('Done. Metrics written to artifacts/metrics.json')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    run(cfg)

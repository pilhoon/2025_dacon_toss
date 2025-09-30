# plan4/infer_submit.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import yaml

def load_df(path, n_rows=None, use_columns=None):
    return pd.read_parquet(path, columns=use_columns) if n_rows is None else \
           pd.read_parquet(path, columns=use_columns).head(n_rows)

def apply_preprocessor(enc, df, categorical_cols, numeric_fill=0):
    X = df.copy()
    if len(categorical_cols) > 0 and enc is not None:
        X[categorical_cols] = enc.transform(X[categorical_cols])
    num_cols = X.columns.difference(categorical_cols)
    X[num_cols] = X[num_cols].fillna(numeric_fill)
    return X

def main(exp_dir, cfg_path, apply_calibration=True):
    exp_dir = Path(exp_dir)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    test_path = cfg["test_path"]
    id_col    = cfg["id_column"]
    sub_name  = cfg["submission_name"]
    fillna_num = cfg["fillna_numeric"]

    # 로드
    booster   = xgb.Booster()
    booster.load_model(str(exp_dir / "artifacts" / "xgb_model.json"))
    enc_path  = exp_dir / "artifacts" / "encoder.joblib"
    enc       = joblib.load(enc_path) if enc_path.exists() else None

    cal_path  = exp_dir / "artifacts" / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if (apply_calibration and cal_path.exists()) else None

    # 테스트 로딩
    test_df = load_df(test_path)
    test_ids = test_df[id_col].values
    X_test = test_df.drop(columns=[id_col])

    # 컬럼 타입
    cat_cols = X_test.select_dtypes(include=["object"]).columns.tolist()

    # 전처리
    X_test = apply_preprocessor(enc, X_test, cat_cols, numeric_fill=fillna_num)

    # 예측
    dtest = xgb.DMatrix(X_test)
    p = booster.predict(dtest)

    # 캘리브레이션 적용(선택)
    if calibrator is not None:
        kind = calibrator["kind"]
        cal  = calibrator["cal"]
        if kind == "isotonic":
            p = cal.transform(p)
        elif kind == "platt":
            from sklearn.linear_model import LogisticRegression
            p = cal.predict_proba(p.reshape(-1,1))[:,1]

    # 안전 클리핑
    p = np.clip(p, 1e-6, 1 - 1e-6)

    sub = pd.DataFrame({id_col: test_ids, "clicked": p})
    out_path = exp_dir / sub_name
    sub.to_csv(out_path, index=False)
    print(f"[Saved] submission -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--apply_calibration", type=lambda s: s.lower()=="true", default=True)
    args = ap.parse_args()
    main(args.exp_dir, args.cfg, args.apply_calibration)

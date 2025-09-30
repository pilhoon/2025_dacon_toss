# plan4/train_cv_xgb.py
import os, json, time, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import joblib
import xgboost as xgb
import yaml

# ----- 점수 함수 (대회 정의) -----
def weighted_logloss(y_true, y_prob, eps=1e-15):
    """
    Weighted LogLoss (양/음성 각 0.5씩 가중).
    """
    p = np.clip(y_prob, eps, 1 - eps)
    y_true = np.asarray(y_true).astype(np.float64)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        # 비정상 케이스 보호
        return float('nan')
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    ll = -(w_pos * (y_true * np.log(p)).sum() + w_neg * ((1 - y_true) * np.log(1 - p)).sum())
    return ll

def competition_score(y_true, y_prob):
    """
    최종 점수 = 0.5 * AP + 0.5 * (1 / (1 + WLL))
    """
    ap = average_precision_score(y_true, y_prob)
    wll = weighted_logloss(y_true, y_prob)
    return ap, wll, 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))

# ----- 유틸 -----
def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def load_df(path, n_rows=None, use_columns=None):
    return pd.read_parquet(path, columns=use_columns) if n_rows is None else \
           pd.read_parquet(path, columns=use_columns).head(n_rows)

def detect_columns(df, id_col, target):
    cats = df.select_dtypes(include=["object"]).columns.tolist()
    nums = df.select_dtypes(exclude=["object"]).columns.tolist()
    for c in [id_col, target]:
        if c in cats: cats.remove(c)
        if c in nums: nums.remove(c)
    return cats, nums

def make_preprocessor(categorical_cols):
    if len(categorical_cols) == 0:
        return None
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)

def apply_preprocessor(enc, df, categorical_cols, numeric_fill=0):
    X = df.copy()
    if len(categorical_cols) > 0:
        # Fill missing values before transform to avoid unknown categories error
        X[categorical_cols] = X[categorical_cols].fillna("missing")
        X[categorical_cols] = enc.transform(X[categorical_cols])
    # 결측 처리
    num_cols = X.columns.difference(categorical_cols)
    X[num_cols] = X[num_cols].fillna(numeric_fill)
    return X

def fit_calibrator(method, oof_pred, y):
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(oof_pred, y)
        return ("isotonic", cal)
    elif method == "platt":
        # Platt scaling: 로지스틱 회귀를 확률-확률 맵핑으로 사용
        lr = LogisticRegression(max_iter=1000)
        lr.fit(oof_pred.reshape(-1,1), y)
        return ("platt", lr)
    else:
        return ("none", None)

def apply_calibrator(kind, cal, p):
    if kind == "isotonic":
        return cal.transform(p)
    elif kind == "platt":
        return cal.predict_proba(p.reshape(-1,1))[:,1]
    return p

# ----- 메인 -----
def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed        = cfg["seed"]
    train_path  = cfg["train_path"]
    test_path   = cfg["test_path"]
    id_col      = cfg["id_column"]
    target      = cfg["target"]
    n_rows      = cfg["n_rows"]
    use_columns = cfg["use_columns"]

    n_splits    = cfg["n_splits"]
    es_rounds   = cfg["early_stopping_rounds"]
    n_rounds    = cfg["num_boost_round"]

    enc_kind    = cfg["categorical_encoding"]
    fillna_num  = cfg["fillna_numeric"]

    use_spw     = cfg["use_scale_pos_weight"]
    xgb_params  = cfg["xgb_params"]

    cal_method  = cfg["calibration"]["method"]
    cal_on_test = cfg["calibration"]["apply_on_test"]

    sub_name    = cfg["submission_name"]

    # 출력 폴더
    exp_root = Path("experiments") / "plan4"
    exp_dir  = exp_root / f'exp_{now_str()}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로딩
    print(f"[Load] {train_path}")
    train_df = load_df(train_path, n_rows=n_rows, use_columns=use_columns)
    assert target in train_df.columns, f"target {target} not found"
    y = train_df[target].astype(int).values
    if id_col in train_df.columns: train_df = train_df.drop(columns=[id_col])
    X_all = train_df.drop(columns=[target])

    # 컬럼 타입
    cat_cols, num_cols = detect_columns(pd.concat([X_all.head(1000)]), id_col, target)
    print(f"[Cols] categorical={len(cat_cols)}, numeric={len(num_cols)}")

    # 인코더 준비
    enc = make_preprocessor(cat_cols)
    if enc is not None:
        enc.fit(X_all[cat_cols].fillna("missing"))
    X_all = apply_preprocessor(enc, X_all, cat_cols, numeric_fill=fillna_num)

    # scale_pos_weight
    spw = None
    if use_spw:
        pos = y.sum()
        neg = len(y) - pos
        spw = (neg / max(pos, 1.0))
        xgb_params = {**xgb_params, "scale_pos_weight": spw}
        print(f"[Imbalance] scale_pos_weight={spw:.2f}")

    # CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=np.float64)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
        print(f"\n[Fold {fold}/{n_splits}] train={len(tr_idx):,}, valid={len(va_idx):,}")

        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)

        booster = xgb.train(
            params=xgb_params,
            dtrain=dtr,
            num_boost_round=n_rounds,
            evals=[(dva, "valid")],
            early_stopping_rounds=es_rounds,
            verbose_eval=100
        )

        p_va = booster.predict(dva, iteration_range=(0, booster.best_iteration+1))
        oof_pred[va_idx] = p_va

        ap, wll, score = competition_score(y_va, p_va)
        print(f"[Fold {fold}] AP={ap:.5f} | WLL={wll:.5f} | score={score:.5f}")
        fold_metrics.append({"fold": fold, "ap": ap, "wll": wll, "score": score})

    # 전체 OOF 점수
    ap, wll, score = competition_score(y, oof_pred)
    print(f"\n[OOF] AP={ap:.5f} | WLL={wll:.5f} | score={score:.5f}")

    # 캘리브레이션 적합 (WLL 개선 목적)
    cal_kind, calibrator = fit_calibrator(cal_method, oof_pred, y)
    if calibrator is not None:
        oof_cal = apply_calibrator(cal_kind, calibrator, oof_pred)
        ap_c, wll_c, score_c = competition_score(y, oof_cal)
        print(f"[OOF-Cal] AP={ap_c:.5f} | WLL={wll_c:.5f} | score={score_c:.5f}")
    else:
        ap_c, wll_c, score_c = ap, wll, score

    # 전체 데이터로 최종 적합
    print("\n[Fit on FULL] training final booster with best params")
    dfull = xgb.DMatrix(X_all, label=y)
    final_booster = xgb.train(
        params=xgb_params,
        dtrain=dfull,
        num_boost_round=int(np.median([m["score"] for m in fold_metrics]) * 0 + n_rounds), # 그대로 사용
        evals=[(dfull, "train")],
        verbose_eval=200
    )

    # 아티팩트 저장
    (exp_dir / "artifacts").mkdir(exist_ok=True)
    final_booster.save_model(str(exp_dir / "artifacts" / "xgb_model.json"))
    if enc is not None:
        joblib.dump(enc, exp_dir / "artifacts" / "encoder.joblib")
    if calibrator is not None:
        joblib.dump({"kind": cal_kind, "cal": calibrator}, exp_dir / "artifacts" / "calibrator.joblib")

    # OOF/메트릭/설정 저장
    pd.DataFrame({
        "oof_pred": oof_pred,
        "y": y
    }).to_parquet(exp_dir / "oof.parquet", index=False)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump({
            "folds": fold_metrics,
            "oof": {"ap": ap, "wll": wll, "score": score},
            "oof_cal": {"ap": ap_c, "wll": wll_c, "score": score_c},
            "scale_pos_weight": spw
        }, f, indent=2)

    with open(exp_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"\n[Saved] {exp_dir}")
    print(f"  - model/encoder/calibrator under artifacts/")
    print(f"  - OOF predictions & metrics saved")
    print("\nNext:")
    print(f"  python plan4/infer_submit.py --exp_dir {exp_dir} --cfg {cfg_path} --apply_calibration {str(cal_on_test)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="path to YAML config")
    args = parser.parse_args()
    main(args.cfg)

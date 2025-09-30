from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

from data import DatasetConfig, load_dataset, summarize_schema
from metrics import compute_metrics
from utils import ensure_dir


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = DatasetConfig(
        train_path=cfg["data"]["train_path"],
        test_path=cfg["data"]["test_path"],
        target=cfg["data"]["target"],
        id_column=cfg["data"]["id_column"],
        use_patterns=cfg["data"].get("use_patterns"),
        exclude_patterns=cfg["data"].get("exclude_patterns"),
        n_rows=cfg["data"].get("n_rows"),
    )

    out_dir = Path(cfg["output"]["dir"]).resolve()
    ensure_dir(str(out_dir))

    train_df, test_df = load_dataset(data_cfg)
    target = data_cfg.target

    # Save schema summary
    schema_df = summarize_schema(train_df)
    schema_df.to_csv(out_dir / "schema_summary.csv", index=False)

    features = [c for c in train_df.columns if c != target]
    X = train_df[features]
    y = train_df[target].astype(int).to_numpy()

    preprocessor = build_preprocessor(X)
    model = HistGradientBoostingClassifier(**cfg["model"]["params"])

    n_splits = cfg["cv"]["n_splits"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=cfg["cv"]["shuffle"], random_state=cfg["cv"]["random_state"])

    oof_pred = np.zeros(len(X), dtype=float)
    fold_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", model),
        ])

        pipe.fit(X_trn, y_trn)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = val_prob
        m = compute_metrics(y_val, val_prob)
        m["fold"] = fold
        fold_metrics.append(m)

    overall = compute_metrics(y, oof_pred)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"folds": fold_metrics, "overall": overall}, f, ensure_ascii=False, indent=2)

    # Save oof predictions
    oof_df = pd.DataFrame({"oof_prob": oof_pred})
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    # Persist final pipeline for quick inference baseline
    try:
        import joblib  # type: ignore
        joblib.dump(pipe, out_dir / "model.joblib")
    except Exception:
        pass

    print("AUC:", overall.get("roc_auc"), "Logloss:", overall.get("logloss"))


if __name__ == "__main__":
    main()



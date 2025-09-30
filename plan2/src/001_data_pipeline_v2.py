from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from plan2.src.dataset import DataConfig, prepare_data, build_vocabs, compute_num_stats
from plan2.src.utils import ensure_dir, load_yaml, save_json
from plan2.src.log_utils import append_md_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="plan2/configs/dcnv2.yaml", help="YAML with data.* keys")
    parser.add_argument("--out", default="plan2/experiments/001_data_v2", help="Output directory for artifacts")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds")
    parser.add_argument("--n-rows", type=int, default=None, help="Optional: limit rows for a quick dry run")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data = cfg["data"]

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    dc = DataConfig(
        train_path=data["train_path"],
        test_path=data["test_path"],
        target=data["target"],
        cat_patterns=data["cat_patterns"],
        num_patterns=data["num_patterns"],
        min_freq=int(data.get("min_freq", 10)),
        max_seq_len=int(data.get("max_seq_len", 0)),
    )

    # Prepare dataframes and column selection
    train_df, test_df, cat_cols, num_cols = prepare_data(dc, n_rows=args.n_rows)

    # Save basic shapes/columns
    save_json({
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "n_cat": len(cat_cols),
        "n_num": len(num_cols),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }, out_dir / "001_columns.json")

    # Class stats
    y = train_df[dc.target].to_numpy().astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_rate = float(n_pos / max(1, n_pos + n_neg))
    save_json({"n_pos": n_pos, "n_neg": n_neg, "pos_rate": pos_rate}, out_dir / "001_class_stats.json")

    # Vocabs and numeric stats
    vocabs: Dict[str, Dict[str, int]] = build_vocabs(train_df, cat_cols, dc.min_freq)
    vocab_sizes = {k: len(v) for k, v in vocabs.items()}
    save_json(vocabs, out_dir / "001_vocabs.json")
    save_json(vocab_sizes, out_dir / "001_vocab_sizes.json")

    num_stats = compute_num_stats(train_df, num_cols)
    save_json(num_stats, out_dir / "001_num_stats.json")

    # Stratified folds (store val assignment only)
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(cfg.get("train", {}).get("seed", 42)))
    fold_assign = np.full(len(train_df), -1, dtype=np.int32)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_assign[val_idx] = fold
    # Save compact CSV and JSON summary
    df_folds = pd.DataFrame({"idx": np.arange(len(fold_assign), dtype=np.int64), "fold": fold_assign})
    df_folds.to_csv(out_dir / "001_folds_val.csv", index=False)

    folds_json = {"n_folds": int(args.folds), "counts": {int(f): int((fold_assign == f).sum()) for f in range(int(args.folds))}}
    save_json(folds_json, out_dir / "001_folds.json")

    # Append to central experiment log
    bullets: List[str] = [
        f"train={tuple(train_df.shape)}, test={tuple(test_df.shape)}",
        f"n_cat={len(cat_cols)}, n_num={len(num_cols)}",
        f"pos_rate={pos_rate:.5f}",
        f"vocab_total={sum(vocab_sizes.values()):,} across {len(vocab_sizes)} fields",
        f"folds={int(args.folds)} | counts={folds_json['counts']}",
        f"out={out_dir.as_posix()}",
    ]
    brief = "prepared vocabs/stats/folds" if args.n_rows is None else f"prepared (n_rows={args.n_rows}) vocabs/stats/folds"
    append_md_entry("plan2/000_EXPERIMENT_LOG.md", "001 Data pipeline v2", brief, bullets)

    print("[OK] Data pipeline artifacts saved to:", out_dir)


if __name__ == "__main__":
    main()

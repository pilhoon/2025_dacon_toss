from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import match_patterns


@dataclass
class DatasetConfig:
    train_path: str
    test_path: str
    target: str
    id_column: str
    use_patterns: Optional[List[str]]
    exclude_patterns: Optional[List[str]]
    n_rows: Optional[int]


def read_parquet_fast(path: str, columns: List[str] | None = None, n_rows: Optional[int] = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    if n_rows is not None:
        table = table.slice(0, n_rows)
    return table.to_pandas(types_mapper={})


def load_dataset(cfg: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    # probe schema to decide columns
    schema = pq.read_table(cfg.train_path, columns=None).schema
    all_columns = [name for name in schema.names]
    # ensure target and id
    if cfg.target not in all_columns:
        raise ValueError(f"Target column '{cfg.target}' not found in train dataset")
    # decide include columns
    feature_columns = [c for c in match_patterns(all_columns, cfg.use_patterns, cfg.exclude_patterns) if c != cfg.target]
    train_cols = feature_columns + [cfg.target]
    train_df = read_parquet_fast(cfg.train_path, columns=train_cols, n_rows=cfg.n_rows)

    test_schema = pq.read_table(cfg.test_path, columns=None).schema
    test_columns = [name for name in test_schema.names]
    if cfg.id_column not in test_columns:
        # Some datasets may not include ID in train, ensure at inference time
        pass
    test_feature_columns = [c for c in match_patterns(test_columns, cfg.use_patterns, cfg.exclude_patterns)]
    test_df = read_parquet_fast(cfg.test_path, columns=test_feature_columns, n_rows=None)

    return train_df, test_df


def summarize_schema(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        num_null = int(series.isna().sum())
        nunique = int(series.nunique(dropna=True))
        sample_values = series.dropna().head(3).tolist()
        summary_rows.append({
            "column": col,
            "dtype": dtype,
            "n_null": num_null,
            "n_unique": nunique,
            "sample": sample_values,
        })
    return pd.DataFrame(summary_rows)



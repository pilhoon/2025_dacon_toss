from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from utils import match_patterns, save_json, load_json


@dataclass
class DataConfig:
    train_path: str
    test_path: str
    target: str
    cat_patterns: List[str]
    num_patterns: List[str]
    min_freq: int = 5
    max_seq_len: int = 0


def read_parquet_cols(path: str, columns: List[str] | None = None, n_rows: int | None = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    if n_rows is not None:
        table = table.slice(0, n_rows)
    return table.to_pandas()


def build_vocabs(df: pd.DataFrame, cat_cols: List[str], min_freq: int) -> Dict[str, Dict[str, int]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        vc = df[c].astype(str).value_counts()
        vocab = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for val, cnt in vc.items():
            if cnt >= min_freq:
                vocab[val] = idx
                idx += 1
        vocabs[c] = vocab
    return vocabs


def save_vocabs(vocabs: Dict[str, Dict[str, int]], path: str) -> None:
    save_json(vocabs, path)


def load_vocabs(path: str) -> Dict[str, Dict[str, int]]:
    return load_json(path)


def encode_categoricals(df: pd.DataFrame, vocabs: Dict[str, Dict[str, int]], cat_cols: List[str]) -> Dict[str, np.ndarray]:
    encoded = {}
    for c in cat_cols:
        vocab = vocabs[c]
        arr = df[c].astype(str).map(lambda x: vocab.get(x, 1)).astype(np.int64).to_numpy()
        encoded[c] = arr
    return encoded


def extract_numericals(df: pd.DataFrame, num_cols: List[str]) -> np.ndarray:
    return df[num_cols].astype(np.float32).to_numpy()


def compute_num_stats(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, List[float]]:
    if not num_cols:
        return {"mean": [], "std": []}
    mean = df[num_cols].astype(np.float32).mean(axis=0).tolist()
    std = (df[num_cols].astype(np.float32).std(axis=0) + 1e-6).tolist()
    return {"mean": mean, "std": std}


def apply_num_stats(df: pd.DataFrame, num_cols: List[str], stats: Optional[Dict[str, List[float]]]) -> np.ndarray:
    if not num_cols:
        return np.empty((len(df), 0), dtype=np.float32)
    arr = df[num_cols].astype(np.float32).to_numpy()
    if stats and stats.get("mean") is not None and stats.get("std") is not None and len(stats["mean"]) == arr.shape[1]:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        arr = (arr - mean) / std
    return arr


class CTRDataset(Dataset):
    def __init__(self, cat_arrays: Dict[str, np.ndarray], num_array: np.ndarray, labels: Optional[np.ndarray] = None):
        self.cat_keys = list(cat_arrays.keys())
        self.cat_arrays = cat_arrays
        self.num_array = num_array
        self.labels = labels

    def __len__(self) -> int:
        return len(self.num_array)

    def __getitem__(self, idx: int):
        cats = {k: torch.as_tensor(v[idx]) for k, v in self.cat_arrays.items()}
        nums = torch.as_tensor(self.num_array[idx])
        if self.labels is None:
            return {"cat": cats, "num": nums}
        else:
            y = torch.as_tensor(self.labels[idx], dtype=torch.float32)
            return {"cat": cats, "num": nums, "y": y}


def prepare_data(cfg: DataConfig, patterns_from_schema: bool = True, n_rows: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    schema = pq.read_table(cfg.train_path).schema
    all_cols = list(schema.names)
    cat_cols = match_patterns(all_cols, cfg.cat_patterns, None)
    num_cols = match_patterns(all_cols, cfg.num_patterns, None)
    use_cols = sorted(set(cat_cols + num_cols + [cfg.target]))
    train_df = read_parquet_cols(cfg.train_path, columns=use_cols, n_rows=n_rows)
    test_schema = pq.read_table(cfg.test_path).schema
    test_cols = list(test_schema.names)
    test_cat_cols = [c for c in cat_cols if c in test_cols]
    test_num_cols = [c for c in num_cols if c in test_cols]
    test_use_cols = sorted(set(test_cat_cols + test_num_cols))
    test_df = read_parquet_cols(cfg.test_path, columns=test_use_cols, n_rows=n_rows)
    return train_df, test_df, cat_cols, num_cols


# Pipeline/lazy dataset: encodes on-the-fly in collate_fn
class CTRDatasetLazy(Dataset):
    def __init__(self, size: int):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> int:
        # return index only; collate_fn will build the batch
        return idx


def make_collate_fn(df: pd.DataFrame, y: Optional[np.ndarray], cat_cols: List[str], num_cols: List[str], vocabs: Dict[str, Dict[str, int]], num_stats: Optional[Dict[str, List[float]]] = None):
    def collate(indices: List[int]):
        batch_df = df.iloc[indices]
        cat_batch: Dict[str, torch.Tensor] = {}
        for c in cat_cols:
            vocab = vocabs[c]
            arr = batch_df[c].astype(str).map(lambda x: vocab.get(x, 1)).to_numpy(dtype=np.int64)
            cat_batch[c] = torch.from_numpy(arr)
        num = torch.from_numpy(apply_num_stats(batch_df, num_cols, num_stats))
        out = {"cat": cat_batch, "num": num}
        if y is not None:
            out["y"] = torch.as_tensor(y[indices], dtype=torch.float32)
        return out

    return collate


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

from plan2.src.dataset import DataConfig, prepare_data, build_vocabs, encode_categoricals, extract_numericals
from plan2.src.utils import load_yaml
from plan2.src.modules.dcnv2 import DCNv2
from plan2.src.modules.xdeepfm import XDeepFM
from plan2.src.modules.ft_transformer import FTTransformer
from plan2.src.modules.din import DIN


def build_model(model_cfg: dict, cat_cardinalities: Dict[str, int], num_dim: int):
    t = model_cfg["type"].lower()
    if t == "dcnv2":
        return DCNv2(cat_cardinalities, num_dim, embed_dim=model_cfg.get("embed_dim", 32), cross_depth=model_cfg.get("cross_depth", 3), mlp_dims=model_cfg.get("mlp_dims"), dropout=model_cfg.get("dropout", 0.0))
    if t == "xdeepfm":
        return XDeepFM(cat_cardinalities, num_dim, embed_dim=model_cfg.get("embed_dim", 32), cin_layers=model_cfg.get("cin_layers"), dnn_layers=model_cfg.get("dnn_layers"), dropout=model_cfg.get("dropout", 0.0))
    if t == "ft_transformer":
        return FTTransformer(cat_cardinalities, num_dim, embed_dim=model_cfg.get("embed_dim", 64), n_layers=model_cfg.get("n_layers", 4), n_heads=model_cfg.get("n_heads", 8), ff_mult=model_cfg.get("ff_mult", 4), dropout=model_cfg.get("dropout", 0.1), stochastic_depth=model_cfg.get("stochastic_depth", 0.0))
    if t == "din":
        return DIN(cat_cardinalities, num_dim, target_item_col=model_cfg.get("target_item_col", "l_feat_14"), embed_dim=model_cfg.get("embed_dim", 32), attn_hidden=model_cfg.get("attn_hidden", 64), mlp_dims=model_cfg.get("mlp_dims"), dropout=model_cfg.get("dropout", 0.1))
    raise ValueError(f"Unknown model type: {t}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = DataConfig(
        train_path=cfg["data"]["train_path"],
        test_path=cfg["data"]["test_path"],
        target=cfg["data"]["target"],
        cat_patterns=cfg["data"]["cat_patterns"],
        num_patterns=cfg["data"]["num_patterns"],
        min_freq=cfg["data"].get("min_freq", 10),
        max_seq_len=cfg["data"].get("max_seq_len", 0),
    )

    train_df, test_df, cat_cols, num_cols = prepare_data(data_cfg)
    vocabs = build_vocabs(train_df, cat_cols, data_cfg.min_freq)

    cat_test = encode_categoricals(test_df, vocabs, cat_cols)
    num_test = extract_numericals(test_df, num_cols)
    cat_cardinalities = {k: len(v) for k, v in vocabs.items()}
    num_dim = num_test.shape[1] if num_cols else 0

    model = build_model(cfg["model"], cat_cardinalities, num_dim)
    device = torch.device("cuda")
    model.to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    probs = np.zeros(len(num_test), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(probs), 262144):
            sl = slice(i, min(i + 262144, len(probs)))
            batch = {"cat": {k: torch.as_tensor(v[sl]).to(device) for k, v in cat_test.items()}, "num": torch.as_tensor(num_test[sl]).to(device)}
            logit = model(batch)
            probs[sl] = torch.sigmoid(logit).float().cpu().numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Attempt to include ID if available in test set
    try:
        id_table = pq.read_table(cfg["data"]["test_path"], columns=["ID"]).to_pandas()
        df_out = pd.DataFrame({"ID": id_table["ID"].to_numpy(), "clicked": probs})
    except Exception:
        df_out = pd.DataFrame({"clicked": probs})
    df_out.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()


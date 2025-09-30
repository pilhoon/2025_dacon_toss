import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from plan2.src.modules.dcnv2 import DCNv2
from plan2.src.dataset import prepare_data, DataConfig, build_vocabs, compute_num_stats

# Load a small sample
print("Loading data...")
data_cfg = DataConfig(
    train_path="data/train.parquet",
    test_path="data/test.parquet",
    target="clicked",
    cat_patterns=["gender", "age_group", "inventory_id", "l_feat_*", "feat_*"],
    num_patterns=["day_of_week", "hour", "history_*"],
    min_freq=10,
    max_seq_len=50
)

train_df, test_df, cat_cols, num_cols = prepare_data(data_cfg, n_rows=10000)
print(f"Train shape: {train_df.shape}")
print(f"Cat cols: {len(cat_cols)}, Num cols: {len(num_cols)}")

# Build vocabs
vocabs = build_vocabs(train_df, cat_cols, data_cfg.min_freq)
num_stats = compute_num_stats(train_df, num_cols)

# Prepare batch
y = train_df["clicked"].to_numpy().astype(np.float32)
X = train_df.drop(columns=["clicked"])

# Simple encoding
cat_encoded = {}
for col in cat_cols:
    vocab = vocabs[col]
    cat_encoded[col] = torch.tensor([vocab.get(str(v), 1) for v in X[col].values[:32]], dtype=torch.long)

num_encoded = torch.zeros((32, len(num_cols)), dtype=torch.float32)
for i, col in enumerate(num_cols):
    vals = X[col].values[:32]
    mean = num_stats[col]["mean"]
    std = num_stats[col]["std"]
    if std > 0:
        num_encoded[:, i] = torch.tensor((vals - mean) / std, dtype=torch.float32)
    else:
        num_encoded[:, i] = torch.tensor(vals - mean, dtype=torch.float32)

batch = {
    "cat": cat_encoded,
    "num": num_encoded,
    "y": torch.tensor(y[:32], dtype=torch.float32)
}

# Build model
cat_cardinalities = {k: len(v) for k, v in vocabs.items()}
model = DCNv2(
    cat_cardinalities=cat_cardinalities,
    num_dim=len(num_cols),
    embed_dim=8,
    cross_depth=2,
    mlp_dims=[64, 32],
    dropout=0.1
)

print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
model.eval()
with torch.no_grad():
    try:
        logits = model(batch)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"Logits mean: {logits.mean():.4f}")
        print(f"Logits std: {logits.std():.4f}")

        # Test loss
        pos_weight = torch.tensor([10.0])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(logits, batch["y"])
        print(f"\nLoss value: {loss.item():.4f}")

        if torch.isnan(loss):
            print("WARNING: Loss is NaN!")
            print(f"Y values: {batch['y'][:10]}")
            print(f"Logits values: {logits[:10]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\nTest completed!")
"""
Fast DCNv2 training with cached data
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pickle
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# Import model
from plan2.src.modules.dcnv2 import DCNv2

print("Loading cached data...")
cache_dir = Path('plan2/cache')
train_X = pd.read_parquet(cache_dir / 'train_X.parquet')
train_y = np.load(cache_dir / 'train_y.npy')
test_X = pd.read_parquet(cache_dir / 'test_X.parquet')

with open(cache_dir / 'columns.pkl', 'rb') as f:
    col_info = pickle.load(f)
    cat_cols = col_info['cat_cols']
    num_cols = col_info['num_cols']

print(f"Train: {train_X.shape}, Test: {test_X.shape}")
print(f"Positive rate: {train_y.mean():.4f}")

# Build vocabularies
print("Building vocabularies...")
vocabs = {}
for col in cat_cols:
    unique_vals = pd.concat([train_X[col], test_X[col]]).astype(str).unique()
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for i, val in enumerate(unique_vals):
        if val not in vocab:
            vocab[val] = len(vocab)
    vocabs[col] = vocab
    print(f"{col}: {len(vocab)} unique values")

# Encode categorical
print("Encoding categorical features...")
cat_encoded = np.zeros((len(train_X), len(cat_cols)), dtype=np.int64)
for i, col in enumerate(cat_cols):
    vocab = vocabs[col]
    cat_encoded[:, i] = train_X[col].astype(str).map(lambda x: vocab.get(x, 0)).values

# Normalize numerical
print("Normalizing numerical features...")
num_encoded = np.zeros((len(train_X), len(num_cols)), dtype=np.float32)
for i, col in enumerate(num_cols):
    vals = train_X[col].values.astype(np.float32)
    mean = vals.mean()
    std = vals.std() + 1e-6
    num_encoded[:, i] = (vals - mean) / std

print(f"Encoded shapes - Cat: {cat_encoded.shape}, Num: {num_encoded.shape}")

# Model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple 2-fold CV
n_folds = 2
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_y))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
    print(f"\n=== FOLD {fold} ===")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create simple model
    cat_cardinalities = {f"cat_{i}": len(vocabs[col]) for i, col in enumerate(cat_cols)}
    model = DCNv2(
        cat_cardinalities=cat_cardinalities,
        num_dim=len(num_cols),
        embed_dim=8,
        cross_depth=2,
        mlp_dims=[64, 32],
        dropout=0.2
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Simple data loaders
    X_train_cat = torch.from_numpy(cat_encoded[train_idx])
    X_train_num = torch.from_numpy(num_encoded[train_idx])
    y_train = torch.from_numpy(train_y[train_idx])

    X_val_cat = torch.from_numpy(cat_encoded[val_idx])
    X_val_num = torch.from_numpy(num_encoded[val_idx])
    y_val = torch.from_numpy(train_y[val_idx])

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    batch_size = 4096
    n_epochs = 2

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Simple batch iteration
        for i in range(0, len(train_idx), batch_size):
            batch_indices = slice(i, min(i + batch_size, len(train_idx)))

            # Prepare batch
            batch_cat_dict = {f"cat_{j}": X_train_cat[batch_indices, j].to(device)
                             for j in range(len(cat_cols))}
            batch_num = X_train_num[batch_indices].to(device)
            batch_y = y_train[batch_indices].to(device)

            batch = {"cat": batch_cat_dict, "num": batch_num}

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch_y)

            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {n_batches}")
                print(f"Outputs: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
                print(f"Y: sum={batch_y.sum():.0f}/{len(batch_y)}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            val_cat_dict = {f"cat_{j}": X_val_cat[:, j].to(device)
                           for j in range(len(cat_cols))}
            val_batch = {"cat": val_cat_dict, "num": X_val_num.to(device)}
            val_outputs = model(val_batch)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()

        # Metrics
        try:
            val_auc = roc_auc_score(y_val, val_probs)
            val_ap = average_precision_score(y_val, val_probs)
            val_logloss = log_loss(y_val, val_probs)
        except:
            val_auc = val_ap = val_logloss = 0

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, AUC={val_auc:.4f}, AP={val_ap:.4f}, LogLoss={val_logloss:.4f}")
        print(f"Val pred stats: mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")

    # Save OOF predictions
    oof_preds[val_idx] = val_probs

# Final evaluation
print(f"\n=== FINAL OOF RESULTS ===")
try:
    oof_auc = roc_auc_score(train_y, oof_preds)
    oof_ap = average_precision_score(train_y, oof_preds)
    oof_logloss = log_loss(train_y, oof_preds)
    print(f"AUC: {oof_auc:.4f}")
    print(f"AP: {oof_ap:.4f}")
    print(f"LogLoss: {oof_logloss:.4f}")
except Exception as e:
    print(f"Error computing metrics: {e}")

print(f"Prediction stats: mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}")
print("Done!")
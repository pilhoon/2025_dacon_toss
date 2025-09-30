"""
Prepare cached data for faster experiments
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import time

print("Loading data...")
t0 = time.time()

# Load only first 1M rows
train_df = pd.read_parquet('data/train.parquet', engine='pyarrow').head(1000000)
test_df = pd.read_parquet('data/test.parquet', engine='pyarrow').head(100000)

print(f"Loaded in {time.time()-t0:.1f}s")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Identify column types
cat_cols = []
num_cols = []

for col in train_df.columns:
    if col == 'clicked':
        continue
    if col in ['gender', 'age_group', 'inventory_id', 'seq'] or col.startswith('l_feat_') or col.startswith('feat_'):
        cat_cols.append(col)
    else:
        num_cols.append(col)

print(f"Categorical: {len(cat_cols)}")
print(f"Numerical: {len(num_cols)}")

# Save as numpy arrays for faster loading
cache_dir = Path('plan2/cache')
cache_dir.mkdir(exist_ok=True)

# Save train
np.save(cache_dir / 'train_y.npy', train_df['clicked'].values.astype(np.float32))
train_df.drop(columns=['clicked']).to_parquet(cache_dir / 'train_X.parquet', engine='pyarrow')

# Save test
test_df.to_parquet(cache_dir / 'test_X.parquet', engine='pyarrow')

# Save column info
with open(cache_dir / 'columns.pkl', 'wb') as f:
    pickle.dump({
        'cat_cols': cat_cols,
        'num_cols': num_cols
    }, f)

print(f"Cached to {cache_dir}")
print("Done!")
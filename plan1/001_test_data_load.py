import sys
sys.path.append('src')

from data import DatasetConfig, load_dataset
import time

# 설정
cfg = DatasetConfig(
    train_path="../data/train.parquet",
    test_path="../data/test.parquet",
    target="clicked",
    id_column="ID",
    use_patterns=["gender", "age_group", "inventory_id"],
    exclude_patterns=[],
    n_rows=1000
)

print("데이터 로딩 시작...")
start = time.time()
train_df, test_df = load_dataset(cfg)
print(f"로딩 완료: {time.time() - start:.2f}초")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()[:10]}...")
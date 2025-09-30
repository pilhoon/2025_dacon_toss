#!/usr/bin/env python3
"""
data_loader.py
Efficient data loading module with caching and parallel processing
"""

import numpy as np
import pandas as pd
import pickle
import hashlib
from pathlib import Path
from joblib import Parallel, delayed, Memory
import multiprocessing
from sklearn.preprocessing import LabelEncoder
import time
import pyarrow.feather as feather
import gc

# Setup joblib memory for caching
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)
memory = Memory(cache_dir, verbose=0)

class DataLoader:
    """Efficient data loader with caching"""

    def __init__(self, cache_dir="./cache", n_jobs=-1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        print(f"DataLoader initialized with {self.n_jobs} workers")

        # File paths for caching
        self.train_cache = self.cache_dir / "train_processed.feather"
        self.test_cache = self.cache_dir / "test_processed.feather"
        self.encoders_cache = self.cache_dir / "encoders.pkl"
        self.feature_info_cache = self.cache_dir / "feature_info.pkl"

    def _compute_hash(self, df):
        """Compute hash of dataframe for cache validation"""
        return hashlib.md5(pd.util.hash_pandas_object(df.head(1000)).values).hexdigest()

    def _parallel_encode_categorical(self, train_df, test_df, cat_cols):
        """Parallel categorical encoding"""
        print(f"Encoding {len(cat_cols)} categorical columns with {self.n_jobs} workers...")

        def encode_column(col):
            le = LabelEncoder()
            # Combine train and test for consistent encoding
            combined = pd.concat([
                train_df[col].fillna('missing').astype(str),
                test_df[col].fillna('missing').astype(str)
            ])
            le.fit(combined)

            train_encoded = le.transform(train_df[col].fillna('missing').astype(str))
            test_encoded = le.transform(test_df[col].fillna('missing').astype(str))

            return col, train_encoded, test_encoded, le

        # Parallel encoding
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(encode_column)(col) for col in cat_cols
        )

        encoders = {}
        for col, train_enc, test_enc, le in results:
            train_df[col] = train_enc
            test_df[col] = test_enc
            encoders[col] = le

        return train_df, test_df, encoders

    def load_raw_data(self):
        """Load raw data from parquet files"""
        print("Loading raw data...")
        t0 = time.time()

        # Use absolute path or find data directory
        base_path = Path('/home/km/work/2025_dacon_toss')
        train_path = base_path / 'data' / 'train.parquet'
        test_path = base_path / 'data' / 'test.parquet'

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        print(f"Raw data loaded in {time.time() - t0:.1f}s")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        return train_df, test_df

    def process_and_cache(self, force_rebuild=False):
        """Process data and cache results"""

        # Check if cache exists and is valid
        if not force_rebuild and all([
            self.train_cache.exists(),
            self.test_cache.exists(),
            self.encoders_cache.exists(),
            self.feature_info_cache.exists()
        ]):
            print("Loading from cache...")
            return self.load_from_cache()

        print("Processing data (this will be cached for future use)...")
        t0 = time.time()

        # Load raw data
        train_df, test_df = self.load_raw_data()

        # Extract labels
        y_train = train_df['clicked'].values.astype(np.float32)
        train_df = train_df.drop(columns=['clicked'])

        # Identify column types
        print("Identifying column types...")
        cat_cols = []
        num_cols = []
        seq_cols = []

        for col in train_df.columns:
            if col == 'ID':
                continue

            sample = train_df[col].dropna().iloc[0] if not train_df[col].isna().all() else None

            if sample is None:
                num_cols.append(col)
            elif isinstance(sample, str):
                if ',' in str(sample):
                    seq_cols.append(col)  # Skip sequence features
                else:
                    cat_cols.append(col)
            else:
                # Check cardinality
                if train_df[col].nunique() < 100:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)

        print(f"Found: {len(cat_cols)} categorical, {len(num_cols)} numeric, {len(seq_cols)} sequence")

        # Drop sequence columns for now
        if seq_cols:
            train_df = train_df.drop(columns=seq_cols)
            test_df = test_df.drop(columns=seq_cols)

        # Process categorical columns in parallel
        encoders = {}
        if cat_cols:
            train_df, test_df, encoders = self._parallel_encode_categorical(
                train_df, test_df, cat_cols
            )

        # Process numeric columns
        print("Processing numeric columns...")
        for col in num_cols:
            if col in train_df.columns:
                # Convert to numeric and fill NaN
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(np.float32)
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0).astype(np.float32)

                # Simple normalization (can be improved)
                max_val = max(train_df[col].max(), test_df[col].max())
                if max_val > 0:
                    train_df[col] = train_df[col] / max_val
                    test_df[col] = test_df[col] / max_val

        # Save feature info
        feature_info = {
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'seq_cols': seq_cols,
            'feature_cols': [c for c in train_df.columns if c != 'ID']
        }

        # Cache everything
        print("Caching processed data...")

        # Use feather for fast I/O
        train_df.reset_index(drop=True).to_feather(self.train_cache)
        test_df.reset_index(drop=True).to_feather(self.test_cache)

        # Save encoders and feature info
        with open(self.encoders_cache, 'wb') as f:
            pickle.dump(encoders, f)

        with open(self.feature_info_cache, 'wb') as f:
            pickle.dump(feature_info, f)

        # Save labels separately
        np.save(self.cache_dir / "y_train.npy", y_train)

        print(f"Processing and caching completed in {time.time() - t0:.1f}s")

        return train_df, test_df, y_train, feature_info, encoders

    def load_from_cache(self):
        """Load processed data from cache"""
        t0 = time.time()

        # Load dataframes
        train_df = feather.read_feather(self.train_cache)
        test_df = feather.read_feather(self.test_cache)

        # Load labels
        y_train = np.load(self.cache_dir / "y_train.npy")

        # Load encoders and feature info
        with open(self.encoders_cache, 'rb') as f:
            encoders = pickle.load(f)

        with open(self.feature_info_cache, 'rb') as f:
            feature_info = pickle.load(f)

        print(f"Loaded from cache in {time.time() - t0:.1f}s")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        return train_df, test_df, y_train, feature_info, encoders

    def get_feature_matrix(self, train_df, test_df, feature_info):
        """Extract feature matrices for training"""

        feature_cols = feature_info['feature_cols']

        # Remove ID column if present
        feature_cols = [c for c in feature_cols if c != 'ID']

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        return X_train, X_test, feature_cols

    def add_engineered_features(self, df, feature_cols):
        """Add engineered features (can be extended)"""
        print("Adding engineered features...")

        new_cols = []

        # Example: Add interaction features for top numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns[:10]

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplication interaction
                new_col = f"{col1}_x_{col2}"
                df[new_col] = df[col1] * df[col2]
                new_cols.append(new_col)

                # Ratio (avoid division by zero)
                new_col = f"{col1}_div_{col2}"
                df[new_col] = df[col1] / (df[col2] + 1e-6)
                new_cols.append(new_col)

        print(f"Added {len(new_cols)} engineered features")

        return df, feature_cols + new_cols


# Singleton instance
_data_loader = None

def get_data_loader():
    """Get or create singleton DataLoader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader

# Convenience function for quick loading
def load_data(force_rebuild=False):
    """Quick data loading function"""
    loader = get_data_loader()
    return loader.process_and_cache(force_rebuild=force_rebuild)


if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")

    # First run - will process and cache
    t0 = time.time()
    train_df, test_df, y_train, feature_info, encoders = load_data()
    print(f"First load: {time.time() - t0:.1f}s")

    # Second run - should load from cache
    t0 = time.time()
    train_df2, test_df2, y_train2, feature_info2, encoders2 = load_data()
    print(f"Second load (from cache): {time.time() - t0:.1f}s")

    # Get feature matrices
    loader = get_data_loader()
    X_train, X_test, feature_cols = loader.get_feature_matrix(train_df, test_df, feature_info)
    print(f"Feature matrices: X_train {X_train.shape}, X_test {X_test.shape}")
# plan4/src/feature_engineering.py
"""
Feature engineering and caching pipeline
Consolidates features from Plan1 and Plan3
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional
import time


class FeatureCache:
    """Cache computed features to avoid redundant processing"""

    def __init__(self, cache_dir: str = "plan4/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, df_hash: str, feature_name: str) -> str:
        """Generate cache key from data hash and feature name"""
        return f"{df_hash}_{feature_name}"

    def save(self, data: pd.DataFrame, key: str):
        """Save features to cache"""
        cache_path = self.cache_dir / f"{key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, key: str) -> Optional[pd.DataFrame]:
        """Load features from cache if exists"""
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash from dataframe shape and sample"""
        hash_str = f"{df.shape}_{df.iloc[:100].values.tobytes()}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]


class FeatureEngineer:
    """Main feature engineering pipeline"""

    def __init__(self, use_cache: bool = True, verbose: bool = True):
        self.cache = FeatureCache() if use_cache else None
        self.verbose = verbose
        self.feature_stats = {}
        self.is_fitted = False
        self.fill_values = {}  # Store median values from training data

    def log(self, message: str):
        """Print message with timestamp"""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic engineered features"""
        self.log("Creating basic features...")

        df_feat = df.copy()

        # Time-based features
        # NOTE: Keep categorical columns as strings, don't convert to codes here
        # The OrdinalEncoder in main script will handle the conversion consistently
        if 'hour' in df.columns:
            # Convert hour to actual numeric values for correct time-based features
            # Assuming hour values are strings like '00', '01', ..., '23'
            hour_numeric = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)
            df_feat['is_night'] = hour_numeric.isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
            df_feat['is_morning'] = hour_numeric.isin([6, 7, 8, 9, 10, 11]).astype(int)
            df_feat['is_afternoon'] = hour_numeric.isin([12, 13, 14, 15, 16, 17]).astype(int)
            df_feat['is_evening'] = hour_numeric.isin([18, 19]).astype(int)

        if 'day_of_week' in df.columns:
            # Map day names to correct numeric values
            dow_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
            dow_numeric = df['day_of_week'].map(dow_mapping)

            # If not found in mapping, try numeric conversion
            if dow_numeric.isna().any():
                dow_numeric = dow_numeric.fillna(pd.to_numeric(df['day_of_week'], errors='coerce'))

            dow_numeric = dow_numeric.fillna(0).astype(int)
            df_feat['is_weekend'] = dow_numeric.isin([5, 6]).astype(int)
            df_feat['is_weekday'] = (~dow_numeric.isin([5, 6])).astype(int)

        # Demographic features
        if 'age_group' in df.columns and 'gender' in df.columns:
            df_feat['age_gender'] = df['age_group'].astype(str) + '_' + df['gender'].astype(str)

        # Sequence features (from Plan3)
        if 'seq' in df.columns:
            # Sequence length (number of items)
            df_feat['seq_length'] = df['seq'].astype(str).apply(lambda x: len(x.split(',')) if pd.notna(x) and x != 'nan' else 0)

            # Sequence diversity (unique items)
            df_feat['seq_unique'] = df['seq'].astype(str).apply(
                lambda x: len(set(x.split(','))) if pd.notna(x) and x != 'nan' else 0
            )

            # Sequence repetition rate
            df_feat['seq_repetition'] = np.where(
                df_feat['seq_length'] > 0,
                1 - df_feat['seq_unique'] / df_feat['seq_length'],
                0
            )

        return df_feat

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features"""
        self.log("Creating statistical features...")

        df_feat = df.copy()

        # Numerical feature statistics
        num_cols = [col for col in df.columns if col.startswith(('l_feat_', 'feat_', 'history_'))]

        if num_cols:
            # Row-wise statistics
            df_feat['num_mean'] = df[num_cols].mean(axis=1)
            df_feat['num_std'] = df[num_cols].std(axis=1)
            df_feat['num_min'] = df[num_cols].min(axis=1)
            df_feat['num_max'] = df[num_cols].max(axis=1)
            df_feat['num_range'] = df_feat['num_max'] - df_feat['num_min']
            df_feat['num_skew'] = df[num_cols].apply(lambda x: x.skew(), axis=1)

            # Zero/missing indicators
            df_feat['num_zeros'] = (df[num_cols] == 0).sum(axis=1)
            df_feat['num_missing'] = df[num_cols].isna().sum(axis=1)
            df_feat['num_nonzero_ratio'] = 1 - df_feat['num_zeros'] / len(num_cols)

        # Feature group statistics
        for prefix in ['l_feat_', 'feat_a_', 'feat_b_', 'feat_c_', 'feat_d_', 'feat_e_', 'history_a_', 'history_b_']:
            group_cols = [col for col in df.columns if col.startswith(prefix)]
            if len(group_cols) > 1:
                df_feat[f'{prefix}mean'] = df[group_cols].mean(axis=1)
                df_feat[f'{prefix}std'] = df[group_cols].std(axis=1)
                df_feat[f'{prefix}max'] = df[group_cols].max(axis=1)

        return df_feat

    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 20) -> pd.DataFrame:
        """Create interaction features between important columns"""
        self.log(f"Creating top {max_interactions} interaction features...")

        df_feat = df.copy()

        # Define important feature pairs based on domain knowledge
        interactions = [
            ('inventory_id', 'hour'),
            ('inventory_id', 'day_of_week'),
            ('age_group', 'hour'),
            ('gender', 'inventory_id'),
            ('age_group', 'gender'),
        ]

        # Create categorical interactions
        # NOTE: Keep as string for OrdinalEncoder to handle consistently
        for col1, col2 in interactions[:max_interactions]:
            if col1 in df.columns and col2 in df.columns:
                feat_name = f'interact_{col1}_{col2}'
                # Create interaction as string concatenation, don't convert to codes here
                df_feat[feat_name] = df[col1].astype(str) + '_' + df[col2].astype(str)

        # Create numerical interactions for top features
        num_cols = [col for col in df.columns if col.startswith('l_feat_')][:5]
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                df_feat[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df_feat[f'{col1}_div_{col2}'] = np.where(df[col2] != 0, df[col1] / df[col2], 0)

        return df_feat

    def handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values consistently

        Args:
            df: DataFrame to process
            fit: If True, compute and store fill values (training mode)
                 If False, use stored fill values (inference mode)
        """
        self.log(f"Handling missing values... (fit={fit})")

        df_clean = df.copy()

        # Categorical columns: fill with 'missing'
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna('missing')

        # Numerical columns: fill with appropriate values
        num_cols = df_clean.select_dtypes(include=['number']).columns

        # Different strategies for different column types
        for col in num_cols:
            if col.startswith('history_'):
                # Historical features: fill with 0 (no history)
                df_clean[col] = df_clean[col].fillna(0)
            elif col.startswith('feat_') or col.startswith('l_feat_'):
                # Feature columns: fill with median
                if fit:
                    # Training mode: compute and store median
                    median_val = df_clean[col].median()
                    if pd.notna(median_val):  # Only store valid medians
                        self.fill_values[col] = median_val
                    else:
                        self.fill_values[col] = 0  # Fallback if all NaN
                    df_clean[col] = df_clean[col].fillna(self.fill_values[col])
                else:
                    # Inference mode: use stored median
                    median_val = self.fill_values.get(col, 0)  # Default to 0 if not found
                    df_clean[col] = df_clean[col].fillna(median_val)
            else:
                # Others: fill with 0
                df_clean[col] = df_clean[col].fillna(0)

        return df_clean

    def fit_transform(self, df: pd.DataFrame, feature_groups: List[str] = None) -> pd.DataFrame:
        """Fit the transformer and transform the data (for training data)"""
        self.is_fitted = True
        return self._transform_internal(df, feature_groups, fit=True)

    def transform(self, df: pd.DataFrame, feature_groups: List[str] = None) -> pd.DataFrame:
        """Transform the data using fitted parameters (for test data)

        If not fitted yet, performs fit_transform automatically (for backward compatibility)
        """
        if not self.is_fitted:
            # For backward compatibility, auto-fit if not fitted
            self.log("Warning: transform() called before fit_transform(). Auto-fitting...")
            return self.fit_transform(df, feature_groups)
        return self._transform_internal(df, feature_groups, fit=False)

    def _transform_internal(self, df: pd.DataFrame, feature_groups: List[str] = None, fit: bool = False) -> pd.DataFrame:
        """Internal transformation pipeline"""

        start_time = time.time()
        mode = "fit_transform" if fit else "transform"
        self.log(f"Starting feature engineering for {len(df)} samples... (mode={mode})")

        # Default to all feature groups
        if feature_groups is None:
            feature_groups = ['basic', 'statistical', 'interaction']

        # Check cache
        df_hash = self.cache.get_data_hash(df) if self.cache else None

        # Initialize with original dataframe
        df_transformed = df.copy()

        # Apply feature groups
        if 'basic' in feature_groups:
            cache_key = f"{df_hash}_basic" if df_hash else None
            if self.cache and cache_key:
                cached = self.cache.load(cache_key)
                if cached is not None:
                    self.log("Loaded basic features from cache")
                    basic_cols = [col for col in cached.columns if col not in df.columns]
                    df_transformed[basic_cols] = cached[basic_cols]
                else:
                    basic_df = self.create_basic_features(df)
                    new_cols = [col for col in basic_df.columns if col not in df.columns]
                    df_transformed[new_cols] = basic_df[new_cols]
                    self.cache.save(basic_df[new_cols], cache_key)
            else:
                basic_df = self.create_basic_features(df)
                new_cols = [col for col in basic_df.columns if col not in df.columns]
                df_transformed[new_cols] = basic_df[new_cols]

        if 'statistical' in feature_groups:
            cache_key = f"{df_hash}_statistical" if df_hash else None
            if self.cache and cache_key:
                cached = self.cache.load(cache_key)
                if cached is not None:
                    self.log("Loaded statistical features from cache")
                    stat_cols = [col for col in cached.columns if col not in df_transformed.columns]
                    df_transformed[stat_cols] = cached[stat_cols]
                else:
                    stat_df = self.create_statistical_features(df)
                    new_cols = [col for col in stat_df.columns if col not in df_transformed.columns]
                    df_transformed[new_cols] = stat_df[new_cols]
                    self.cache.save(stat_df[new_cols], cache_key)
            else:
                stat_df = self.create_statistical_features(df)
                new_cols = [col for col in stat_df.columns if col not in df_transformed.columns]
                df_transformed[new_cols] = stat_df[new_cols]

        if 'interaction' in feature_groups:
            inter_df = self.create_interaction_features(df_transformed)
            new_cols = [col for col in inter_df.columns if col not in df_transformed.columns]
            df_transformed[new_cols] = inter_df[new_cols]

        # Handle missing values with fit parameter
        df_transformed = self.handle_missing_values(df_transformed, fit=fit)

        # Store feature statistics
        self.feature_stats = {
            'n_original': len(df.columns),
            'n_engineered': len(df_transformed.columns) - len(df.columns),
            'n_total': len(df_transformed.columns),
            'time_elapsed': time.time() - start_time
        }

        self.log(f"Feature engineering complete: {self.feature_stats['n_engineered']} new features created")
        self.log(f"Total features: {self.feature_stats['n_total']}, Time: {self.feature_stats['time_elapsed']:.2f}s")

        return df_transformed


if __name__ == "__main__":
    # Test feature engineering pipeline
    print("Testing Feature Engineering Pipeline")
    print("=" * 60)

    # Load sample data
    print("\nLoading sample data...")
    df = pd.read_parquet('data/train.parquet')
    df_sample = df.head(10000)

    # Initialize feature engineer
    fe = FeatureEngineer(use_cache=True, verbose=True)

    # Transform data
    df_transformed = fe.transform(df_sample)

    print("\n" + "=" * 60)
    print("Feature Engineering Summary:")
    print(f"  Original features: {fe.feature_stats['n_original']}")
    print(f"  Engineered features: {fe.feature_stats['n_engineered']}")
    print(f"  Total features: {fe.feature_stats['n_total']}")
    print(f"  Processing time: {fe.feature_stats['time_elapsed']:.2f}s")

    # Show sample of new features
    new_cols = [col for col in df_transformed.columns if col not in df.columns][:10]
    print(f"\nSample of new features: {new_cols}")

    # Test cache
    print("\n" + "=" * 60)
    print("Testing cache (should be faster)...")
    df_sample2 = df.head(10000)
    fe2 = FeatureEngineer(use_cache=True, verbose=True)
    df_transformed2 = fe2.transform(df_sample2)
    print(f"Cache test time: {fe2.feature_stats['time_elapsed']:.2f}s")
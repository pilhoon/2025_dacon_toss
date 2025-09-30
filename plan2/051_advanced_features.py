import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans
import warnings
import gc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')
sys.path.append('..')
from src.data_loader import DataLoader

def create_advanced_features(train_df, test_df):
    """
    Create advanced feature engineering
    """
    print("="*60)
    print("Advanced Feature Engineering")
    print("="*60)

    all_features = []

    # 1. Statistical aggregations per categorical
    print("\n1. Creating statistical aggregations...")
    categorical_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' or train_df[col].nunique() < 100]
    categorical_cols = [col for col in categorical_cols if col not in ['ID', 'target']][:20]  # Top 20 categoricals

    numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ['ID', 'target']][:30]  # Top 30 numericals

    for cat_col in categorical_cols[:10]:  # Limit to avoid memory issues
        for num_col in numerical_cols[:10]:
            # Mean encoding
            mean_enc = train_df.groupby(cat_col)[num_col].mean()
            train_df[f'{cat_col}_{num_col}_mean'] = train_df[cat_col].map(mean_enc)
            test_df[f'{cat_col}_{num_col}_mean'] = test_df[cat_col].map(mean_enc)

            # Std encoding
            std_enc = train_df.groupby(cat_col)[num_col].std()
            train_df[f'{cat_col}_{num_col}_std'] = train_df[cat_col].map(std_enc)
            test_df[f'{cat_col}_{num_col}_std'] = test_df[cat_col].map(std_enc)

            all_features.extend([f'{cat_col}_{num_col}_mean', f'{cat_col}_{num_col}_std'])

    print(f"Created {len(all_features)} aggregation features")

    # 2. Target encoding with regularization
    print("\n2. Creating target encodings...")
    if 'target' in train_df.columns:
        for cat_col in categorical_cols[:15]:
            # Calculate target encoding with smoothing
            target_mean = train_df['target'].mean()
            agg = train_df.groupby(cat_col)['target'].agg(['sum', 'count'])
            smoothing = 100
            agg['smooth_mean'] = (agg['sum'] + smoothing * target_mean) / (agg['count'] + smoothing)

            train_df[f'{cat_col}_target_enc'] = train_df[cat_col].map(agg['smooth_mean'])
            test_df[f'{cat_col}_target_enc'] = test_df[cat_col].map(agg['smooth_mean'])
            test_df[f'{cat_col}_target_enc'].fillna(target_mean, inplace=True)

            all_features.append(f'{cat_col}_target_enc')

    # 3. Frequency encoding
    print("\n3. Creating frequency encodings...")
    for col in categorical_cols[:20]:
        freq = train_df[col].value_counts(normalize=True)
        train_df[f'{col}_freq'] = train_df[col].map(freq)
        test_df[f'{col}_freq'] = test_df[col].map(freq).fillna(0)
        all_features.append(f'{col}_freq')

    # 4. Interaction features
    print("\n4. Creating interaction features...")
    for i in range(len(numerical_cols[:15])):
        for j in range(i+1, len(numerical_cols[:15])):
            col1, col2 = numerical_cols[i], numerical_cols[j]

            # Multiplication
            train_df[f'{col1}_X_{col2}'] = train_df[col1] * train_df[col2]
            test_df[f'{col1}_X_{col2}'] = test_df[col1] * test_df[col2]

            # Division (with small epsilon to avoid division by zero)
            train_df[f'{col1}_div_{col2}'] = train_df[col1] / (train_df[col2] + 1e-8)
            test_df[f'{col1}_div_{col2}'] = test_df[col1] / (test_df[col2] + 1e-8)

            all_features.extend([f'{col1}_X_{col2}', f'{col1}_div_{col2}'])

    # 5. Polynomial features for important columns
    print("\n5. Creating polynomial features...")
    for col in numerical_cols[:10]:
        train_df[f'{col}_squared'] = train_df[col] ** 2
        test_df[f'{col}_squared'] = test_df[col] ** 2

        train_df[f'{col}_cubed'] = train_df[col] ** 3
        test_df[f'{col}_cubed'] = test_df[col] ** 3

        train_df[f'{col}_sqrt'] = np.sqrt(np.abs(train_df[col]))
        test_df[f'{col}_sqrt'] = np.sqrt(np.abs(test_df[col]))

        train_df[f'{col}_log'] = np.log1p(np.abs(train_df[col]))
        test_df[f'{col}_log'] = np.log1p(np.abs(test_df[col]))

        all_features.extend([f'{col}_squared', f'{col}_cubed', f'{col}_sqrt', f'{col}_log'])

    # 6. Clustering features
    print("\n6. Creating clustering features...")
    cluster_features = numerical_cols[:20]
    X_cluster_train = train_df[cluster_features].fillna(0).values
    X_cluster_test = test_df[cluster_features].fillna(0).values

    for n_clusters in [5, 10, 20]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_df[f'cluster_{n_clusters}'] = kmeans.fit_predict(X_cluster_train)
        test_df[f'cluster_{n_clusters}'] = kmeans.predict(X_cluster_test)

        # Distance to each cluster center
        train_distances = kmeans.transform(X_cluster_train)
        test_distances = kmeans.transform(X_cluster_test)

        for i in range(n_clusters):
            train_df[f'dist_cluster_{n_clusters}_{i}'] = train_distances[:, i]
            test_df[f'dist_cluster_{n_clusters}_{i}'] = test_distances[:, i]
            all_features.append(f'dist_cluster_{n_clusters}_{i}')

        all_features.append(f'cluster_{n_clusters}')

    # 7. PCA features
    print("\n7. Creating PCA features...")
    pca_features = numerical_cols[:30]
    X_pca_train = train_df[pca_features].fillna(0).values
    X_pca_test = test_df[pca_features].fillna(0).values

    for n_comp in [5, 10, 20]:
        pca = PCA(n_components=n_comp, random_state=42)
        pca_train = pca.fit_transform(X_pca_train)
        pca_test = pca.transform(X_pca_test)

        for i in range(n_comp):
            train_df[f'pca_{n_comp}_{i}'] = pca_train[:, i]
            test_df[f'pca_{n_comp}_{i}'] = pca_test[:, i]
            all_features.append(f'pca_{n_comp}_{i}')

    # 8. Row-wise statistics
    print("\n8. Creating row-wise statistics...")
    num_cols_subset = numerical_cols[:30]

    train_df['row_sum'] = train_df[num_cols_subset].sum(axis=1)
    test_df['row_sum'] = test_df[num_cols_subset].sum(axis=1)

    train_df['row_mean'] = train_df[num_cols_subset].mean(axis=1)
    test_df['row_mean'] = test_df[num_cols_subset].mean(axis=1)

    train_df['row_std'] = train_df[num_cols_subset].std(axis=1)
    test_df['row_std'] = test_df[num_cols_subset].std(axis=1)

    train_df['row_skew'] = train_df[num_cols_subset].skew(axis=1)
    test_df['row_skew'] = test_df[num_cols_subset].skew(axis=1)

    train_df['row_kurt'] = train_df[num_cols_subset].kurtosis(axis=1)
    test_df['row_kurt'] = test_df[num_cols_subset].kurtosis(axis=1)

    train_df['row_median'] = train_df[num_cols_subset].median(axis=1)
    test_df['row_median'] = test_df[num_cols_subset].median(axis=1)

    train_df['row_max'] = train_df[num_cols_subset].max(axis=1)
    test_df['row_max'] = test_df[num_cols_subset].max(axis=1)

    train_df['row_min'] = train_df[num_cols_subset].min(axis=1)
    test_df['row_min'] = test_df[num_cols_subset].min(axis=1)

    train_df['row_range'] = train_df['row_max'] - train_df['row_min']
    test_df['row_range'] = test_df['row_max'] - test_df['row_min']

    all_features.extend(['row_sum', 'row_mean', 'row_std', 'row_skew', 'row_kurt',
                        'row_median', 'row_max', 'row_min', 'row_range'])

    # 9. Count features
    print("\n9. Creating count features...")
    for col in categorical_cols[:10]:
        counts = train_df[col].value_counts()
        train_df[f'{col}_count'] = train_df[col].map(counts)
        test_df[f'{col}_count'] = test_df[col].map(counts).fillna(0)
        all_features.append(f'{col}_count')

    # 10. Null pattern features
    print("\n10. Creating null pattern features...")
    train_df['null_count'] = train_df[numerical_cols].isnull().sum(axis=1)
    test_df['null_count'] = test_df[numerical_cols].isnull().sum(axis=1)

    train_df['null_ratio'] = train_df['null_count'] / len(numerical_cols)
    test_df['null_ratio'] = test_df['null_count'] / len(numerical_cols)

    all_features.extend(['null_count', 'null_ratio'])

    print(f"\nTotal new features created: {len(all_features)}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Clean up
    gc.collect()

    return train_df, test_df, all_features


def process_and_save():
    """Process data and save with advanced features"""
    # Load data
    print("Loading data...")
    loader = DataLoader(cache_dir='cache')

    train_data, test_data = loader.load_raw_data()
    if hasattr(loader, 'save_to_cache'):
        loader.save_to_cache(train_data, test_data)

    print(f"Original train shape: {train_data.shape}, test shape: {test_data.shape}")

    # Create advanced features
    train_enhanced, test_enhanced, new_features = create_advanced_features(
        train_data.copy(), test_data.copy()
    )

    # Save enhanced data
    print("\nSaving enhanced data...")
    train_enhanced.to_pickle('plan2/051_train_enhanced.pkl')
    test_enhanced.to_pickle('plan2/051_test_enhanced.pkl')

    # Save feature list
    with open('plan2/051_new_features.txt', 'w') as f:
        for feat in new_features:
            f.write(f"{feat}\n")

    print(f"Enhanced data saved!")
    print(f"Final train shape: {train_enhanced.shape}, test shape: {test_enhanced.shape}")

    return train_enhanced, test_enhanced


if __name__ == "__main__":
    train, test = process_and_save()
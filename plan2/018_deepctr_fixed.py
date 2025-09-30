#!/usr/bin/env python3
"""
Fixed DeepCTR models - handle NaN and CUDA errors properly
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# DeepCTR-Torch imports
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import FiBiNET, AutoInt, DCN, xDeepFM, DeepFM

import warnings
warnings.filterwarnings('ignore')

# Set CUDA environment for debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def prepare_data_for_deepctr(n_samples=50000):
    """Prepare data in DeepCTR format with proper cleaning"""
    print("Preparing data for DeepCTR...")

    # Load cached data
    cache_dir = 'plan2/cache'
    df = pd.read_parquet(f'{cache_dir}/train_X.parquet').head(n_samples)
    y = np.load(f'{cache_dir}/train_y.npy')[:n_samples]

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Handle NaN values FIRST
    print(f"NaN count before cleaning: {df.isna().sum().sum()}")
    df = df.fillna(0)  # Fill NaN with 0
    print(f"NaN count after cleaning: {df.isna().sum().sum()}")

    # Identify feature types
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Limit features
    sparse_features = sparse_features[:10]  # Reduce to 10 for stability
    dense_features = dense_features[:10]

    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        # Convert to string and handle NaN
        df[feat] = df[feat].astype(str).fillna('unknown')
        df[feat] = lbe.fit_transform(df[feat])
        # Ensure indices are within bounds
        df[feat] = df[feat].clip(0, len(lbe.classes_) - 1)

    # Process dense features - ensure proper range
    for feat in dense_features:
        # Convert to numeric first
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
        # Remove outliers using quantiles
        q01 = df[feat].quantile(0.01)
        q99 = df[feat].quantile(0.99)
        df[feat] = df[feat].clip(q01, q99)

    # Normalize dense features to [0, 1]
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])

    # Double check for NaN after scaling
    df[dense_features] = df[dense_features].fillna(0.5)

    # Verify ranges
    print("\nFeature ranges after processing:")
    for feat in sparse_features[:3]:
        print(f"  {feat}: [{df[feat].min()}, {df[feat].max()}]")
    for feat in dense_features[:3]:
        print(f"  {feat}: [{df[feat].min():.3f}, {df[feat].max():.3f}]")

    # Create feature columns for DeepCTR
    fixlen_feature_columns = []

    # Sparse features - fixed embedding dim
    embedding_dim = 8
    for feat in sparse_features:
        vocab_size = int(df[feat].max()) + 2  # +2 for safety
        fixlen_feature_columns.append(
            SparseFeat(feat,
                      vocabulary_size=vocab_size,
                      embedding_dim=embedding_dim)
        )

    # Dense features
    for feat in dense_features:
        fixlen_feature_columns.append(DenseFeat(feat, 1))

    # Get feature names
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # Train-validation split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    y_train = y[train.index]
    y_test = y[test.index]

    # Create model input - ensure proper dtypes
    train_model_input = {}
    test_model_input = {}

    for name in feature_names:
        if name in sparse_features:
            train_model_input[name] = train[name].values.astype(np.int32)
            test_model_input[name] = test[name].values.astype(np.int32)
        else:
            train_model_input[name] = train[name].values.astype(np.float32)
            test_model_input[name] = test[name].values.astype(np.float32)

    return (train_model_input, y_train, test_model_input, y_test,
            linear_feature_columns, dnn_feature_columns)

def test_model(model_class, model_name, train_input, y_train, test_input, y_test,
               linear_cols, dnn_cols, **model_params):
    """Test a DeepCTR model with error handling"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # Create model
        if model_name == "DCN":
            model = model_class(
                linear_feature_columns=linear_cols,
                dnn_feature_columns=dnn_cols,
                task='binary',
                device=device,
                cross_num=2,
                dnn_hidden_units=(64, 32),  # Smaller network
                dnn_dropout=0.3,
                l2_reg_embedding=1e-5
            )
        else:
            model = model_class(
                linear_feature_columns=linear_cols,
                dnn_feature_columns=dnn_cols,
                task='binary',
                device=device,
                dnn_hidden_units=(64, 32),  # Smaller network
                dnn_dropout=0.3,
                l2_reg_embedding=1e-5,
                **model_params
            )

        print(f"Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Compile model
        model.compile("adam",
                     "binary_crossentropy",
                     metrics=["auc"])

        # Train with smaller batch size
        history = model.fit(train_input, y_train,
                           batch_size=512,  # Smaller batch
                           epochs=5,  # Fewer epochs
                           verbose=1,
                           validation_split=0.1)

        # Predict
        pred_probs = model.predict(test_input, batch_size=512)

        # Handle any NaN in predictions
        if np.isnan(pred_probs).any():
            print(f"Warning: NaN in predictions, replacing with 0.5")
            pred_probs = np.nan_to_num(pred_probs, nan=0.5)

        # Evaluate
        test_auc = roc_auc_score(y_test, pred_probs)
        test_ap = average_precision_score(y_test, pred_probs)
        test_logloss = log_loss(y_test, np.clip(pred_probs, 1e-7, 1-1e-7))

        print(f"\nResults for {model_name}:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  AP: {test_ap:.4f}")
        print(f"  LogLoss: {test_logloss:.4f}")
        print(f"  Prediction stats: mean={pred_probs.mean():.4f}, std={pred_probs.std():.4f}")

        return test_auc, test_ap, test_logloss

    except Exception as e:
        print(f"Error training {model_name}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return 0, 0, float('inf')

def main():
    print("FIXED DEEPCTR MODEL COMPARISON")
    print("="*60)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Prepare data
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_data_for_deepctr(n_samples=30000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Models to test (simpler ones first)
    models_to_test = [
        (DeepFM, "DeepFM", {}),  # Simplest and most stable
        (DCN, "DCN", {}),
        (AutoInt, "AutoInt", {"att_layer_num": 1, "att_head_num": 2, "att_res": False}),
        (FiBiNET, "FiBiNET", {"bilinear_type": "field_all"}),
        # (xDeepFM, "xDeepFM", {"cin_layer_size": (64, 64)})  # Most complex
    ]

    results = {}

    for model_class, model_name, params in models_to_test:
        # Clear CUDA cache before each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        auc, ap, logloss = test_model(
            model_class, model_name,
            train_input, y_train,
            test_input, y_test,
            linear_cols, dnn_cols,
            **params
        )
        results[model_name] = {
            'auc': auc,
            'ap': ap,
            'logloss': logloss
        }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    print("\n| Model     | AUC    | AP     | LogLoss |")
    print("|-----------|--------|--------|---------|")
    for model_name, metrics in results.items():
        if metrics['auc'] > 0:  # Only show successful models
            print(f"| {model_name:9s} | {metrics['auc']:.4f} | {metrics['ap']:.4f} | {metrics['logloss']:.4f} |")

    print("\n" + "-"*40)
    print("Plan1 XGBoost benchmark: AUC=0.7430")

    # Find best model
    successful_models = {k: v for k, v in results.items() if v['auc'] > 0}
    if successful_models:
        best_model = max(successful_models.items(), key=lambda x: x[1]['auc'])
        print(f"\nBest DeepCTR model: {best_model[0]} with AUC={best_model[1]['auc']:.4f}")

        if best_model[1]['auc'] > 0.70:
            print("✅ Achieved competitive performance!")
        else:
            print(f"📊 Gap to XGBoost: {0.7430 - best_model[1]['auc']:.4f}")

    # Save results
    import json
    with open('plan2/experiments/deepctr_fixed_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
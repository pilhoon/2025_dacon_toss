#!/usr/bin/env python3
"""
State-of-the-art CTR models using DeepCTR-Torch
Models to test:
1. FiBiNET - Feature Importance-based Bilinear Network
2. AutoInt - Automatic Feature Interaction
3. DCN-V2 - Deep & Cross Network V2
4. xDeepFM - eXtreme Deep Factorization Machine
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# DeepCTR-Torch imports
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import FiBiNET, AutoInt, DCN, xDeepFM

import warnings
warnings.filterwarnings('ignore')

def prepare_data_for_deepctr(n_samples=100000):
    """Prepare data in DeepCTR format"""
    print("Preparing data for DeepCTR...")

    # Load cached data
    cache_dir = 'plan2/cache'
    df = pd.read_parquet(f'{cache_dir}/train_X.parquet').head(n_samples)
    y = np.load(f'{cache_dir}/train_y.npy')[:n_samples]

    print(f"Data shape: {df.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Identify feature types
    sparse_features = []
    dense_features = []

    for col in df.columns:
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            sparse_features.append(col)
        else:
            dense_features.append(col)

    # Limit features for stability
    sparse_features = sparse_features[:15]  # Top 15 categorical
    dense_features = dense_features[:10]    # Top 10 numerical

    print(f"Using {len(sparse_features)} sparse and {len(dense_features)} dense features")

    # Process sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat].astype(str))

    # Process dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])

    # Create feature columns for DeepCTR
    fixlen_feature_columns = []

    # Sparse features - use same embedding_dim for all
    embedding_dim = 8  # Fixed embedding dimension for all sparse features
    for feat in sparse_features:
        fixlen_feature_columns.append(
            SparseFeat(feat,
                      vocabulary_size=df[feat].nunique() + 1,
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

    # Create model input
    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    return (train_model_input, y_train, test_model_input, y_test,
            linear_feature_columns, dnn_feature_columns)

def test_model(model_class, model_name, train_input, y_train, test_input, y_test,
               linear_cols, dnn_cols, **model_params):
    """Test a DeepCTR model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    device = 'cpu'  # Use CPU to avoid CUDA errors

    # Create model
    if model_name == "DCN":
        # DCN has different parameter names
        model = model_class(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            cross_num=2,
            dnn_hidden_units=(128, 64),
            dnn_dropout=0.2
        )
    else:
        # Common parameters for other models
        model = model_class(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task='binary',
            device=device,
            dnn_hidden_units=(128, 64),
            dnn_dropout=0.2,
            **model_params
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compile model
    model.compile("adam",
                  "binary_crossentropy",
                  metrics=["auc", "logloss"])

    # Train
    try:
        history = model.fit(train_input, y_train,
                           batch_size=2048,
                           epochs=10,
                           verbose=2,
                           validation_split=0.1)

        # Predict
        pred_probs = model.predict(test_input, batch_size=2048)

        # Evaluate
        test_auc = roc_auc_score(y_test, pred_probs)
        test_ap = average_precision_score(y_test, pred_probs)
        test_logloss = log_loss(y_test, pred_probs)

        print(f"\nResults for {model_name}:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  AP: {test_ap:.4f}")
        print(f"  LogLoss: {test_logloss:.4f}")
        print(f"  Prediction stats: mean={pred_probs.mean():.4f}, std={pred_probs.std():.4f}")

        return test_auc, test_ap, test_logloss

    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return 0, 0, float('inf')

def main():
    print("DEEPCTR-TORCH MODEL COMPARISON")
    print("="*60)

    # Prepare data
    (train_input, y_train, test_input, y_test,
     linear_cols, dnn_cols) = prepare_data_for_deepctr(n_samples=50000)

    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")

    # Models to test
    models_to_test = [
        (FiBiNET, "FiBiNET", {"bilinear_type": "interaction"}),
        (AutoInt, "AutoInt", {"att_layer_num": 2, "att_head_num": 2}),
        (DCN, "DCN", {}),  # Parameters set differently
        (xDeepFM, "xDeepFM", {"cin_layer_size": (128, 128)})
    ]

    results = {}

    for model_class, model_name, params in models_to_test:
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

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    print("\n| Model     | AUC    | AP     | LogLoss |")
    print("|-----------|--------|--------|---------|")
    for model_name, metrics in results.items():
        print(f"| {model_name:9s} | {metrics['auc']:.4f} | {metrics['ap']:.4f} | {metrics['logloss']:.4f} |")

    # Compare with XGBoost
    print("\n" + "-"*40)
    print("Plan1 XGBoost: AUC=0.7430, AP≈0.25")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\nBest DeepCTR model: {best_model[0]} with AUC={best_model[1]['auc']:.4f}")

    if best_model[1]['auc'] > 0.70:
        print("✅ Achieved competitive performance with XGBoost!")
    else:
        print("❌ Still below XGBoost performance")

    # Save results
    import json
    with open('plan2/experiments/deepctr_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to plan2/experiments/deepctr_results.json")

if __name__ == "__main__":
    main()
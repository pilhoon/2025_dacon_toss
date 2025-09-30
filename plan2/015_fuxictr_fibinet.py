#!/usr/bin/env python3
"""
FiBiNET implementation using FuxiCTR
FiBiNET: Feature Importance-based Bilinear Network
Paper: https://arxiv.org/abs/1905.09433
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from fuxictr.pytorch.models import FiBiNET
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.features import FeatureMap
import yaml
import json

# Set seeds
seed_everything(seed=42)

def prepare_data_for_fuxictr():
    """Prepare data in FuxiCTR format"""
    print("Preparing data for FuxiCTR...")

    # Load cached data
    cache_dir = 'plan2/cache'
    X = pd.read_parquet(f'{cache_dir}/train_X.parquet').head(100000)
    y = np.load(f'{cache_dir}/train_y.npy')[:100000]

    # Add target column
    X['label'] = y

    print(f"Data shape: {X.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    # Split features
    categorical_cols = []
    numerical_cols = []

    for col in X.columns:
        if col == 'label':
            continue
        if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    print(f"Categorical: {len(categorical_cols)}, Numerical: {len(numerical_cols)}")

    # Create feature map configuration
    feature_map = {
        "features": []
    }

    # Add categorical features
    for col in categorical_cols[:20]:  # Use top 20 categorical
        feature_map["features"].append({
            "name": col,
            "type": "categorical",
            "source": "user",
            "vocab_size": X[col].nunique() + 1,  # +1 for unknown
            "embedding_dim": min(50, (X[col].nunique() + 1) // 2)
        })

    # Add numerical features
    for col in numerical_cols[:20]:  # Use top 20 numerical
        feature_map["features"].append({
            "name": col,
            "type": "numeric",
            "source": "user"
        })

    # Add label
    feature_map["features"].append({
        "name": "label",
        "type": "label"
    })

    # Encode categorical features
    for col in categorical_cols[:20]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Normalize numerical features
    scaler = MinMaxScaler()
    X[numerical_cols[:20]] = scaler.fit_transform(X[numerical_cols[:20]])

    # Train-validation split
    train_data, val_data = train_test_split(X, test_size=0.2, random_state=42, stratify=X['label'])

    return train_data, val_data, feature_map

def create_fibinet_model(feature_map):
    """Create FiBiNET model with FuxiCTR"""

    # Model configuration
    model_config = {
        "model": "FiBiNET",
        "dataset_id": "toss_ctr",
        "model_id": "FiBiNET_001",

        # Model architecture
        "embedding_dim": 16,
        "hidden_units": [256, 128, 64],
        "hidden_activations": "relu",

        # FiBiNET specific
        "bilinear_type": "field_interaction",
        "reduction_ratio": 3,

        # Regularization
        "net_dropout": 0.2,
        "batch_norm": True,

        # Training
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 2048,
        "epochs": 10,

        # Loss
        "loss": "binary_crossentropy",
        "task": "binary_classification",

        # Metrics
        "metrics": ["AUC", "logloss"],

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    return model_config

class SimpleFiBiNET(torch.nn.Module):
    """Simplified FiBiNET implementation"""
    def __init__(self, field_dims, embed_dim=16, reduction_ratio=3,
                 mlp_dims=[256, 128, 64], dropout=0.2):
        super().__init__()

        # Embeddings
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(field_dim, embed_dim) for field_dim in field_dims
        ])

        num_fields = len(field_dims)

        # SENET layer for feature importance
        self.senet_excitation = torch.nn.Sequential(
            torch.nn.Linear(num_fields, num_fields // reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(num_fields // reduction_ratio, num_fields),
            torch.nn.Sigmoid()
        )

        # Bilinear interaction layer
        self.bilinear = torch.nn.Bilinear(embed_dim, embed_dim, embed_dim)

        # DNN layers
        input_dim = num_fields * embed_dim + num_fields * (num_fields - 1) // 2 * embed_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in mlp_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, categorical_x, numerical_x):
        # Get embeddings
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(categorical_x[:, i]))

        # Stack embeddings
        emb_matrix = torch.stack(embeddings, dim=1)  # [batch, num_fields, embed_dim]

        # SENET: Calculate feature importance
        Z = torch.mean(emb_matrix, dim=-1)  # [batch, num_fields]
        A = self.senet_excitation(Z)  # [batch, num_fields]
        A = A.unsqueeze(-1)  # [batch, num_fields, 1]

        # Apply importance weights
        V = emb_matrix * A  # [batch, num_fields, embed_dim]

        # Bilinear interactions
        interactions = []
        num_fields = V.shape[1]
        for i in range(num_fields):
            for j in range(i+1, num_fields):
                interaction = self.bilinear(V[:, i, :], V[:, j, :])
                interactions.append(interaction)

        # Concatenate all features
        flat_embeddings = V.reshape(V.size(0), -1)
        if interactions:
            interaction_features = torch.stack(interactions, dim=1).reshape(V.size(0), -1)
            combined = torch.cat([flat_embeddings, interaction_features], dim=1)
        else:
            combined = flat_embeddings

        # MLP
        output = self.mlp(combined)
        return output.squeeze()

def train_fibinet():
    """Train FiBiNET model"""
    print("="*60)
    print("FiBiNET MODEL TRAINING")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    print("\nPreparing data...")
    train_data, val_data, feature_map = prepare_data_for_fuxictr()

    # Extract features for our custom implementation
    cat_cols = [f for f in feature_map["features"] if f.get("type") == "categorical" and f["name"] != "label"]
    num_cols = [f for f in feature_map["features"] if f.get("type") == "numeric"]

    # Prepare tensors
    X_train_cat = torch.LongTensor(train_data[[c["name"] for c in cat_cols]].values)
    X_train_num = torch.FloatTensor(train_data[[c["name"] for c in num_cols]].values)
    y_train = torch.FloatTensor(train_data['label'].values)

    X_val_cat = torch.LongTensor(val_data[[c["name"] for c in cat_cols]].values)
    X_val_num = torch.FloatTensor(val_data[[c["name"] for c in num_cols]].values)
    y_val = torch.FloatTensor(val_data['label'].values)

    # Get field dimensions
    field_dims = [c["vocab_size"] for c in cat_cols]

    # Create model
    model = SimpleFiBiNET(
        field_dims=field_dims,
        embed_dim=16,
        reduction_ratio=3,
        mlp_dims=[128, 64, 32],
        dropout=0.2
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    batch_size = 1024
    n_epochs = 20
    best_auc = 0

    print("\nStarting training...")
    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        # Mini-batch training
        indices = torch.randperm(len(X_train_cat))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]

            batch_cat = X_train_cat[batch_idx].to(device)
            batch_num = X_train_num[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            outputs = model(batch_cat, batch_num)

            # Check for NaN
            if torch.isnan(outputs).any():
                print(f"NaN detected at epoch {epoch}")
                continue

            loss = criterion(outputs, batch_y)

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_cat.to(device), X_val_num.to(device))
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()

        # Metrics
        from sklearn.metrics import roc_auc_score, average_precision_score

        val_auc = roc_auc_score(y_val.numpy(), val_probs)
        val_ap = average_precision_score(y_val.numpy(), val_probs)

        if epoch % 5 == 0:
            avg_loss = np.mean(train_losses) if train_losses else 0
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, AUC={val_auc:.4f}, AP={val_ap:.4f}")
            print(f"  Pred stats: mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")

        scheduler.step()

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'plan2/experiments/fibinet_best.pth')

    print("\n" + "="*60)
    print(f"Best validation AUC: {best_auc:.4f}")

    # Compare with XGBoost
    print("\n--- Comparison with Plan1 XGBoost ---")
    print(f"FiBiNET AUC: {best_auc:.4f}")
    print(f"XGBoost AUC: 0.7430 (from Plan1)")
    print(f"Difference: {0.7430 - best_auc:.4f}")

    if best_auc > 0.70:
        print("\n✅ FiBiNET achieved competitive performance!")
    else:
        print("\n❌ FiBiNET needs further tuning to match XGBoost")

    return best_auc

if __name__ == "__main__":
    auc = train_fibinet()

    print("\n" + "="*60)
    print("FiBiNET TRAINING COMPLETE")
    print("="*60)
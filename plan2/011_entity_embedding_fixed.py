#!/usr/bin/env python3
"""
Entity Embeddings with proper initialization and stable training
Key improvements:
1. Proper embedding initialization (small values)
2. Batch-wise class balancing
3. Learning rate warmup
4. Gradient accumulation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter
import math

# Force deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
torch.manual_seed(42)

class EntityEmbeddingDataset(Dataset):
    """Dataset for entity embeddings"""
    def __init__(self, cat_data, num_data, labels=None):
        self.cat_data = torch.LongTensor(cat_data)
        self.num_data = torch.FloatTensor(num_data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.cat_data[idx], self.num_data[idx], self.labels[idx]
        return self.cat_data[idx], self.num_data[idx]

class EntityEmbeddingModel(nn.Module):
    """Entity Embedding Neural Network with stability features"""

    def __init__(self, cat_dims, num_continuous, emb_dims=None, hidden_layers=[64, 32],
                 dropout=0.3, use_batchnorm=True):
        super().__init__()

        # Default embedding dimensions (rule of thumb: min(50, (cat_dim+1)//2))
        if emb_dims is None:
            emb_dims = [min(50, (cat_dim + 1) // 2) for cat_dim in cat_dims]

        # Create embeddings with proper initialization
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        for cat_dim, emb_dim in zip(cat_dims, emb_dims):
            emb = nn.Embedding(cat_dim, emb_dim)
            # Initialize with small normal distribution
            nn.init.normal_(emb.weight, mean=0, std=0.05)
            self.embeddings.append(emb)
            total_emb_dim += emb_dim

        # Input dimension
        self.input_dim = total_emb_dim + num_continuous

        # Build network layers
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(prev_dim, 1)
        # Initialize output layer with small weights
        nn.init.normal_(self.output.weight, mean=0, std=0.02)
        nn.init.constant_(self.output.bias, -1.5)  # Bias towards negative class

    def forward(self, cat_data, num_data):
        # Embed categorical variables
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_data[:, i]))

        # Concatenate embeddings with numerical features
        x = torch.cat(embeddings + [num_data], dim=1)

        # Pass through network
        x = self.layers(x)
        out = self.output(x)

        # Clamp output to prevent overflow in sigmoid
        out = torch.clamp(out, min=-15, max=15)

        return out.squeeze()

def prepare_entity_data(n_samples=100000):
    """Prepare data for entity embeddings"""
    print("Preparing data for entity embeddings...")

    # Load data
    df = pd.read_parquet('data/train.parquet', engine='pyarrow').head(n_samples)
    y = df['clicked'].values.astype(np.float32)
    X = df.drop(columns=['clicked'])

    # Separate categorical and numerical columns
    cat_cols = []
    num_cols = []

    for col in X.columns:
        if col.startswith(('gender', 'age_group', 'inventory_id', 'seq', 'l_feat_', 'feat_')):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    print(f"Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

    # Encode categorical variables (0 will be reserved for unknown)
    cat_data = []
    cat_dims = []

    for col in cat_cols:
        # Create mapping
        unique_vals = X[col].unique()
        val_to_idx = {val: idx + 1 for idx, val in enumerate(unique_vals)}  # Start from 1

        # Encode
        encoded = X[col].map(val_to_idx).fillna(0).astype(int).values
        cat_data.append(encoded)
        cat_dims.append(len(unique_vals) + 1)  # +1 for unknown

    cat_data = np.column_stack(cat_data)

    # Normalize numerical features
    num_data = X[num_cols].values.astype(np.float32)

    # Robust normalization
    for i in range(num_data.shape[1]):
        col = num_data[:, i]
        # Remove outliers
        q1, q99 = np.percentile(col, [1, 99])
        col = np.clip(col, q1, q99)
        # Standardize
        mean, std = col.mean(), col.std() + 1e-6
        num_data[:, i] = (col - mean) / std

    # Clip to reasonable range
    num_data = np.clip(num_data, -5, 5)

    print(f"Data shapes - Cat: {cat_data.shape}, Num: {num_data.shape}")
    print(f"Positive rate: {y.mean():.4f}")

    return cat_data, num_data, y, cat_dims

def create_balanced_sampler(labels, batch_size):
    """Create a weighted sampler for balanced batches"""
    class_counts = Counter(labels)
    class_weights = {0: 1.0 / class_counts[0], 1: 1.0 / class_counts[1]}
    sample_weights = [class_weights[int(label)] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

    return sampler

def train_entity_embeddings():
    """Train entity embeddings model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    cat_data, num_data, y, cat_dims = prepare_entity_data(n_samples=100000)

    # Train-validation split
    cat_train, cat_val, num_train, num_val, y_train, y_val = train_test_split(
        cat_data, num_data, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}")

    # Create datasets
    train_dataset = EntityEmbeddingDataset(cat_train, num_train, y_train)
    val_dataset = EntityEmbeddingDataset(cat_val, num_val, y_val)

    # Create balanced sampler for training
    train_sampler = create_balanced_sampler(y_train, batch_size=512)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = EntityEmbeddingModel(
        cat_dims=cat_dims,
        num_continuous=num_data.shape[1],
        emb_dims=None,  # Use default
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        use_batchnorm=True
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function - use regular BCE since we're balancing batches
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer with different learning rates for embeddings and other layers
    embedding_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'embeddings' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': embedding_params, 'lr': 0.01},  # Higher LR for embeddings
        {'params': other_params, 'lr': 0.001}      # Lower LR for other layers
    ], weight_decay=1e-5)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 ** ((epoch - warmup_epochs) // 10)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print("\nStarting training...")
    best_auc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(50):
        # Training
        model.train()
        train_losses = []
        gradient_accumulation_steps = 2
        accumulated_loss = 0

        for batch_idx, (cat_batch, num_batch, y_batch) in enumerate(train_loader):
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(cat_batch, num_batch)

            # Check for NaN
            if torch.isnan(outputs).any():
                print(f"NaN detected in outputs at epoch {epoch}, batch {batch_idx}")
                # Skip this batch
                continue

            loss = criterion(outputs, y_batch)

            # Gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Check gradient norms
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                if total_norm < 100:  # Only update if gradients are reasonable
                    optimizer.step()
                    train_losses.append(accumulated_loss)
                else:
                    print(f"Skipping update due to large gradient norm: {total_norm:.2f}")

                optimizer.zero_grad()
                accumulated_loss = 0

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for cat_batch, num_batch, y_batch in val_loader:
                cat_batch = cat_batch.to(device)
                num_batch = num_batch.to(device)

                outputs = model(cat_batch, num_batch)
                preds = torch.sigmoid(outputs).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(y_batch.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # Handle NaN predictions
        if np.isnan(val_preds).any():
            print(f"NaN in validation predictions, replacing with 0.5")
            val_preds = np.nan_to_num(val_preds, nan=0.5)

        # Compute metrics
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
            val_ap = average_precision_score(val_targets, val_preds)
        except:
            val_auc = val_ap = 0.5

        # Update learning rate
        scheduler.step()

        # Logging
        if epoch % 5 == 0 or val_auc > best_auc:
            avg_loss = np.mean(train_losses) if train_losses else 0
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, AUC={val_auc:.4f}, AP={val_ap:.4f}")
            print(f"  Predictions: mean={val_preds.mean():.4f}, std={val_preds.std():.4f}")
            print(f"  LR: {scheduler.get_last_lr()}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save model
            torch.save(model.state_dict(), 'plan2/experiments/entity_embedding_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n" + "="*60)
    print(f"Best validation AUC: {best_auc:.4f}")

    if best_auc > 0.65:
        print("SUCCESS! Entity embeddings trained successfully!")
        return True
    else:
        print("Model trained but performance is low. Need more tuning.")
        return False

if __name__ == "__main__":
    print("ENTITY EMBEDDINGS WITH STABLE TRAINING")
    print("="*60)

    success = train_entity_embeddings()

    if not success:
        print("\nTrying alternative configuration...")
        # Could try different hyperparameters here
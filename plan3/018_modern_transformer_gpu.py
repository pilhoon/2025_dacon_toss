#!/usr/bin/env python
"""
Modern Transformer Architecture for Tabular Data - Maximizing GPU Memory Usage
Using multiple Transformer blocks with cross-attention and self-attention
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
from multiprocessing import cpu_count
import math

class TabularEmbedding(nn.Module):
    """Advanced embedding layer for tabular data"""

    def __init__(self, num_features, d_model=512):
        super().__init__()
        self.feature_embedding = nn.Linear(num_features, d_model)
        self.positional_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Feature embedding
        x = self.feature_embedding(x)
        # Add positional information
        x = x.unsqueeze(1)  # [batch, 1, d_model]
        x = x + self.positional_embedding
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention and feed-forward"""

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for feature interaction"""

    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        attn_out, _ = self.cross_attn(x, context, context)
        x = self.norm(x + self.dropout(attn_out))
        return x

class ModernTransformerModel(nn.Module):
    """Modern Transformer architecture optimized for tabular data and GPU memory usage"""

    def __init__(self, num_features, n_layers=12, d_model=768, n_heads=12, d_ff=3072):
        super().__init__()

        # Embedding layer
        self.embedding = TabularEmbedding(num_features, d_model)

        # Create feature tokens for different aspects
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.feature_tokens = nn.Parameter(torch.randn(1, 8, d_model))  # 8 learnable feature tokens

        # Multiple Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=0.1)
            for _ in range(n_layers)
        ])

        # Cross-attention layers for feature interaction (every 3 layers)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout=0.1)
            for _ in range(n_layers // 3)
        ])

        # Additional parallel branch with different configuration
        self.parallel_transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads//2, d_ff//2, dropout=0.15)
            for _ in range(n_layers // 2)
        ])

        # Pooling and output layers
        self.pool_attn = nn.Linear(d_model, 1)

        # Multiple prediction heads for ensemble
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(3)
        ])

        # Final ensemble layer
        self.ensemble = nn.Linear(3, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Embed input features
        x_embed = self.embedding(x)  # [batch, 1, d_model]

        # Expand cls and feature tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        feature_tokens = self.feature_tokens.expand(batch_size, -1, -1)

        # Concatenate all tokens
        x_main = torch.cat([cls_tokens, x_embed, feature_tokens], dim=1)  # [batch, 10, d_model]

        # Main transformer branch
        cross_attn_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            x_main = block(x_main)

            # Apply cross-attention every 3 layers
            if (i + 1) % 3 == 0 and cross_attn_idx < len(self.cross_attn_blocks):
                x_main = self.cross_attn_blocks[cross_attn_idx](x_main, feature_tokens)
                cross_attn_idx += 1

        # Parallel transformer branch (on original embedding)
        x_parallel = x_embed
        for block in self.parallel_transformer:
            x_parallel = block(x_parallel)

        # Attention pooling for main branch
        attn_weights = F.softmax(self.pool_attn(x_main), dim=1)
        x_main_pooled = torch.sum(x_main * attn_weights, dim=1)

        # Simple pooling for parallel branch
        x_parallel_pooled = x_parallel.squeeze(1)

        # Combine features
        combined = torch.cat([x_main_pooled, x_parallel_pooled], dim=1)

        # Multiple prediction heads
        predictions = []
        for head in self.heads:
            pred = head(combined)
            predictions.append(pred)

        # Ensemble predictions
        stacked_preds = torch.stack(predictions, dim=1).squeeze(-1)
        final_output = self.ensemble(stacked_preds)

        return torch.sigmoid(final_output).squeeze()


def process_features(df):
    """Process features"""
    print("Processing features...")
    processed = df.copy()

    # Process f_1 column
    if 'f_1' in processed.columns:
        processed['f_1_count'] = processed['f_1'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        processed['f_1_first'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0)
        processed['f_1_last'] = processed['f_1'].apply(lambda x: int(str(x).split(',')[-1]) if pd.notna(x) and str(x) else 0)
        processed['f_1_unique'] = processed['f_1'].apply(
            lambda x: len(set(str(x).split(','))) if pd.notna(x) else 0
        )
        processed = processed.drop('f_1', axis=1)

    # Convert categorical columns
    from sklearn.preprocessing import LabelEncoder
    for col in processed.columns:
        if col == 'clicked':
            continue
        if processed[col].dtype == 'object':
            le = LabelEncoder()
            processed[col] = le.fit_transform(processed[col].astype(str))

    processed = processed.fillna(0)

    if 'clicked' in processed.columns:
        feature_cols = [col for col in processed.columns if col != 'clicked']
    else:
        feature_cols = list(processed.columns)

    print(f"  Total features: {len(feature_cols)}")
    return processed, feature_cols


def train_transformer_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=4096):
    """Train modern transformer model"""
    print("\nTraining Modern Transformer Model...")
    print(f"Batch size: {batch_size:,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors and move to GPU
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)

    # Initialize model with large configuration
    num_features = X_train.shape[1]
    model = ModernTransformerModel(
        num_features=num_features,
        n_layers=12,  # 12 transformer layers
        d_model=768,  # Hidden dimension
        n_heads=12,   # Attention heads
        d_ff=3072     # Feed-forward dimension
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(val_labels, val_preds)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Check GPU memory usage
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val AUC: {auc:.4f}, "
                  f"GPU Mem: {gpu_mem:.1f}/{gpu_reserved:.1f} GB")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val AUC: {auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'plan3/018_transformer_model.pth')

    # Load best model
    model.load_state_dict(torch.load('plan3/018_transformer_model.pth'))
    print(f"\nBest validation loss: {best_val_loss:.4f}")

    return model


def create_submission(model, scaler, feature_cols, device):
    """Create submission"""
    print("\n" + "="*80)
    print("Creating submission...")
    print("="*80)

    # Load test data
    print("Loading test data...")
    test_data = pd.read_parquet('data/test.parquet')
    print(f"Test data shape: {test_data.shape}")

    # Process features
    test_processed, test_feature_cols = process_features(test_data)

    # Align features with training
    for col in feature_cols:
        if col not in test_processed.columns:
            test_processed[col] = 0

    # Select and order columns to match training
    X_test = test_processed[feature_cols].values.astype(np.float32)
    X_test = scaler.transform(X_test)

    # Make predictions
    print("Making predictions...")
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)

    # Predict in batches
    batch_size = 8192
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_test_t), batch_size):
            batch = X_test_t[i:i+batch_size]
            batch_pred = model(batch)
            predictions.extend(batch_pred.cpu().numpy())

    final_pred = np.array(predictions)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data.index,
        'clicked': final_pred
    })

    # Save submission
    output_path = 'plan3/018_transformer_submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"✓ Submission saved to: {output_path}")

    # Print statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean: {final_pred.mean():.6f}")
    print(f"  Std: {final_pred.std():.6f}")
    print(f"  Min: {final_pred.min():.6f}")
    print(f"  Max: {final_pred.max():.6f}")

    return submission


def main():
    """Main execution"""
    print("="*80)
    print("MODERN TRANSFORMER MODEL FOR CTR PREDICTION")
    print("Using advanced Transformer architecture to maximize GPU usage")
    print("="*80)

    # Check resources
    mem_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Available memory: {mem_gb:.1f} GB")
    print(f"Available CPUs: {cpu_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load FULL training data
    print("\nLoading FULL training data...")
    train_data = pd.read_parquet('data/train.parquet')
    print(f"Full dataset size: {len(train_data):,} samples")
    print(f"Positive rate: {train_data['clicked'].mean():.4f}")

    # Process features
    train_processed, feature_cols = process_features(train_data)

    # Split data
    X = train_processed[feature_cols].values.astype(np.float32)
    y = train_processed['clicked'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train):,}")
    print(f"Validation size: {len(X_val):,}")

    # Train model with reduced batch size to avoid OOM
    model = train_transformer_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=2000)

    # Check final GPU usage
    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Final GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Create submission
    submission = create_submission(model, scaler, feature_cols, device)

    print("\n" + "="*80)
    print("MODERN TRANSFORMER MODEL COMPLETE")
    print("="*80)

    return submission


if __name__ == "__main__":
    submission = main()
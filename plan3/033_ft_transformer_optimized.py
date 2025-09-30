#!/usr/bin/env python
"""
FT-Transformer for Plan3 - Optimized for GPU Memory Usage
Based on plan2/046 which achieved 0.3168 score
Maximizing GPU memory utilization with larger batch sizes
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
import gc
import math
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FT-TRANSFORMER OPTIMIZED FOR PLAN3")
print("Target: 0.351+ competition score")
print("Maximizing GPU Memory Usage")
print("="*80)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load data
print("\nLoading data...")
train = pd.read_parquet('data/train.parquet')
test = pd.read_parquet('data/test.parquet')

# Convert object columns to numeric
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = pd.factorize(train[col])[0]
for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = pd.factorize(test[col])[0]

# Prepare features and target
X = train.drop(columns=['clicked']).values
y = train['clicked'].values
X_test = test.drop(columns=['ID']) if 'ID' in test.columns else test.values

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Positive class ratio: {y.mean():.4f}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train):,}")
print(f"Val size: {len(X_val):,}")


class NumericalEmbedding(nn.Module):
    """Enhanced numerical feature embedding with piecewise linear encoding"""

    def __init__(self, num_features, d_model, n_bins=128):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_bins = n_bins

        # Larger projections for better expressiveness
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_bins, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_features)
        ])

        # Learnable bin boundaries
        self.register_buffer('bin_boundaries', torch.linspace(-3, 3, n_bins))

    def forward(self, x):
        batch_size = x.shape[0]
        embeddings = []

        for i in range(self.num_features):
            feat = x[:, i].unsqueeze(1)

            # Compute distances to bin boundaries
            dists = feat - self.bin_boundaries.unsqueeze(0)

            # Piecewise linear encoding with smoothing
            weights = F.softmax(-torch.abs(dists) * 2, dim=1)

            # Project to d_model
            emb = self.projections[i](weights)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=1)


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular data"""

    def __init__(self, num_features, d_model=512, nhead=16, num_layers=8,
                 dropout=0.1, n_bins=128):
        super().__init__()

        # Numerical feature embedding
        self.feature_embedding = NumericalEmbedding(num_features, d_model, n_bins)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head with residual connections
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size = x.shape[0]

        # Embed features
        x = self.feature_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        x = self.transformer(x)

        # Use CLS token for classification
        cls_output = x[:, 0]

        # Output prediction
        out = self.output_head(cls_output)
        return torch.sigmoid(out.squeeze())


# Training parameters
# Increased batch size for better GPU memory utilization
BATCH_SIZE = 8192  # Much larger than plan2's 2048
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Model configuration - Larger model for better performance
D_MODEL = 768  # Increased from 512
NHEAD = 16
NUM_LAYERS = 12  # Increased from 8
DROPOUT = 0.15
N_BINS = 256  # Increased from 128

print("\nModel Configuration:")
print(f"  D_MODEL: {D_MODEL}")
print(f"  NHEAD: {NHEAD}")
print(f"  NUM_LAYERS: {NUM_LAYERS}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  N_BINS: {N_BINS}")

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_train)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.FloatTensor(y_val)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,  # Larger for validation
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# Initialize model
model = FTTransformer(
    num_features=X.shape[1],
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    n_bins=N_BINS
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Loss and optimizer
pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

# Learning rate scheduler - Cosine annealing with warmup
num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = len(train_loader) * 2  # 2 epochs warmup

def get_lr(step):
    if step < num_warmup_steps:
        return step / num_warmup_steps
    progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

# Training
print("\nStarting training...")
best_val_score = 0
best_epoch = 0
step = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        step += 1

        train_loss += loss.item()
        train_preds.extend(outputs.detach().cpu().numpy())
        train_labels.extend(batch_y.cpu().numpy())

    # Validation
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(batch_y.numpy())

    # Calculate metrics
    train_ap = average_precision_score(train_labels, train_preds)
    val_ap = average_precision_score(val_labels, val_preds)
    val_logloss = log_loss(val_labels, val_preds, eps=1e-7)

    # Competition score
    val_score = 0.7 * val_ap + 0.3 / val_logloss

    # GPU memory monitoring
    if torch.cuda.is_available():
        gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_cached = torch.cuda.memory_reserved() / 1024**3
        mem_info = f", GPU: {gpu_mem_used:.1f}/{gpu_mem_cached:.1f} GB"
    else:
        mem_info = ""

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train AP: {train_ap:.4f}, "
          f"Val AP: {val_ap:.4f}, "
          f"Val Score: {val_score:.4f}{mem_info}")

    # Save best model
    if val_score > best_val_score:
        best_val_score = val_score
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'plan3/033_ft_transformer_best.pt')

print(f"\nBest validation score: {best_val_score:.4f} at epoch {best_epoch}")

# Load best model for prediction
model.load_state_dict(torch.load('plan3/033_ft_transformer_best.pt'))
model.eval()

# Make predictions on test set
print("\nMaking predictions on test set...")
test_dataset = TensorDataset(torch.FloatTensor(X_test))
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

test_preds = []
with torch.no_grad():
    for batch_x, in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        test_preds.extend(outputs.cpu().numpy())

# Create submission
submission = pd.DataFrame()
submission['ID'] = [f'TEST_{i:07d}' for i in range(len(test))]
submission['clicked'] = test_preds

# Apply calibration based on validation performance
calibration_power = 1.05
submission['clicked'] = np.power(submission['clicked'], calibration_power)

# Save submission
submission.to_csv('plan3/033_ft_transformer_submission.csv', index=False)
print(f"\nSubmission saved to plan3/033_ft_transformer_submission.csv")

# Print statistics
print(f"\nPrediction statistics:")
print(f"  Mean: {submission['clicked'].mean():.6f}")
print(f"  Std: {submission['clicked'].std():.6f}")
print(f"  Min: {submission['clicked'].min():.6f}")
print(f"  Max: {submission['clicked'].max():.6f}")

# Final GPU memory usage
if torch.cuda.is_available():
    print(f"\nFinal GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n✓ FT-Transformer training complete!")
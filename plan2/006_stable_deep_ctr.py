"""
Stable Deep CTR Model with improved training
Key improvements:
1. Better weight initialization
2. Focal loss for class imbalance
3. Layer normalization
4. Residual connections
5. Gradient clipping
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
from pathlib import Path
import json

# Custom modules
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)

class StableDeepCTR(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=16, hidden_dims=[256, 128, 64]):
        super().__init__()

        # Embeddings with proper initialization
        self.embeddings = nn.ModuleList()
        for dim in cat_dims:
            emb = nn.Embedding(dim, emb_dim)
            nn.init.normal_(emb.weight, mean=0, std=0.01)  # Small initialization
            self.embeddings.append(emb)

        # Input dimension
        input_dim = len(cat_dims) * emb_dim + num_dim

        # Deep network with residual connections
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(0.2)

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.deep = nn.Sequential(*layers)

        # Add residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1]) for _ in range(2)
        ])

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cat_features, num_features):
        # Embed categorical features
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_features[:, i]))

        # Concatenate all features
        x = torch.cat(embeddings + [num_features], dim=-1)

        # Input normalization
        x = self.input_bn(x)
        x = self.input_dropout(x)

        # Deep network
        x = self.deep(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output
        return self.output(x).squeeze(-1)

def prepare_data(n_samples=500000):
    """Load and prepare data with proper preprocessing"""
    print("Loading data...")
    t0 = time.time()

    # Use cached data if available
    cache_dir = Path('plan2/cache')
    if (cache_dir / 'train_X.parquet').exists():
        train_X = pd.read_parquet(cache_dir / 'train_X.parquet').head(n_samples)
        train_y = np.load(cache_dir / 'train_y.npy')[:n_samples]
        print(f"Loaded cached data in {time.time()-t0:.1f}s")
    else:
        df = pd.read_parquet('data/train.parquet', engine='pyarrow').head(n_samples)
        train_y = df['clicked'].values.astype(np.float32)
        train_X = df.drop(columns=['clicked'])
        print(f"Loaded fresh data in {time.time()-t0:.1f}s")

    print(f"Data shape: {train_X.shape}, Positive rate: {train_y.mean():.4f}")

    # Identify columns
    cat_cols = []
    num_cols = []
    for col in train_X.columns:
        if col in ['gender', 'age_group', 'inventory_id', 'seq'] or \
           col.startswith('l_feat_') or col.startswith('feat_'):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    print(f"Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

    # Encode categorical
    cat_encoded = []
    cat_dims = []
    for col in cat_cols:
        le = LabelEncoder()
        # Add 1 to leave 0 for padding
        encoded = le.fit_transform(train_X[col].astype(str)) + 1
        cat_encoded.append(encoded)
        cat_dims.append(len(le.classes_) + 2)  # +2 for padding and unknown

    cat_encoded = np.column_stack(cat_encoded).astype(np.int64)

    # Normalize numerical
    scaler = StandardScaler()
    num_encoded = scaler.fit_transform(train_X[num_cols].values.astype(np.float32))

    return cat_encoded, num_encoded, train_y, cat_dims, len(num_cols)

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """Train with stable settings"""
    model = model.to(device)

    # Use focal loss for class imbalance
    criterion = FocalLoss(alpha=2.0, gamma=2.0)

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_aucs = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0

        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_cat, batch_num)
            loss = criterion(outputs, batch_y)

            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, batch {n_batches}")
                print(f"Outputs stats: min={outputs.min():.4f}, max={outputs.max():.4f}")
                return None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)

                outputs = model(batch_cat, batch_num)
                preds = torch.sigmoid(outputs).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Metrics
        val_auc = roc_auc_score(all_targets, all_preds)
        val_ap = average_precision_score(all_targets, all_preds)
        val_logloss = log_loss(all_targets, all_preds)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}, LogLoss: {val_logloss:.4f}")
        print(f"  Pred stats: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

    return train_losses, val_aucs

def main():
    print("="*60)
    print("STABLE DEEP CTR MODEL")
    print("="*60)

    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    cat_encoded, num_encoded, targets, cat_dims, num_dim = prepare_data(n_samples=500000)

    # Train-val split
    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        cat_encoded, num_encoded, targets,
        test_size=0.2, random_state=42, stratify=targets
    )

    print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}")

    # Create data loaders
    batch_size = 4096
    train_dataset = TensorDataset(
        torch.from_numpy(X_cat_train),
        torch.from_numpy(X_num_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_cat_val),
        torch.from_numpy(X_num_val),
        torch.from_numpy(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = StableDeepCTR(
        cat_dims=cat_dims,
        num_dim=num_dim,
        emb_dim=16,
        hidden_dims=[256, 128, 64]
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\nTraining...")
    train_losses, val_aucs = train_model(
        model, train_loader, val_loader,
        device, epochs=10, lr=0.001
    )

    if train_losses is None:
        print("Training failed due to NaN loss")
        return

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch_cat, batch_num, batch_y in val_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            outputs = model(batch_cat, batch_num)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Final metrics
    final_auc = roc_auc_score(all_targets, all_preds)
    final_ap = average_precision_score(all_targets, all_preds)
    final_logloss = log_loss(all_targets, all_preds)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"AUC: {final_auc:.6f}")
    print(f"AP: {final_ap:.6f}")
    print(f"LogLoss: {final_logloss:.6f}")
    print(f"Pred mean: {all_preds.mean():.6f} (target: ~0.019)")
    print(f"Pred std: {all_preds.std():.6f} (target: >0.05)")

    # Estimate competition score
    wll = final_logloss * 10  # Rough estimate
    score = 0.5 * final_ap + 0.5 * (1 / (1 + wll))
    print(f"\nEstimated competition score: {score:.6f} (target: >0.349)")

    # Save results
    output_dir = Path('plan2/experiments/007_stable_deep_ctr')
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {
        'final_auc': float(final_auc),
        'final_ap': float(final_ap),
        'final_logloss': float(final_logloss),
        'estimated_score': float(score),
        'pred_mean': float(all_preds.mean()),
        'pred_std': float(all_preds.std())
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pth')
    print(f"\nModel and results saved to {output_dir}")

if __name__ == "__main__":
    main()
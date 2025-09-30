"""
TabNet-inspired model for stable training
TabNet is specifically designed for tabular data with built-in feature selection
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

class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim

    def forward(self, x):
        output = self.fc(x)
        return output[:, :self.output_dim] * torch.sigmoid(output[:, self.output_dim:])

class AttentiveTransformer(nn.Module):
    """Feature attention mechanism from TabNet"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.sparse = nn.Linear(output_dim, input_dim)

    def forward(self, x, prior=None):
        h = self.fc(x)
        h = self.bn(h)
        mask = torch.sigmoid(self.sparse(h))
        if prior is not None:
            mask = mask * prior
        return mask

class TabNetBlock(nn.Module):
    """Simplified TabNet decision block"""
    def __init__(self, input_dim, output_dim, virtual_batch_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Feature transformer
        self.feat_transformer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            GLU(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            GLU(output_dim, output_dim)
        )

        # Attention transformer
        self.att_transformer = AttentiveTransformer(input_dim, output_dim)

    def forward(self, x, mask_prior=None):
        # Attention
        mask = self.att_transformer(x, mask_prior)
        masked_x = x * mask

        # Feature transformation
        out = self.feat_transformer(masked_x)
        return out, mask

class SimpleTabNet(nn.Module):
    """Simplified TabNet for binary classification"""
    def __init__(self, cat_dims, num_dim, emb_dim=8,
                 n_steps=3, step_dim=64, output_dim=64):
        super().__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList()
        for dim in cat_dims:
            emb = nn.Embedding(dim, emb_dim)
            # Xavier initialization scaled down
            bound = 1 / np.sqrt(dim)
            nn.init.uniform_(emb.weight, -bound, bound)
            self.embeddings.append(emb)

        # Input dimension
        input_dim = len(cat_dims) * emb_dim + num_dim

        # Initial batch norm
        self.initial_bn = nn.BatchNorm1d(input_dim)

        # TabNet steps
        self.steps = nn.ModuleList()
        for _ in range(n_steps):
            self.steps.append(TabNetBlock(input_dim, step_dim))

        # Final layers
        self.final = nn.Sequential(
            nn.Linear(step_dim * n_steps, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, 1)
        )

        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.final[-1].weight, gain=0.01)
        nn.init.zeros_(self.final[-1].bias)

    def forward(self, cat_features, num_features):
        # Embed categorical features
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(cat_features[:, i]))

        # Concatenate all features
        x = torch.cat(embeddings + [num_features], dim=-1)
        x = self.initial_bn(x)

        # TabNet steps with attention
        outputs = []
        mask = None
        for step in self.steps:
            out, mask = step(x, mask)
            outputs.append(out)

        # Aggregate outputs
        final_out = torch.cat(outputs, dim=-1)

        # Final prediction
        return self.final(final_out).squeeze(-1)

def weighted_binary_cross_entropy(output, target, pos_weight=10.0):
    """Custom weighted BCE that's more stable"""
    # Clamp output to prevent extreme values
    output = torch.clamp(output, min=-10, max=10)

    loss = F.binary_cross_entropy_with_logits(
        output, target,
        reduction='none',
        pos_weight=torch.tensor([pos_weight], device=output.device)
    )

    # Additional clamping to prevent explosion
    loss = torch.clamp(loss, max=10.0)
    return loss.mean()

def prepare_data(n_samples=500000):
    """Load and prepare data with robust preprocessing"""
    print("Loading data...")
    t0 = time.time()

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

    # Encode categorical with frequency threshold
    cat_encoded = []
    cat_dims = []
    for col in cat_cols:
        # Frequency encoding to reduce vocabulary
        value_counts = train_X[col].value_counts()
        # Keep only values with frequency > 5
        frequent_values = value_counts[value_counts > 5].index

        le = LabelEncoder()
        # Map infrequent values to 'RARE'
        col_values = train_X[col].apply(lambda x: x if x in frequent_values else 'RARE')
        encoded = le.fit_transform(col_values.astype(str)) + 1
        cat_encoded.append(encoded)
        cat_dims.append(len(le.classes_) + 2)

    cat_encoded = np.column_stack(cat_encoded).astype(np.int64)

    # Robust numerical scaling
    num_data = train_X[num_cols].values.astype(np.float32)

    # Clip outliers at 99th percentile
    for i in range(num_data.shape[1]):
        p99 = np.percentile(num_data[:, i], 99)
        p1 = np.percentile(num_data[:, i], 1)
        num_data[:, i] = np.clip(num_data[:, i], p1, p99)

    # Standard scaling
    scaler = StandardScaler()
    num_encoded = scaler.fit_transform(num_data)

    # Final clipping to [-5, 5] range
    num_encoded = np.clip(num_encoded, -5, 5).astype(np.float32)

    return cat_encoded, num_encoded, train_y, cat_dims, len(num_cols)

def train_model(model, train_loader, val_loader, device, epochs=15):
    """Train with careful monitoring"""
    model = model.to(device)

    # Lower learning rate for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

    # Reduce on plateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_auc = 0
    patience = 5
    patience_counter = 0

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

            # Use custom loss
            loss = weighted_binary_cross_entropy(outputs, batch_y, pos_weight=5.0)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at epoch {epoch}, batch {n_batches}")
                print(f"Loss: {loss.item()}, Outputs range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                continue

            loss.backward()

            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            print("No valid batches in this epoch")
            break

        avg_train_loss = train_loss / n_batches

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)

                outputs = model(batch_cat, batch_num)
                # Clamp outputs before sigmoid to prevent overflow
                outputs = torch.clamp(outputs, min=-10, max=10)
                preds = torch.sigmoid(outputs).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Metrics
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
            val_ap = average_precision_score(all_targets, all_preds)
            val_logloss = log_loss(all_targets, np.clip(all_preds, 1e-7, 1-1e-7))
        except:
            print("Error computing metrics")
            val_auc = val_ap = val_logloss = 0

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}, LogLoss: {val_logloss:.4f}")
        print(f"  Pred stats: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}")

        # Scheduler step
        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'plan2/experiments/best_tabnet.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model

def main():
    print("="*60)
    print("TABNET-INSPIRED MODEL")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    cat_encoded, num_encoded, targets, cat_dims, num_dim = prepare_data(n_samples=300000)

    # Train-val split
    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        cat_encoded, num_encoded, targets,
        test_size=0.2, random_state=42, stratify=targets
    )

    print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}")
    print(f"Train positive rate: {y_train.mean():.4f}")

    # Create data loaders with smaller batch size
    batch_size = 2048
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
    model = SimpleTabNet(
        cat_dims=cat_dims,
        num_dim=num_dim,
        emb_dim=8,  # Small embeddings
        n_steps=3,
        step_dim=32,  # Smaller hidden dimensions
        output_dim=32
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\nTraining...")
    model = train_model(model, train_loader, val_loader, device, epochs=15)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch_cat, batch_num, batch_y in val_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            outputs = model(batch_cat, batch_num)
            outputs = torch.clamp(outputs, min=-10, max=10)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Final metrics
    final_auc = roc_auc_score(all_targets, all_preds)
    final_ap = average_precision_score(all_targets, all_preds)
    final_logloss = log_loss(all_targets, np.clip(all_preds, 1e-7, 1-1e-7))

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"AUC: {final_auc:.6f}")
    print(f"AP: {final_ap:.6f}")
    print(f"LogLoss: {final_logloss:.6f}")
    print(f"Pred mean: {all_preds.mean():.6f} (target: ~0.019)")
    print(f"Pred std: {all_preds.std():.6f} (target: >0.05)")

    # Competition score estimate
    wll_estimate = final_logloss * (5.0)  # Account for pos_weight
    score = 0.5 * final_ap + 0.5 * (1 / (1 + wll_estimate))
    print(f"\nEstimated competition score: {score:.6f} (target: >0.349)")

    # Save results
    output_dir = Path('plan2/experiments/008_tabnet')
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

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
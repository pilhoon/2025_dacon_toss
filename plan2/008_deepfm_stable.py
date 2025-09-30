"""
DeepFM: Factorization Machine + Deep Network
Most stable CTR model with proven performance
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import time
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class CTRDataset(Dataset):
    """Custom dataset that handles preprocessing on-the-fly"""
    def __init__(self, data_path, indices=None, is_train=True, n_samples=None):
        df = pd.read_parquet(data_path, engine='pyarrow')
        if n_samples:
            df = df.head(n_samples)
        if indices is not None:
            df = df.iloc[indices]

        self.is_train = is_train
        if is_train:
            self.y = df['clicked'].values.astype(np.float32)
            self.X = df.drop(columns=['clicked'])
        else:
            self.X = df
            self.y = None

        # Simple feature processing
        self.sparse_features = []
        self.dense_features = []

        for col in self.X.columns:
            if col.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat')):
                self.sparse_features.append(col)
            else:
                self.dense_features.append(col)

        # Build vocabularies for sparse features
        self.vocab = {}
        for col in self.sparse_features:
            unique_vals = self.X[col].unique()
            self.vocab[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}  # 0 for unknown

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get sparse features (categorical)
        sparse_vals = []
        for col in self.sparse_features:
            val = self.X[col].iloc[idx]
            sparse_vals.append(self.vocab[col].get(val, 0))
        sparse_tensor = torch.tensor(sparse_vals, dtype=torch.long)

        # Get dense features (numerical)
        dense_vals = self.X[self.dense_features].iloc[idx].values.astype(np.float32)
        # Simple normalization
        dense_vals = (dense_vals - dense_vals.mean()) / (dense_vals.std() + 1e-8)
        dense_vals = np.clip(dense_vals, -3, 3)  # Clip to prevent extreme values
        dense_tensor = torch.tensor(dense_vals, dtype=torch.float32)

        if self.is_train:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return sparse_tensor, dense_tensor, label
        else:
            return sparse_tensor, dense_tensor

class FM(nn.Module):
    """Factorization Machine layer"""
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, inputs):
        # inputs: [batch_size, num_features, embedding_dim]
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        cross = 0.5 * (square_of_sum - sum_of_square)
        if self.reduce_sum:
            cross = torch.sum(cross, dim=1, keepdim=True)
        return cross

class DeepFM(nn.Module):
    """DeepFM model with stable initialization"""
    def __init__(self, sparse_feature_dims, dense_feature_dim,
                 embedding_dim=8, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()

        self.sparse_feature_dims = sparse_feature_dims
        self.dense_feature_dim = dense_feature_dim

        # Embeddings for sparse features - smaller initialization
        self.embeddings = nn.ModuleList()
        for dim in sparse_feature_dims:
            emb = nn.Embedding(dim + 1, embedding_dim, padding_idx=0)  # +1 for unknown
            # Initialize with smaller weights
            nn.init.uniform_(emb.weight.data, -0.001, 0.001)
            emb.weight.data[0] = 0  # padding
            self.embeddings.append(emb)

        # Linear weights for first order
        self.linear_sparse = nn.ModuleList()
        for dim in sparse_feature_dims:
            linear = nn.Embedding(dim + 1, 1, padding_idx=0)
            nn.init.uniform_(linear.weight.data, -0.001, 0.001)
            linear.weight.data[0] = 0
            self.linear_sparse.append(linear)

        self.linear_dense = nn.Linear(dense_feature_dim, 1)
        nn.init.uniform_(self.linear_dense.weight.data, -0.001, 0.001)

        # FM component
        self.fm = FM()

        # Deep component
        input_dim = len(sparse_feature_dims) * embedding_dim + dense_feature_dim

        # Batch normalization at input
        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.deep = nn.Sequential(*layers)
        self.deep_output = nn.Linear(prev_dim, 1)

        # Initialize deep output with small weights
        nn.init.uniform_(self.deep_output.weight.data, -0.001, 0.001)
        nn.init.zeros_(self.deep_output.bias.data)

        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sparse_inputs, dense_inputs):
        batch_size = sparse_inputs.size(0)

        # First order - sparse
        linear_sparse_output = []
        for i, linear in enumerate(self.linear_sparse):
            linear_sparse_output.append(linear(sparse_inputs[:, i]))
        linear_sparse_output = torch.cat(linear_sparse_output, dim=1).sum(dim=1, keepdim=True)

        # First order - dense
        linear_dense_output = self.linear_dense(dense_inputs)

        # Second order - FM on embeddings
        emb_list = []
        for i, emb in enumerate(self.embeddings):
            emb_list.append(emb(sparse_inputs[:, i]))
        emb_matrix = torch.stack(emb_list, dim=1)  # [batch, num_features, emb_dim]
        fm_output = self.fm(emb_matrix)

        # Deep part
        deep_input_sparse = torch.cat([e(sparse_inputs[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        deep_input = torch.cat([deep_input_sparse, dense_inputs], dim=1)
        deep_input = self.input_bn(deep_input)
        deep_output = self.deep(deep_input)
        deep_output = self.deep_output(deep_output)

        # Combine all parts
        output = self.bias + linear_sparse_output + linear_dense_output + fm_output + deep_output
        return output.squeeze(-1)

def train_deepfm():
    """Train DeepFM with k-fold CV"""
    print("="*60)
    print("DeepFM TRAINING")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load small sample first
    n_samples = 200000
    print(f"Loading {n_samples} samples...")

    # Use custom dataset
    full_dataset = CTRDataset('data/train.parquet', n_samples=n_samples)
    n_sparse = len(full_dataset.sparse_features)
    n_dense = len(full_dataset.dense_features)
    sparse_dims = [len(full_dataset.vocab[col]) for col in full_dataset.sparse_features]

    print(f"Sparse features: {n_sparse}, Dense features: {n_dense}")
    print(f"Positive rate: {full_dataset.y.mean():.4f}")

    # K-fold CV
    n_folds = 2
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(full_dataset))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)), full_dataset.y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Create data loaders
        train_dataset = CTRDataset('data/train.parquet', indices=train_idx, n_samples=n_samples)
        val_dataset = CTRDataset('data/train.parquet', indices=val_idx, n_samples=n_samples)

        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)

        # Create model
        model = DeepFM(
            sparse_feature_dims=sparse_dims,
            dense_feature_dim=n_dense,
            embedding_dim=4,  # Very small embeddings
            hidden_dims=[64, 32],  # Smaller network
            dropout=0.3
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))  # Moderate weight
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        # Training
        best_auc = 0
        patience = 3
        patience_counter = 0

        for epoch in range(10):
            # Train
            model.train()
            train_loss = 0
            n_batches = 0

            for sparse, dense, labels in train_loader:
                sparse = sparse.to(device)
                dense = dense.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(sparse, dense)

                # Check for NaN
                if torch.isnan(outputs).any():
                    print(f"NaN detected in outputs at epoch {epoch}")
                    continue

                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print(f"NaN loss at epoch {epoch}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                print("No valid batches")
                break

            avg_train_loss = train_loss / n_batches

            # Validation
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for sparse, dense, labels in val_loader:
                    sparse = sparse.to(device)
                    dense = dense.to(device)

                    outputs = model(sparse, dense)
                    probs = torch.sigmoid(outputs).cpu().numpy()

                    val_preds.extend(probs)
                    val_labels.extend(labels.numpy())

            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)

            # Remove NaN predictions
            valid_mask = ~np.isnan(val_preds)
            if valid_mask.sum() == 0:
                print("All predictions are NaN")
                break

            val_preds = val_preds[valid_mask]
            val_labels = val_labels[valid_mask]

            # Metrics
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
                val_ap = average_precision_score(val_labels, val_preds)
                val_logloss = log_loss(val_labels, np.clip(val_preds, 1e-7, 1-1e-7))
            except:
                val_auc = val_ap = val_logloss = 0

            print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, AUC={val_auc:.4f}, AP={val_ap:.4f}")
            print(f"  Pred stats: mean={val_preds.mean():.4f}, std={val_preds.std():.4f}")

            scheduler.step()

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model and get OOF predictions
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)

        model.eval()
        val_all_preds = []
        with torch.no_grad():
            for sparse, dense, labels in val_loader:
                sparse = sparse.to(device)
                dense = dense.to(device)
                outputs = model(sparse, dense)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_all_preds.extend(probs)

        oof_preds[val_idx] = np.array(val_all_preds)

        fold_metrics.append({
            'fold': fold,
            'auc': best_auc,
            'ap': val_ap
        })

    # Final evaluation
    valid_mask = ~np.isnan(oof_preds)
    if valid_mask.sum() > 0:
        final_auc = roc_auc_score(full_dataset.y[valid_mask], oof_preds[valid_mask])
        final_ap = average_precision_score(full_dataset.y[valid_mask], oof_preds[valid_mask])
        final_logloss = log_loss(full_dataset.y[valid_mask],
                                 np.clip(oof_preds[valid_mask], 1e-7, 1-1e-7))

        print("\n" + "="*60)
        print("FINAL OOF RESULTS")
        print("="*60)
        print(f"AUC: {final_auc:.6f}")
        print(f"AP: {final_ap:.6f}")
        print(f"LogLoss: {final_logloss:.6f}")

        # Estimate competition score
        wll_estimate = final_logloss * 3.0
        score = 0.5 * final_ap + 0.5 * (1 / (1 + wll_estimate))
        print(f"\nEstimated competition score: {score:.6f} (target: >0.349)")

        # Save results
        output_dir = Path('plan2/experiments/009_deepfm')
        output_dir.mkdir(exist_ok=True, parents=True)

        results = {
            'final_auc': float(final_auc),
            'final_ap': float(final_ap),
            'final_logloss': float(final_logloss),
            'estimated_score': float(score),
            'fold_metrics': fold_metrics
        }

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_dir}")
    else:
        print("All predictions are NaN - training failed")

if __name__ == "__main__":
    train_deepfm()
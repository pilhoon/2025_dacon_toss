import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import warnings
import gc
import time
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')
sys.path.append('..')
from src.data_loader import DataLoader as CompetitionDataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SAINT(nn.Module):
    """
    SAINT: Self-Attention and Intersample Attention Networks for Tabular Data
    """
    def __init__(
        self,
        num_continuous,
        num_categories,
        cat_cardinalities,
        dim=128,
        depth=6,
        heads=8,
        dim_head=16,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_ratio=4,
        cont_embeddings='MLP',
        intersample_attention=True,
        mixup_alpha=0.2
    ):
        super().__init__()

        self.num_continuous = num_continuous
        self.num_categories = num_categories
        self.intersample_attention = intersample_attention
        self.mixup_alpha = mixup_alpha

        # Continuous feature embedding
        if cont_embeddings == 'MLP':
            self.cont_embeddings = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim // 2, dim)
                ) for _ in range(num_continuous)
            ])
        else:
            self.cont_embeddings = nn.ModuleList([
                nn.Linear(1, dim) for _ in range(num_continuous)
            ])

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, dim)
            for cardinality in cat_cardinalities
        ])

        # Column embeddings
        total_features = num_continuous + num_categories
        self.column_embeddings = nn.Parameter(torch.randn(total_features, dim))

        # Self-attention blocks
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(dim, heads, dim_head, attn_dropout, ff_dropout, mlp_hidden_ratio)
            for _ in range(depth)
        ])

        # Intersample attention blocks
        if intersample_attention:
            self.intersample_blocks = nn.ModuleList([
                IntersampleAttentionBlock(dim, heads, dim_head, attn_dropout, ff_dropout)
                for _ in range(depth // 2)
            ])

        # Final layers
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, cont_features, cat_features, apply_mixup=True):
        batch_size = cont_features.shape[0]

        # Mixup augmentation during training
        if self.training and apply_mixup and np.random.random() < 0.5:
            lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size).to(device)
            cont_features = lambda_ * cont_features + (1 - lambda_) * cont_features[index]
            if cat_features is not None:
                # For categorical, we randomly choose one or the other
                mask = torch.rand(batch_size, 1).to(device) < lambda_
                mask = mask.expand_as(cat_features)
                cat_features = torch.where(mask, cat_features, cat_features[index])

        embeddings = []

        # Embed continuous features
        for i, embed_layer in enumerate(self.cont_embeddings):
            embeddings.append(embed_layer(cont_features[:, i:i+1]))

        # Embed categorical features
        if self.num_categories > 0:
            for i, embed_layer in enumerate(self.cat_embeddings):
                embeddings.append(embed_layer(cat_features[:, i]))

        # Stack embeddings
        x = torch.stack(embeddings, dim=1)  # (batch_size, num_features, dim)

        # Add column embeddings
        x = x + self.column_embeddings.unsqueeze(0)

        # Apply self-attention and intersample attention blocks
        for i, self_attn_block in enumerate(self.self_attn_blocks):
            x = self_attn_block(x)

            # Apply intersample attention at certain depths
            if self.intersample_attention and i < len(self.intersample_blocks):
                x = self.intersample_blocks[i](x)

        # Pool across features
        x = x.mean(dim=1)  # (batch_size, dim)

        # Final prediction
        x = self.norm(x)
        return self.to_logits(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout, mlp_hidden_ratio):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=attn_dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_hidden_ratio),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mlp_hidden_ratio, dim),
            nn.Dropout(ff_dropout)
        )

    def forward(self, x):
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Feed-forward
        x = x + self.ff(self.norm2(x))

        return x


class IntersampleAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=attn_dropout, batch_first=True
        )

    def forward(self, x):
        # x: (batch_size, num_features, dim)
        batch_size, num_features, dim = x.shape

        # Transpose for intersample attention
        x_t = x.transpose(0, 1)  # (num_features, batch_size, dim)

        # Apply attention across samples
        normed = self.norm(x_t)
        attn_out, _ = self.attn(normed, normed, normed)
        x_t = x_t + attn_out

        # Transpose back
        x = x_t.transpose(0, 1)  # (batch_size, num_features, dim)

        return x


def calculate_competition_score(y_true, y_pred, k=0.01):
    """Calculate competition score"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # AP Score
    ap_score = average_precision_score(y_true, y_pred)

    # Weighted Log Loss
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    n_positives = np.sum(y_true == 1)
    n_negatives = np.sum(y_true == 0)
    total = len(y_true)

    weight_positive = k * total / n_positives if n_positives > 0 else 0
    weight_negative = (1 - k) * total / n_negatives if n_negatives > 0 else 0

    wll = -(weight_positive * np.sum(y_true * np.log(y_pred)) +
            weight_negative * np.sum((1 - y_true) * np.log(1 - y_pred))) / total

    return 0.7 * ap_score + 0.3 / wll, ap_score, wll


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def train_saint():
    """Train SAINT model"""
    print("="*60)
    print("SAINT: Self-Attention and Intersample Attention")
    print("Advanced tabular deep learning model")
    print("="*60)

    # Load data
    print("\nLoading data...")
    loader = CompetitionDataLoader(cache_dir='cache')

    # Check for enhanced features
    if os.path.exists('plan2/051_train_enhanced.pkl'):
        print("Loading enhanced features...")
        train_data = pd.read_pickle('plan2/051_train_enhanced.pkl')
        test_data = pd.read_pickle('plan2/051_test_enhanced.pkl')
    else:
        train_data, test_data = loader.load_raw_data()

    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['ID', 'target']]
    X = train_data[feature_cols].values
    y = train_data['target'].values
    X_test = test_data[feature_cols].values

    # Identify categorical and continuous columns
    categorical_cols = []
    continuous_cols = []
    categorical_cardinalities = []

    for i, col in enumerate(feature_cols):
        if train_data[col].dtype == 'object' or train_data[col].nunique() < 100:
            categorical_cols.append(i)
            # Encode categorical
            le = LabelEncoder()
            combined = np.concatenate([train_data[col].values, test_data[col].values])
            le.fit(combined)
            X[:, i] = le.transform(train_data[col].values)
            X_test[:, i] = le.transform(test_data[col].values)
            categorical_cardinalities.append(len(le.classes_))
        else:
            continuous_cols.append(i)

    print(f"\nContinuous features: {len(continuous_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    # Separate features
    X_cont = X[:, continuous_cols].astype(np.float32)
    X_test_cont = X_test[:, continuous_cols].astype(np.float32)

    if len(categorical_cols) > 0:
        X_cat = X[:, categorical_cols].astype(np.int64)
        X_test_cat = X_test[:, categorical_cols].astype(np.int64)
    else:
        X_cat = None
        X_test_cat = None

    # Standardize continuous features
    scaler = StandardScaler()
    X_cont = scaler.fit_transform(X_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # Replace NaN with 0
    X_cont = np.nan_to_num(X_cont, 0)
    X_test_cont = np.nan_to_num(X_test_cont, 0)

    print(f"Positive rate: {y.mean():.4f}")

    # 5-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_predictions = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold}/5")
        print(f"{'='*60}")

        # Split data
        X_train_cont = X_cont[train_idx]
        X_val_cont = X_cont[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        if X_cat is not None:
            X_train_cat = X_cat[train_idx]
            X_val_cat = X_cat[val_idx]
        else:
            X_train_cat = None
            X_val_cat = None

        print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

        # Convert to tensors
        train_cont_tensor = torch.FloatTensor(X_train_cont).to(device)
        val_cont_tensor = torch.FloatTensor(X_val_cont).to(device)
        train_target = torch.FloatTensor(y_train).to(device)
        val_target = torch.FloatTensor(y_val).to(device)

        if X_train_cat is not None:
            train_cat_tensor = torch.LongTensor(X_train_cat).to(device)
            val_cat_tensor = torch.LongTensor(X_val_cat).to(device)
        else:
            train_cat_tensor = None
            val_cat_tensor = None

        # Create datasets
        batch_size = 2048
        train_dataset = TensorDataset(
            train_cont_tensor,
            train_cat_tensor if train_cat_tensor is not None else train_cont_tensor,
            train_target
        )
        val_dataset = TensorDataset(
            val_cont_tensor,
            val_cat_tensor if val_cat_tensor is not None else val_cont_tensor,
            val_target
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)

        # Initialize model
        model = SAINT(
            num_continuous=len(continuous_cols),
            num_categories=len(categorical_cols),
            cat_cardinalities=categorical_cardinalities,
            dim=256,
            depth=8,
            heads=8,
            dim_head=32,
            attn_dropout=0.15,
            ff_dropout=0.15,
            mlp_hidden_ratio=4,
            cont_embeddings='MLP',
            intersample_attention=True,
            mixup_alpha=0.3
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        # Mixed precision
        scaler_amp = GradScaler()

        # Training
        best_score = -np.inf
        best_epoch = 0
        patience_counter = 0
        patience = 10

        for epoch in range(1, 31):
            # Training
            model.train()
            train_loss = 0

            for cont, cat, targets in train_loader:
                optimizer.zero_grad()

                with autocast():
                    if X_cat is None:
                        outputs = model(cont, None)
                    else:
                        outputs = model(cont, cat)
                    loss = criterion(outputs.squeeze(), targets)

                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_predictions = []

            with torch.no_grad():
                for cont, cat, _ in val_loader:
                    with autocast():
                        if X_cat is None:
                            outputs = model(cont, None, apply_mixup=False)
                        else:
                            outputs = model(cont, cat, apply_mixup=False)
                    val_predictions.append(torch.sigmoid(outputs).cpu().numpy())

            val_predictions = np.concatenate(val_predictions).flatten()

            # Calculate metrics
            val_score, val_ap, val_wll = calculate_competition_score(y_val, val_predictions)

            print(f"Epoch {epoch}/30:")
            print(f"  Train Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  Val AP: {val_ap:.4f}, Val WLL: {val_wll:.4f}")
            print(f"  Val Competition Score: {val_score:.6f}")

            scheduler.step()

            # Save best model
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                patience_counter = 0
                print(f"  -> New best score!")
                torch.save(model.state_dict(), f'plan2/053_saint_fold{fold}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model and generate test predictions
        model.load_state_dict(torch.load(f'plan2/053_saint_fold{fold}.pth'))
        model.eval()

        test_cont_tensor = torch.FloatTensor(X_test_cont).to(device)
        if X_test_cat is not None:
            test_cat_tensor = torch.LongTensor(X_test_cat).to(device)
        else:
            test_cat_tensor = None

        # Test time augmentation
        tta_preds = []
        for _ in range(5):
            test_preds = []
            with torch.no_grad():
                for i in range(0, len(test_cont_tensor), batch_size*2):
                    batch_cont = test_cont_tensor[i:i+batch_size*2]
                    if test_cat_tensor is not None:
                        batch_cat = test_cat_tensor[i:i+batch_size*2]
                        with autocast():
                            outputs = model(batch_cont, batch_cat, apply_mixup=False)
                    else:
                        with autocast():
                            outputs = model(batch_cont, None, apply_mixup=False)
                    test_preds.append(torch.sigmoid(outputs).cpu().numpy())
            tta_preds.append(np.concatenate(test_preds).flatten())

        fold_predictions = np.mean(tta_preds, axis=0)
        test_predictions.append(fold_predictions)

        print(f"\nFold {fold} completed. Best epoch: {best_epoch}, Best score: {best_score:.6f}")

        # Clean up
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # Average predictions
    final_predictions = np.mean(test_predictions, axis=0)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'target': final_predictions
    })

    submission.to_csv('plan2/053_saint_submission.csv', index=False)
    print("\nSaved to plan2/053_saint_submission.csv")

    print("\n" + "="*60)
    print("Final Results:")
    print(f"Test predictions: mean={final_predictions.mean():.6f}, std={final_predictions.std():.6f}")
    print(f"Min={final_predictions.min():.6f}, Max={final_predictions.max():.6f}")
    print("="*60)

    return final_predictions


if __name__ == "__main__":
    import os
    predictions = train_saint()
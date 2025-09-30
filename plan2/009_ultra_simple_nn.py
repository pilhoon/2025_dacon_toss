"""
Ultra Simple Neural Network with maximum stability
- No embeddings (use one-hot or hash encoding)
- Simple MLP
- Batch-wise normalization
- No pos_weight (use sampling instead)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
import gc

print("Ultra Simple NN - Maximum Stability")
print("="*50)

# Load preprocessed cache data
print("Loading cached data...")
cache_dir = 'plan2/cache'
train_X = pd.read_parquet(f'{cache_dir}/train_X.parquet').head(100000)
train_y = np.load(f'{cache_dir}/train_y.npy')[:100000]

print(f"Data shape: {train_X.shape}")
print(f"Positive rate: {train_y.mean():.4f}")

# Feature hashing for categorical (instead of embeddings)
print("Feature hashing...")
cat_cols = [c for c in train_X.columns if c.startswith(('gender', 'age', 'inventory', 'seq', 'l_feat', 'feat'))]
num_cols = [c for c in train_X.columns if c not in cat_cols]

# Hash categorical features to fixed dimension
n_hash_features = 1000
hasher = FeatureHasher(n_features=n_hash_features, input_type='string')
cat_data = train_X[cat_cols].astype(str)
cat_dict_list = cat_data.to_dict('records')
cat_hashed = hasher.transform(cat_dict_list).toarray().astype(np.float32)

# Normalize numerical features
scaler = StandardScaler()
num_data = scaler.fit_transform(train_X[num_cols].values.astype(np.float32))

# Combine features
X = np.hstack([cat_hashed, num_data]).astype(np.float32)
print(f"Final feature dimension: {X.shape[1]}")

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(
    X, train_y, test_size=0.2, random_state=42, stratify=train_y
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # First layer with batch norm
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dims[0]

        # Hidden layers
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer - initialize with very small weights
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -3.0)  # Start with negative bias for imbalanced data

    def forward(self, x):
        x = self.layers(x)
        return self.output(x).squeeze()

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(X.shape[1], hidden_dims=[128, 64, 32]).to(device)
print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Balanced sampling
pos_indices = np.where(y_train.numpy() == 1)[0]
neg_indices = np.where(y_train.numpy() == 0)[0]

# Undersample negatives to balance
n_pos = len(pos_indices)
balanced_neg_indices = np.random.choice(neg_indices, size=n_pos*2, replace=False)  # 2:1 ratio
balanced_indices = np.concatenate([pos_indices, balanced_neg_indices])
np.random.shuffle(balanced_indices)

X_train_balanced = X_train[balanced_indices]
y_train_balanced = y_train[balanced_indices]

print(f"Balanced training set: {len(y_train_balanced)} samples")
print(f"Balanced positive rate: {y_train_balanced.mean():.4f}")

# Data loaders
train_dataset = TensorDataset(X_train_balanced, y_train_balanced)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Training with simple BCE loss (no pos_weight since we balanced the data)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Training loop
print("\nTraining...")
for epoch in range(20):
    # Train
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        # Check for NaN
        if torch.isnan(outputs).any():
            print(f"NaN in outputs at epoch {epoch}")
            # Reset model
            model = SimpleNN(X.shape[1], hidden_dims=[128, 64, 32]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            break

        loss = criterion(outputs, batch_y)

        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch}")
            break

        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_preds.extend(probs)

    val_preds = np.array(val_preds)

    # Remove NaN if any
    if np.isnan(val_preds).any():
        print(f"NaN in validation predictions at epoch {epoch}")
        val_preds = np.nan_to_num(val_preds, nan=0.5)

    # Metrics
    val_auc = roc_auc_score(y_val.numpy(), val_preds)
    val_ap = average_precision_score(y_val.numpy(), val_preds)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: AUC={val_auc:.4f}, AP={val_ap:.4f}")
        print(f"  Pred stats: mean={val_preds.mean():.4f}, std={val_preds.std():.4f}")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Final AUC: {val_auc:.4f}")
print(f"Final AP: {val_ap:.4f}")

# Estimate score
wll_estimate = 0.5  # Conservative estimate
score_estimate = 0.5 * val_ap + 0.5 * (1/(1+wll_estimate))
print(f"Estimated competition score: {score_estimate:.4f}")

if val_auc > 0.7:  # Only save if reasonable
    print("\nModel performed reasonably well!")
    torch.save(model.state_dict(), 'plan2/experiments/simple_nn.pth')
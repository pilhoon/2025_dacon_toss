
from typing import Optional, Dict, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FTTokenEmbedding(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(num_features, d_model))
        nn.init.xavier_uniform_(self.proj)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.proj.unsqueeze(0)

class CLSPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        return torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)

class TabTransformer(nn.Module):
    def __init__(self, num_features: int, d_model: int=128, nhead: int=8, num_layers: int=4, dim_feedforward: int=256, dropout: float=0.1):
        super().__init__()
        self.embed = FTTokenEmbedding(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_pool = CLSPool(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.embed(x)
        tok = self.cls_pool(tok)
        z = self.encoder(tok)[:,0]
        z = self.norm(z)
        logit = self.head(z).squeeze(-1)
        return logit

def train_tab_transformer(X_train, y_train, X_valid, y_valid, device: str='cuda', max_epochs: int=10,
                          batch_size: int=65536, lr: float=2e-4, d_model: int=128, nhead: int=8, num_layers: int=4,
                          dim_feedforward: int=256, dropout: float=0.1, sample_weight=None):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_valid = np.asarray(y_valid, dtype=np.float32)

    n_features = X_train.shape[1]
    model = TabTransformer(n_features, d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith('cuda')))
    bce = nn.BCEWithLogitsLoss(reduction='none')

    import torch.utils.data as tud
    class DS(tud.Dataset):
        def __init__(self, X, y, w):
            self.X, self.y = X, y
            self.w = w if w is not None else np.ones_like(y, dtype=np.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            return self.X[i], self.y[i], self.w[i]

    train_loader = tud.DataLoader(DS(X_train, y_train, sample_weight), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    valid_loader = tud.DataLoader(DS(X_valid, y_valid, None), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    best_state = None
    best_val = float('inf')
    for epoch in range(1, max_epochs+1):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.startswith('cuda'))):
                logits = model(xb)
                loss = (bce(logits, yb) * wb).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad():
            losses = []
            for xb, yb, _ in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                p = torch.sigmoid(model(xb)).clamp(1e-7, 1-1e-7)
                losses.append(nn.functional.binary_cross_entropy(p, yb, reduction='mean').item())
            val = float(np.mean(losses))
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(X):
        XT = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(XT)).detach().cpu().numpy().astype('float32')
        return out
    return model, predict

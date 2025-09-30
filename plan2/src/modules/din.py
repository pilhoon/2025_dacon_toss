from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class DIN(nn.Module):
    def __init__(self, cat_cardinalities: Dict[str, int], num_dim: int, target_item_col: str, embed_dim: int = 32, attn_hidden: int = 64, mlp_dims=None, dropout: float = 0.1):
        super().__init__()
        mlp_dims = mlp_dims or [256, 128, 64]
        self.cat_keys = list(cat_cardinalities.keys())
        self.embeds = nn.ModuleDict({k: nn.Embedding(v, embed_dim) for k, v in cat_cardinalities.items()})
        self.target_item_col = target_item_col
        self.num_proj = nn.Linear(num_dim, embed_dim) if num_dim > 0 else None
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 2, attn_hidden), nn.ReLU(), nn.Linear(attn_hidden, 1)
        )
        layers = []
        dim = embed_dim * 2 + (embed_dim if self.num_proj is not None else 0)
        for h in mlp_dims:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 1)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor] | torch.Tensor]) -> torch.Tensor:
        cat: Dict[str, torch.Tensor] = batch["cat"]  # type: ignore
        num: torch.Tensor = batch["num"]  # type: ignore
        target = self.embeds[self.target_item_col](cat[self.target_item_col])  # [B, E]
        # Simple placeholder: average behavior embedding from any history_* if exists
        hist_keys = [k for k in self.cat_keys if k.startswith("history_")]
        if len(hist_keys) > 0:
            h_embs = [self.embeds[k](cat[k]) for k in hist_keys]
            hist = torch.stack(h_embs, dim=1).mean(dim=1)
        else:
            hist = torch.zeros_like(target)
        # attention score (simplified)
        a = self.attn(torch.cat([target, hist], dim=-1))  # [B, 1]
        user_vec = a * hist
        parts = [target, user_vec]
        if self.num_proj is not None:
            parts.append(self.num_proj(num))
        x = torch.cat(parts, dim=-1)
        x = self.mlp(x)
        logit = self.out(x)
        return logit.squeeze(-1)



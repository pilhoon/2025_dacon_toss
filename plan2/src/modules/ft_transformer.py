from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FTTransformer(nn.Module):
    def __init__(self, cat_cardinalities: Dict[str, int], num_dim: int, embed_dim: int = 64, n_layers: int = 4, n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1, stochastic_depth: float = 0.0):
        super().__init__()
        self.cat_keys = list(cat_cardinalities.keys())
        self.embeds = nn.ModuleDict({k: nn.Embedding(v, embed_dim) for k, v in cat_cardinalities.items()})
        self.num_proj = nn.Linear(num_dim, embed_dim) if num_dim > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * ff_mult, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor] | torch.Tensor]) -> torch.Tensor:
        cat: Dict[str, torch.Tensor] = batch["cat"]  # type: ignore
        num: torch.Tensor = batch["num"]  # type: ignore
        tokens = [self.embeds[k](cat[k]) for k in self.cat_keys]
        if self.num_proj is not None:
            tokens.append(self.num_proj(num))
        x = torch.stack(tokens, dim=1)  # [B, T, E]
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        cls_out = x[:, 0]
        logit = self.out(cls_out)
        return logit.squeeze(-1)



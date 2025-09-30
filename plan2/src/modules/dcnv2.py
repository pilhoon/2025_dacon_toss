from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for layer in self.layers:
            x = x + layer(x) * x0
        return x


class DCNv2(nn.Module):
    def __init__(self, cat_cardinalities: Dict[str, int], num_dim: int, embed_dim: int = 32, cross_depth: int = 3, mlp_dims: List[int] | None = None, dropout: float = 0.0):
        super().__init__()
        mlp_dims = mlp_dims or [256, 128, 64]
        self.cat_keys = list(cat_cardinalities.keys())
        self.embeds = nn.ModuleDict({k: nn.Embedding(v, embed_dim) for k, v in cat_cardinalities.items()})
        input_dim = len(self.embeds) * embed_dim + num_dim
        self.cross = CrossNetworkV2(input_dim, cross_depth)
        layers: List[nn.Module] = []
        dim = input_dim
        for h in mlp_dims:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 1)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor] | torch.Tensor]) -> torch.Tensor:
        cat: Dict[str, torch.Tensor] = batch["cat"]  # type: ignore
        num: torch.Tensor = batch["num"]  # type: ignore
        embs = [self.embeds[k](cat[k]) for k in self.cat_keys]
        x = torch.cat([*(embs), num], dim=-1)
        x = self.cross(x)
        x = self.mlp(x)
        logit = self.out(x)
        return logit.squeeze(-1)



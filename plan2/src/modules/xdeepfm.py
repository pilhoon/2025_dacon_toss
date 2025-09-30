from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class CIN(nn.Module):
    def __init__(self, field_dim: int, layer_sizes: List[int]):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.field_dim = field_dim
        self.filters = nn.ModuleList()
        prev_dim = field_dim
        for h in layer_sizes:
            self.filters.append(nn.Conv1d(in_channels=prev_dim * field_dim, out_channels=h, kernel_size=1))
            prev_dim = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, E]
        xs = []
        xk = x
        B, F, E = x.shape
        for conv in self.filters:
            # outer product along embedding dim via pairwise interactions
            z = torch.einsum('bfe,bge->bfge', x, xk)  # [B, F, F, E]
            z = z.reshape(B, F * F, E)  # [B, F*F, E]
            z = z.transpose(1, 2)  # [B, E, F*F]
            z = conv(z)  # [B, H, F*F] with kernel=1
            z = torch.relu(z)
            z = torch.sum(z, dim=-1)  # [B, H]
            xs.append(z)
            xk = z.unsqueeze(2).repeat(1, 1, E)  # approximate feature map to keep dims
        return torch.cat(xs, dim=1)


class XDeepFM(nn.Module):
    def __init__(self, cat_cardinalities: Dict[str, int], num_dim: int, embed_dim: int = 32, cin_layers: List[int] | None = None, dnn_layers: List[int] | None = None, dropout: float = 0.0):
        super().__init__()
        cin_layers = cin_layers or [128, 128]
        dnn_layers = dnn_layers or [256, 128, 64]
        self.cat_keys = list(cat_cardinalities.keys())
        self.embeds = nn.ModuleDict({k: nn.Embedding(v, embed_dim) for k, v in cat_cardinalities.items()})
        self.field_dim = len(self.embeds)
        self.cin = CIN(self.field_dim, cin_layers)
        dnn_input = self.field_dim * embed_dim + num_dim
        layers: List[nn.Module] = []
        dim = dnn_input
        for h in dnn_layers:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        self.dnn = nn.Sequential(*layers)
        self.out = nn.Linear(dim + sum(cin_layers), 1)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor] | torch.Tensor]) -> torch.Tensor:
        cat: Dict[str, torch.Tensor] = batch["cat"]  # type: ignore
        num: torch.Tensor = batch["num"]  # type: ignore
        embs = [self.embeds[k](cat[k]) for k in self.cat_keys]
        x = torch.stack(embs, dim=1)  # [B, F, E]
        cin_feat = self.cin(x)
        dnn_in = torch.cat([x.flatten(start_dim=1), num], dim=-1)
        dnn_feat = self.dnn(dnn_in)
        out = self.out(torch.cat([dnn_feat, cin_feat], dim=-1))
        return out.squeeze(-1)



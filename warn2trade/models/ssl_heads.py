"""Self-supervised heads on top of the anomaly latent z: contrastive projection + masked-window reconstruction."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLHeads(nn.Module):
    def __init__(self, d_model: int, lookback: int, price_dim: int, proj_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.L, self.F = int(lookback), int(price_dim)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim))
        self.dec = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, self.L * self.F))

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z), dim=-1)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z).view(-1, self.L, self.F)

"""Per-modality encoders. Every encoder maps its modality to a single (B, d_model) token."""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def liq_transform(liq: torch.Tensor) -> torch.Tensor:
    """Raw liquidity state [rel half-spread, depth, adv_frac] -> O(1) features [log half-spread in bps, depth, log adv_frac]."""
    return torch.stack([torch.log(liq[..., 0] * 1e4 + 1e-3), liq[..., 1], torch.log(liq[..., 2] + 1e-6)], -1)


class PatchTSEncoder(nn.Module):
    """PatchTST-style patching + Transformer over a (B, L, F) multivariate window (channel-mixed patches)."""

    def __init__(self, in_dim: int, d_model: int, patch_len: int = 8, stride: int = 4, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1, max_patches: int = 512):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.proj = nn.Linear(in_dim * patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Fdim = x.shape
        if L < self.patch_len:
            x = F.pad(x, (0, 0, self.patch_len - L, 0))
        patches = x.unfold(1, self.patch_len, self.stride)            # (B, P, F, patch_len)
        P = patches.shape[1]
        patches = patches.permute(0, 1, 3, 2).reshape(B, P, self.patch_len * Fdim)
        h = self.proj(patches) + self.pos[:, :P]
        h = self.encoder(h)
        return self.norm(h[:, -1] + h.mean(1))                           # last patch + global context


class LOBConvEncoder(nn.Module):
    """Temporal 1D conv stack over LOB snapshots (B, L, Fl) -> (B, d_model). Fl = levels x 4 (+ engineered)."""

    def __init__(self, in_dim: int, d_model: int, channels: int = 64, n_blocks: int = 3, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        layers = []
        c_in = in_dim
        for b in range(n_blocks):
            layers += [nn.Conv1d(c_in, channels, kernel, padding=(kernel // 2) * 2 ** b, dilation=2 ** b),
                       nn.GELU(), nn.Dropout(dropout)]
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(channels * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))                                  # (B, C, L)
        return self.out(torch.cat([h[:, :, -1], h.mean(-1)], -1))


class TimeDecayTextEncoder(nn.Module):
    """Attention pooling over pre-computed document embeddings with a learned recency decay.

    score_m = <q, W e_m> / sqrt(d) - lambda * age_m ; lambda = exp(log_decay) > 0.
    Empty slots are masked; a sample with no documents returns a learned 'silence' token.
    """

    def __init__(self, in_dim: int, d_model: int, init_decay: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.log_decay = nn.Parameter(torch.tensor(math.log(init_decay)))
        self.silence = nn.Parameter(torch.zeros(d_model))
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, e: torch.Tensor, mask: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        h = self.proj(e)                                                 # (B, M, d)
        score = (h @ self.query) * self.scale - torch.exp(self.log_decay) * age
        score = score.masked_fill(~mask, -1e4)
        attn = torch.softmax(score, -1) * mask.float()
        pooled = (attn.unsqueeze(-1) * h).sum(1)
        has = mask.any(1, keepdim=True).float()
        return has * pooled + (1.0 - has) * self.silence


class NeighborAttentionEncoder(nn.Module):
    """Single-query cross attention from the focal asset token to K neighbour embeddings (GAT-like)."""

    def __init__(self, in_dim: int, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))              # always-present key avoids NaN softmax
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, focal: torch.Tensor, nb: torch.Tensor, nb_mask: torch.Tensor) -> torch.Tensor:
        B = focal.shape[0]
        kv = torch.cat([self.null.expand(B, 1, -1), self.proj(nb)], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=nb.device), ~nb_mask], 1)
        out, _ = self.attn(focal.unsqueeze(1), kv, kv, key_padding_mask=pad)
        return self.norm(out.squeeze(1) + focal)

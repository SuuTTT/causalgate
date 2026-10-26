"""Stage 2 - Temporal Dynamics Extrapolator (TDE).

Given the anomaly latent z, the anomaly probability a and the current liquidity state, predict for every
horizon h in the grid:
    * quantiles q_tau(h) of the post-warning cumulative log return I(h)   (direction + magnitude + uncertainty)
    * the liquidity state l_hat(h)                                        (spread / depth / ADV after the warning)
    * a time-to-peak distribution over the horizon grid                  (alpha-decay horizon h*)
Quantiles are monotone in tau by construction (base + cumulative softplus increments) so no crossing penalty is needed.
The network predicts q_norm in units of sigma * sqrt(h) (so losses and gradients are O(1) and comparable to the BCE);
`q` = q_norm * sigma * sqrt(h) is in log-return units for the policy, the plug-in rule and the backtest.
"""
from __future__ import annotations
from typing import Dict, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import liq_transform


class TemporalDynamicsExtrapolator(nn.Module):
    def __init__(self, d_model: int, horizons: Sequence[int], taus: Sequence[float], liq_dim: int = 3, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.H, self.Q = len(horizons), len(taus)
        self.register_buffer("taus", torch.tensor(sorted(taus), dtype=torch.float32))
        self.register_buffer("horizons", torch.tensor(list(horizons), dtype=torch.float32))
        self.h_emb = nn.Parameter(torch.randn(self.H, d_model) * 0.02)
        self.cond = nn.Sequential(nn.Linear(d_model + 1 + liq_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.dec = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, self.Q + liq_dim + 1))
        self.liq_dim = liq_dim

    def forward(self, z: torch.Tensor, a_prob: torch.Tensor, liq: torch.Tensor, sigma: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z.shape[0]
        c = self.cond(torch.cat([z, a_prob.unsqueeze(-1), liq_transform(liq)], -1))      # (B, d)
        x = torch.cat([c.unsqueeze(1).expand(B, self.H, -1), self.h_emb.unsqueeze(0).expand(B, -1, -1)], -1)
        out = self.dec(x)                                                  # (B, H, Q + liq + 1)
        raw_q = out[..., : self.Q]
        base = raw_q[..., :1]
        q_norm = torch.cat([base, base + torch.cumsum(F.softplus(raw_q[..., 1:]), -1)], -1)   # sigma units, monotone in tau
        scale = (sigma.clamp_min(1e-6).view(B, 1, 1) * torch.sqrt(self.horizons).view(1, -1, 1))
        return {"q": q_norm * scale, "q_norm": q_norm, "liq_hat": out[..., self.Q : self.Q + self.liq_dim], "tpeak_logit": out[..., -1]}


def expected_impact(q: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """Trapezoid approximation of E[I(h)] from quantiles (B, H, Q) -> (B, H)."""
    w = torch.zeros_like(taus)
    w[1:] += 0.5 * (taus[1:] - taus[:-1])
    w[:-1] += 0.5 * (taus[1:] - taus[:-1])
    w = w / w.sum()
    return (q * w).sum(-1)


def tail_mean(q: torch.Tensor, taus: torch.Tensor, alpha: float, side: str) -> torch.Tensor:
    """Mean of the lower (side='lower') or upper tail quantiles with tau <= alpha / tau >= 1 - alpha. (B, H)"""
    if side == "lower":
        m = (taus <= alpha + 1e-6).float()
    else:
        m = (taus >= 1.0 - alpha - 1e-6).float()
    if m.sum() == 0:
        m = torch.zeros_like(taus)
        m[0 if side == "lower" else -1] = 1.0
    return (q * m).sum(-1) / m.sum()

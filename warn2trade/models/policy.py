"""Stage 3 - Risk-Aware Execution Policy (REP).

Learned policy: (a, impact quantiles, liquidity forecast, time-to-peak, previous weight, sigma) ->
    w        target weight in [-w_max, w_max] (soft-gated by an abstain probability so the 'no trade' action is differentiable)
    hold_p   distribution over the horizon grid (holding period / alpha-decay horizon)
    kappa_p  distribution over execution schedules (1 = immediate, k = TWAP over k bars)
Plug-in Bayes policy: closed-form decision from the same inputs; used as (i) an interpretable baseline and (ii) the
initialisation target of the learned policy (behaviour-cloning warm start).
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .extrapolator import expected_impact, tail_mean
from .encoders import liq_transform


def policy_features(a_prob: torch.Tensor, q: torch.Tensor, liq_hat: torch.Tensor, tpeak_logit: torch.Tensor,
                    prev_w: torch.Tensor, sigma: torch.Tensor, liq: torch.Tensor) -> torch.Tensor:
    B = a_prob.shape[0]
    return torch.cat([a_prob.unsqueeze(-1), q.reshape(B, -1), liq_hat.reshape(B, -1), torch.softmax(tpeak_logit, -1),
                      prev_w.unsqueeze(-1), torch.log(sigma.unsqueeze(-1) * 1e2 + 1e-4), liq_transform(liq)], -1)


def policy_in_dim(H: int, Q: int, liq_dim: int) -> int:
    return 1 + H * Q + H * liq_dim + H + 1 + 1 + liq_dim


class RiskAwarePolicy(nn.Module):
    def __init__(self, in_dim: int, n_horizons: int, n_exec: int = 4, w_max: float = 1.0, hidden: int = 256,
                 use_abstain: bool = True, dropout: float = 0.1):
        super().__init__()
        self.w_max, self.use_abstain, self.H, self.n_exec = w_max, use_abstain, n_horizons, n_exec
        self.body = nn.Sequential(nn.BatchNorm1d(in_dim, affine=False), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.size = nn.Linear(hidden, 1)
        self.gate = nn.Linear(hidden, 1)
        self.hold = nn.Linear(hidden, n_horizons)
        self.exec = nn.Linear(hidden, n_exec)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.body(x)
        abstain = torch.sigmoid(self.gate(h)).squeeze(-1) if self.use_abstain else torch.zeros(x.shape[0], device=x.device)
        w = self.w_max * torch.tanh(self.size(h)).squeeze(-1) * (1.0 - abstain)
        return {"w": w, "abstain": abstain, "hold_p": torch.softmax(self.hold(h), -1), "kappa_p": torch.softmax(self.exec(h), -1)}


class BayesPlugInPolicy(nn.Module):
    """Decision-theoretic plug-in rule (no parameters).

    For each horizon h:  net_long(h)  = E[I(h)] - c_rt - lam * CVaR_alpha(-I(h))
                         net_short(h) = -E[I(h)] - c_rt - lam * CVaR_alpha(+I(h))
    Trade iff a >= a_min and max_h max(net_long, net_short) > 0 ; h* = argmax ; direction = better side ;
    size = clip( kelly_frac * mu_h / var_h , -w_max, w_max ) with var from the inter-quartile range.
    """

    def __init__(self, taus: torch.Tensor, cost_rt: float = 0.002, lam_cvar: float = 0.5, alpha: float = 0.1,
                 kelly_frac: float = 0.25, w_max: float = 1.0, a_min: float = 0.5, n_exec: int = 4):
        super().__init__()
        self.register_buffer("taus", taus)
        self.cost_rt, self.lam, self.alpha, self.kelly, self.w_max, self.a_min, self.n_exec = cost_rt, lam_cvar, alpha, kelly_frac, w_max, a_min, n_exec

    @torch.no_grad()
    def forward(self, a_prob: torch.Tensor, q: torch.Tensor, cost_rt: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mu = expected_impact(q, self.taus)                                  # (B, H)
        c = self.cost_rt if cost_rt is None else cost_rt.unsqueeze(-1)
        cvar_long = -tail_mean(q, self.taus, self.alpha, "lower")
        cvar_short = tail_mean(q, self.taus, self.alpha, "upper")
        net_long = mu - c - self.lam * cvar_long
        net_short = -mu - c - self.lam * cvar_short
        best = torch.maximum(net_long, net_short)
        h_star = best.argmax(-1)
        bi = torch.arange(q.shape[0], device=q.device)
        direction = torch.where(net_long[bi, h_star] >= net_short[bi, h_star], 1.0, -1.0)
        iqr = (tail_mean(q, self.taus, 0.25, "upper") - tail_mean(q, self.taus, 0.25, "lower"))[bi, h_star]
        var = (iqr / 1.349).pow(2).clamp_min(1e-8)
        size = (self.kelly * mu[bi, h_star].abs() / var).clamp(0.0, self.w_max)
        trade = (a_prob >= self.a_min) & (best[bi, h_star] > 0)
        w = torch.where(trade, direction * size, torch.zeros_like(size))
        hold_p = F.one_hot(h_star, q.shape[1]).float()
        kappa_p = torch.zeros(q.shape[0], self.n_exec, device=q.device)
        kappa_p[:, 0] = 1.0
        return {"w": w, "abstain": (~trade).float(), "hold_p": hold_p, "kappa_p": kappa_p}

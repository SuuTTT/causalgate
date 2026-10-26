"""Loss zoo for the closed-loop objective.

L = L_det (utility-weighted) + b1 L_imp (pinball) + b2 L_liq + b3 L_peak - b4 U(policy) + b5 Omega(turnover) + b6 L_svdd
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def utility_weighted_bce(logit: torch.Tensor, y: torch.Tensor, exec_alpha: torch.Tensor, rho: float = 2.0,
                         pos_weight: Optional[float] = None) -> torch.Tensor:
    """BCE where positive samples are up-weighted by their executable alpha (normalised to mean 1 over positives).

    omega = 1 + rho * A_ex / mean_pos(A_ex)  for y = 1 ; omega = 1 for y = 0.
    Un-monetisable anomalies (liquidity freeze, move smaller than cost) keep weight ~1 so the detector learns to
    prioritise *actionable* warnings. rho = 0 recovers plain (class-weighted) BCE for the ablation.
    """
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pw)
    if rho <= 0:
        return bce.mean()
    pos = y > 0.5
    denom = exec_alpha[pos].mean().clamp_min(1e-8) if pos.any() else torch.tensor(1.0, device=logit.device)
    omega = torch.where(pos, 1.0 + rho * exec_alpha / denom, torch.ones_like(exec_alpha))
    return (omega * bce).sum() / omega.sum()


def pinball_loss(q: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """q (B, H, Q), target (B, H), taus (Q,). Optional per-sample weight (B,) e.g. the anomaly probability."""
    u = target.unsqueeze(-1) - q
    loss = torch.maximum(taus * u, (taus - 1.0) * u).mean(-1).mean(-1)   # (B,)
    if weight is not None:
        return (weight * loss).sum() / weight.sum().clamp_min(1e-8)
    return loss.mean()


def direction_loss(dir_logit: torch.Tensor, impact: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE on sign(I(h*)) for anomaly samples only. Forces the shared latent to encode direction so that the
    extrapolator's quantiles are not the unconditional (symmetric) impact distribution."""
    pos = y > 0.5
    if not pos.any():
        return dir_logit.sum() * 0.0
    hs = impact[pos].abs().argmax(-1)
    target = (impact[pos].gather(1, hs[:, None]).squeeze(1) > 0).float()
    return F.binary_cross_entropy_with_logits(dir_logit[pos], target)


def peak_time_ce(tpeak_logit: torch.Tensor, impact: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on argmax_h |I(h)| restricted to anomaly samples."""
    pos = y > 0.5
    if not pos.any():
        return tpeak_logit.sum() * 0.0
    target = impact[pos].abs().argmax(-1)
    return F.cross_entropy(tpeak_logit[pos], target)


def liquidity_loss(liq_hat: torch.Tensor, fut_liq: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(liq_hat, fut_liq, reduction="none").mean((-1, -2))
    if weight is not None:
        return (weight * loss).sum() / weight.sum().clamp_min(1e-8)
    return loss.mean()


def differentiable_sortino(pnl: torch.Tensor, target: float = 0.0, eps: float = 1e-6) -> torch.Tensor:
    """Batch Sortino ratio: mean(pnl - target) / sqrt(mean(min(pnl - target, 0)^2)). Higher is better."""
    ex = pnl - target
    downside = torch.sqrt((torch.clamp(ex, max=0.0) ** 2).mean() + eps)
    return ex.mean() / downside


def differentiable_sharpe(pnl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return pnl.mean() / (pnl.std() + eps)


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Rockafellar-Uryasev CVaR_alpha of the loss (-pnl); nu fixed at the empirical VaR (sub-gradient friendly)."""
    loss = -pnl
    nu = torch.quantile(loss.detach(), 1.0 - alpha)
    return nu + torch.clamp(loss - nu, min=0.0).mean() / alpha


def turnover_penalty(w: torch.Tensor, prev_w: torch.Tensor) -> torch.Tensor:
    return (w - prev_w).abs().mean()


def svdd_regularizer(z: torch.Tensor, center: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One-class pull of normal latents towards the centre (Deep SVDD); anomalies are pushed out (hinge)."""
    d2 = ((z - center) ** 2).sum(-1)
    normal = (1.0 - y) * d2
    anom = y * torch.clamp(1.0 - d2, min=0.0)
    return (normal + anom).mean()


def utility(pnl: torch.Tensor, lam_down: float = 1.0, lam_cvar: float = 0.5, alpha: float = 0.05) -> torch.Tensor:
    """U = E[pnl] - lam_down * sqrt(LPM2) - lam_cvar * CVaR_alpha(-pnl). Maximise."""
    lpm2 = torch.sqrt((torch.clamp(pnl, max=0.0) ** 2).mean() + 1e-8)
    return pnl.mean() - lam_down * lpm2 - lam_cvar * cvar_loss(pnl, alpha)

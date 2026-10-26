"""The four loss families of the paper.

    L_sup   (anchors)              deep_sad_loss + lam_dev * deviation_loss           -> supervised_anchor_loss
    L_cons  (unlabelled)           FixMatch consistency with a dynamic (FlexMatch) threshold, conformal gate on
                                   pseudo-positives and per-sample outcome / monetisability weights omega
    L_ssl   (unlabelled)           NT-Xent between weak & strong views + masked reconstruction
    L_trade (all, closed loop)     -U(pnl) + lam_turnover * turnover  where pnl comes from the differentiable event
                                   backtest (latency, costs, liquidity freezes)
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

from ..models.losses import utility, turnover_penalty
from ..backtest.engine import event_pnl
from ..backtest.costs import CostModel


# ----------------------------------------------------------------------------- L_sup
def deep_sad_loss(z: torch.Tensor, center: torch.Tensor, y: torch.Tensor, eta: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Ruff et al. (ICLR 2020): normals pulled to the centre, labelled anomalies pushed out via (d^2 + eps)^-1."""
    d2 = ((z - center) ** 2).sum(-1)
    return ((1.0 - y) * d2 + y * eta * (d2 + eps).reciprocal()).mean()


def deviation_loss(logit: torch.Tensor, y: torch.Tensor, margin: float = 5.0, n_ref: int = 5000,
                   gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """DevNet (Pang et al., KDD 2019): z-score of the anomaly score against a Gaussian reference; hinge for anomalies."""
    ref = torch.randn(n_ref, generator=gen).to(logit.device)
    dev = (logit - ref.mean()) / (ref.std() + 1e-8)
    return ((1.0 - y) * dev.abs() + y * torch.clamp(margin - dev, min=0.0)).mean()


def supervised_anchor_loss(z: torch.Tensor, center: torch.Tensor, logit: torch.Tensor, y: torch.Tensor, eta: float = 1.0,
                           lam_dev: float = 0.5, margin: float = 5.0, gen: Optional[torch.Generator] = None,
                           lam_sad: float = 1.0) -> Dict[str, torch.Tensor]:
    """lam_sad * Deep SAD (latent) + lam_dev * DevNet deviation (score head). Faithful baselines: Deep SAD = (1, 0) and is
    scored by ||z - c||^2; DevNet = (0, 1) and is scored by the head."""
    sad = deep_sad_loss(z, center, y, eta)
    dev = deviation_loss(logit, y, margin, gen=gen)
    return {"loss": lam_sad * sad + lam_dev * dev, "sad": sad, "dev": dev}


# ----------------------------------------------------------------------------- L_cons
class DynamicThreshold:
    """FlexMatch-style curriculum threshold for binary pseudo-labels.

    status_c = EMA of the fraction of unlabelled samples confidently (>= tau) assigned to class c;
    thr_c = tau * M(status_c / max_c status_c) with the convex mapping M(x) = x / (2 - x), floored at `floor`.
    Classes the model has not learnt yet get a lower threshold so their pseudo-labels are not starved.
    """

    def __init__(self, tau: float = 0.95, momentum: float = 0.99, floor: float = 0.5):
        self.tau, self.m, self.floor = tau, momentum, floor
        self.status = torch.ones(2)

    def thresholds(self) -> torch.Tensor:
        s = self.status / self.status.max().clamp_min(1e-8)
        return torch.clamp(self.tau * (s / (2.0 - s)), min=self.floor)

    def per_sample(self, hard: torch.Tensor) -> torch.Tensor:
        return self.thresholds().to(hard.device)[hard.long()]

    @torch.no_grad()
    def update(self, conf: torch.Tensor, hard: torch.Tensor) -> None:
        for c in (0, 1):
            frac = ((conf >= self.tau) & (hard == c)).float().mean().item()
            self.status[c] = self.m * self.status[c] + (1.0 - self.m) * frac


def consistency_loss(logit_strong: torch.Tensor, p_teacher: torch.Tensor, threshold, weight: Optional[torch.Tensor] = None,
                     gate: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FixMatch BCE on hard pseudo-labels from the teacher (weak view), masked by confidence >= threshold.

    `gate` (bool, B) is the conformal gate: a pseudo-positive that fails it is dropped. Returns (loss, mask_rate, hard).
    """
    hard = (p_teacher >= 0.5).float()
    conf = torch.maximum(p_teacher, 1.0 - p_teacher)
    thr = threshold if torch.is_tensor(threshold) else torch.full_like(conf, float(threshold))
    mask = (conf >= thr).float()
    if gate is not None:
        mask = mask * (gate.float() + (1.0 - hard)).clamp(max=1.0)
    w = mask if weight is None else mask * weight
    bce = F.binary_cross_entropy_with_logits(logit_strong, hard, reduction="none")
    return (w * bce).sum() / w.sum().clamp_min(1.0), mask.mean(), hard


# ----------------------------------------------------------------------------- L_ssl
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """SimCLR / TS2Vec instance-level contrastive loss between two views (z already L2-normalised)."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), -1e4)
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, target)


def masked_reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return ((x_hat - x) ** 2 * m).sum() / (m.sum() * x.shape[-1]).clamp_min(1.0)


# ----------------------------------------------------------------------------- closed-loop weights
def outcome_consistency_weights(impact: torch.Tensor, sigma: torch.Tensor, horizons: Sequence[int], m_out: float = 2.0,
                                kappa: float = 4.0, floor: float = 0.1) -> torch.Tensor:
    """Realised post-warning footprint as a *delayed weak label*: pseudo-positives whose max |I(h)| / (sigma sqrt h)
    stays below m_out are down-weighted towards `floor` (training windows only; never used at test time)."""
    hz = torch.tensor(list(horizons), dtype=impact.dtype, device=impact.device)
    z = (impact.abs() / (sigma.unsqueeze(-1) * hz.sqrt() + 1e-8)).max(-1).values
    return floor + (1.0 - floor) * torch.sigmoid(kappa * (z - m_out))


def executable_alpha_t(impact: torch.Tensor, cost_rt: float) -> torch.Tensor:
    return torch.clamp(impact.abs() - cost_rt, min=0.0).max(-1).values


def monetisable_weights(exec_alpha: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
    return 1.0 + rho * exec_alpha / exec_alpha.mean().clamp_min(1e-8)


# ----------------------------------------------------------------------------- L_trade
def closed_loop_trade_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], horizons: Sequence[int], latency: int,
                           cost_model: CostModel, lam_down: float = 1.0, lam_cvar: float = 0.5, alpha: float = 0.05,
                           lam_turnover: float = 0.01) -> Dict[str, torch.Tensor]:
    r = event_pnl(out, batch, horizons, latency, cost_model, hard=False)
    pnl = r["pnl"]
    u = utility(pnl, lam_down, lam_cvar, alpha)
    prev_w = batch.get("prev_w", torch.zeros_like(r["w"]))
    to = turnover_penalty(r["w"], prev_w)
    return {"loss": -u + lam_turnover * to, "utility": u, "pnl": pnl, "turnover": to, "frozen_rate": r["frozen"].float().mean()}

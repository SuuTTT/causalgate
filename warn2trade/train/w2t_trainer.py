"""Trainer v2 (Warn2Trade track). Subclasses the shared Trainer; fixes from the CCF-A readiness review:

* targets measured from the FILL bar (t + latency), the bar the trade actually starts from:
      I_fill(h) = log m[t+latency+h] - log m[t+latency]   (computed from fut_ret)
  used for the quantile, peak-time, direction and executable-alpha terms; A_ex is zeroed when the entry bar is untradable;
* label_free comparators: no det / dir / peak / svdd terms and an unweighted quantile loss;
* the trading utility is computed with inverse-inclusion weights (normal bars were sub-sampled in TRAIN with probability
  `data.neg_subsample`), so the policy is optimised for the true event base rate, not a 4x-inflated one.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F

from .trainer import Trainer, to_device
from ..models import losses as Lz
from ..models.encoders import liq_transform
from ..backtest import event_pnl, CostModel


def fill_impact(batch: Dict[str, torch.Tensor], horizons, latency: int) -> torch.Tensor:
    cum = torch.cumsum(batch["fut_ret"], 1)
    base = cum[:, latency - 1]
    idx = torch.tensor([latency + h - 1 for h in horizons], device=cum.device)
    return cum[:, idx] - base.unsqueeze(1)


def weighted_utility(pnl: torch.Tensor, w: torch.Tensor, gamma: float, lam_cvar: float = 0.0, alpha: float = 0.05) -> torch.Tensor:
    """Separable mean-semivariance utility  U = E_w[r] - (gamma/2) E_w[min(r, 0)^2]  (+ optional CVaR term).

    Separable means each decision contributes on its own: whether a trade with mean m and downside second moment d is
    worth taking (m > gamma d / 2) does not depend on how many other bars are abstentions. The earlier
    mean - sqrt(LPM2) utility was not separable: at a true 5 % event base rate it ranked "never trade" above every policy
    (smoke test 2026-10-06: zero trades), because sqrt() shrinks the mean faster than the penalty as decisions get sparser.
    """
    w = w / w.sum().clamp_min(1e-12)
    u = (w * pnl).sum() - 0.5 * gamma * (w * torch.clamp(pnl, max=0.0) ** 2).sum()
    if lam_cvar > 0:
        loss = -pnl
        order = torch.argsort(loss.detach(), descending=True)
        cw = torch.cumsum(w[order], 0)
        k = int(torch.searchsorted(cw, torch.tensor(alpha, device=cw.device)).clamp(max=len(cw) - 1))
        nu = loss.detach()[order][k]
        u = u - lam_cvar * (nu + (w * torch.clamp(loss - nu, min=0.0)).sum() / alpha)
    return u


def utility_from_cfg(pnl: torch.Tensor, lc, w: torch.Tensor = None) -> torch.Tensor:
    w = torch.ones_like(pnl) if w is None else w
    return weighted_utility(pnl * float(lc.get("pnl_scale", 100.0)), w, float(lc.get("gamma", 0.1)),
                            float(lc.get("lam_cvar", 0.0)), float(lc.get("alpha", 0.05)))


class SafeCost(CostModel):
    """CostModel with a gradient-safe square-root impact term: d/dx sqrt(x) is infinite at x = 0, and an exactly-zero
    position (saturated abstain gate) then yields inf * 0 = NaN gradients that poison every parameter."""

    def __init__(self, base: CostModel):
        super().__init__(base.commission_bps, base.use_half_spread, base.impact_eta, base.delay_zeta, base.notional)
        self.mult = float(getattr(base, "mult", 1.0))

    def __call__(self, dw_abs, half_spread, sigma, adv_frac, kappa):
        c = self.commission_bps * 1e-4 * dw_abs
        if self.use_half_spread:
            c = c + half_spread * dw_abs
        part = dw_abs * self.notional / (adv_frac.clamp_min(1e-6) * kappa.clamp_min(1.0))
        c = c + self.impact_eta * sigma * torch.sqrt(part.clamp_min(1e-12)) * dw_abs
        c = c + self.delay_zeta * sigma * torch.sqrt(kappa.clamp_min(1.0)) * dw_abs
        return self.mult * c


class W2TTrainer(Trainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cost = SafeCost(self.cost)
        self.neg_keep = float(self.cfg["data"].get("neg_subsample") or 1.0)
        self.label_free = bool(getattr(self.model, "label_free", False))

    def _targets(self, batch):
        I = fill_impact(batch, self.horizons, self.latency)
        c_rt = self.cost.round_trip_estimate(batch["liq"][:, 0], batch["sigma"], batch["liq"][:, 2])
        a_ex = torch.clamp(I.abs() - c_rt.unsqueeze(-1), min=0.0).max(-1).values
        a_ex = a_ex * batch["fut_tradable"][:, self.latency - 1].float()
        return I, a_ex

    def compute_losses(self, out, batch, weights):
        L: Dict[str, torch.Tensor] = {}
        lc = self.tcfg.get("loss", {})
        I, a_ex = self._targets(batch)
        y = batch["y"]
        use_y = not (self.label_free or self.ab.get("no_anomaly_guidance", False))
        if use_y and weights.get("det", 0) > 0:
            L["det"] = Lz.utility_weighted_bce(out["logit"], y, a_ex, rho=float(lc.get("rho", 2.0)), pos_weight=lc.get("pos_weight"))
        if use_y and weights.get("dir", 0) > 0 and "dir_logit" in out:
            L["dir"] = Lz.direction_loss(out["dir_logit"], I, y)
        if use_y and weights.get("svdd", 0) > 0:
            L["svdd"] = Lz.svdd_regularizer(out["z"], self.model.detector.center, y)
        if weights.get("imp", 0) > 0 and not getattr(self.model, "raw_policy", False):
            hz = torch.tensor(self.horizons, device=I.device, dtype=torch.float32)
            target_n = I / (batch["sigma"].clamp_min(1e-6).unsqueeze(-1) * hz.sqrt())
            wgt = None if self.label_free else out["a_prob"].detach() + float(lc.get("imp_floor", 0.1))
            L["imp"] = Lz.pinball_loss(out["q_norm"], target_n, self.taus, weight=wgt)
        if weights.get("liq", 0) > 0 and not getattr(self.model, "raw_policy", False):
            fut = batch["fut_liq"][:, [self.latency + h - 1 for h in self.horizons]]
            wl = None if self.label_free else out["a_prob"].detach() + 0.1
            L["liq"] = Lz.liquidity_loss(out["liq_hat"], liq_transform(fut), weight=wl)
        if use_y and weights.get("peak", 0) > 0:
            L["peak"] = Lz.peak_time_ce(out["tpeak_logit"], I, y)
        if weights.get("utility", 0) > 0:
            r = event_pnl(out, batch, self.horizons, self.latency, self.cost, hard=False)
            iw = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, 1.0 / max(self.neg_keep, 1e-6)))
            L["utility"] = -utility_from_cfg(r["pnl"], lc, iw)
            if weights.get("turnover", 0) > 0:
                L["turnover"] = out["w"].abs().mean()      # gross-exposure (size) penalty; netting is handled at evaluation
        L["total"] = sum(weights[k] * v for k, v in L.items() if k in weights)
        if not torch.is_tensor(L["total"]):
            L["total"] = sum(p.sum() * 0.0 for p in self.model.parameters())
        return L

    @torch.no_grad()
    def validate(self, weights):
        """Model selection on the (un-subsampled) validation fold with the SAME utility the policy is trained on."""
        self.model.eval()
        if weights.get("utility", 0) > 0:
            pnls = [event_pnl(self.model(b), b, self.horizons, self.latency, self.cost, hard=True)["pnl"]
                    for b in (to_device(bb, self.device) for bb in self.loaders["val"])]
            return float(utility_from_cfg(torch.cat(pnls), self.tcfg.get("loss", {})))
        tot, n = 0.0, 0
        for bb in self.loaders["val"]:
            b = to_device(bb, self.device)
            tot += float(self.compute_losses(self.model(b), b, weights)["total"]); n += 1
        return -tot / max(n, 1)

    def _init_center(self):
        if self.label_free:
            return
        super()._init_center()

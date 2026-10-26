"""Warn2Trade track baselines, v2 (fair tuning + published anomaly-to-trade methods).

1. Validation grids (attribute GRID) for the existing event baselines; scripts/w2t_run.py picks the combination that
   maximises the hard-decision trading utility U on the validation fold — the same criterion the model is selected on.
2. `ae_gated` — Al Ridhawi et al. 2026-style (arXiv:2603.19136): an autoencoder's reconstruction error gates a
   multi-horizon forecaster (here DLinear); trade the forecast direction only when the AE flags the window.
3. `dmn_cpd` — Wood, Roberts & Zohren 2022 (JFDS) "slow momentum with fast reversion": an LSTM Deep Momentum Network
   trained on the Sharpe ratio with change-point features. APPROXIMATION: change-point severity/location come from
   Bayesian online change-point detection (Adams & MacKay 2007) on the normalised return window instead of the paper's
   Gaussian-process CPD module; report it as "DMN + BOCPD (CPD-DMN-style)".
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .event_base import BaseBaseline, register_event_baseline
from .rules import ZScoreMomentum, ZScoreReversal
from .dlinear import DLinearBaseline, DLinear
from .detect_then_rule import DetectThenRule

# ----------------------------------------------------------------------------- 1. tuning grids
ZScoreMomentum.GRID = {"thr": [1.5, 2.0, 2.5, 3.0, 4.0], "hold_index": [0, 1, 2, 3, 4, 5]}
ZScoreReversal.GRID = ZScoreMomentum.GRID
DLinearBaseline.GRID = {"k_trade": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]}
DetectThenRule.GRID = {"thr": [0.3, 0.5, 0.7, 0.9], "hold_index": [0, 1, 2, 3, 4, 5], "rule": ["momentum", "reversal"]}


def _to(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


# ----------------------------------------------------------------------------- 2. AE-gated forecaster
class _AE(nn.Module):
    def __init__(self, L: int, F_in: int, h: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(L * F_in, 128), nn.GELU(), nn.Linear(128, h))
        self.dec = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, L * F_in))

    def forward(self, x):
        return self.dec(self.enc(x)).view_as(x)


@register_event_baseline("ae_gated")
class AEGatedForecaster(BaseBaseline):
    name = "ae_gated"
    paper = "Al Ridhawi, Haj Ali, Al Osman 2026 (arXiv:2603.19136)-style AE-gated forecaster"
    GRID = {"gate_q": [0.8, 0.9, 0.95, 0.99], "k_trade": [0.25, 0.5, 1.0, 2.0]}

    def __init__(self, cfg, dims, horizons, device):
        super().__init__(cfg, dims, horizons, device)
        L, Fp = int(cfg["lookback"]), dims["price"]
        self.ae = _AE(L, Fp).to(device)
        self.fc = DLinear(L, Fp, len(horizons)).to(device)
        self.epochs, self.lr = int(cfg.get("epochs", 8)), float(cfg.get("lr", 1e-3))
        self.gate_q, self.k_trade = 0.95, 1.0
        self.err_q = None

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        opt = torch.optim.Adam(list(self.ae.parameters()) + list(self.fc.parameters()), lr=self.lr)
        for _ in range(self.epochs):
            for b in train:
                b = _to(b, self.device)
                x = b["price"]
                l_ae = F.mse_loss(self.ae(x), x)
                l_fc = F.mse_loss(self.fc(x), b["impact"] / (b["sigma"].unsqueeze(-1) + 1e-8))
                opt.zero_grad(); (l_ae + l_fc).backward(); opt.step()
        errs = []
        with torch.no_grad():
            for b in train:
                x = b["price"].to(self.device)
                errs.append(((self.ae(x) - x) ** 2).mean((1, 2)).cpu())
        self.err_ref = torch.cat(errs)

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, sigma = batch["price"], batch["sigma"]
        err = ((self.ae(x) - x) ** 2).mean((1, 2))
        thr = torch.quantile(self.err_ref, self.gate_q).to(x.device)
        pred = self.fc(x)                                                     # sigma units
        hz = torch.tensor(self.horizons, device=x.device, dtype=torch.float32)
        snr = pred.abs() / hz.sqrt()
        h_star = snr.argmax(-1)
        bi = torch.arange(x.shape[0], device=x.device)
        trade = (err > thr) & (snr[bi, h_star] > self.k_trade)
        w = torch.where(trade, torch.sign(pred[bi, h_star]) * self.w_max, torch.zeros_like(err))
        score = (err / (torch.median(self.err_ref).to(x.device) + 1e-12)).clamp(max=1e3)
        return {"a_prob": score, "w": w, "abstain": (~trade).float(), "hold_p": F.one_hot(h_star, len(self.horizons)).float(),
                "kappa_p": self._immediate(x.shape[0])}


# ----------------------------------------------------------------------------- 3. DMN + BOCPD
def bocpd_features(r: torch.Tensor, hazard: float = 1.0 / 30, prior_var: float = 1.0) -> torch.Tensor:
    """Vectorised BOCPD (Gaussian, unknown mean, unit variance) over a (B, L) window of sigma-normalised returns.
    Returns (B, 2): [severity = P(run length < 5) at the last step, location = expected run length / L]."""
    B, L = r.shape
    dev = r.device
    logR = torch.full((B, L + 1), -1e9, device=dev); logR[:, 0] = 0.0
    n = torch.zeros(B, L + 1, device=dev); s = torch.zeros(B, L + 1, device=dev)
    lh, l1h = float(np.log(hazard)), float(np.log(1 - hazard))
    for t in range(L):
        x = r[:, t:t + 1]
        post_var = 1.0 / (1.0 / prior_var + n)
        mu = post_var * s
        pred_var = post_var + 1.0
        logp = -0.5 * (np.log(2 * np.pi) + torch.log(pred_var) + (x - mu) ** 2 / pred_var)
        grow = logR + logp + l1h
        cp = torch.logsumexp(logR + logp + lh, 1, keepdim=True)
        newR = torch.cat([cp, grow[:, :-1]], 1)
        logR = newR - torch.logsumexp(newR, 1, keepdim=True)
        n = torch.cat([torch.zeros(B, 1, device=dev), n[:, :-1] + 1], 1)
        s = torch.cat([torch.zeros(B, 1, device=dev), s[:, :-1] + x], 1)
    R = logR.exp()
    rl = torch.arange(L + 1, device=dev, dtype=torch.float32)
    return torch.stack([R[:, :5].sum(1), (R * rl).sum(1) / L], 1)


class _DMN(nn.Module):
    def __init__(self, F_in: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(F_in + 2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, cpd):
        z = torch.cat([x, cpd.unsqueeze(1).expand(-1, x.shape[1], -1)], -1)
        h, _ = self.lstm(z)
        return torch.tanh(self.head(self.drop(h[:, -1]))).squeeze(-1)


@register_event_baseline("dmn_cpd")
class DMNCPD(BaseBaseline):
    name = "dmn_cpd"
    paper = "Wood, Roberts, Zohren 2022 (JFDS) CPD + Deep Momentum Network; CPD approximated by BOCPD"
    GRID = {"pos_floor": [0.0, 0.1, 0.25, 0.5], "hold_index": [0, 1, 2]}

    def __init__(self, cfg, dims, horizons, device):
        super().__init__(cfg, dims, horizons, device)
        self.net = _DMN(dims["price"]).to(device)
        self.epochs, self.lr = int(cfg.get("epochs", 8)), float(cfg.get("lr", 1e-3))
        self.ret_col = int(cfg.get("ret_col", 0))
        self.pos_floor, self.hold_index = 0.0, 0

    def _cpd(self, x):
        return bocpd_features(x[:, -min(x.shape[1], 63):, self.ret_col])

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for b in train:
                b = _to(b, self.device)
                w = self.net(b["price"], self._cpd(b["price"]))
                nxt = b["fut_ret"][:, 1]                               # return realised after a 1-bar latency
                pnl = w * nxt / (b["sigma"] + 1e-8)
                loss = -pnl.mean() / (pnl.std() + 1e-6)                # Sharpe loss (Lim, Zohren, Roberts 2019)
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["price"]
        cpd = self._cpd(x)
        w = self.net(x, cpd) * self.w_max
        trade = w.abs() > self.pos_floor
        w = torch.where(trade, w, torch.zeros_like(w))
        B = x.shape[0]
        return {"a_prob": cpd[:, 0], "w": w, "abstain": (~trade).float(), "hold_p": self._hold_fixed(B, self.hold_index),
                "kappa_p": self._immediate(B)}

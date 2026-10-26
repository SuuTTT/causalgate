"""Event-level differentiable backtest (training surrogate + evaluation) and a panel portfolio backtest.

Timing convention (latency delta >= 1 bar):
    decision at bar t  ->  fill at the close of bar t + delta  ->  exit at the close of bar t + delta + h
    fut_ret[:, j] = log return from bar t+j to t+j+1 ; fut_liq[:, j] / fut_tradable[:, j] refer to bar t+1+j.
If the entry bar is untradable (halt / limit-hit), the trade is *not* entered (PnL 0, flagged `frozen`).
If the exit bar is untradable the exit is delayed to the next tradable bar within the window.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch

from .costs import CostModel


def _next_tradable_index(tradable: torch.Tensor) -> torch.Tensor:
    """For each (b, j) the smallest j' >= j with tradable[b, j'] (or last index if none)."""
    B, J = tradable.shape
    idx = torch.full((B, J), J - 1, dtype=torch.long, device=tradable.device)
    nxt = torch.full((B,), J - 1, dtype=torch.long, device=tradable.device)
    for j in range(J - 1, -1, -1):
        nxt = torch.where(tradable[:, j], torch.full_like(nxt, j), nxt)
        idx[:, j] = nxt
    return idx


def event_pnl(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], horizons: Sequence[int], latency: int,
              cost_model: CostModel, hard: bool = False) -> Dict[str, torch.Tensor]:
    """Returns dict(pnl (B,), pnl_h (B,H), gross_h (B,H), cost (B,), frozen (B,), w (B,), hold (B,), kappa (B,))."""
    assert latency >= 1, "latency must be >= 1 bar (decisions cannot be filled at the decision bar)"
    fut_ret, fut_liq, fut_trad = batch["fut_ret"], batch["fut_liq"], batch["fut_tradable"]
    sigma = batch["sigma"]
    B, Hmax = fut_ret.shape
    H = len(horizons)
    w, hold_p, kappa_p = out["w"], out["hold_p"], out["kappa_p"]
    n_exec = kappa_p.shape[1]
    exec_grid = torch.arange(1, n_exec + 1, device=w.device, dtype=torch.float32)
    if hard:
        if "abstain" in out:
            w = torch.where(out["abstain"] > 0.5, torch.zeros_like(w), w)
        hold_p = torch.nn.functional.one_hot(hold_p.argmax(-1), H).float()
        kappa = exec_grid[kappa_p.argmax(-1)]
    else:
        kappa = (kappa_p * exec_grid).sum(-1)
    cum = torch.cumsum(fut_ret, 1)
    entry_j = latency - 1
    nxt = _next_tradable_index(fut_trad)
    exit_j = torch.tensor([latency + h - 1 for h in horizons], device=w.device)
    assert int(exit_j.max()) < Hmax, "fut_ret window too short for max(horizons) + latency"
    exit_idx = nxt[:, exit_j]                                            # (B, H) delayed exits
    bi = torch.arange(B, device=w.device).unsqueeze(1)
    ret_h = cum[bi, exit_idx] - cum[:, entry_j].unsqueeze(1)             # (B, H) from fill to exit
    gross_h = w.unsqueeze(1) * ret_h
    dw = w.abs()
    entry_cost = cost_model(dw, fut_liq[:, entry_j, 0], sigma, fut_liq[:, entry_j, 2], kappa)
    exit_cost = cost_model(dw.unsqueeze(1), fut_liq[bi, exit_idx, 0], sigma.unsqueeze(1), fut_liq[bi, exit_idx, 2], torch.ones_like(gross_h))
    pnl_h = gross_h - entry_cost.unsqueeze(1) - exit_cost
    frozen = ~fut_trad[:, entry_j]
    pnl_h = torch.where(frozen.unsqueeze(1), torch.zeros_like(pnl_h), pnl_h)
    pnl = (hold_p * pnl_h).sum(-1)
    hold_bars = (hold_p * torch.tensor(list(horizons), device=w.device, dtype=torch.float32)).sum(-1)
    return {"pnl": pnl, "pnl_h": pnl_h, "gross_h": gross_h, "cost": entry_cost + (hold_p * exit_cost).sum(-1),
            "frozen": frozen, "w": w, "hold": hold_bars, "kappa": kappa}


class EventBacktester:
    """Collects hard-decision event trades from model outputs into a tidy DataFrame."""

    def __init__(self, horizons: Sequence[int], latency: int, cost_model: CostModel):
        self.horizons, self.latency, self.cost_model = list(horizons), latency, cost_model

    @torch.no_grad()
    def collect(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> pd.DataFrame:
        r = event_pnl(out, batch, self.horizons, self.latency, self.cost_model, hard=True)
        g = lambda k: r[k].detach().cpu().numpy()
        b = lambda k: batch[k].detach().cpu().numpy()
        return pd.DataFrame({
            "t": b("t"), "asset": b("asset"), "y": b("y"), "score": out["a_prob"].detach().cpu().numpy(),
            "w": g("w"), "hold": g("hold"), "kappa": g("kappa"), "pnl": g("pnl"), "cost": g("cost"), "frozen": g("frozen"),
            "bars_to_peak": b("bars_to_peak"), "exit_t": b("t") + self.latency + g("hold").astype(int),
            "oracle_alpha": np.abs(b("impact")).max(-1),            # best |move| an oracle could have monetised (for ACR)
        })

    @staticmethod
    def to_bar_returns(trades: pd.DataFrame, n_bars: int, t0: int = 0) -> np.ndarray:
        """Book each trade's net PnL on its exit bar -> a per-bar strategy return series of length n_bars."""
        r = np.zeros(n_bars)
        tr = trades[trades["w"] != 0]
        idx = np.clip(tr["exit_t"].to_numpy() - t0, 0, n_bars - 1)
        np.add.at(r, idx, tr["pnl"].to_numpy())
        return r


class PortfolioBacktester:
    """Vectorised panel backtest: target weights (T, N) -> strategy returns with latency and costs."""

    def __init__(self, cost_model: CostModel, latency: int = 1):
        self.cost_model, self.latency = cost_model, latency

    @torch.no_grad()
    def run(self, weights: np.ndarray, logret: np.ndarray, liq: np.ndarray, sigma: np.ndarray, tradable: np.ndarray,
            kappa: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        T, N = weights.shape
        w_tgt = np.nan_to_num(weights)
        w_eff = np.zeros_like(w_tgt)
        held = np.zeros(N)
        turnover = np.zeros(T)
        cost = np.zeros(T)
        pnl = np.zeros(T)
        kap = np.ones((T, N)) if kappa is None else kappa
        for t in range(T):
            src = t - self.latency
            desired = w_tgt[src] if src >= 0 else held
            can = tradable[t] & ~np.isnan(logret[t])
            new = np.where(can, desired, held)
            dw = np.abs(new - held)
            c = self.cost_model(torch.as_tensor(dw), torch.as_tensor(liq[t, :, 0]), torch.as_tensor(np.nan_to_num(sigma[t])),
                                torch.as_tensor(liq[t, :, 2]), torch.as_tensor(kap[t])).numpy()
            cost[t] = np.nansum(c)
            turnover[t] = dw.sum()
            pnl[t] = np.nansum(held * np.nan_to_num(logret[t])) - cost[t]
            held = new
            w_eff[t] = held
        equity = np.cumsum(pnl)
        return {"ret": pnl, "equity": equity, "turnover": turnover, "cost": cost, "weights": w_eff}

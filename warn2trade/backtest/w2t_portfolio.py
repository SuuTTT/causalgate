"""Position-aware, netted, mark-to-market portfolio evaluation (replaces exit-booked event PnL for all trading tables).

Fixes from the CCF-A readiness review:
  * no position stacking: a new decision on an asset REPLACES that asset's target position (|w| <= w_max per slot);
    an abstention (w == 0) leaves the existing target untouched until its holding period ends;
  * mark-to-market: P&L accrues every bar from held positions, not on the exit bar;
  * execution schedule kappa changes fill TIMING (the position moves toward its target over kappa bars), not only cost;
  * participation cap: per bar |dw| <= cap * adv_frac * kappa / notional, otherwise the order is partially filled;
  * direction-aware tradability (meta can_buy / can_sell), long-only where meta shortable is False, borrow fee on shorts;
  * frozen entry = not filled at that bar but retried on the next tradable bar inside the decision's window
    (default; `entry_policy="cancel"` reproduces the old cancel-on-freeze as a sensitivity);
  * risk ratios are computed on returns aggregated to calendar days (bars_per_day), annualised with days_per_year.

Each asset owns a 1/N capital slot; portfolio return per bar = mean over assets of held_w * r - costs - borrow.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch

from .costs import CostModel


def _bars_per_day(freq: str, bars_per_year: int, days_per_year: int) -> int:
    if freq in ("daily", "1d", "filing-day"):
        return 1
    return max(1, int(round(bars_per_year / days_per_year)))


def build_target_panel(trades: pd.DataFrame, T: int, N: int, t0: int, latency: int, w_max: float):
    """Decisions -> per-bar target weights (T, N) and execution horizon kappa (T, N), with replacement (no stacking)."""
    target = np.full((T, N), np.nan)      # NaN = "no active instruction, flatten"
    kap = np.ones((T, N))
    tr = trades.sort_values("t")
    for t, i, w, h, k in zip(tr["t"].to_numpy(), tr["asset"].to_numpy(), tr["w"].to_numpy(), tr["hold"].to_numpy(), tr["kappa"].to_numpy()):
        if w == 0:
            continue
        a = int(t) - t0 + latency
        b = a + max(1, int(round(h)))
        if a >= T:
            continue
        target[max(a, 0):min(b, T), int(i)] = float(np.clip(w, -w_max, w_max))
        kap[max(a, 0):min(b, T), int(i)] = max(1.0, float(k))
    return np.nan_to_num(target, nan=0.0), kap


def simulate(target: np.ndarray, kap: np.ndarray, logret: np.ndarray, liq: np.ndarray, sigma: np.ndarray, tradable: np.ndarray,
             cost: CostModel, can_buy: Optional[np.ndarray] = None, can_sell: Optional[np.ndarray] = None,
             shortable: Optional[np.ndarray] = None, borrow_bps_per_bar: float = 0.0, participation_cap: float = 0.1,
             entry_policy: str = "retry") -> Dict[str, np.ndarray]:
    """All arrays (T, N) over the evaluation window. logret[t] = log return from bar t-1 to bar t (applied to the position
    held at the end of bar t-1)."""
    T, N = target.shape
    held = np.zeros(N)
    ret = np.zeros(T); cst = np.zeros(T); brw = np.zeros(T); turn = np.zeros(T); gross = np.zeros(T)
    W = np.zeros((T, N))
    blocked = 0
    if shortable is not None:
        target = np.where(shortable[None, :], target, np.clip(target, 0.0, None))
    for t in range(T):
        r = np.nan_to_num(logret[t])
        g = held * r
        gross[t] = g.mean()
        brw[t] = (np.clip(-held, 0.0, None) * borrow_bps_per_bar * 1e-4).mean()
        desired = target[t]
        gap = desired - held
        step = gap / np.maximum(kap[t], 1.0) if entry_policy == "retry" else gap
        # participation cap (fraction of the bar's volume we may take)
        adv = np.clip(np.nan_to_num(liq[t, :, 2], nan=1e-6), 1e-6, None)
        max_dw = participation_cap * adv * np.maximum(kap[t], 1.0) / max(cost.notional, 1e-9)
        step = np.clip(step, -max_dw, max_dw)
        ok = tradable[t].copy()
        if can_buy is not None:
            ok &= ~((step > 0) & ~can_buy[t])
        if can_sell is not None:
            ok &= ~((step < 0) & ~can_sell[t])
        blocked += int(((np.abs(step) > 1e-12) & ~ok).sum())
        step = np.where(ok, step, 0.0)
        c = cost(torch.as_tensor(np.abs(step)), torch.as_tensor(np.nan_to_num(liq[t, :, 0], nan=0.0)),
                 torch.as_tensor(np.nan_to_num(sigma[t], nan=0.0)), torch.as_tensor(adv), torch.as_tensor(np.maximum(kap[t], 1.0))).numpy()
        cst[t] = c.mean()
        turn[t] = np.abs(step).mean()
        ret[t] = gross[t] - cst[t] - brw[t]
        held = held + step
        W[t] = held
    return {"ret": ret, "gross": gross, "cost": cst, "borrow": brw, "turnover": turn, "weights": W, "blocked_orders": blocked}


def daily(x: np.ndarray, bars_per_day: int) -> np.ndarray:
    if bars_per_day <= 1:
        return x
    n = len(x) // bars_per_day * bars_per_day
    out = x[:n].reshape(-1, bars_per_day).sum(1)
    if n < len(x):
        out = np.append(out, x[n:].sum())
    return out


def risk_summary(daily_ret: np.ndarray, days_per_year: int, daily_turnover: Optional[np.ndarray] = None,
                 bench_daily: Optional[np.ndarray] = None) -> Dict[str, float]:
    r = np.asarray(daily_ret, dtype=float)
    out: Dict[str, float] = {}
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    dd = np.sqrt(np.mean(np.clip(r, None, 0.0) ** 2))
    eq = np.cumsum(r)
    mdd = float((eq - np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]).min()) if len(r) else np.nan
    ann = r.mean() * days_per_year
    out["CR"] = float(np.expm1(r.sum()))
    out["ARR"] = float(ann)
    out["AVol"] = float(sd * np.sqrt(days_per_year)) if np.isfinite(sd) else np.nan
    out["SR"] = float(r.mean() / sd * np.sqrt(days_per_year)) if np.isfinite(sd) and sd > 0 else np.nan
    out["SoR"] = float(r.mean() / dd * np.sqrt(days_per_year)) if dd > 0 else np.nan
    out["MDD"] = mdd                                  # in log-return units of the 1/N-slot portfolio
    out["Calmar"] = float(ann / abs(mdd)) if mdd < 0 else np.nan
    nz = r[r != 0]
    out["WinRate"] = float((nz > 0).mean()) if len(nz) else np.nan
    out["active_days"] = float((r != 0).mean()) if len(r) else np.nan
    if daily_turnover is not None:
        out["Turnover"] = float(np.mean(daily_turnover) * days_per_year)
    if bench_daily is not None and len(bench_daily) == len(r):
        a = r - bench_daily
        te = a.std(ddof=1)
        out["IR"] = float(a.mean() / te * np.sqrt(days_per_year)) if te > 0 else np.nan
    out["n_days"] = int(len(r))
    return out


def evaluate_portfolio(trades: pd.DataFrame, panel, fold_test: np.ndarray, sigma_full: np.ndarray, cost: CostModel,
                       latency: int, w_max: float, freq: str, bars_per_year: int, days_per_year: int,
                       participation_cap: float = 0.1, entry_policy: str = "retry") -> Dict[str, Any]:
    t0, t1 = int(fold_test[0]), int(fold_test[-1]) + 1
    T, N = t1 - t0, panel.N
    target, kap = build_target_panel(trades, T, N, t0, latency, w_max)
    lr = np.full((T, N), np.nan)
    lm = np.log(panel.mid)
    lr[:] = lm[t0:t1] - lm[t0 - 1:t1 - 1]
    meta = panel.meta or {}
    sl = lambda k: (np.asarray(meta[k])[t0:t1] if k in meta and meta[k] is not None else None)
    sim = simulate(target, kap, lr, panel.liq[t0:t1], sigma_full[t0:t1], panel.tradable[t0:t1], cost,
                   sl("can_buy"), sl("can_sell"), np.asarray(meta["shortable"]) if "shortable" in meta else None,
                   float(meta.get("borrow_bps_per_bar", 0.0)), participation_cap, entry_policy)
    bpd = _bars_per_day(freq, bars_per_year, days_per_year)
    d_ret = daily(sim["ret"], bpd)
    d_to = daily(sim["turnover"], bpd)
    bench = daily(np.nanmean(np.nan_to_num(lr, nan=0.0) * panel.tradable[t0:t1], axis=1), bpd)
    out = risk_summary(d_ret, days_per_year, d_to, bench)
    out["cost_share"] = float(sim["cost"].sum() / max(np.abs(sim["gross"]).sum(), 1e-12))
    out["blocked_orders"] = int(sim["blocked_orders"])
    out["max_abs_slot_weight"] = float(np.abs(sim["weights"]).max()) if sim["weights"].size else 0.0
    return {"summary": out, "daily_ret": d_ret, "bench_daily": bench}

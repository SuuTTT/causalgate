"""Date-shift probe (Leakage Audit Card item 3).

Take the frozen decision series w_{t,i} (T, N) that a pipeline produced and execute it shifted by k bars,
k in {-K..K}: k > 0 delays the signal (w_{t-k} executed at the fill bar of t), k < 0 lets it peek (w_{t+|k|}).
An honest edge decays smoothly with delay and improves (or stays) when allowed to peek; a leaked edge collapses
as soon as the signal leaves the bar it was leaked from. Gross and net (flat bps) PnL are supported; the fill
convention is the engine's: decision at t, fill at the close of t + latency, exit at t + latency + hold.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np

from warn2trade.backtest import metrics as WM
from ..eval.metrics import date_shift_asymmetry


def shifted_bar_returns(weight: np.ndarray, mid: np.ndarray, latency: int, hold: int, shift: int, cost_bps: float = 0.0,
                        n_assets_norm: bool = True) -> np.ndarray:
    """Per-bar strategy returns (booked on the exit bar) of the weight series shifted by `shift` bars."""
    w = np.nan_to_num(np.asarray(weight, dtype=np.float64))
    T, N = w.shape
    ws = np.zeros_like(w)
    if shift > 0:
        ws[shift:] = w[:-shift]
    elif shift < 0:
        ws[:shift] = w[-shift:]
    else:
        ws = w
    lp = np.log(np.asarray(mid, dtype=np.float64))
    r = np.zeros(T)
    for t in range(T - latency - hold):
        pnl = ws[t] * (lp[t + latency + hold] - lp[t + latency]) - 2.0 * cost_bps * 1e-4 * np.abs(ws[t])
        r[t + latency + hold] += np.nansum(pnl)
    return r / (N if n_assets_norm else 1.0)


def date_shift_profile(weight: np.ndarray, mid: np.ndarray, latency: int, hold: int, bars_per_year: int,
                       shifts: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3), cost_bps: float = 0.0) -> Dict[str, object]:
    prof = {int(k): WM.sharpe(shifted_bar_returns(weight, mid, latency, hold, int(k), cost_bps), bars_per_year) for k in shifts}
    out: Dict[str, object] = {"profile": prof}
    out.update(date_shift_asymmetry(prof))
    return out

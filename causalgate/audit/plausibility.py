"""Sharpe plausibility (Leakage Audit Card item 5): PSR, DSR with the declared trial count, haircut, and the two
flags of the plan: `placebo_leak` (the unchanged pipeline earns on martingale-difference surrogates) and
`implausible_sharpe` (DSR p < 0.05 AND SR above the 99th percentile of the placebo distribution AND above a
declared ceiling). Uses warn2trade.backtest.metrics.{probabilistic_sharpe, deflated_sharpe} unchanged.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
from scipy import stats

from warn2trade.backtest import metrics as WM


def sharpe_plausibility(r: np.ndarray, bars_per_year: int, n_trials: int = 1, sr_var_across_trials: float = 0.0,
                        placebo_srs: Optional[Sequence[float]] = None, sr_ceiling: float = 3.0) -> Dict[str, float]:
    """`sr_var_across_trials` is the variance of the *annualised* Sharpe ratio across the declared trials; it is converted
    to per-bar units (divided by bars_per_year) before WM.deflated_sharpe, which works per bar."""
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    sr_var_bar = float(sr_var_across_trials) / float(bars_per_year)
    out: Dict[str, float] = {"SR": WM.sharpe(r, bars_per_year) if r.size > 2 else float("nan"), "n_bars": int(r.size), "n_trials": int(n_trials)}
    out["PSR"] = WM.probabilistic_sharpe(r) if r.size > 2 else float("nan")
    out["DSR"] = WM.deflated_sharpe(r, int(n_trials), sr_var_bar) if r.size > 2 else float("nan")
    out["DSR_p_value"] = float(1.0 - out["DSR"]) if np.isfinite(out["DSR"]) else float("nan")
    # Harvey & Liu (2015)-style haircut: shrink the t-stat for the number of trials (Bonferroni-type, conservative)
    if r.size > 2 and np.isfinite(out["SR"]):
        t_stat = out["SR"] / np.sqrt(bars_per_year) * np.sqrt(r.size)
        t_adj = haircut_t(t_stat, n_trials)
        out["SR_haircut"] = float(np.sign(out["SR"]) * t_adj / np.sqrt(r.size) * np.sqrt(bars_per_year))
    else:
        out["SR_haircut"] = float("nan")
    out["sr_var_across_trials_annual"] = float(sr_var_across_trials)
    p99 = float("nan"); placebo_leak = False
    if placebo_srs is not None:
        p = np.asarray(placebo_srs, dtype=np.float64); p = p[np.isfinite(p)]
        if p.size:
            p99 = float(np.quantile(p, 0.99))
            placebo_leak = bool(p.size >= 5 and (p > 0).mean() >= 0.9 and p.mean() > 0)
    out["placebo_p99"] = p99
    out["placebo_leak"] = bool(placebo_leak)
    out["implausible_sharpe"] = bool(np.isfinite(out["DSR_p_value"]) and out["DSR_p_value"] < 0.05 and np.isfinite(out["SR"])
                                     and out["SR"] > sr_ceiling and (not np.isfinite(p99) or out["SR"] > p99))
    return out


def haircut_t(t_stat: float, n_trials: int) -> float:
    """Bonferroni haircut of |t| for n_trials, computed in the survival-function (log) domain so it stays finite for
    large t; clamped to [0, |t|]."""
    a = abs(float(t_stat)); m = max(int(n_trials), 1)
    if not np.isfinite(a):
        return a
    log_p_adj = np.log(2.0) + stats.norm.logsf(a) + np.log(m)
    if log_p_adj >= 0.0:
        return 0.0
    log_half = log_p_adj - np.log(2.0)
    if log_half > -700:
        t_adj = float(stats.norm.isf(np.exp(log_half)))
    else:                                                   # isf underflow: solve logsf(x) = log_half by bisection
        lo, hi = 0.0, a
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if stats.norm.logsf(mid) > log_half:
                lo = mid
            else:
                hi = mid
        t_adj = 0.5 * (lo + hi)
    return float(min(max(t_adj, 0.0), a))

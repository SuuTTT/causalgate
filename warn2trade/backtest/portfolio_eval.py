"""Portfolio-level economics for warning signals (label-scarcity track; new file, shared modules untouched).

Why: event-level PnL booked on the exit bar is neither netted nor mark-to-market nor capital-normalised, so Sharpe-type
ratios computed from it are not a portfolio's. Here every strategy is a (T, N) target-weight path run through one
sequential netted backtest with latency, tradability, the shared CostModel (per capital slot), and borrow cost on shorts.

Strategies
    netted_from_trades   the event trades of a rule (learned policy or the shared score-to-trade adapter), netted per
                         asset (sum of active trade weights, clipped to [-1, 1] of the asset's 1/N capital slot)
    derisk_overlay       long-only, implementable without borrow: equal-weight the universe, sell an asset while a warning
                         is active (score >= val threshold, for `hold` bars), re-enter after. Evaluated as ACTIVE return
                         against the same equal-weight portfolio run through the same costs (benchmark-adjusted).
Inference
    timing_permutation_pvalue  per-asset circular shifts of the signal (keeps count, run lengths, cross-section) -> null
                               distribution of the statistic; p = (1 + #{null >= obs}) / (n + 1)
    sharpe_moments / deflated_sharpe_from_moments  per-run moments so DSR can be computed over all trials of a sweep
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd
import torch
from scipy import stats

from .costs import CostModel


def _panel_slices(panel, t0: int, n_bars: int):
    T = panel.T
    t1 = min(T, t0 + n_bars)
    lm = np.log(panel.mid)
    logret = np.full((t1 - t0, panel.N), np.nan)
    lo = max(t0, 1)
    logret[lo - t0:] = lm[lo:t1] - lm[lo - 1:t1 - 1]
    return logret, panel.liq[t0:t1].astype(np.float64), panel.tradable[t0:t1]


def netted_backtest(target: np.ndarray, logret: np.ndarray, liq: np.ndarray, sigma: np.ndarray, tradable: np.ndarray,
                    cost_model: CostModel, latency: int = 1, borrow_bps_annual: float = 0.0, bars_per_year: int = 252,
                    slot_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """target[u, i] = decision-time portfolio weight. The position decided at u is filled at the close of u + latency and
    earns returns from u + latency + 1 on (same convention as the event backtest). Untradable bars freeze the position.
    Costs are evaluated per capital slot (weight * slot_scale, slot_scale = N for 1/N slots) then scaled back."""
    n, N = target.shape
    held = np.zeros(N)
    held_path = np.zeros((n, N))
    dw = np.zeros((n, N))
    pnl_gross = np.zeros(n)
    r = np.nan_to_num(logret)
    for u in range(n):
        pnl_gross[u] = float(held @ r[u])
        src = u - latency
        desired = target[src] if src >= 0 else np.zeros(N)
        can = tradable[u] & np.isfinite(logret[u])
        new = np.where(can, desired, held)
        dw[u] = np.abs(new - held)
        held = new
        held_path[u] = held
    c = cost_model(torch.as_tensor(dw * slot_scale), torch.as_tensor(np.nan_to_num(liq[..., 0])), torch.as_tensor(np.nan_to_num(sigma)),
                   torch.as_tensor(np.nan_to_num(liq[..., 2], nan=1.0)), torch.ones(n, N, dtype=torch.float64)).numpy() / slot_scale
    cost = np.nansum(c, axis=1)
    borrow = np.clip(-held_path, 0, None).sum(1) * borrow_bps_annual * 1e-4 / bars_per_year
    borrow = np.concatenate([[0.0], borrow[:-1]])                      # borrow accrues on the position held during the bar
    ret = pnl_gross - cost - borrow
    return {"ret": ret, "gross": pnl_gross, "cost": cost, "borrow": borrow, "turnover": dw.sum(1), "held": held_path}


def trades_to_target(trades: pd.DataFrame, t0: int, n_bars: int, N: int, cap: float = 1.0) -> np.ndarray:
    """Decision-time targets: a trade decided at t with weight w and holding `hold` bars targets w on decisions t..t+hold-1.
    Overlapping trades on one asset are summed then clipped to the slot ([-cap, cap]); portfolio weight = slot / N."""
    tgt = np.zeros((n_bars, N))
    tr = trades[trades["w"] != 0]
    for t, a, w, h in zip(tr["t"].to_numpy(), tr["asset"].to_numpy(), tr["w"].to_numpy(), tr["hold"].to_numpy()):
        lo = int(t) - t0
        hi = min(n_bars, lo + max(1, int(round(h))))
        if 0 <= lo < n_bars:
            tgt[lo:hi, int(a)] += float(w)
    return np.clip(tgt, -cap, cap) / N


def fire_matrix(trades: pd.DataFrame, t0: int, n_bars: int, N: int, thr: float) -> np.ndarray:
    f = np.zeros((n_bars, N), dtype=bool)
    sel = trades[trades["score"] >= thr]
    lo = sel["t"].to_numpy().astype(int) - t0
    ok = (lo >= 0) & (lo < n_bars)
    f[lo[ok], sel["asset"].to_numpy().astype(int)[ok]] = True
    return f


def overlay_targets(fire: np.ndarray, valid: np.ndarray, hold: int):
    """Equal-weight benchmark over valid assets; overlay drops an asset while a warning fired within the last `hold` bars."""
    n, N = fire.shape
    active = np.zeros_like(fire)
    for k in range(max(1, hold)):
        active[k:] |= fire[: n - k] if k else fire
    nvalid = np.maximum(valid.sum(1, keepdims=True), 1)
    bench = valid / nvalid
    return bench, bench * (~active)


def _sr(r: np.ndarray, bpy: int) -> float:
    sd = np.nanstd(r, ddof=1)
    return float(np.nanmean(r) / sd * np.sqrt(bpy)) if sd > 0 else float("nan")


def timing_permutation_pvalue(signal: np.ndarray, stat_fn: Callable[[np.ndarray], float], n_perm: int, seed: int, min_shift: int = 5) -> Dict[str, float]:
    obs = stat_fn(signal)
    if not np.isfinite(obs) or n_perm <= 0:
        return {"obs": obs, "p": float("nan"), "null_mean": float("nan")}
    rng = np.random.default_rng(seed)
    n = signal.shape[0]
    null = []
    for _ in range(n_perm):
        shifts = rng.integers(min(min_shift, n - 1), max(min_shift + 1, n - min_shift), size=signal.shape[1])
        perm = np.stack([np.roll(signal[:, i], int(s)) for i, s in enumerate(shifts)], 1)
        null.append(stat_fn(perm))
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    p = float((1 + (null >= obs).sum()) / (len(null) + 1)) if len(null) else float("nan")
    return {"obs": float(obs), "p": p, "null_mean": float(null.mean()) if len(null) else float("nan")}


def sharpe_moments(r: np.ndarray) -> Dict[str, float]:
    r = r[np.isfinite(r)]
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return {"sr_bar": float("nan"), "n_bars": float(len(r)), "skew": float("nan"), "kurt": float("nan")}
    return {"sr_bar": float(np.mean(r) / np.std(r, ddof=1)), "n_bars": float(len(r)),
            "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False))}


def deflated_sharpe_from_moments(sr: float, n: float, skew: float, kurt: float, sr_trials: np.ndarray) -> float:
    """DSR (Bailey & Lopez de Prado 2014) from per-bar moments; sr_trials = per-bar SRs of all configurations tried."""
    sr_trials = np.asarray(sr_trials, dtype=np.float64)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    if not np.isfinite(sr) or n < 3 or len(sr_trials) < 2:
        return float("nan")
    euler = 0.5772156649
    k = len(sr_trials)
    z1, z2 = stats.norm.ppf(1 - 1.0 / k), stats.norm.ppf(1 - 1.0 / (k * np.e))
    sr_star = np.sqrt(np.var(sr_trials, ddof=1)) * ((1 - euler) * z1 + euler * z2)
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def evaluate_portfolio(panel, trades: pd.DataFrame, t0: int, n_bars: int, thr: float, cost_model: CostModel, latency: int,
                       bars_per_year: int, sigma: np.ndarray, cfg: Dict, seed: int, with_null: bool) -> Dict[str, float]:
    """All portfolio metrics for one (rule, cost) evaluation of one fold. sigma = realised vol (T, N) of the full panel."""
    from . import metrics as M
    N = panel.N
    logret, liq, trad = _panel_slices(panel, t0, n_bars)
    n = logret.shape[0]
    sig = sigma[t0:t0 + n]
    borrow = float(cfg.get("borrow_bps_annual", 300.0))
    hold = int(cfg.get("overlay_hold", 5))
    n_perm = int(cfg.get("n_perm", 200)) if with_null else 0
    out: Dict[str, float] = {}

    tgt = trades_to_target(trades, t0, n, N, float(cfg.get("slot_cap", 1.0)))
    run = lambda T_: netted_backtest(T_, logret, liq, sig, trad, cost_model, latency, borrow, bars_per_year, float(N))
    net = run(tgt)
    tr_stats = M.summarize_trading(net["ret"], bars_per_year, net["turnover"])
    out.update({f"net_{k}": v for k, v in tr_stats.items()})
    out["net_borrow_share"] = float(net["borrow"].sum() / max(np.abs(net["gross"]).sum(), 1e-12))
    out.update({f"net_{k}": v for k, v in sharpe_moments(net["ret"]).items()})
    if n_perm:
        pv = timing_permutation_pvalue(tgt, lambda T_: _sr(run(T_)["ret"], bars_per_year), n_perm, seed)
        out["net_SR_perm_p"], out["net_SR_null_mean"] = pv["p"], pv["null_mean"]

    valid = np.isfinite(panel.mid[t0:t0 + n])
    fire = fire_matrix(trades, t0, n, N, thr)
    bench_t, over_t = overlay_targets(fire, valid, hold)
    b = netted_backtest(bench_t, logret, liq, sig, trad, cost_model, latency, 0.0, bars_per_year, float(N))
    o = netted_backtest(over_t, logret, liq, sig, trad, cost_model, latency, 0.0, bars_per_year, float(N))
    active = o["ret"] - b["ret"]
    out.update({"ovl_SR": _sr(o["ret"], bars_per_year), "bench_SR": _sr(b["ret"], bars_per_year),
                "ovl_active_ARR": float(np.nanmean(active) * bars_per_year), "ovl_IR": _sr(active, bars_per_year),
                "ovl_MDD": M.max_drawdown(o["ret"]), "bench_MDD": M.max_drawdown(b["ret"]),
                "ovl_turnover": float(o["turnover"].sum()), "ovl_exposure": float(over_t.sum(1).mean())})
    out.update({f"ovl_active_{k}": v for k, v in sharpe_moments(active).items()})
    if n_perm:
        def ir_of(F):
            _, ot = overlay_targets(F, valid, hold)
            return _sr(netted_backtest(ot, logret, liq, sig, trad, cost_model, latency, 0.0, bars_per_year, float(N))["ret"] - b["ret"], bars_per_year)
        pv = timing_permutation_pvalue(fire, ir_of, n_perm, seed + 1)
        out["ovl_IR_perm_p"], out["ovl_IR_null_mean"] = pv["p"], pv["null_mean"]
    return out

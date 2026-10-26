"""Dual evaluation metric matrix.

A. Detection & signal quality : AUC-PR, AUC-ROC, best-F1, F1@thr, P@K, R@K, Warning Lead Time (WLT),
                                False Discovery Penalty (FDP), Executable Alpha Ratio (EAR)
B. Trading & execution        : CR, ARR, SR, SoR, MDD, Calmar, IR, Win rate, Turnover, Probabilistic/Deflated Sharpe
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


# ----------------------------------------------------------------------------- A. detection
def auc_pr(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def auc_roc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def best_f1(y: np.ndarray, s: np.ndarray):
    if y.sum() == 0:
        return float("nan"), 0.5
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    k = int(np.nanargmax(f1[:-1]))
    return float(f1[k]), float(thr[k])


def f1_at(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    pred = s >= thr
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float((~pred & (y == 1)).sum())
    prec = tp / max(tp + fp, 1e-12); rec = tp / max(tp + fn, 1e-12)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    tn = float((~pred & (y == 0)).sum())
    prec0 = tn / max(tn + fn, 1e-12); rec0 = tn / max(tn + fp, 1e-12)
    f1_0 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1, "macro_f1": 0.5 * (f1 + f1_0)}


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    top = np.argsort(-s)[:k]
    return float(y[top].mean()) if k > 0 else float("nan")


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def warning_lead_time(y: np.ndarray, s: np.ndarray, bars_to_peak: np.ndarray, thr: float) -> Dict[str, float]:
    """WLT = mean signed bars from a true-positive warning to the event's peak impact; share of early warnings."""
    tp = (s >= thr) & (y == 1) & ~np.isnan(bars_to_peak)
    if tp.sum() == 0:
        return {"wlt": float("nan"), "wlt_early_rate": float("nan")}
    btp = bars_to_peak[tp]
    return {"wlt": float(btp.mean()), "wlt_early_rate": float((btp > 0).mean())}


def false_discovery_penalty(pnl: np.ndarray, traded: np.ndarray, y: np.ndarray) -> float:
    """FDP = sum of losses on trades triggered by false warnings / total gross |PnL| of all trades. In [0, 1]."""
    fp = traded & (y == 0)
    denom = np.abs(pnl[traded]).sum()
    return float(np.clip(-pnl[fp], 0, None).sum() / denom) if denom > 0 else float("nan")


def executable_alpha_ratio(pnl: np.ndarray, traded: np.ndarray, y: np.ndarray) -> float:
    """EAR = share of traded true anomalies whose net PnL after latency and costs is positive."""
    tp = traded & (y == 1)
    return float((pnl[tp] > 0).mean()) if tp.sum() > 0 else float("nan")


# ----------------------------------------------------------------------------- B. trading
def cumulative_return(r: np.ndarray) -> float:
    return float(np.expm1(np.nansum(r)))


def annualized_return(r: np.ndarray, bars_per_year: int) -> float:
    return float(np.nanmean(r) * bars_per_year)


def annualized_vol(r: np.ndarray, bars_per_year: int) -> float:
    return float(np.nanstd(r, ddof=1) * np.sqrt(bars_per_year))


def sharpe(r: np.ndarray, bars_per_year: int, rf: float = 0.0) -> float:
    ex = r - rf / bars_per_year
    sd = np.nanstd(ex, ddof=1)
    return float(np.nanmean(ex) / sd * np.sqrt(bars_per_year)) if sd > 0 else float("nan")


def sortino(r: np.ndarray, bars_per_year: int, target: float = 0.0) -> float:
    ex = r - target / bars_per_year
    dd = np.sqrt(np.nanmean(np.clip(ex, None, 0.0) ** 2))
    return float(np.nanmean(ex) / dd * np.sqrt(bars_per_year)) if dd > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    eq = np.exp(np.nancumsum(r))
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def calmar(r: np.ndarray, bars_per_year: int) -> float:
    mdd = abs(max_drawdown(r))
    return float(annualized_return(r, bars_per_year) / mdd) if mdd > 0 else float("nan")


def information_ratio(r: np.ndarray, bench: np.ndarray, bars_per_year: int) -> float:
    active = r - bench
    te = np.nanstd(active, ddof=1)
    return float(np.nanmean(active) / te * np.sqrt(bars_per_year)) if te > 0 else float("nan")


def win_rate(r: np.ndarray) -> float:
    nz = r[r != 0]
    return float((nz > 0).mean()) if len(nz) else float("nan")


def turnover(to: np.ndarray, bars_per_year: int) -> float:
    return float(np.nanmean(to) * bars_per_year)


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P[SR_true > sr_benchmark] accounting for skew/kurtosis and sample size (per-bar SR)."""
    T = len(r)
    sd = np.nanstd(r, ddof=1)
    if not np.isfinite(sd) or sd <= 0 or T < 3:
        return float("nan")
    sr = np.nanmean(r) / sd
    g3, g4 = stats.skew(r, nan_policy="omit"), stats.kurtosis(r, fisher=False, nan_policy="omit")
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom)) if denom > 0 else float("nan")


def deflated_sharpe(r: np.ndarray, n_trials: int, sr_var_across_trials: float) -> float:
    """DSR (Bailey & Lopez de Prado 2014): PSR against the expected max SR of n_trials draws (per-bar units)."""
    if n_trials <= 1 or sr_var_across_trials <= 0:
        return probabilistic_sharpe(r, 0.0)
    euler = 0.5772156649
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_star = np.sqrt(sr_var_across_trials) * ((1 - euler) * z1 + euler * z2)
    return probabilistic_sharpe(r, sr_star)


def summarize_trading(r: np.ndarray, bars_per_year: int, to: Optional[np.ndarray] = None, bench: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {"CR": cumulative_return(r), "ARR": annualized_return(r, bars_per_year), "AVol": annualized_vol(r, bars_per_year),
           "SR": sharpe(r, bars_per_year), "SoR": sortino(r, bars_per_year), "MDD": max_drawdown(r), "Calmar": calmar(r, bars_per_year),
           "WinRate": win_rate(r), "PSR": probabilistic_sharpe(r)}
    if to is not None:
        out["Turnover"] = turnover(to, bars_per_year)
    if bench is not None:
        out["IR"] = information_ratio(r, bench, bars_per_year)
    return out


def summarize_detection(trades: pd.DataFrame, k_frac: float = 0.01) -> Dict[str, float]:
    y, s = trades["y"].to_numpy(), trades["score"].to_numpy()
    f1, thr = best_f1(y, s)
    k = max(1, int(k_frac * len(y)))
    out = {"AUC_PR": auc_pr(y, s), "AUC_ROC": auc_roc(y, s), "F1_best": f1, "thr": thr,
           "P@K": precision_at_k(y, s, k), "R@K": recall_at_k(y, s, k)}
    out.update({f"{k_}_@thr": v for k_, v in f1_at(y, s, thr).items()})
    out.update(warning_lead_time(y, s, trades["bars_to_peak"].to_numpy(), thr))
    traded = trades["w"].to_numpy() != 0
    pnl = trades["pnl"].to_numpy()
    out["FDP"] = false_discovery_penalty(pnl, traded, y)
    out["EAR"] = executable_alpha_ratio(pnl, traded, y)
    out["trade_rate"] = float(traded.mean())
    out["frozen_rate"] = float(trades["frozen"].to_numpy()[traded].mean()) if traded.any() else float("nan")
    out["avg_hold"] = float(trades.loc[traded, "hold"].mean()) if traded.any() else float("nan")
    return out


# ----------------------------------------------------------------------------- C. label-scarcity & monetisation metrics
def _events_from_trades(trades: pd.DataFrame, max_gap: int = 1) -> np.ndarray:
    """Group consecutive oracle-positive bars of the same asset into events. Returns event ids (-1 for y == 0)."""
    eid = np.full(len(trades), -1, dtype=np.int64)
    order = np.lexsort((trades["t"].to_numpy(), trades["asset"].to_numpy()))
    a, t, y = trades["asset"].to_numpy()[order], trades["t"].to_numpy()[order], trades["y"].to_numpy()[order]
    cur, prev_a, prev_t = -1, None, None
    for pos, idx in enumerate(order):
        if y[pos] != 1:
            continue
        if prev_a != a[pos] or prev_t is None or t[pos] - prev_t > max_gap + 1:
            cur += 1
        eid[idx] = cur
        prev_a, prev_t = a[pos], t[pos]
    return eid


def warning_to_alpha_delay(trades: pd.DataFrame, thr: float, latency: int = 1, max_gap: int = 1) -> Dict[str, float]:
    """WAD = latency - bars_to_peak at the *first* above-threshold warning of each event (bars).

    WAD <= 0: the fill lands at or before the peak impact (alpha still capturable); WAD > 0: priced in before the fill.
    Also reports the share of events warned ahead of the peak and the event-level recall at this threshold.
    """
    eid = _events_from_trades(trades, max_gap)
    n_events = int(eid.max() + 1) if len(eid) else 0
    if n_events == 0:
        return {"WAD_mean": float("nan"), "WAD_median": float("nan"), "WAD_ahead_rate": float("nan"), "event_recall": float("nan"), "n_events": 0}
    s, t, btp = trades["score"].to_numpy(), trades["t"].to_numpy(), trades["bars_to_peak"].to_numpy()
    wads = []
    for e in range(n_events):
        rows = np.where((eid == e) & (s >= thr) & ~np.isnan(btp))[0]
        if len(rows) == 0:
            continue
        first = rows[np.argmin(t[rows])]
        wads.append(latency - btp[first])
    wads = np.asarray(wads, dtype=np.float64)
    if len(wads) == 0:
        return {"WAD_mean": float("nan"), "WAD_median": float("nan"), "WAD_ahead_rate": float("nan"), "event_recall": 0.0, "n_events": n_events}
    return {"WAD_mean": float(wads.mean()), "WAD_median": float(np.median(wads)), "WAD_ahead_rate": float((wads <= 0).mean()),
            "event_recall": float(len(wads) / n_events), "n_events": n_events}


def alpha_capture_ratio(trades: pd.DataFrame) -> float:
    """ACR = net PnL realised on traded true anomalies / oracle |peak impact| of those same bars (requires `oracle_alpha`)."""
    if "oracle_alpha" not in trades:
        return float("nan")
    tp = (trades["w"].to_numpy() != 0) & (trades["y"].to_numpy() == 1)
    denom = trades.loc[tp, "oracle_alpha"].abs().sum()
    return float(trades.loc[tp, "pnl"].sum() / denom) if denom > 0 else float("nan")


def label_efficiency_score(k_values: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    """Label Efficiency Score from a label-budget curve (K labelled anchors per class -> metric).

    LES_slope : OLS slope of the metric against log2(K + 1)  (gain per doubling of the label budget)
    AULC      : area under the curve over log2(K + 1), normalised by the range -> mean metric across budgets
    K_80      : smallest K reaching 80% of the best value on the curve
    """
    k = np.asarray(k_values, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(k) & np.isfinite(v)
    k, v = k[ok], v[ok]
    if len(k) < 2:
        return {"LES_slope": float("nan"), "AULC": float("nan"), "K_80": float("nan")}
    order = np.argsort(k)
    k, v = k[order], v[order]
    x = np.log2(k + 1.0)
    slope = float(np.polyfit(x, v, 1)[0])
    aulc = float(np.trapz(v, x) / max(x[-1] - x[0], 1e-12))
    target = 0.8 * np.nanmax(v)
    reach = np.where(v >= target)[0]
    return {"LES_slope": slope, "AULC": aulc, "K_80": float(k[reach[0]]) if len(reach) else float("nan")}


def cost_resilience(bps: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    """Breakeven cost (largest bps with a positive metric, linearly interpolated) and the slope per bps."""
    b, v = np.asarray(bps, dtype=np.float64), np.asarray(values, dtype=np.float64)
    ok = np.isfinite(b) & np.isfinite(v)
    b, v = b[ok], v[ok]
    if len(b) < 2:
        return {"breakeven_bps": float("nan"), "slope_per_bps": float("nan")}
    order = np.argsort(b)
    b, v = b[order], v[order]
    slope = float(np.polyfit(b, v, 1)[0])
    if (v > 0).all():
        be = float("inf")
    elif (v <= 0).all():
        be = 0.0
    else:
        i = int(np.where(v > 0)[0][-1])
        if i + 1 < len(b):
            be = float(b[i] + (b[i + 1] - b[i]) * v[i] / (v[i] - v[i + 1]))
        else:
            be = float(b[i])
    return {"breakeven_bps": be, "slope_per_bps": slope}

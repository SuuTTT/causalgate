"""Warn2Trade detection and bridge metrics, corrected per the CCF-A readiness review.

* Thresholds are chosen on the VALIDATION fold only (theta_F1 = val-F1-optimal score threshold) and applied to test.
* Warnings = test rows with score >= theta_val. Bridge metrics condition on warnings, not on trades:
    FDP  = sum over FALSE warnings of (-pnl)_+   /   sum over ALL warnings of |pnl|      (pnl = 0 when the policy abstained)
    EAR  = share of DETECTED true events whose first trade inside [first warning, t_peak] has net pnl > 0
           (an event that was warned but not traded counts as not monetised)
    coverage = traded warnings / warnings
* Warning Lead Time is per EVENT: WLT_e = t_peak - t_first_warning over detected events (signed bars), plus the event
  detection rate. Events come from panel.meta["events"] (exogenous tables) or are reconstructed from label runs.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def events_from_panel(panel) -> List[Dict[str, Any]]:
    ev = (panel.meta or {}).get("events")
    if ev:
        g = lambda e, *ks: next(e[k] for k in ks if k in e)        # accept both key conventions used in this repo
        return [dict(asset_index=int(g(e, "asset_index", "asset")), t_onset=int(g(e, "t_onset", "onset_t")),
                     t_peak=int(g(e, "t_peak", "peak_t")), direction=float(e.get("direction", 0.0))) for e in ev]
    out = []
    y, btp = panel.labels, panel.bars_to_peak
    for i in range(panel.N):
        col = y[:, i].astype(bool)
        if not col.any():
            continue
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            b = btp[s, i]
            t_peak = int(s + b) if np.isfinite(b) else int(e - 1)
            out.append(dict(asset_index=i, t_onset=int(s), t_peak=t_peak, direction=0.0))
    return out


def val_threshold(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float(np.quantile(s, 0.99))
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def detection_metrics(test: pd.DataFrame, theta: float, k_frac: float = 0.01) -> Dict[str, float]:
    y, s = test["y"].to_numpy(), test["score"].to_numpy()
    out = {"theta_val": theta}
    out["AUC_PR"] = float(average_precision_score(y, s)) if 0 < y.sum() else np.nan
    out["AUC_ROC"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan
    out["pos_rate"] = float(y.mean())
    pred = s >= theta
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float((~pred & (y == 1)).sum())
    prec = tp / max(tp + fp, 1e-12); rec = tp / max(tp + fn, 1e-12)
    out.update({"P@thr": prec, "R@thr": rec, "F1@thr": 2 * prec * rec / max(prec + rec, 1e-12), "warn_rate": float(pred.mean())})
    k = max(1, int(k_frac * len(y)))
    top = np.argsort(-s)[:k]
    out["P@K"] = float(y[top].mean()); out["R@K"] = float(y[top].sum() / max(y.sum(), 1))
    return out


def bridge_metrics(test: pd.DataFrame, theta: float, events: List[Dict[str, Any]], t_lo: int, t_hi: int,
                   lead_window: int) -> Dict[str, float]:
    s = test["score"].to_numpy()
    warned = s >= theta
    pnl = np.where(test["w"].to_numpy() != 0, test["pnl"].to_numpy(), 0.0)
    traded = test["w"].to_numpy() != 0
    y = test["y"].to_numpy()
    out: Dict[str, float] = {}
    denom = np.abs(pnl[warned]).sum()
    out["FDP"] = float(np.clip(-pnl[warned & (y == 0)], 0, None).sum() / denom) if denom > 0 else np.nan
    out["coverage"] = float(traded[warned].mean()) if warned.any() else np.nan
    out["trade_rate"] = float(traded.mean())
    out["frozen_rate"] = float(test["frozen"].to_numpy()[traded].mean()) if traded.any() else np.nan
    # event level
    by_asset = {a: g.sort_values("t") for a, g in test.assign(_w=warned, _pnl=pnl, _tr=traded).groupby("asset")}
    n_ev = det = mon = 0
    leads = []
    for e in events:
        if not (t_lo <= e["t_peak"] < t_hi):
            continue
        g = by_asset.get(e["asset_index"])
        n_ev += 1
        if g is None:
            continue
        win = g[(g["t"] >= e["t_onset"] - lead_window) & (g["t"] <= e["t_peak"])]
        ww = win[win["_w"]]
        if len(ww) == 0:
            continue
        det += 1
        t_first = int(ww["t"].iloc[0])
        leads.append(e["t_peak"] - t_first)
        tr = win[(win["t"] >= t_first) & win["_tr"]]
        if len(tr) and float(tr["_pnl"].iloc[0]) > 0:
            mon += 1
    out["n_events"] = n_ev
    out["event_detect_rate"] = det / n_ev if n_ev else np.nan
    out["WLT"] = float(np.mean(leads)) if leads else np.nan
    out["WLT_early_rate"] = float(np.mean(np.array(leads) > 0)) if leads else np.nan
    out["EAR"] = mon / det if det else np.nan
    return out


def threshold_mismatch(val: pd.DataFrame, test: pd.DataFrame, theta_f1: float) -> Dict[str, float]:
    """T8: utility-optimal threshold on val (sum of pnl of trades whose score >= theta) vs the F1-optimal one."""
    s, pnl = val["score"].to_numpy(), np.where(val["w"].to_numpy() != 0, val["pnl"].to_numpy(), 0.0)
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 60)))
    util = [pnl[s >= g].sum() for g in grid]
    theta_u = float(grid[int(np.argmax(util))]) if len(grid) else theta_f1
    st, pt = test["score"].to_numpy(), np.where(test["w"].to_numpy() != 0, test["pnl"].to_numpy(), 0.0)
    return {"theta_U": theta_u, "theta_F1": theta_f1, "test_pnl_at_thetaU": float(pt[st >= theta_u].sum()),
            "test_pnl_at_thetaF1": float(pt[st >= theta_f1].sum())}

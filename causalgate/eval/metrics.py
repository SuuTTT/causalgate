"""CausalGate metrics (plan Sec. 7), on top of warn2trade.backtest.metrics (which stays untouched):

    leakage_premium        LP_M(m, D) = M_{P-O} - M_{P-C}; channel-wise LP^{(l)} from one-switch runs
    deployability_gap      DG = M_backtest - M_replay (zero for a causal pipeline, Prop. 1)
    ts_auc                 ADIA Structural Break Real-Time Edition metric: per-step cross-sectional AUC weighted by n_pos * n_neg
    vus_pr                 Volume Under the PR Surface over label buffers (Paparrizos et al., VLDB 2022; our implementation
                           with linear-ramp buffers, no point-adjust)
    point_adjust           the PA convention (Xu et al. 2018 / Su et al. 2019) -- used ONLY to reproduce published numbers
    event_f1               event-wise F1 without PA
    kendall_tau_rankings   rank agreement between two protocol rankings
    alarm_rate_error       |realised alarm rate - alpha|
    placebo_rank           where the real Sharpe sits in the placebo distribution
    date_shift_asymmetry   collapse / peek-gain indices of a date-shift profile
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

from warn2trade.backtest import metrics as WM

CHANNELS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")


# --------------------------------------------------------------------------------------- leakage premium / DG
def leakage_premium(m_published: Mapping[str, float], m_causal: Mapping[str, float], keys: Sequence[str] = ("SR", "AUC_PR", "EAR", "F1_headline", "ARR")) -> Dict[str, float]:
    """LP_M = M_{P-O} - M_{P-C}. The F1 key is `F1_headline` (PA-F1 at the test-best threshold under a PA protocol,
    otherwise the F1 of the pipeline's own realised alarms), so LP_F1 carries L2/L3/L4. `F1_best` (test-label-optimal
    threshold, no PA, identical construction in every protocol) is an oracle diagnostic only; pass it explicitly."""
    out = {}
    for k in keys:
        a, b = m_published.get(k, np.nan), m_causal.get(k, np.nan)
        out[f"LP_{k}"] = float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
    return out


def channel_premiums(m_causal: Mapping[str, float], one_switch: Mapping[str, Mapping[str, float]], key: str = "SR") -> Dict[str, float]:
    """one_switch[channel] = metrics with only that channel toggled from P-C. LP^{(l)} = M_l - M_{P-C}."""
    base = m_causal.get(key, np.nan)
    return {ch: (float(one_switch[ch].get(key, np.nan) - base) if ch in one_switch else float("nan")) for ch in CHANNELS}


def deployability_gap(m_backtest: Mapping[str, float], m_replay: Mapping[str, float], keys: Sequence[str] = ("SR", "ARR", "n_trades")) -> Dict[str, float]:
    return {f"DG_{k}": float(m_backtest.get(k, np.nan) - m_replay.get(k, np.nan)) for k in keys}


# --------------------------------------------------------------------------------------- detection
def ts_auc(steps: Iterable[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Time-stratified AUC: weighted mean over steps of the cross-sectional AUC-ROC, weights n_pos * n_neg
    (ADIA Lab Structural Break Challenge, Real-Time Edition, 2026). `steps` yields (y_step, score_step)."""
    num, den = 0.0, 0.0
    for y, s in steps:
        y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * roc_auc_score(y, s); den += w
    return float(num / den) if den > 0 else float("nan")


def ts_auc_from_panel(y: np.ndarray, s: np.ndarray) -> float:
    """(T, N) labels and scores -> TS-AUC over bars (each bar is one cross-sectional step)."""
    return ts_auc((y[t], s[t]) for t in range(y.shape[0]))


def point_adjust(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """PA: if any bar of a contiguous anomalous segment is flagged, the whole segment counts as detected (L4)."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(bool).copy()
    T = len(y)
    t = 0
    while t < T:
        if y[t] == 1:
            u = t
            while u < T and y[u] == 1:
                u += 1
            if pred[t:u].any():
                pred[t:u] = True
            t = u
        else:
            t += 1
    return pred


def f1_with_optional_pa(y: np.ndarray, s: np.ndarray, thr: float, pa: bool = False) -> Dict[str, float]:
    pred = np.asarray(s) >= thr
    if pa:
        pred = point_adjust(y, pred)
    y = np.asarray(y).astype(int)
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float((~pred & (y == 1)).sum())
    p = tp / max(tp + fp, 1e-12); r = tp / max(tp + fn, 1e-12)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}


def _segments(y: np.ndarray) -> List[Tuple[int, int]]:
    y = np.asarray(y).astype(int)
    segs, t, T = [], 0, len(y)
    while t < T:
        if y[t] == 1:
            u = t
            while u < T and y[u] == 1:
                u += 1
            segs.append((t, u)); t = u
        else:
            t += 1
    return segs


def pa_event_panel(Y: np.ndarray, S: np.ndarray, M: np.ndarray, thr: float, alarms: Optional[np.ndarray] = None) -> Dict[str, float]:
    """PA-F1 and event-wise F1 on a (T, N) panel, segmenting PER ASSET along time (never across the cross-section).
    Y labels, S scores, M observed-cell mask (unobserved cells: y = 0, no alarm; they also break segments because a
    segment is a run of observed positive bars). Alarms = S >= thr unless `alarms` (T, N bool) is given. TP/FP/FN and
    segment hit counts are pooled over assets."""
    Y = np.asarray(Y).astype(int); M = np.asarray(M).astype(bool)
    if alarms is None:
        S = np.asarray(S, dtype=np.float64)
        P = np.isfinite(S) & (np.nan_to_num(S, nan=-np.inf) >= thr)
    else:
        P = np.asarray(alarms).astype(bool)
    Y = np.where(M, Y, 0); P = P & M
    tp = fp = fn = 0.0; hit = nseg = 0; in_seg = n_alarm = 0.0
    for j in range(Y.shape[1]):
        y, p = Y[:, j], P[:, j]
        pa = point_adjust(y, p)
        tp += float((pa & (y == 1)).sum()); fp += float((pa & (y == 0)).sum()); fn += float((~pa & (y == 1)).sum())
        segs = _segments(y)
        nseg += len(segs); hit += sum(1 for a, b in segs if p[a:b].any())
        in_seg += float((p & (y == 1)).sum()); n_alarm += float(p.sum())
    pr = tp / max(tp + fp, 1e-12); rc = tp / max(tp + fn, 1e-12)
    out = {"PA_precision": pr, "PA_recall": rc, "PA_f1": 2 * pr * rc / max(pr + rc, 1e-12)}
    if nseg == 0:
        out.update({"event_precision": float("nan"), "event_recall": float("nan"), "event_f1": float("nan"), "n_events": 0})
    else:
        rec = hit / nseg; prec = in_seg / n_alarm if n_alarm > 0 else 0.0
        out.update({"event_precision": prec, "event_recall": rec, "event_f1": 2 * prec * rec / max(prec + rec, 1e-12), "n_events": nseg})
    return out


def event_f1(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Event-wise F1 without PA: recall = share of anomalous segments with >= 1 alarm; precision = share of alarm
    bars that fall inside some anomalous segment."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(bool)
    segs = _segments(y)
    if not segs:
        return {"event_precision": float("nan"), "event_recall": float("nan"), "event_f1": float("nan"), "n_events": 0}
    hit = sum(1 for a, b in segs if pred[a:b].any())
    rec = hit / len(segs)
    prec = float((pred & (y == 1)).sum() / max(pred.sum(), 1)) if pred.any() else 0.0
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"event_precision": prec, "event_recall": rec, "event_f1": f1, "n_events": len(segs)}


def _soft_ap(w: np.ndarray, s: np.ndarray) -> float:
    """Average precision with soft (in [0,1]) labels: precision/recall over descending-score thresholds. Tied scores
    form ONE threshold (cumulative counts taken at the last index of each distinct score), the construction of
    sklearn.metrics.precision_recall_curve; on binary labels this equals average_precision_score and does not depend
    on the row order inside a tie group."""
    w = np.asarray(w, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    tot = w.sum()
    if tot <= 0 or len(w) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, w = s[order], w[order]
    idx = np.r_[np.nonzero(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    tp = np.cumsum(w)[idx]
    prec = tp / (idx + 1)
    rec = tp / tot
    drec = np.diff(np.r_[0.0, rec])
    return float((prec * drec).sum())


def vus_pr(y: np.ndarray, s: np.ndarray, max_buffer: Optional[int] = None, n_buffers: int = 8) -> float:
    """VUS-PR: mean over ramp half-widths h in a grid of n_buffers distinct integers in [0, max_buffer] of the
    range-based AP with labels extended by a linear ramp of half-width h on both sides of every anomalous segment
    (no point-adjust). Each half-width is counted once (h = 0, i.e. plain AP, is one grid point, not two).
    Default max_buffer = max(4, median segment length) so that 1-bar events still get a non-trivial volume.
    Univariate y, s of shape (T,); NaN scores are excluded."""
    y = np.asarray(y).astype(float); s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    if ok.sum() == 0 or y[ok].sum() == 0:
        return float("nan")
    T = len(y)
    segs = _segments(y)
    if max_buffer is None:
        max_buffer = int(max(4, np.median([b - a for a, b in segs]))) if segs else 4
    halves = np.unique(np.linspace(0, max_buffer, n_buffers).round().astype(int))
    aps = []
    for half in halves:
        w = y.copy()
        half = int(half)
        if half > 0:
            for a, b in segs:
                for d in range(1, half + 1):
                    v = 1.0 - d / (half + 1)
                    if a - d >= 0:
                        w[a - d] = max(w[a - d], v)
                    if b - 1 + d < T:
                        w[b - 1 + d] = max(w[b - 1 + d], v)
        aps.append(_soft_ap(w[ok], s[ok]))
    return float(np.nanmean(aps))


def vus_pr_panel(y: np.ndarray, s: np.ndarray, **kw) -> float:
    """Mean VUS-PR over assets of a (T, N) panel (assets without positives are skipped)."""
    vals = [vus_pr(y[:, i], s[:, i], **kw) for i in range(y.shape[1]) if y[:, i].sum() > 0]
    return float(np.nanmean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------------------- rankings / rates
def kendall_tau_rankings(score_a: Mapping[str, float], score_b: Mapping[str, float]) -> Dict[str, float]:
    """Kendall tau between two method -> metric maps (higher = better in both)."""
    keys = [k for k in score_a if k in score_b and np.isfinite(score_a[k]) and np.isfinite(score_b[k])]
    if len(keys) < 3:
        return {"kendall_tau": float("nan"), "p_value": float("nan"), "n": len(keys)}
    tau, p = stats.kendalltau([score_a[k] for k in keys], [score_b[k] for k in keys])
    return {"kendall_tau": float(tau), "p_value": float(p), "n": len(keys)}


def rank_methods(scores: Mapping[str, float]) -> Dict[str, int]:
    keys = sorted([k for k in scores if np.isfinite(scores[k])], key=lambda k: -scores[k])
    return {k: r + 1 for r, k in enumerate(keys)}


def alarm_rate_error(alarms: np.ndarray, alpha: float) -> float:
    a = np.asarray(alarms, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(abs(a.mean() - alpha)) if a.size else float("nan")


def placebo_rank(sr_real: float, sr_placebo: Sequence[float]) -> Dict[str, float]:
    p = np.asarray(sr_placebo, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0 or not np.isfinite(sr_real):
        return {"placebo_mean": float("nan"), "placebo_p99": float("nan"), "placebo_p_value": float("nan"), "placebo_frac_positive": float("nan")}
    return {"placebo_mean": float(p.mean()), "placebo_p99": float(np.quantile(p, 0.99)), "placebo_p_value": float((p >= sr_real).mean()),
            "placebo_frac_positive": float((p > 0).mean()), "placebo_t_stat": float(p.mean() / max(p.std(ddof=1) / np.sqrt(p.size), 1e-12)) if p.size > 1 else float("nan")}


def date_shift_asymmetry(profile: Mapping[int, float]) -> Dict[str, float]:
    """profile: shift k (bars) -> Sharpe. k < 0 = signal allowed to peek k bars; k > 0 = signal delayed k bars.
    collapse_ratio = 1 - max_{k != 0} SR_k / SR_0 (both sides);
    delay_collapse = 1 - max_{k>0} SR_k / SR_0  (near 1: the edge vanishes as soon as the signal is delayed);
    peek_ratio     = max_{k<0} SR_k / SR_0      (> 1: peeking improves the edge, as it should for an honest signal);
    peek_gain = mean_{k<0} SR_k - SR_0; delay_decay = SR_0 - mean_{k>0} SR_k.
    collapse_flag = SR_0 > 0 and delay_collapse > 0.75 and peeking does NOT improve the edge (no k<0 points, or
    max_{k<0} SR_k <= SR_0); requires at least one k<0 point (peek_tested). The two sides are tested independently: a one-bar causal edge that grows with peeking
    is not flagged; an edge that lives only at k = 0 on both sides is."""
    sr0 = profile.get(0, np.nan)
    neg = [v for k, v in profile.items() if k < 0 and np.isfinite(v)]
    pos = [v for k, v in profile.items() if k > 0 and np.isfinite(v)]
    off = neg + pos
    good = np.isfinite(sr0) and sr0 > 0
    out = {"SR_0": float(sr0), "peek_gain": float(np.mean(neg) - sr0) if neg and np.isfinite(sr0) else float("nan"),
           "delay_decay": float(sr0 - np.mean(pos)) if pos and np.isfinite(sr0) else float("nan"),
           "collapse_ratio": float(1.0 - max(off) / sr0) if off and good else float("nan"),
           "delay_collapse": float(1.0 - max(pos) / sr0) if pos and good else float("nan"),
           "peek_ratio": float(max(neg) / sr0) if neg and good else float("nan")}
    peek_improves = bool(neg) and good and max(neg) > sr0
    out["peek_tested"] = bool(neg)
    # without peek (k<0) shifts the peek side is untested -> never flagged (explicit, not vacuous)
    out["collapse_flag"] = bool(neg and np.isfinite(out["delay_collapse"]) and out["delay_collapse"] > 0.75 and not peek_improves)
    return out


def summarize_detection_no_pa(y: np.ndarray, s: np.ndarray, thr: Optional[float] = None) -> Dict[str, float]:
    """AUC-PR / AUC-ROC / F1 without point-adjust, plus event-wise F1 at `thr` (best-F1 threshold if None)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    out = {"AUC_PR": WM.auc_pr(y[ok], s[ok]) if ok.any() else float("nan"), "AUC_ROC": WM.auc_roc(y[ok], s[ok]) if ok.any() else float("nan")}
    if thr is None:
        f1, thr = WM.best_f1(y[ok], s[ok]) if ok.any() else (float("nan"), np.inf)
        out["F1_best"] = f1
    out["thr"] = float(thr)
    out.update(f1_with_optional_pa(y[ok], s[ok], thr, pa=False))
    out.update({f"PA_{k}": v for k, v in f1_with_optional_pa(y[ok], s[ok], thr, pa=True).items()})
    out.update(event_f1(y[ok], s[ok] >= thr))
    return out

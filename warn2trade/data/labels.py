"""Weak anomaly labels, impact curves and the executable-alpha weight that closes the loop.

All functions operate on (T, N) arrays with time on axis 0. Forward-looking quantities are
NaN in the last `max(horizons)` bars; EventDataset never samples there.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def log_returns(mid: np.ndarray) -> np.ndarray:
    r = np.full_like(mid, np.nan, dtype=np.float64)
    r[1:] = np.log(mid[1:]) - np.log(mid[:-1])
    return r


def rolling_mean_std(x: np.ndarray, window: int):
    """Causal rolling mean/std along axis 0 (NaN-aware, min_periods = window // 2)."""
    T = x.shape[0]
    mean = np.full_like(x, np.nan, dtype=np.float64)
    std = np.full_like(x, np.nan, dtype=np.float64)
    minp = max(2, window // 2)
    for t in range(T):
        lo = max(0, t - window + 1)
        seg = x[lo : t + 1]
        cnt = np.sum(~np.isnan(seg), axis=0)
        ok = cnt >= minp
        with np.errstate(invalid="ignore"):
            m = np.nanmean(seg, axis=0)
            s = np.nanstd(seg, axis=0)
        mean[t] = np.where(ok, m, np.nan)
        std[t] = np.where(ok, s, np.nan)
    return mean, std


def realized_vol(mid: np.ndarray, window: int = 60) -> np.ndarray:
    _, s = rolling_mean_std(log_returns(mid), window)
    return s


def impact_curves(mid: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
    """I[t, i, k] = log(mid[t + h_k]) - log(mid[t]) : forward cumulative log return at each horizon."""
    T, N = mid.shape
    out = np.full((T, N, len(horizons)), np.nan, dtype=np.float64)
    lm = np.log(mid)
    for k, h in enumerate(horizons):
        out[: T - h, :, k] = lm[h:] - lm[: T - h]
    return out


def abnormal_move_labels(
    mid: np.ndarray,
    volume: np.ndarray,
    horizon: int = 5,
    k_ret: float = 3.0,
    k_vol: float = 3.0,
    window: int = 60,
    lead: int = 3,
) -> np.ndarray:
    """Rule-based weak label used when no curated event table exists.

    A bar (t, i) is an anomaly onset if the forward |cum return| over `horizon` exceeds k_ret * sigma * sqrt(horizon)
    AND the forward max volume z-score exceeds k_vol. Labels are then extended `lead` bars backwards so the detector
    learns to fire *before* the move. This label is forward-looking by construction (it defines the event); it is
    a training target only and never enters the backtest as a feature.
    """
    r = log_returns(mid)
    sigma = realized_vol(mid, window)
    vmean, vstd = rolling_mean_std(np.log1p(volume), window)
    vz = (np.log1p(volume) - vmean) / (vstd + 1e-8)
    T, N = mid.shape
    fwd = np.full((T, N), np.nan)
    fwd_vz = np.full((T, N), np.nan)
    lm = np.log(mid)
    fwd[: T - horizon] = lm[horizon:] - lm[: T - horizon]
    for t in range(T - horizon):
        fwd_vz[t] = np.nanmax(vz[t + 1 : t + 1 + horizon], axis=0)
    onset = (np.abs(fwd) > k_ret * sigma * np.sqrt(horizon)) & (fwd_vz > k_vol)
    onset = np.nan_to_num(onset, nan=False).astype(bool)
    labels = onset.copy()
    for d in range(1, lead + 1):
        labels[:-d] |= onset[d:]
    return labels.astype(np.int8)


def bars_to_peak_from_labels(labels: np.ndarray, impact: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
    """Ground-truth signed distance to peak impact for labelled bars (NaN elsewhere).

    For a labelled bar we take argmax_h |I(h)| over the horizon grid. This is >= min(horizons) for early warnings.
    Dataset builders with a curated event table (true onset / peak timestamps) should overwrite this with the exact
    t_peak - t, which can be negative for late bars. See labels in datasets/*.py.
    """
    hz = np.asarray(horizons)
    k = np.nanargmax(np.where(np.isnan(impact), -np.inf, np.abs(impact)), axis=-1)
    btp = hz[k].astype(np.float64)
    btp[labels == 0] = np.nan
    btp[np.isnan(impact).all(-1)] = np.nan
    return btp


def executable_alpha(impact: np.ndarray, cost_est: np.ndarray) -> np.ndarray:
    """A_ex[t, i] = max_h (|I(h)| - c_hat)_+ : the best net-of-cost absolute move an oracle could monetise.

    Used as the per-sample weight in the utility-weighted detection loss. `cost_est` is (T, N) round-trip cost in
    log-return units (e.g. 2 * (commission + half-spread + impact estimate)).
    """
    net = np.abs(impact) - cost_est[..., None]
    return np.nanmax(np.clip(net, 0.0, None), axis=-1)

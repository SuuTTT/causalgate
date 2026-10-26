"""Shared helpers for the Qlib-bundle builders (csi300_limit, qlib_alpha158).

Qlib .bin layout: <root>/features/<code_lower>/<field>.day.bin = float32 little-endian; element 0 = start index into
calendars/day.txt, elements 1.. = values on consecutive calendar days (NaN where missing). Prices are ADJUSTED
(raw * $factor); volume is adjusted (raw / $factor), so raw price = $close / $factor and price * volume is invariant.

Universe gating (why `assemble_universe` exists)
    EventDataset only drops a decision point (t, i) when mid / sigma / impact / price_feat[t] is NaN, but it reads the
    look-back window price_feat[t-L+1 : t+1] and the forward window mid[t+1 : t+hmax] unchecked. A naive "mid = NaN
    when not a member" would therefore feed NaN into models (look-back) and into the differentiable backtest (forward
    returns of the last member bars). We therefore:
      * keep mid finite on member bars and for `tail` bars after every membership spell (forward window of the last
        member bars; the stock is still listed and could be exited; after delisting mid is the last close, flat),
      * set price_feat = NaN on those tail bars (so they are never decision points; tail bars that are reached by a
        member bar's look-back cannot exist, see below),
      * keep price_feat finite (0 = cross-sectional median after rank normalisation) on all other bars,
      * if a tail bar would fall inside the look-back of a later valid decision bar (the asset re-enters the universe
        shortly after leaving), the gap between the two spells is BRIDGED (treated as in-universe). This is the only
        place where the universe uses information after t (the re-entry date); the number of bridged asset-bars is
        reported in meta["n_bridged_bars"] and is tiny.
"""
from __future__ import annotations
import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------------------- I/O
def read_calendar(bin_root: str) -> pd.DatetimeIndex:
    with open(os.path.join(bin_root, "calendars", "day.txt")) as f:
        return pd.DatetimeIndex(pd.to_datetime([ln.strip() for ln in f if ln.strip()]))


def read_field(bin_root: str, code: str, field: str, n_cal: int) -> np.ndarray:
    out = np.full(n_cal, np.nan)
    p = os.path.join(bin_root, "features", code.lower(), f"{field}.day.bin")
    if not os.path.exists(p):
        return out
    a = np.fromfile(p, dtype="<f4")
    if a.size < 2:
        return out
    st = int(a[0])
    v = a[1:].astype(np.float64)
    n = min(len(v), n_cal - st)
    if n > 0:
        out[st:st + n] = v[:n]
    return out


def load_fields(bin_root: str, codes: Sequence[str], fields: Sequence[str], i0: int, i1: int, n_cal: int) -> Dict[str, np.ndarray]:
    out = {f: np.full((i1 - i0, len(codes)), np.nan) for f in fields}
    for j, c in enumerate(codes):
        for f in fields:
            out[f][:, j] = read_field(bin_root, c, f, n_cal)[i0:i1]
    return out


def has_features(bin_root: str, code: str) -> bool:
    return os.path.exists(os.path.join(bin_root, "features", code.lower(), "close.day.bin"))


def parse_membership(path: str) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    spells: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    with open(path) as f:
        for ln in f:
            p = ln.strip().split("\t")
            if len(p) < 3:
                continue
            spells.setdefault(p[0], []).append((pd.Timestamp(p[1]), pd.Timestamp(p[2])))
    return spells


def membership_mask(spells, codes: Sequence[str], dates: pd.DatetimeIndex) -> np.ndarray:
    m = np.zeros((len(dates), len(codes)), dtype=bool)
    d = dates.values
    for j, c in enumerate(codes):
        for (a, b) in spells.get(c, []):
            m[:, j] |= (d >= np.datetime64(a)) & (d <= np.datetime64(b))
    return m


# ------------------------------------------------------------------------------------------------------- features
def _df(x):
    return pd.DataFrame(x)


def _r(x, w, minp, how):
    return getattr(_df(x).rolling(w, min_periods=minp), how)().to_numpy()


def ffill(x: np.ndarray) -> np.ndarray:
    return _df(x).ffill().to_numpy()


def compute_features(o, h, l, c, v, dv, traded, extended: bool = False) -> Tuple[List[np.ndarray], List[str]]:
    """Alpha158-lite on adjusted OHLCV. Every value at t uses bars <= t only (trailing pandas windows).
    Non-trading bars (suspension / missing) are NaN in the inputs; rolling stats skip them."""
    c = np.where(traded, c, np.nan); o = np.where(traded, o, np.nan)
    h = np.where(traded, h, np.nan); l = np.where(traded, l, np.nan)
    cf = ffill(c)                                         # last traded close (for returns across suspensions)
    lc = np.log(cf)
    r1 = np.full_like(lc, np.nan); r1[1:] = lc[1:] - lc[:-1]
    r1 = np.where(traded, r1, np.nan)
    sig = _r(r1, 60, 20, "std")
    sig = np.where(sig > 1e-6, sig, np.nan)
    eps = 1e-12

    def ret_h(k):
        out = np.full_like(lc, np.nan); out[k:] = lc[k:] - lc[:-k]
        return out / (sig * np.sqrt(k))

    lv = np.log1p(np.where(traded, v, np.nan))
    ldv = np.log1p(np.where(traded, dv, np.nan))
    vz20 = (lv - _r(lv, 20, 10, "mean")) / (_r(lv, 20, 10, "std") + 1e-6)
    vz60 = (lv - _r(lv, 60, 20, "mean")) / (_r(lv, 60, 20, "std") + 1e-6)
    dvz20 = (ldv - _r(ldv, 20, 10, "mean")) / (_r(ldv, 20, 10, "std") + 1e-6)
    hl = np.log(h / l) / sig
    cpos = (c - l) / (h - l + eps * c)
    cpos = np.where(h > l, cpos, 0.5)
    oc = np.log(c / o) / sig
    prev_c = np.full_like(cf, np.nan); prev_c[1:] = cf[:-1]
    gap = np.log(o / prev_c) / sig
    ush = (h - np.maximum(o, c)) / c / sig
    lsh = (np.minimum(o, c) - l) / c / sig
    mx20, mn20 = _r(c, 20, 10, "max"), _r(c, 20, 10, "min")
    mx60, mn60 = _r(c, 60, 20, "max"), _r(c, 60, 20, "min")
    feats = [ret_h(1), ret_h(5), ret_h(10), ret_h(20), vz20, vz60, dvz20, hl, cpos, oc, gap, ush, lsh,
             np.log(c / mx20) / sig, np.log(c / mn20) / sig, np.log(c / mx60) / sig, np.log(c / mn60) / sig,
             np.log(c / _r(c, 5, 3, "mean")) / sig, np.log(c / _r(c, 20, 10, "mean")) / sig,
             np.log(c / _r(c, 60, 20, "mean")) / sig, _r(r1, 20, 10, "std") / sig]
    names = ["ret1_z", "ret5_z", "ret10_z", "ret20_z", "vol_z20", "vol_z60", "dvol_z20", "hl_range_z", "close_pos",
             "oc_z", "gap_z", "upper_shadow_z", "lower_shadow_z", "dist_max20_z", "dist_min20_z", "dist_max60_z",
             "dist_min60_z", "ma5_z", "ma20_z", "ma60_z", "vol_ratio_20_60"]
    dlv = np.full_like(lv, np.nan); dlv[1:] = lv[1:] - lv[:-1]
    feats.append(_df(r1).rolling(20, min_periods=10).corr(_df(dlv)).to_numpy())          # Alpha158 CORD-like
    names.append("corr_ret_dvol20")
    pos = _r(np.clip(r1, 0, None), 20, 10, "sum"); tot = _r(np.abs(r1), 20, 10, "sum")
    feats.append(pos / (tot + 1e-12)); names.append("sump20")
    if extended:
        hh20, ll20 = _r(h, 20, 10, "max"), _r(l, 20, 10, "min")
        feats += [ret_h(60), _r(r1, 20, 10, "skew"),
                  _bars_since_ext(c, 20, True), _bars_since_ext(c, 20, False),
                  _r((r1 > 0).astype(float) + np.where(np.isnan(r1), np.nan, 0.0), 20, 10, "mean"),
                  _r(np.where(traded, v, np.nan), 20, 10, "std") / (_r(np.where(traded, v, np.nan), 20, 10, "mean") + 1e-12),
                  _r(np.abs(r1) * np.where(traded, v, np.nan), 20, 10, "std") / (_r(np.abs(r1) * np.where(traded, v, np.nan), 20, 10, "mean") + 1e-12),
                  (c - ll20) / (hh20 - ll20 + 1e-12 * c)]
        names += ["ret60_z", "skew20", "imax20", "imin20", "cntp20", "vstd20", "wvma20", "rsv20"]
    return feats, names


def _bars_since_ext(x: np.ndarray, w: int, is_max: bool) -> np.ndarray:
    """(bars since the trailing-w max (or min) of x) / w  -- Alpha158 IMAX / IMIN. NaN if < w/2 finite values."""
    from numpy.lib.stride_tricks import sliding_window_view
    T, N = x.shape
    out = np.full((T, N), np.nan)
    fill = -np.inf if is_max else np.inf
    xf = np.where(np.isfinite(x), x, fill)
    cnt = _roll_count(np.isfinite(x), w)
    for j0 in range(0, N, 128):
        win = sliding_window_view(xf[:, j0:j0 + 128], w, axis=0)          # (T-w+1, n, w)
        k = win.argmax(-1) if is_max else win.argmin(-1)
        out[w - 1:, j0:j0 + 128] = (w - 1 - k) / float(w)
    out[cnt < w // 2] = np.nan
    return out


def rank_normalise(feats: List[np.ndarray], ref: np.ndarray) -> np.ndarray:
    """Per day, map each value to its mid-rank percentile within the SAME-DAY reference set (point-in-time index
    members with a finite value), scaled to [-1, 1]. Non-members are placed against the members' distribution.
    NaN (no data / warm-up) -> 0 (= cross-sectional median)."""
    T, N = ref.shape
    out = np.zeros((T, N, len(feats)), dtype=np.float32)
    for k, x in enumerate(feats):
        x = np.where(np.isfinite(x), x, np.nan)
        for t in range(T):
            row = x[t]
            fin = np.isfinite(row)
            m = ref[t] & fin
            n = int(m.sum())
            if n < 2:
                continue
            s = np.sort(row[m])
            vals = row[fin]
            lo = np.searchsorted(s, vals, "left"); hi = np.searchsorted(s, vals, "right")
            out[t, fin, k] = (2.0 * ((lo + hi) / 2.0) / n - 1.0).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------------------- liquidity
def corwin_schultz_half_spread(h, l, c, traded, window: int = 20) -> np.ndarray:
    """Corwin & Schultz (2012) two-day estimator on (t-1, t) with the overnight-gap adjustment of day t's range,
    negatives set to 0, trailing `window`-bar mean, divided by 2 (half-spread), clipped to [1e-5, 0.05]."""
    h = np.where(traded, h, np.nan); l = np.where(traded, l, np.nan)
    cprev = np.full_like(c, np.nan); cprev[1:] = ffill(np.where(traded, c, np.nan))[:-1]
    up = np.where(cprev < l, l - cprev, 0.0); dn = np.where(cprev > h, cprev - h, 0.0)
    up = np.nan_to_num(up); dn = np.nan_to_num(dn)
    h2, l2 = h - up + dn, l - up + dn
    h1 = np.full_like(h, np.nan); l1 = np.full_like(l, np.nan)
    h1[1:], l1[1:] = h[:-1], l[:-1]
    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
    gamma = np.log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), np.clip(s, 0.0, None), np.nan)
    s = _r(s, window, 5, "mean") / 2.0
    return np.clip(s, 1e-5, 0.05)


def fill_asset(x: np.ndarray, default: float) -> np.ndarray:
    x = ffill(np.where(np.isfinite(x), x, np.nan))
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=0) if np.isfinite(x).any() else np.full(x.shape[1], np.nan)
    glob = float(np.nanmedian(med)) if np.isfinite(med).any() else default
    med = np.where(np.isfinite(med), med, glob)
    return np.where(np.isfinite(x), x, med[None, :])


def build_liq(h, l, c, v, dv, traded) -> np.ndarray:
    hs = corwin_schultz_half_spread(h, l, c, traded)
    dvt = np.where(traded, dv, np.nan)
    depth = np.where(traded, dv, 0.0) / _r(dvt, 60, 10, "median")
    vt = np.where(traded, v, np.nan)
    adv = np.where(traded, v, 0.0) / _r(vt, 20, 5, "mean")
    hs = np.clip(fill_asset(hs, 1e-3), 1e-5, 0.05)
    depth = np.clip(fill_asset(depth, 1.0), 0.0, 1e3)
    adv = np.clip(fill_asset(adv, 1.0), 1e-4, 1e3)
    return np.stack([hs, depth, adv], -1).astype(np.float32)


# ------------------------------------------------------------------------------------------- universe gating
def _roll_count(x: np.ndarray, w: int) -> np.ndarray:
    """Number of True in x[t-w+1 : t+1] along axis 0."""
    cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x.astype(np.int64)]), 0)
    t = np.arange(x.shape[0])
    return cs[t + 1] - cs[np.maximum(t + 1 - w, 0)]


def _fwd_any(x: np.ndarray, H: int) -> np.ndarray:
    """any(x[t+1 : t+1+H]) along axis 0 (beyond the end counts as False)."""
    cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x.astype(np.int64)]), 0)
    T = x.shape[0]
    t = np.arange(T)
    return (cs[np.minimum(t + 1 + H, T)] - cs[t + 1]) > 0


def assemble_universe(active: np.ndarray, close_ff: np.ndarray, horizons: Sequence[int], tail: int = 30,
                      lookback: int = 64, vol_window: int = 60, max_iter: int = 200):
    """Returns (mid, tail_mask, active_final, n_bridged). See the module docstring."""
    A = active & np.isfinite(close_ff)
    T, N = A.shape
    A0 = A.copy()
    minp = max(2, vol_window // 2)
    hz = list(horizons)
    for _ in range(max_iter):
        ends = A & ~np.vstack([A[1:], np.zeros((1, N), bool)])
        since_end = np.full((T, N), 10 ** 9)
        last = np.full(N, -10 ** 9)
        for t in range(T):
            since_end[t] = t - last
            last = np.where(ends[t], t, last)
        tail_m = ~A & (since_end <= tail) & np.isfinite(close_ff)
        ext = A | tail_m
        mid = np.where(ext, close_ff, np.nan)
        fin = np.isfinite(mid)
        lr_fin = np.zeros((T, N), bool); lr_fin[1:] = fin[1:] & fin[:-1]
        sig_ok = _roll_count(lr_fin, vol_window) >= minp
        imp_ok = fin.copy()
        for h in hz:
            sh = np.zeros((T, N), bool); sh[:T - h] = fin[h:]
            imp_ok &= sh
        valid = fin & sig_ok & imp_ok & ~tail_m
        bad_lb = valid & (_roll_count(tail_m, lookback) > 0)
        bad_fw = valid & _fwd_any(~fin, tail)
        bad_fw[T - tail:] = False
        bad = bad_lb | bad_fw
        if not bad.any():
            return mid, tail_m, A, int((A & ~A0).sum())
        for i in np.where(bad.any(0))[0]:
            t = int(np.argmax(bad[:, i]))
            if bad_lb[t, i]:
                prev = np.where(A[:t, i])[0]
                prev = prev[prev < t]
                # last member bar before the tail bars that sit in the look-back window
                win_tail = np.where(tail_m[max(0, t - lookback + 1):t + 1, i])[0] + max(0, t - lookback + 1)
                j = int(win_tail.min())
                e = int(prev[prev < j].max()) if (prev < j).any() else j - 1
                A[e + 1:t, i] = True
            else:
                A[t:min(T, t + tail + 1), i] = True
            A[:, i] &= np.isfinite(close_ff[:, i])
    raise RuntimeError("assemble_universe did not converge")


# ------------------------------------------------------------------------------------------------------- cache
def save_cache(path: str, arrays: Dict[str, np.ndarray], meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta_arr = {f"meta__{k}": v for k, v in meta.items() if isinstance(v, np.ndarray)}
    meta_rest = {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)}
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays, **meta_arr, meta_pickle=np.frombuffer(pickle.dumps(meta_rest), dtype=np.uint8))
    os.replace(tmp, path)


def load_cache(path: str):
    z = np.load(path, allow_pickle=False)
    meta = pickle.loads(z["meta_pickle"].tobytes())
    arrays = {}
    for k in z.files:
        if k == "meta_pickle":
            continue
        if k.startswith("meta__"):
            meta[k[6:]] = z[k]
        else:
            arrays[k] = z[k]
    return arrays, meta

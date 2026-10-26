"""Martingale-difference placebo (Leakage Audit Card item 4; Nikolopoulos 2026, arXiv:2604.15531).

Run the *unchanged* pipeline on surrogate panels with no predictability but the same volatility structure:
    sign_flip        r'_t = eps_t * r_t, eps_t = +-1 i.i.d.  (keeps |r_t|, hence volatility clustering)
    block_bootstrap  returns resampled in blocks (keeps short-range dependence inside blocks, breaks the signal)
    garch_sign       filtered-residual surrogate: EWMA (RiskMetrics) standardised residuals z_t = r_t / sigma_t are permuted
                     in time and sign-flipped, then re-scaled by a vol path regenerated from the new returns with the
                     same filter (keeps the marginal of z and volatility clustering, breaks any predictability; distinct
                     from sign_flip, which keeps |r_t| at its original bar)
Missing mids (suspensions, pre-listing) are kept missing: surrogate returns are built on each asset's finite bars only.
Labels, documents and tradability are re-derived by the caller's pipeline from the surrogate prices when the pipeline
builds them from prices; otherwise they are kept, which keeps the test conservative (a pipeline that reads labels
or documents illegally can still earn on the surrogate -> detected by FPIT, not here).
A significantly positive placebo Sharpe (mean over surrogates, or a high share of positive draws) indicates leakage.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from warn2trade.data.schema import Panel
from ..data.pit import clone_panel
from ..eval.metrics import placebo_rank

SURROGATES = ("sign_flip", "block_bootstrap", "garch_sign")


def surrogate_panel(panel: Panel, rng: np.random.Generator, method: str = "sign_flip", block: int = 10, keep_from: int = 0) -> Panel:
    """Surrogate copy of the panel: prices rebuilt from surrogate log returns from bar `keep_from` on; the return
    feature column (meta['ret_col']) is rebuilt consistently, other raw features keep their values. Per asset, only
    finite bars carry returns (a return spanning a gap is booked on the first finite bar after it); the original NaN
    mask of mid is re-applied, and an asset is anchored on its last finite mid at or before keep_from (or its first
    finite mid after it)."""
    assert method in SURROGATES, method
    q = clone_panel(panel)
    T, N = q.mid.shape
    mid = np.asarray(panel.mid, dtype=np.float64)
    nanmask = ~np.isfinite(mid)
    col = int(q.meta.get("ret_col", 0))
    for n in range(N):
        fin = np.where(~nanmask[:, n])[0]
        if fin.size < 2:
            continue
        before = fin[fin <= keep_from]
        anchor = int(before[-1]) if before.size else int(fin[0])
        after = fin[fin > anchor]
        if after.size == 0:
            continue
        lm = np.log(mid[np.concatenate([[anchor], after]), n])
        seg = np.diff(lm)                                       # returns on finite bars after the anchor
        new = _surrogate_1d(seg[:, None], rng, method, block)[:, 0]
        q.mid[after, n] = mid[anchor, n] * np.exp(np.cumsum(new))
        if q.meta.get("raw_features", False):
            q.price_feat[after, n, col] = new.astype(q.price_feat.dtype)
    q.mid[nanmask] = np.nan
    q.meta = dict(q.meta); q.meta["surrogate"] = method
    return q


def _surrogate_1d(seg: np.ndarray, rng: np.random.Generator, method: str, block: int) -> np.ndarray:
    if method == "sign_flip":
        return rng.choice([-1.0, 1.0], size=seg.shape) * seg
    if method == "block_bootstrap":
        n = seg.shape[0]
        idx: List[int] = []
        while len(idx) < n:
            s = int(rng.integers(0, max(n - block, 0) + 1))      # inclusive of the last block start
            idx.extend(range(s, min(s + block, n)))
        return seg[np.asarray(idx[:n])]
    return _garch_sign(seg, rng)


def _ewma_filter(r: np.ndarray, lam: float, var0: np.ndarray) -> np.ndarray:
    """Pre-return EWMA volatility path sigma_t (uses r_0..r_{t-1}); non-finite returns leave the variance unchanged."""
    var = var0.copy(); vols = np.zeros_like(r)
    for t in range(r.shape[0]):
        vols[t] = np.sqrt(var)
        ok = np.isfinite(r[t])
        var = np.where(ok, lam * var + (1 - lam) * np.where(ok, r[t], 0.0) ** 2, var)
    return vols


def _garch_sign(r: np.ndarray, rng: np.random.Generator, lam: float = 0.94) -> np.ndarray:
    """Filtered-residual surrogate: z = r / sigma (EWMA filter), permute z in time and flip its sign, then rebuild the
    returns recursively with the same filter so the volatility path responds to the new returns."""
    T, N = r.shape
    if T == 0:
        return r.copy()
    var0 = np.nanvar(r[: max(20, T // 10)], axis=0)
    var0 = np.where(np.isfinite(var0) & (var0 > 0), var0, 1e-8) + 1e-12
    z = r / _ewma_filter(r, lam, var0)
    out = np.zeros_like(r)
    for n in range(N):
        zn = z[:, n][np.isfinite(z[:, n])]
        if zn.size == 0:
            continue
        zs = rng.permutation(zn) if zn.size >= T else rng.choice(zn, size=T)
        zs = zs[:T] * rng.choice([-1.0, 1.0], size=T)
        var = float(var0[n])
        for t in range(T):
            out[t, n] = zs[t] * np.sqrt(var)
            var = lam * var + (1 - lam) * out[t, n] ** 2
    return out


def placebo_distribution(run_fn: Callable[[Panel], float], panel: Panel, n: int = 20, seed: int = 0, method: str = "sign_flip",
                         keep_from: int = 0, block: int = 10) -> Dict[str, object]:
    """run_fn(panel) -> Sharpe (or any scalar metric). Returns the surrogate distribution and the rank of the real run."""
    rng = np.random.default_rng(seed)
    real = float(run_fn(panel))
    srs: List[float] = []
    for _ in range(n):
        srs.append(float(run_fn(surrogate_panel(panel, rng, method, block, keep_from))))
    out: Dict[str, object] = {"real": real, "placebo": srs, "method": method, "n": n}
    out.update(placebo_rank(real, srs))
    p = np.asarray(srs, dtype=np.float64); p = p[np.isfinite(p)]
    out["placebo_leak_flag"] = bool(p.size >= 5 and (p > 0).mean() >= 0.9 and p.mean() > 0)
    return out

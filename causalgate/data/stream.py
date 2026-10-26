"""Streaming (strictly causal) normalisers with as-of stamps: Welford mean/variance, EWMA mean/variance, and the
P-squared (P^2) quantile estimator (Jain & Chlamtac 1985) for a streaming median / MAD. Every estimator is an
AsOfStamped operator: its state after consuming bar t is stamped `as_of = t`, and the Filtration Guard refuses to
apply it at any decision bar < t.

Conventions
    * `transform_series(X)` with X of shape (T, ...) walks the series forward; the output at bar t uses the state
      built from bars < t (`include_current=False`, default) or <= t. Both are F_t-measurable.
    * The first `warmup` bars are *abstained* (NaN), never back-filled (plan Sec. 3.2).
    * `OfflineScaler` is the leaky counterpart (fit on an index range, stamped at the range end) used to emulate L1.
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np

from .pit import AsOfStamped


# --------------------------------------------------------------------------------------- Welford
class Welford(AsOfStamped):
    """Running mean / variance over K parallel streams (vector state), numerically stable."""

    def __init__(self, k: int):
        self.n = np.zeros(k)
        self.mean = np.zeros(k)
        self.m2 = np.zeros(k)

    def update(self, x: np.ndarray, t: Optional[int] = None) -> None:
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x)
        self.n[ok] += 1
        d = np.where(ok, x - self.mean, 0.0)
        self.mean = self.mean + np.where(ok, d / np.maximum(self.n, 1), 0.0)
        self.m2 = self.m2 + np.where(ok, d * (x - self.mean), 0.0)
        if t is not None:
            self.stamp(t)

    @property
    def var(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n > 1, self.m2 / np.maximum(self.n - 1, 1), np.nan)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


# --------------------------------------------------------------------------------------- EWMA
class EWMA(AsOfStamped):
    """Exponentially weighted mean / variance with half-life in bars (bias-corrected during warm-up).

    The mean starts at the first observation; the variance recursion starts at 0, so after n observations its
    expectation is (1 - lam^(n-1)) sigma^2. `var` divides by that factor (the usual EW debiasing)."""

    def __init__(self, k: int, halflife: float = 60.0):
        self.lam = float(np.exp(np.log(0.5) / max(halflife, 1e-6)))
        self.n = np.zeros(k)
        self.mean = np.zeros(k)
        self.var_ = np.zeros(k)

    def update(self, x: np.ndarray, t: Optional[int] = None) -> None:
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x)
        lam = self.lam
        first = ok & (self.n == 0)
        self.mean = np.where(first, x, self.mean)
        upd = ok & ~first
        d = np.where(upd, x - self.mean, 0.0)
        self.mean = self.mean + np.where(upd, (1 - lam) * d, 0.0)
        self.var_ = np.where(upd, lam * (self.var_ + (1 - lam) * d * d), self.var_)
        self.n[ok] += 1
        if t is not None:
            self.stamp(t)

    @property
    def var(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = 1.0 - self.lam ** np.maximum(self.n - 1, 0)
            return np.where(self.n > 1, self.var_ / np.maximum(corr, 1e-12), np.nan)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


# --------------------------------------------------------------------------------------- P^2 quantile
class P2Quantile(AsOfStamped):
    """Jain & Chlamtac (1985) P^2 algorithm, vectorised over K parallel streams, one target quantile p.

    Five markers per stream; exact for the first five observations, then O(1) per update with no storage.
    """

    def __init__(self, k: int, p: float = 0.5):
        self.k, self.p = int(k), float(p)
        self.n = np.zeros(k, dtype=int)
        self.q = np.zeros((5, k))          # marker heights
        self.npos = np.zeros((5, k))       # marker positions (1-based as in the paper)
        self.buf = np.full((5, k), np.nan) # first five observations
        self.dn = np.array([0.0, p / 2, p, (1 + p) / 2, 1.0])
        self.dnpos = np.tile(np.array([1.0, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5.0])[:, None], (1, k))

    def update(self, x: np.ndarray, t: Optional[int] = None) -> None:
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x)
        for j in np.nonzero(ok)[0]:
            self._update_one(int(j), float(x[j]))
        if t is not None:
            self.stamp(t)

    def _update_one(self, j: int, x: float) -> None:
        n = self.n[j]
        if n < 5:
            self.buf[n, j] = x
            self.n[j] += 1
            if self.n[j] == 5:
                self.q[:, j] = np.sort(self.buf[:, j])
                self.npos[:, j] = np.arange(1, 6)
            return
        q = self.q[:, j]
        npos = self.npos[:, j]
        if x < q[0]:
            q[0] = x; kcell = 0
        elif x >= q[4]:
            q[4] = x; kcell = 3
        else:
            kcell = int(np.searchsorted(q, x, side="right") - 1)
            kcell = min(max(kcell, 0), 3)
        npos[kcell + 1 :] += 1
        self.dnpos[:, j] += self.dn
        for i in range(1, 4):
            d = self.dnpos[i, j] - npos[i]
            if (d >= 1 and npos[i + 1] - npos[i] > 1) or (d <= -1 and npos[i - 1] - npos[i] < -1):
                s = 1.0 if d > 0 else -1.0
                qp = self._parabolic(q, npos, i, s)
                if q[i - 1] < qp < q[i + 1]:
                    q[i] = qp
                else:
                    q[i] = self._linear(q, npos, i, s)
                npos[i] += s
        self.n[j] += 1

    @staticmethod
    def _parabolic(q, n, i, s):
        return q[i] + s / (n[i + 1] - n[i - 1]) * ((n[i] - n[i - 1] + s) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
                                                   + (n[i + 1] - n[i] - s) * (q[i] - q[i - 1]) / (n[i] - n[i - 1]))

    @staticmethod
    def _linear(q, n, i, s):
        si = int(s)
        return q[i] + s * (q[i + si] - q[i]) / (n[i + si] - n[i])

    @property
    def value(self) -> np.ndarray:
        out = np.full(self.k, np.nan)
        full = self.n >= 5
        out[full] = self.q[2, full]
        part = (self.n > 0) & ~full
        for j in np.nonzero(part)[0]:
            out[j] = np.nanquantile(self.buf[: self.n[j], j], self.p)
        return out


# --------------------------------------------------------------------------------------- normalisers
class StreamingNormaliser(AsOfStamped):
    """Causal per-stream standardisation of a (T, ...) array along axis 0.

    mode = 'welford' : (x - running mean) / running std                 (expanding window)
           'ewma'    : (x - EWMA mean) / EWMA std                        (half-life in bars)
           'robust'  : (x - P2 median) / (1.4826 * P2 median of |x - median|)   (streaming MAD)
    include_current=False (default): the statistics used at bar t come from bars < t (an outlier does not dilute its
    own z-score). The first `warmup` observations of each stream abstain (NaN).
    """

    def __init__(self, mode: str = "welford", warmup: int = 60, halflife: float = 120.0, include_current: bool = False, eps: float = 1e-8):
        assert mode in ("welford", "ewma", "robust"), mode
        self.mode, self.warmup, self.halflife, self.include_current, self.eps = mode, int(warmup), float(halflife), bool(include_current), eps
        self._est = None
        self._mad = None
        self._shape: Optional[Tuple[int, ...]] = None

    def reset(self) -> "StreamingNormaliser":
        """Forget all estimator state (and the as-of stamp): the next transform_series starts a fresh causal sweep."""
        self._est = None
        self._mad = None
        self.as_of = -1
        return self

    def _init(self, k: int) -> None:
        if self.mode == "welford":
            self._est = Welford(k)
        elif self.mode == "ewma":
            self._est = EWMA(k, self.halflife)
        else:
            self._est = P2Quantile(k, 0.5)
            self._mad = P2Quantile(k, 0.5)

    def _loc_scale(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mode == "robust":
            loc = self._est.value
            scale = 1.4826 * self._mad.value
            n = self._est.n.astype(float)
        else:
            loc, scale, n = self._est.mean, self._est.std, self._est.n
        return loc, scale, n

    def _feed(self, x: np.ndarray, t: Optional[int]) -> None:
        if self.mode == "robust":
            med = self._est.value
            self._est.update(x, t)
            self._mad.update(np.abs(x - np.where(np.isfinite(med), med, x)), t)
        else:
            self._est.update(x, t)

    def transform_series(self, X: np.ndarray, t0: int = 0) -> np.ndarray:
        """X: (T, ...) -> Z: (T, ...), causal along axis 0. `t0` is the bar index of X[0] for the as-of stamp.
        The estimator state *continues* from any previous call (X is read as the continuation of the stream); call
        `reset()` first to sweep an unrelated / re-built series from scratch."""
        X = np.asarray(X, dtype=np.float64)
        T = X.shape[0]
        shape = X.shape[1:]
        k = int(np.prod(shape)) if shape else 1
        if self._est is None:
            self._init(k)
        Z = np.full(X.shape, np.nan)
        for t in range(T):
            x = X[t].reshape(k)
            if self.include_current:
                self._feed(x, t0 + t)
            loc, scale, n = self._loc_scale()
            with np.errstate(invalid="ignore", divide="ignore"):
                z = (x - loc) / np.maximum(scale, self.eps)
            z[n < self.warmup] = np.nan
            if not self.include_current:
                self._feed(x, t0 + t)
            Z[t] = z.reshape(shape)
        self.stamp(t0 + T - 1)
        return Z

    def transform_one(self, x: np.ndarray, t: int) -> np.ndarray:
        """Online: standardise one new observation with the state built so far, then absorb it."""
        x = np.asarray(x, dtype=np.float64)
        k = x.size
        if self._est is None:
            self._init(k)
        self.check(t)
        if self.include_current:
            self._feed(x.reshape(k), t)
        loc, scale, n = self._loc_scale()
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (x.reshape(k) - loc) / np.maximum(scale, self.eps)
        z[n < self.warmup] = np.nan
        if not self.include_current:
            self._feed(x.reshape(k), t)
        return z.reshape(x.shape)


class OfflineScaler(AsOfStamped):
    """Global z-score fitted on an index range (the L1 convention). Stamped at the last bar of the fit range, so the
    Filtration Guard flags its use at any earlier decision bar."""

    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, X: np.ndarray, rows: Optional[Sequence[int]] = None) -> "OfflineScaler":
        X = np.asarray(X, dtype=np.float64)
        sub = X if rows is None else X[np.asarray(rows)]
        self.mean = np.nanmean(sub, axis=0)
        self.std = np.nanstd(sub, axis=0)
        self.stamp(int(X.shape[0] - 1 if rows is None else max(rows)))
        return self

    def transform(self, X: np.ndarray, t: Optional[int] = None) -> np.ndarray:
        if t is not None:
            self.check(t)
        return (np.asarray(X, dtype=np.float64) - self.mean) / np.maximum(self.std, self.eps)


def streaming_zscore(X: np.ndarray, mode: str = "welford", warmup: int = 60, halflife: float = 120.0, include_current: bool = False) -> np.ndarray:
    """Convenience: causal standardisation of a (T, N, F) panel array (every (asset, feature) is its own stream)."""
    return StreamingNormaliser(mode, warmup, halflife, include_current).transform_series(X)

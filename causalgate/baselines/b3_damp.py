"""B3 DAMP - Discord Aware Matrix Profile (Lu, Wu, Keogh, KDD 2022, "Matrix Profile XXIV: Scaling Time Series Anomaly
Detection to Trillions of Datapoints and Ultra-fast Arriving Data Streams") as a score -> alarm -> trade baseline.

The authors' reference code is MATLAB (DAMP_2_0.m, distributed from the project site); the Python port we cite is
https://github.com/sihohan/DAMP (damp.py, utils.py; numpy only). Nothing is installed here (no stumpy / numba), so the
left matrix profile and the DAMP search are re-implemented in numpy (status 'reimplemented'); when
third_party/DAMP/damp.py is importable the official-port `DAMP_2_0` is used for the full-series profile (status
'official' only if every (asset, feature) stream succeeds - the port raises on constant regions and requires m > 10).

Two scorers:
    window-local   causalgate.models.scorer.LeftMatrixProfileScorer: nearest earlier m-subsequence inside the lookback
                   window (always available; used when the panel is not attached).
    full-series    DAMP Algorithm 1 on the panel's causally normalised streams, one stream per (asset, feature),
                   profile indexed by the subsequence END bar so that lmp[t] is F_t-measurable:
                   * prefix: the first 16 m queries after t_start are brute force against the whole past
                     (damp.py ~L92-111 "Handle the prefix", MASS over time_series[:i]);
                   * then backward search: nearest X = 2^nextpow2(8 m) bars first, doubling X until either a distance
                     below best-so-far is found (stop: approximate value) or the beginning of the series is reached
                     (exact value, best-so-far updated)  (damp.py ~L152-200 "Classic DAMP", Cases 1-3);
                   * forward pruning with `lookahead` (damp.py ~L115-133, ~L205-223) is a compute shortcut that reads
                     bars > t; it is NOT used (lookahead = 0 is "a pure online algorithm with no pruning", damp.py ~L113).
                   Features are combined by averaging the per-feature profiles (DAMP is univariate).
DAMP is L1-invariant by construction (every subsequence is z-normalised, utils.py MASS_V2), so the published and causal
modes differ only through L2 (threshold) and the warm-up abstention of the streaming normaliser.
"""
from __future__ import annotations
import importlib
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader

from .common import ScoreToTradeBaseline, register
from ..data.pit import AsOfStamped
from ..models.scorer import LeftMatrixProfileScorer

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party")


def _import_official(repo: str, module: str, attr: str) -> Tuple[Optional[Any], str]:
    root = os.path.join(THIRD_PARTY, repo)
    if not os.path.isdir(root):
        return None, f"third_party/{repo} not found"
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr), "ok"
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


class _ModelStamp(AsOfStamped):
    pass


def _nextpow2(x: float) -> int:
    return int(np.ceil(np.log2(max(abs(x), 1.0))))


def _znorm_windows(x: np.ndarray, m: int) -> np.ndarray:
    """(T - m + 1, m) z-normalised subsequences of x; NaN inside a subsequence makes its row NaN."""
    w = sliding_window_view(np.asarray(x, dtype=np.float64), m)
    mu = w.mean(1, keepdims=True)
    sd = w.std(1, keepdims=True)
    return (w - mu) / np.maximum(sd, 1e-8)


class DampStream:
    """DAMP left matrix profile of one univariate stream, computed incrementally (`extend(x, t_hi)` reads x[:t_hi+1]
    only). lmp[t] = z-normalised Euclidean distance from x[t-m+1..t] to its nearest neighbour among the subsequences
    ending <= t - m (non-overlapping, as MASS over time_series[:i] in damp.py ~L108)."""

    def __init__(self, T: int, m: int, t_start: int, prefix_mult: int = 16, init_mult: int = 8, exact: bool = False, max_lookback: Optional[int] = None):
        self.T, self.m, self.t_start = int(T), int(m), int(t_start)
        self.prefix_mult, self.init_mult, self.exact, self.max_lookback = int(prefix_mult), int(init_mult), bool(exact), max_lookback
        self.lmp = np.full(self.T, np.nan)
        self.best = -np.inf
        self.next_t = self.m - 1
        self.n_exact = 0
        self.n_approx = 0

    @staticmethod
    def _dist(cand: np.ndarray, q: np.ndarray) -> float:
        if cand.shape[0] == 0:
            return np.nan
        d = np.sqrt(((cand - q) ** 2).sum(1))
        return float(np.nanmin(d)) if np.isfinite(d).any() else np.nan

    def extend(self, x: np.ndarray, t_hi: int) -> None:
        t_hi = min(int(t_hi), self.T - 1)
        if t_hi < self.next_t:
            return
        m = self.m
        subs = _znorm_windows(x[: t_hi + 1], m)                     # row e ends at bar e + m - 1
        for t in range(self.next_t, t_hi + 1):
            e = t - m + 1
            hi = e - m                                               # last admissible candidate row (ends at t - m)
            q = subs[e]
            if hi < 0 or not np.isfinite(q).all():
                continue                                             # NaN = no candidate / warm-up abstention
            lo_all = 0 if self.max_lookback is None else max(0, hi + 1 - int(self.max_lookback))
            if self.exact or t < self.t_start + self.prefix_mult * m:
                val = self._dist(subs[lo_all : hi + 1], q)           # prefix / exact: brute force over the whole past
                self.lmp[t] = val
                self.n_exact += 1
                if np.isfinite(val) and val > self.best:
                    self.best = val                                  # damp.py ~L111 best_so_far = max(left_mp)
                continue
            X = 2 ** _nextpow2(self.init_mult * m)                   # damp.py ~L152
            val = np.inf
            seg_hi = hi
            while True:
                seg_lo = max(lo_all, hi + 1 - X)
                d = self._dist(subs[seg_lo : seg_hi + 1], q)
                if np.isfinite(d):
                    val = min(val, d)
                if seg_lo <= lo_all:                                 # Case 1: reached the beginning -> exact
                    self.lmp[t] = val if np.isfinite(val) else np.nan
                    self.n_exact += 1
                    if np.isfinite(val) and val > self.best:
                        self.best = val
                    break
                if val < self.best:                                  # a closer neighbour exists -> cannot be the discord
                    self.lmp[t] = val
                    self.n_approx += 1
                    break
                seg_hi = seg_lo + m - 2                              # next segment overlaps by m - 1 rows (no boundary miss)
                X *= 2                                               # damp.py ~L200 "Otherwise expand the search"
        self.next_t = t_hi + 1


# --------------------------------------------------------------------------------------- baseline
@register("cg_damp")
class DampBaseline(ScoreToTradeBaseline):
    name = "cg_damp"
    paper = "Lu, Wu, Keogh. Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions of Datapoints and Ultra-fast Arriving Data Streams (DAMP). KDD 2022"
    official_repo = "https://github.com/sihohan/DAMP (python port of the authors' MATLAB DAMP_2_0)"
    status = "reimplemented"
    year = 2022
    cutoff_date = None
    conventions = {
        "L1": "official: none - every subsequence is z-normalised inside MASS_V2 (utils.py), so DAMP is invariant to any per-feature affine "
              "scaling; the port reads the raw series (damp.py __main__ np.loadtxt). Here the scaffold's L1 only decides the warm-up abstention.",
        "L2": "official: no threshold - DAMP returns the left matrix profile and the top discord (damp.py ~L229 'Get top discord'); the paper "
              "evaluates discord ranks. published mode here: train_quantile (a causal reference has no published test-set rule); "
              "causal: delayed-ACI gate.",
        "L3": "official: subsequence length m and location_to_start_processing are user inputs (damp.py argparse: 100 / 400 on the demo "
              "series); here m from cfg, t_start = first training bar.",
        "L4": "official: none (discord ranking); the UCR archive protocol scores a hit if the top discord falls in the labelled region.",
        "L5": "not applicable.",
        "L6": "not applicable.",
        "L7": "official: with lookahead = 0 the algorithm is 'pure online' (damp.py ~L113 comment); with lookahead > 0 the forward pruning "
              "reads up to lookahead future bars (only to skip computations and assign lmp[i-1] - 1e-5 to pruned points, ~L100, ~L144). "
              "Here lookahead = 0; score at t, fill at t + latency.",
        "L8": "none.",
    }
    unspecified = [
        "Multivariate input: DAMP is univariate; per-feature profiles are averaged (cfg['model']['features'] selects a subset).",
        "Panel-level profile is computed on the scaffold-normalised streams (z-normalisation makes the choice immaterial apart from NaNs).",
        "Candidate exclusion: non-overlapping (ends <= t - m), as the port's MASS over time_series[:i]; the port's first backward segment "
        "time_series[i-X+1:i+1] also contains overlapping subsequences - not replicated.",
        "Prefix length 16 m and initial backward window 2^nextpow2(8 m) taken from the port (damp.py ~L95, ~L152); cfg exposes both.",
        "Values below best-so-far are approximate by design (DAMP only guarantees exactness for discord candidates); cfg['model']['exact'] = true "
        "disables the early stop (full left profile).",
        "Scale: raw z-normalised distance; the window-local fallback divides by the training median (LeftMatrixProfileScorer).",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons, device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "train_quantile")
        cfg.setdefault("published_norm", "train")
        super().__init__(cfg, dims, horizons, device)
        m = dict(cfg.get("model", {}))
        self.m = int(m.get("m", 8))
        self.prefix_mult = int(m.get("prefix_mult", 16)); self.init_mult = int(m.get("init_mult", 8))
        self.exact = bool(m.get("exact", False)); self.max_lookback = m.get("max_lookback", None)
        self.use_series = bool(m.get("use_series_profile", True)) and self.panel is not None
        feats = m.get("features", "all")
        self.features: Optional[List[int]] = None if feats == "all" else [int(f) for f in feats]
        self.local = LeftMatrixProfileScorer(self.dims["price"], self.L, {"m": self.m}, device)
        self.official_fn, reason = (_import_official("DAMP", "damp", "DAMP_2_0") if m.get("use_official", True) else (None, "disabled by cfg"))
        self.diag["official_import"] = reason
        self.streams: Dict[Tuple[int, int], DampStream] = {}
        self._profile: Optional[np.ndarray] = None            # (T, N) averaged over features; NaN = not available
        self._pending: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.t_start = 0
        self._stamp = _ModelStamp()
        self.guard.register(self._stamp)

    # ---- (t, asset) bookkeeping so that _score can read the panel profile
    def _windows(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        self._pending = (batch["t"].cpu().numpy(), batch["asset"].cpu().numpy())
        return super()._windows(batch)

    def _collect(self, loader: DataLoader):
        Xs, ys, ts, iis = [], [], [], []
        for b in loader:
            Xs.append(self._windows(b)); ys.append(b["y"].cpu().numpy()); ts.append(b["t"].cpu().numpy()); iis.append(b["asset"].cpu().numpy())
        if not Xs:
            self._pending = None
            return np.zeros((0, self.L, self.dims["price"]), np.float32), np.zeros(0), np.zeros(0, dtype=int)
        t = np.concatenate(ts); i = np.concatenate(iis)
        self._pending = (t, i)
        return np.concatenate(Xs), np.concatenate(ys), t

    # ---- full-series profile
    def _feature_idx(self) -> List[int]:
        Fd = self.dims["price"]
        return list(range(Fd)) if self.features is None else [f for f in self.features if 0 <= f < Fd]

    def _official_profile(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Full-series left MP through the official port (lookahead=0 -> pure online); indexed by END bar."""
        if self.official_fn is None or self.m <= 10 or not np.isfinite(x).all():
            return None
        try:
            left_mp, _, _ = self.official_fn(time_series=x, subsequence_length=self.m, stride=1,
                                             location_to_start_processing=max(self.t_start, self.m), lookahead=0, enable_output=False)
        except Exception as e:                                   # noqa: BLE001  (constant regions -> Exception)
            self.diag["official_import"] = f"DAMP_2_0 failed on a stream: {type(e).__name__}; numpy DAMP used"
            return None
        out = np.full(x.shape[0], np.nan)
        out[self.m - 1 :] = np.asarray(left_mp, dtype=np.float64)[: x.shape[0] - self.m + 1]
        out[out == 0] = np.nan                                   # the port leaves unprocessed positions at 0
        return out

    def _ensure_profile(self, t_hi: int) -> None:
        Z = self._panel_Z()                                      # (T, N, F) scaffold-normalised
        T, N, _ = Z.shape
        if self._profile is None:
            self._profile = np.full((T, N), np.nan)
            self._official_ok = self.official_fn is not None
        feats = self._feature_idx()
        for i in range(N):
            per_feat = []
            for f in feats:
                key = (i, f)
                if key not in self.streams:
                    st = DampStream(T, self.m, self.t_start, self.prefix_mult, self.init_mult, self.exact, self.max_lookback)
                    off = self._official_profile(Z[:, i, f].astype(np.float64)) if self.official_fn is not None else None
                    if off is not None:
                        st.lmp = off; st.next_t = T                  # the official port computed the whole (online) profile
                    else:
                        self._official_ok = False
                    self.streams[key] = st
                st = self.streams[key]
                st.extend(Z[:, i, f].astype(np.float64), t_hi)
                per_feat.append(st.lmp)
            if per_feat:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN warm-up columns
                    self._profile[:, i] = np.nanmean(np.stack(per_feat, 0), 0)
        self.status = "official" if (self._official_ok and self.official_fn is not None) else "reimplemented"

    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        if self._pending is not None and len(self._pending[0]):
            self.t_start = int(self._pending[0].min())
        if len(X):
            self.local.fit(X, t_end)
        if self.use_series:
            self._ensure_profile(int(t_end))
            n_ex = sum(s.n_exact for s in self.streams.values()); n_ap = sum(s.n_approx for s in self.streams.values())
            self.diag.update({"damp_exact": n_ex, "damp_approx": n_ap, "t_start": self.t_start})
        self._stamp.stamp(int(t_end))
        self.diag["status"] = self.status

    def _score(self, X: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float64)
        if len(Xn) == 0:
            return np.zeros(0)
        if self.use_series and self._pending is not None and len(self._pending[0]) == len(Xn):
            t, i = self._pending
            self._ensure_profile(int(t.max()))
            s = self._profile[t, i].astype(np.float64)
        else:
            s = self.local.score(Xn)
        s[np.isnan(Xn[:, -1]).any(-1)] = np.nan
        return s

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._stamp.check(int(batch["t"].min()))
        return super().predict(batch)

"""Weak labeling functions (LFs) over a Panel -> votes in {-1 (normal), 0 (abstain), +1 (anomaly)}.

These are the noisy market proxies of the paper: trading halts / limit hits, extreme forward jumps, volume
bursts, spread blow-outs, social-media bursts, and any unsupervised detector score. They are *training-time
labelers*: an LF may look forward (`looks_forward` bars) because it defines a pseudo-label, not a feature. The
walk-forward purge must therefore be >= max(looks_forward) so no LF vote inside the train fold peeks into the test
fold. Models never receive LF votes in forward().
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .schema import Panel
from .labels import log_returns, realized_vol, rolling_mean_std

VOTE_POS, VOTE_NEG, VOTE_ABSTAIN = 1, -1, 0


@dataclass
class LabelingFunction:
    name: str
    fn: Callable[[Panel], np.ndarray]
    looks_forward: int = 0

    def __call__(self, panel: Panel) -> np.ndarray:
        v = np.asarray(self.fn(panel))
        assert v.shape == panel.mid.shape, f"LF {self.name}: votes {v.shape} != panel {panel.mid.shape}"
        return v.astype(np.int8)


def _zscore_causal(x: np.ndarray, window: int) -> np.ndarray:
    m, s = rolling_mean_std(x, window)
    return (x - m) / (s + 1e-8)


def _forward_max(x: np.ndarray, lookahead: int) -> np.ndarray:
    """max over bars [t, t + lookahead] (NaN-aware)."""
    if lookahead <= 0:
        return x
    T = x.shape[0]
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(T):
        seg = x[t : min(T, t + lookahead + 1)]
        with np.errstate(all="ignore"):
            out[t] = np.nanmax(seg, axis=0) if np.isfinite(seg).any() else np.nan
    return out


def _votes(pos: np.ndarray, neg: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.zeros(pos.shape, dtype=np.int8)
    v[np.nan_to_num(pos, nan=False).astype(bool)] = VOTE_POS
    if neg is not None:
        v[np.nan_to_num(neg, nan=False).astype(bool) & (v == 0)] = VOTE_NEG
    return v


# ----------------------------------------------------------------------------- the LF zoo
def lf_forward_jump(horizon: int = 5, k_pos: float = 3.0, k_neg: float = 0.75, window: int = 60) -> LabelingFunction:
    """|forward cum log return over `horizon`| in units of sigma*sqrt(h): > k_pos -> anomaly, < k_neg -> normal."""
    def fn(p: Panel) -> np.ndarray:
        T = p.T
        lm = np.log(p.mid)
        sigma = realized_vol(p.mid, window)
        fwd = np.full_like(p.mid, np.nan)
        fwd[: T - horizon] = lm[horizon:] - lm[: T - horizon]
        z = np.abs(fwd) / (sigma * np.sqrt(horizon) + 1e-12)
        return _votes(z > k_pos, z < k_neg)
    return LabelingFunction("fwd_jump", fn, looks_forward=horizon)


def lf_volume_spike(k_pos: float = 3.0, k_neg: float = 0.5, window: int = 60, lookahead: int = 0,
                    vol_feat: str = "vol_z") -> LabelingFunction:
    """Volume z-score (from meta['volume'] if present, else the `vol_feat` price feature) > k_pos -> anomaly."""
    def fn(p: Panel) -> np.ndarray:
        vol = p.meta.get("volume")
        if vol is not None:
            z = _zscore_causal(np.log1p(np.asarray(vol, dtype=np.float64)), window)
        else:
            names = list(p.meta.get("feat_names", []))
            if vol_feat not in names:
                raise ValueError("lf_volume_spike needs meta['volume'] or a price feature named %r" % vol_feat)
            z = p.price_feat[..., names.index(vol_feat)].astype(np.float64)
        z = _forward_max(z, lookahead)
        return _votes(z > k_pos, z < k_neg)
    return LabelingFunction("volume_spike", fn, looks_forward=lookahead)


def lf_halt_or_limit(lookahead: int = 3) -> LabelingFunction:
    """Trading halt / limit hit / liquidity freeze within [t, t + lookahead] -> anomaly; never votes normal."""
    def fn(p: Panel) -> np.ndarray:
        untr = (~p.tradable).astype(np.float64)
        return _votes(_forward_max(untr, lookahead) > 0.5)
    return LabelingFunction("halt_or_limit", fn, looks_forward=lookahead)


def lf_spread_blowout(k_pos: float = 3.0, window: int = 60, lookahead: int = 0) -> LabelingFunction:
    """Relative half-spread z-score blow-out (liquidity vacuum) -> anomaly."""
    def fn(p: Panel) -> np.ndarray:
        z = _zscore_causal(np.log(p.liq[..., 0].astype(np.float64) + 1e-12), window)
        return _votes(_forward_max(z, lookahead) > k_pos)
    return LabelingFunction("spread_blowout", fn, looks_forward=lookahead)


def lf_social_burst(k_pos: float = 2.5, window: int = 60, silence_is_normal: bool = False) -> LabelingFunction:
    """Burst in the number of social / news documents attached to (t, i). Abstains if the panel has no text."""
    def fn(p: Panel) -> np.ndarray:
        if p.text_mask is None:
            return np.zeros(p.mid.shape, dtype=np.int8)
        cnt = p.text_mask.sum(-1).astype(np.float64)
        z = _zscore_causal(cnt, window)
        neg = (cnt == 0) if silence_is_normal else None
        return _votes(z > k_pos, neg)
    return LabelingFunction("social_burst", fn)


def lf_from_score(score: np.ndarray, hi_q: float = 0.99, lo_q: float = 0.5, name: str = "score",
                  looks_forward: int = 0) -> LabelingFunction:
    """Wrap any (T, N) anomaly score (reconstruction error, Isolation Forest, ...) into an LF via quantiles."""
    def fn(p: Panel) -> np.ndarray:
        s = np.asarray(score, dtype=np.float64)
        fin = s[np.isfinite(s)]
        hi, lo = np.quantile(fin, hi_q), np.quantile(fin, lo_q)
        return _votes(s >= hi, s <= lo)
    return LabelingFunction(name, fn, looks_forward=looks_forward)


_LF_FACTORIES: Dict[str, Callable[..., LabelingFunction]] = {
    "fwd_jump": lf_forward_jump, "volume_spike": lf_volume_spike, "halt_or_limit": lf_halt_or_limit,
    "spread_blowout": lf_spread_blowout, "social_burst": lf_social_burst,
}


def labeling_functions_from_config(items: Sequence[Dict[str, Any]]) -> List[LabelingFunction]:
    lfs = []
    for it in items:
        it = dict(it)
        name = it.pop("name")
        if name not in _LF_FACTORIES:
            raise KeyError(f"unknown labeling function {name!r}; available: {sorted(_LF_FACTORIES)}")
        lfs.append(_LF_FACTORIES[name](**it))
    return lfs


# ----------------------------------------------------------------------------- vote matrix + diagnostics
def build_vote_matrix(panel: Panel, lfs: Sequence[LabelingFunction]) -> np.ndarray:
    """(T, N, J) int8 votes."""
    return np.stack([lf(panel) for lf in lfs], -1)


def lf_report(votes: np.ndarray, names: Sequence[str], y_true: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Coverage / polarity / conflict per LF (+ empirical precision & recall if oracle labels are given).

    conflict_j = share of bars covered by LF j where the majority of the *other* non-abstaining LFs disagrees.
    """
    V = votes.reshape(-1, votes.shape[-1]).astype(np.int16)
    rows = []
    for j, n in enumerate(names):
        cov = V[:, j] != 0
        others = np.delete(V, j, axis=1)
        maj = np.sign(others.sum(1))
        conflict = ((maj != 0) & cov & (maj != V[:, j])).sum() / max(cov.sum(), 1)
        r = {"lf": n, "coverage": float(cov.mean()), "pos_rate": float((V[:, j] == 1).mean()),
             "neg_rate": float((V[:, j] == -1).mean()), "conflict": float(conflict)}
        if y_true is not None:
            y = y_true.reshape(-1)
            pp = V[:, j] == 1
            r["precision_pos"] = float(y[pp].mean()) if pp.any() else np.nan
            r["recall_pos"] = float(pp[y == 1].mean()) if (y == 1).any() else np.nan
            pn = V[:, j] == -1
            r["precision_neg"] = float((1 - y[pn]).mean()) if pn.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

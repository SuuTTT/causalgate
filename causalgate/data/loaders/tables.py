"""Defensive table readers shared by the real-data loaders (column names matched case-insensitively with documented
fallbacks; nothing is synthesised when a column is missing -- the caller decides whether that is fatal)."""
from __future__ import annotations
import os
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd

OHLCV_ALIASES: Dict[str, Sequence[str]] = {
    "date": ("date", "datetime", "timestamp", "time", "day"),
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close", "price", "last"),
    "adj_close": ("adj close", "adj_close", "adjclose", "adjusted close", "adjusted_close"),
    "volume": ("volume", "vol"),
}


def norm_name(c: str) -> str:
    return str(c).strip().lower().replace("-", " ").replace("_", " ")


def pick_col(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    """First column whose normalised name equals one of `aliases` (normalised); None if absent."""
    cols = list(columns)
    normed = {norm_name(c): c for c in cols}
    for a in aliases:
        a_n = norm_name(a)
        if a_n in normed:
            return normed[a_n]
    return None


def read_ohlcv_csv(path: str, date_col_fallback_first: bool = True) -> pd.DataFrame:
    """CSV -> DataFrame indexed by a tz-naive calendar date with columns open/high/low/close/adj_close/volume
    (missing optional columns -> absent; `adj_close` falls back to `close`). Column names are matched via OHLCV_ALIASES."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    dcol = pick_col(cols, OHLCV_ALIASES["date"])
    if dcol is None:
        if not date_col_fallback_first:
            raise ValueError(f"{path}: no date column among {cols}")
        dcol = cols[0]
    out = pd.DataFrame(index=pd.to_datetime(df[dcol], errors="coerce").dt.tz_localize(None).dt.normalize())
    for key in ("open", "high", "low", "close", "adj_close", "volume"):
        c = pick_col(cols, OHLCV_ALIASES[key])
        if c is not None:
            out[key] = pd.to_numeric(df[c].values, errors="coerce")
    if "close" not in out:
        raise ValueError(f"{path}: no close column among {cols}")
    if "adj_close" not in out:
        out["adj_close"] = out["close"]
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def find_first(root: str, candidates: Sequence[str]) -> Optional[str]:
    """First existing path among root/<candidate>; None if none exists."""
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def align_frames(frames: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, field: str) -> np.ndarray:
    """(T, N) float64 array of `field` for every asset in `frames` (NaN where missing), N in dict order."""
    T, N = len(dates), len(frames)
    out = np.full((T, N), np.nan)
    for j, (_, df) in enumerate(frames.items()):
        if field in df:
            out[:, j] = df[field].reindex(dates).to_numpy(dtype=np.float64)
    return out

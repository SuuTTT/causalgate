"""Regulatory enforcement anchors = the ultra-sparse, high-confidence positives of the paper.

Sources: SEC Litigation Releases / Administrative Proceedings (https://www.sec.gov/litigation/litreleases),
CSRC Administrative Sanction Decisions (http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zdgk.shtml), exchange
disciplinary notices, auditor-verified rug-pull lists. Each case gives (instrument, conduct window, publication date).

Anchor table schema (CSV, one row per case x instrument). Bar indices refer to Panel.times.
    case_id          str   "SEC-LR-25678", "CSRC-2023-45", ...
    source           str   "SEC" | "CSRC" | "EXCHANGE" | "AUDITOR" | "SYNTHETIC"
    asset            str   must match an entry of Panel.assets
    anomaly_class    str   "pump_dump" | "spoofing" | "insider" | "rug_pull" | "wash_trading" | "accounting" | "bear_raid"
    onset_t          int   first bar of the conduct window named in the complaint / decision
    peak_t           float bar of peak market impact (NaN -> argmax |impact| inside the window)
    end_t            int   last bar of the conduct window
    release_t        int   bar at which the case became PUBLIC (litigation release / decision publication date)
    date_resolution  str   "day" | "month" | "quarter"  (SEC complaints frequently give month-level windows)
    url              str   optional

Release-date-aware availability (the realistic protocol):
    A(t) = {k : release_t_k <= t}.
At training time t only cases already published exist. Enforcement typically lags the conduct by 1-4 years, so
the common "naive" protocol (use every case whose conduct ended before t) leaks regulator hindsight into the
training set. We expose three protocols and report the gap as a finding:
    release  realistic  (default)
    naive    conduct ended before t, publication ignored (leaky; what most prior work implicitly does)
    oracle   all cases, including future conduct -> evaluation labels only, never training
"""
from __future__ import annotations
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

REQUIRED_COLS = ("case_id", "source", "asset", "anomaly_class", "onset_t", "peak_t", "end_t", "release_t")
OPTIONAL_COLS = ("date_resolution", "url", "notes")
PROTOCOLS = ("release", "naive", "oracle")


# ----------------------------------------------------------------------------- loading / validation
def validate_anchor_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"anchor table missing columns {missing}; required = {REQUIRED_COLS}")
    df = df.copy().reset_index(drop=True)
    for c in ("onset_t", "end_t", "release_t"):
        df[c] = df[c].astype(int)
    df["peak_t"] = pd.to_numeric(df["peak_t"], errors="coerce").astype(float)
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = "" if c != "date_resolution" else "day"
    bad_window = df["end_t"] < df["onset_t"]
    if bad_window.any():
        raise ValueError(f"end_t < onset_t for cases {df.loc[bad_window, 'case_id'].tolist()}")
    early = df["release_t"] < df["end_t"]
    if early.any():
        warnings.warn(f"{int(early.sum())} case(s) published before the conduct window ended "
                      f"(trading suspension / TRO cases). They are kept; check {df.loc[early, 'case_id'].tolist()[:5]}")
    return df


def load_anchor_table(path: str) -> pd.DataFrame:
    return validate_anchor_table(pd.read_csv(path))


def dates_to_bar_index(dates: Iterable, times: np.ndarray, side: str = "left") -> np.ndarray:
    """Map calendar dates to bar indices of a Panel whose `times` are int64 ns since epoch (sorted).

    side='left'  -> first bar at or after the date (use for onset / release: information arrives at that bar)
    side='right' -> last bar at or before the date (use for end_t)
    """
    ts = pd.to_datetime(pd.Series(list(dates))).values.astype("datetime64[ns]").astype(np.int64)
    if side == "left":
        idx = np.searchsorted(times, ts, side="left")
    else:
        idx = np.searchsorted(times, ts, side="right") - 1
    return np.clip(idx, 0, len(times) - 1)


# ----------------------------------------------------------------------------- availability protocols
def available_anchors(df: pd.DataFrame, as_of_t: int, protocol: str = "release") -> pd.DataFrame:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    if protocol == "release":
        return df[df["release_t"] <= as_of_t]
    if protocol == "naive":
        return df[df["end_t"] <= as_of_t]
    return df


def sample_k_shot(df: pd.DataFrame, k_per_class: Optional[int], seed: int) -> pd.DataFrame:
    """K anchors per anomaly class, NESTED across K: one seeded permutation per class (keyed on case_id so it does not
    depend on which other cases are available), the first K cases are taken. K=1 subset of K=2 subset of ... of Full,
    so the label-budget curve measures the value of *additional* cases, not a re-draw. `seed` is the budget seed and
    must be decoupled from the model-initialisation seed. All rows of a selected case (multi-instrument cases) are kept.
    The effective number of cases per class is min(K, available); callers record it (see budget_report)."""
    if k_per_class is None or len(df) == 0:
        return df
    parts = []
    for cls, g in df.groupby("anomaly_class", sort=True):
        cases = sorted(g["case_id"].unique().tolist())
        keys = np.array([_case_key(c, seed) for c in cases])
        chosen = [cases[i] for i in np.argsort(keys)[: int(k_per_class)]]
        parts.append(g[g["case_id"].isin(chosen)])
    return pd.concat(parts) if parts else df.iloc[:0]


def _case_key(case_id: str, seed: int) -> float:
    """Deterministic pseudo-random key per (case, seed): the permutation of a class's cases is stable when new cases
    are published later (nestedness across folds as well as across K)."""
    import hashlib
    h = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def budget_report(available: pd.DataFrame, shot: pd.DataFrame, k_per_class: Optional[int]) -> dict:
    """Requested vs effective cases per class; `capped` flags classes where fewer cases exist than requested."""
    out = {"k_requested": "Full" if k_per_class is None else int(k_per_class),
           "n_available_cases": int(available["case_id"].nunique()) if len(available) else 0,
           "k_eff_cases": int(shot["case_id"].nunique()) if len(shot) else 0}
    capped = []
    for cls, g in available.groupby("anomaly_class"):
        n = g["case_id"].nunique()
        if k_per_class is not None and n < int(k_per_class):
            capped.append(f"{cls}:{n}<{k_per_class}")
    out["capped"] = ";".join(capped)
    return out


# ----------------------------------------------------------------------------- anchors -> (T, N) arrays
def anchor_label_arrays(df: pd.DataFrame, T: int, N: int, asset_index: Dict[str, int], lead: int = 0,
                        impact: Optional[np.ndarray] = None, horizons: Optional[Sequence[int]] = None,
                        class_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Rasterise anchors onto the panel grid.

    labels[t, i] = 1 on [onset_t - lead, peak_t] (the warning window: a detector should fire before the peak),
    bars_to_peak[t, i] = peak_t - t on [onset_t - lead, end_t] (negative after the peak),
    anomaly_class[t, i] in {-1, 0..C-1}, event_id[t, i] = row position in df (-1 elsewhere).
    If peak_t is NaN it is set to argmax_h |impact| inside the window when `impact` (T, N, H) is given, else to the
    window mid-point.
    """
    labels = np.zeros((T, N), dtype=np.int8)
    btp = np.full((T, N), np.nan)
    classes = class_names if class_names is not None else sorted(df["anomaly_class"].unique().tolist())
    cmap = {c: k for k, c in enumerate(classes)}
    cls = np.full((T, N), -1, dtype=np.int16)
    eid = np.full((T, N), -1, dtype=np.int32)
    for k, row in enumerate(df.itertuples(index=False)):
        if row.asset not in asset_index:
            warnings.warn(f"anchor {row.case_id}: asset {row.asset!r} not in panel, skipped")
            continue
        i = asset_index[row.asset]
        on, en = int(max(0, row.onset_t)), int(min(T - 1, row.end_t))
        if on > T - 1 or en < 0:
            continue
        peak = row.peak_t
        if peak is None or (isinstance(peak, float) and np.isnan(peak)):
            if impact is not None and horizons is not None:
                hz = np.asarray(list(horizons))
                seg = np.abs(impact[on : en + 1, i])                       # (W, H)
                seg = np.where(np.isnan(seg), -np.inf, seg)
                if np.isfinite(seg).any():
                    w, h = np.unravel_index(int(np.argmax(seg)), seg.shape)
                    peak = on + w + hz[h]
                else:
                    peak = (on + en) / 2.0
            else:
                peak = (on + en) / 2.0
        peak = int(min(T - 1, max(on, round(float(peak)))))
        lo = max(0, on - lead)
        labels[lo : peak + 1, i] = 1
        ts = np.arange(lo, en + 1)
        btp[lo : en + 1, i] = peak - ts
        cls[lo : en + 1, i] = cmap.get(row.anomaly_class, -1)
        eid[lo : en + 1, i] = k
    return {"labels": labels, "bars_to_peak": btp, "anomaly_class": cls, "event_id": eid, "class_names": classes}


def confident_normal_pool(exclusion: np.ndarray, margin: int, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Bars at temporal distance > `margin` (same asset) from every excluded bar (anchor windows, LF positives)."""
    ex = exclusion.astype(bool)
    dil = ex.copy()
    for d in range(1, int(margin) + 1):
        dil[d:] |= ex[:-d]
        dil[:-d] |= ex[d:]
    pool = ~dil
    if valid is not None:
        pool &= valid
    return pool


# ----------------------------------------------------------------------------- robustness / synthetic helpers
def jitter_anchor_dates(df: pd.DataFrame, rng: np.random.Generator, max_jitter: int) -> pd.DataFrame:
    """Emulate month-resolution complaints: shift onset/end/peak by U{-max_jitter..max_jitter} bars (per case)."""
    out = df.copy()
    j = rng.integers(-max_jitter, max_jitter + 1, size=len(out))
    for c in ("onset_t", "end_t"):
        out[c] = (out[c] + j).clip(lower=0)
    out["peak_t"] = out["peak_t"] + j
    out["end_t"] = np.maximum(out["end_t"], out["onset_t"])
    return out


def synthetic_anchor_table(panel, rng: np.random.Generator, release_lag: Tuple[int, int] = (120, 500),
                           source: str = "SYNTHETIC") -> pd.DataFrame:
    """Turn the planted events of the synthetic Panel into an anchor table with an enforcement lag."""
    events = panel.meta.get("events")
    if events is None:
        raise ValueError("panel.meta['events'] missing; rebuild the synthetic panel with the current builder")
    rows = []
    for k, e in enumerate(events):
        lag = int(rng.integers(release_lag[0], release_lag[1] + 1))
        rows.append({"case_id": f"SYN-{k:04d}", "source": source, "asset": panel.assets[int(e["asset"])],
                     "anomaly_class": "pump_dump" if e["direction"] > 0 else "bear_raid",
                     "onset_t": int(e["onset_t"]), "peak_t": float(e["peak_t"]), "end_t": int(e["end_t"]),
                     "release_t": int(e["end_t"]) + lag, "date_resolution": "day", "url": ""})
    cols = list(REQUIRED_COLS) + ["date_resolution", "url"]
    return validate_anchor_table(pd.DataFrame(rows, columns=cols))


def anchor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and the enforcement lag distribution (bars between end_t and release_t)."""
    g = df.assign(lag=df["release_t"] - df["end_t"]).groupby("anomaly_class")
    return g.agg(n_cases=("case_id", "nunique"), n_assets=("asset", "nunique"), lag_median=("lag", "median"),
                 lag_min=("lag", "min"), lag_max=("lag", "max"), window_median=("end_t", lambda s: float(np.median(s - df.loc[s.index, "onset_t"]))))

"""Verification sheet (dates, human-checked) -> bar-indexed anchor table, with an explicit match report.

Sheet columns (one row per case x instrument; template: docs/sparse_anchor_sheet_template.csv)
    case_id, source (SEC|CSRC|...), ticker, anomaly_class, onset_date, end_date, peak_date (optional), release_date,
    date_resolution (day|month|quarter), url, verifier (initials), second_verifier (optional), notes
Rules
    onset/release -> first bar at or after the date; end/peak -> last bar at or before the date
    release after the panel's last bar -> release_t = T (never available inside the panel; NOT clipped to T - 1)
    conduct window entirely outside the panel -> dropped and reported
    ticker not in the panel -> dropped and reported (the per-class match rate goes into the paper's dataset table)
    month resolution -> onset = first day of the month, end = last day of the month (callers pass dates that way)
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from ..anchors import validate_anchor_table


def _to_ns(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, errors="coerce").values.astype("datetime64[ns]").astype(np.int64)


def sheet_to_anchors(sheet: pd.DataFrame, times: np.ndarray, assets, asset_map: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (anchor_table, report). `times` = Panel.times (int64 ns, sorted); `assets` = Panel.assets."""
    df = sheet.copy()
    T = len(times)
    aset = set(assets)
    df["asset"] = df["ticker"].astype(str).map(asset_map) if asset_map else df["ticker"].astype(str)
    status = np.array(["ok"] * len(df), dtype=object)
    status[~df["asset"].isin(aset).to_numpy()] = "ticker_not_in_panel"
    on, en, rel = _to_ns(df["onset_date"]), _to_ns(df["end_date"]), _to_ns(df["release_date"])
    pk = _to_ns(df["peak_date"]) if "peak_date" in df else np.full(len(df), np.iinfo(np.int64).min)
    nat = np.iinfo(np.int64).min
    status[(on == nat) | (en == nat) | (rel == nat)] = np.where(status[(on == nat) | (en == nat) | (rel == nat)] == "ok", "bad_date", status[(on == nat) | (en == nat) | (rel == nat)])
    out_of_panel = (en < times[0]) | (on > times[-1])
    status[out_of_panel & (status == "ok")] = "window_outside_panel"
    df["onset_t"] = np.clip(np.searchsorted(times, on, "left"), 0, T - 1)
    df["end_t"] = np.clip(np.searchsorted(times, en, "right") - 1, 0, T - 1)
    rel_t = np.searchsorted(times, rel, "left")                    # == T when published after the panel ends
    df["release_t"] = rel_t
    peak_t = np.where(pk == nat, np.nan, np.searchsorted(times, pk, "right") - 1).astype(float)
    df["peak_t"] = peak_t
    df["status"] = status
    for c, d in (("date_resolution", "day"), ("url", ""), ("source", "")):
        if c not in df:
            df[c] = d
    ok = df[df["status"] == "ok"]
    table = validate_anchor_table(ok[["case_id", "source", "asset", "anomaly_class", "onset_t", "peak_t", "end_t", "release_t", "date_resolution", "url"]]) if len(ok) else ok
    rep = (df.groupby(["anomaly_class", "status"])["case_id"].nunique().unstack(fill_value=0))
    rep["match_rate"] = rep.get("ok", 0) / rep.sum(axis=1)
    return table, rep.reset_index()

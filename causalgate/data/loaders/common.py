"""Helpers shared by the D1-D10 loaders.

Every loader is a BaseDatasetBuilder registered (tolerantly) as "cg_<name>" and returns a Panel that
    * carries RAW (un-normalised) price features so that L1 is switchable downstream:
      column 0 = log return (meta['ret_col'] = 0), then log volume, high-low range, ...  (meta['feat_names'])
    * has PIT fields attached with causalgate.data.pit.attach_availability:
      avail_text from document timestamps, avail_label from label release / event-end dates, universe as-of,
      avail_feat for restated / late fundamentals, and tradable from halt / limit / restriction calendars
    * sets meta = {freq, bars_per_year, horizons, raw_features: True, ret_col, feat_names, pit_notes, source, licence}
    * raises FileNotFoundError with the download recipe (scripts/cg_prepare_data.py) when raw files are missing.
      Nothing is ever fabricated: no loader synthesises prices, documents or labels.
"""
from __future__ import annotations
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.schema import Panel
from warn2trade.data.labels import abnormal_move_labels, impact_curves, bars_to_peak_from_labels
from warn2trade.utils.registry import DATASETS
from ..pit import attach_availability, default_availability, NEVER


def register_dataset(name: str):
    def deco(cls):
        try:
            DATASETS.register(name)(cls)
        except KeyError:
            pass
        return cls
    return deco


def require(root: str, paths: Iterable[str], recipe: str) -> None:
    missing = [p for p in paths if not os.path.exists(os.path.join(root, p))]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under {root}.\nDownload recipe:\n{recipe}\n(see scripts/cg_prepare_data.py)")


# ------------------------------------------------------------------------------------------- time alignment
NAT_NS = np.iinfo(np.int64).min                   # sentinel the loaders' _parse_ts use for unparseable timestamps


def next_bar_index(bar_times: np.ndarray, event_times: np.ndarray, side: str = "left") -> np.ndarray:
    """Availability bar of each event.

    BOUNDARY CONVENTION (used by every loader): an event is available at the first bar whose close time is
    >= the event time ("at or after"), so an event stamped exactly at a close is knowable at that close; one ns
    later it moves to the next bar. This matches plan Sec. 2.1 (a(x) <= t). `side` has numpy's meaning
    (side="right" gives the strict "close > event" variant). Events after the last bar, and the NaT sentinel
    NAT_NS (int64 min), get index len(bar_times): never knowable inside the panel -- callers should drop and
    count such rows. `bar_times` must be sorted (int64 ns or comparable)."""
    bt = np.asarray(bar_times)
    et = np.asarray(event_times)
    idx = np.searchsorted(bt, et, side=side)
    if et.dtype.kind in "iu":
        idx = np.where(et == NAT_NS, len(bt), idx)
    return idx


def to_ns(x) -> np.ndarray:
    return pd.to_datetime(x, utc=True).values.astype("datetime64[ns]").astype(np.int64)


def session_close_ns(dates, close_local: str = "16:00", tz: str = "America/New_York") -> np.ndarray:
    """Bar close timestamps (UTC ns) for a list of session dates at a local close time."""
    idx = pd.to_datetime(pd.Series(dates)).dt.tz_localize(None)
    ts = pd.to_datetime(idx.dt.strftime("%Y-%m-%d") + " " + close_local).dt.tz_localize(tz).dt.tz_convert("UTC")
    return ts.values.astype("datetime64[ns]").astype(np.int64)


# ------------------------------------------------------------------------------------------- features
def raw_price_features(close: np.ndarray, volume: Optional[np.ndarray] = None, high: Optional[np.ndarray] = None,
                       low: Optional[np.ndarray] = None, spread: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """(T, N, F) raw features: [log_ret, log_volume, hl_range, log_spread_bps] (missing inputs are dropped)."""
    T, N = close.shape
    lr = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr[1:] = np.log(close[1:]) - np.log(close[:-1])
    feats, names = [lr], ["log_ret"]
    if volume is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            feats.append(np.log1p(np.where(volume > 0, volume, np.nan))); names.append("log_volume")
    if high is not None and low is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            feats.append((high - low) / close); names.append("hl_range")
    if spread is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            feats.append(np.log(np.clip(spread, 1e-6, None) * 1e4)); names.append("log_spread_bps")
    return np.stack(feats, -1).astype(np.float32), names


def liquidity_fields(close: np.ndarray, volume: Optional[np.ndarray], high: Optional[np.ndarray] = None, low: Optional[np.ndarray] = None,
                     half_spread: Optional[np.ndarray] = None, adv_window: int = 20, notional_frac: float = 0.05) -> np.ndarray:
    """(T, N, 3): [relative half-spread, depth proxy (=1), ADV fraction]. Half-spread proxy = 0.1 * (high-low)/close / 2
    when quotes are absent (StockNet recipe); ADV fraction = participation proxy ref_t / dv_t where dv_t is the
    trailing `adv_window`-bar mean dollar volume (bars <= t) and ref_t is the CROSS-SECTIONAL median of dv over
    assets at the same bar t (causal: F_t-measurable; no full-sample statistic). Clipped to [1e-4, 1]."""
    T, N = close.shape
    if half_spread is None:
        if high is not None and low is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                half_spread = 0.1 * 0.5 * (high - low) / close
        else:
            half_spread = np.full((T, N), 5e-4)
    half_spread = np.nan_to_num(np.asarray(half_spread, dtype=np.float64), nan=5e-4)
    half_spread = np.clip(half_spread, 1e-5, 0.05)
    if volume is None:
        adv_frac = np.full((T, N), 0.02)
    else:
        dv = pd.DataFrame(np.nan_to_num(volume * close)).rolling(adv_window, min_periods=1).mean().to_numpy()
        pos = np.where(dv > 0, dv, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(pos, axis=1, keepdims=True)                 # per-bar cross-sectional median (bar t only)
            ref = np.where(np.isfinite(ref), ref, 1.0)
            ratio = ref / np.maximum(np.where(dv > 0, dv, 1.0), 1e-9)
        adv_frac = np.clip(np.where(dv > 0, ratio, 0.02), 1e-4, 1.0)
    return np.stack([half_spread, np.ones((T, N)), adv_frac], -1).astype(np.float32)


def rule_labels_and_peaks(close: np.ndarray, volume: Optional[np.ndarray], horizons: Sequence[int], horizon: int = 5,
                          k_ret: float = 3.0, k_vol: float = 3.0, window: int = 60, lead: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """abnormal_move_labels + bars_to_peak + avail_label. labels[t] = OR_{d<=lead} onset[t+d] and onset[s] uses data
    up to s + horizon, so avail_label = t + max(horizon + lead, max(horizons)) (bars_to_peak uses impact curves to
    t + max(horizons)). Returns (labels int8, bars_to_peak, avail_label)."""
    vol = volume if volume is not None else np.ones_like(close)
    labels = abnormal_move_labels(close, np.nan_to_num(vol), horizon, k_ret, k_vol, window, lead)
    imp = impact_curves(close, horizons)
    btp = bars_to_peak_from_labels(labels, imp, horizons)
    T, N = close.shape
    avail_label = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None] + int(max(int(horizon) + int(lead), max(horizons))), (T, N)).copy()
    return labels, btp, avail_label


def event_table_labels(T: int, N: int, events: pd.DataFrame, lead: int, horizons: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curated event table -> labels / bars_to_peak / avail_label. `events` columns: asset (int), onset_t, peak_t, end_t
    (bar indices); label on [onset_t - lead, peak_t], bars_to_peak exact (negative after the peak), label knowable at end_t.
    An end_t beyond the panel (event still open at the last bar) is stamped NEVER rather than clipped to T-1."""
    labels = np.zeros((T, N), dtype=np.int8)
    btp = np.full((T, N), np.nan)
    avail = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None] + int(max(horizons)), (T, N)).copy()
    for r in events.itertuples(index=False):
        i, t0, pk, en = int(r.asset), int(r.onset_t), int(r.peak_t), int(r.end_t)
        lo, hi = max(0, t0 - lead), min(T - 1, pk)
        labels[lo : hi + 1, i] = 1
        rng = np.arange(lo, min(T - 1, en) + 1)
        btp[rng, i] = pk - rng
        avail[rng, i] = en if en <= T - 1 else NEVER
    return labels, btp, avail


def docs_to_slots(T: int, N: int, M: int, D: int, docs: pd.DataFrame, bar_times_ns: np.ndarray, embeddings: np.ndarray,
                  max_age_bars: int = 5) -> Dict[str, np.ndarray]:
    """Place documents into Panel.text slots by their *publication* time.

    docs columns: asset (int), publish_ns (int64 UTC), emb_row (row into `embeddings`). A document is placed at its
    availability bar a = first bar close >= publish time (NaT publish times are dropped), and repeated (with growing age) for up to `max_age_bars`
    bars so that the window encoder sees recent documents; avail_text = a for every copy. Returns text, text_mask,
    text_age, avail_text (and keeps the M most recent per (t, i))."""
    text = np.zeros((T, N, M, D), dtype=np.float32)
    mask = np.zeros((T, N, M), dtype=bool)
    age = np.zeros((T, N, M), dtype=np.float32)
    avail = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None, None], (T, N, M)).copy()
    a_bar = next_bar_index(bar_times_ns, docs["publish_ns"].to_numpy())
    order = np.argsort(a_bar, kind="stable")
    fill = np.zeros((T, N), dtype=int)
    for j in order[::-1]:                         # most recent first
        a = int(a_bar[j]); i = int(docs["asset"].iloc[j]); e = int(docs["emb_row"].iloc[j])
        if a >= T:
            continue
        for k in range(max_age_bars):
            t = a + k
            if t >= T:
                break
            slot = fill[t, i]
            if slot >= M:
                continue
            text[t, i, slot] = embeddings[e]; mask[t, i, slot] = True; age[t, i, slot] = k; avail[t, i, slot] = a
            fill[t, i] += 1
    return {"text": text, "text_mask": mask, "text_age": age, "avail_text": avail}


def finish_panel(panel: Panel, avail_label: Optional[np.ndarray] = None, avail_text: Optional[np.ndarray] = None,
                 universe: Optional[np.ndarray] = None, avail_feat: Optional[np.ndarray] = None, avail_graph: Optional[np.ndarray] = None) -> Panel:
    panel.validate()
    attach_availability(panel, avail_label=avail_label, avail_text=avail_text, universe=universe, avail_feat=avail_feat, avail_graph=avail_graph)
    default_availability(panel)
    return panel


def stack_to_calendar(panel: Panel, order: Optional[Sequence[int]] = None, gap: int = 200, name: str = "CALENDAR") -> Panel:
    """Concatenate the columns of a stacked, NOT time-aligned panel (one column per event window / segment, all
    starting at row 0) into a single calendar-ordered series of N*(T+gap) bars with one asset.

    Columns are placed in `order` (default: column order, which must be chronological), separated by `gap` empty rows
    (NaN mid / features, untradable, label 0 knowable at once) so that no lookback window, return, impact horizon or label
    crosses a boundary. Decision order then equals calendar order, which a stacked layout cannot guarantee: there,
    sequential operators (gates, normalisers) would process a later window's bar t before an earlier window's bar t+1,
    and any cross-column statistic at bar t would mix later windows into earlier decisions. PIT arrays are shifted by
    each window's offset (NEVER stays NEVER). Liquidity fields are kept per row (they must be trailing statistics of the
    same column). Text / LOB / graph are concatenated the same way when present."""
    from ..pit import get_avail
    T, N = panel.mid.shape
    order = np.arange(N) if order is None else np.asarray(order)
    fin = np.isfinite(panel.mid)
    lens = np.array([int(np.nonzero(fin[:, j])[0].max()) + 1 if fin[:, j].any() else 0 for j in order])   # trim trailing padding
    off = np.concatenate([[0], np.cumsum(lens + gap)[:-1]]).astype(int)
    Tc = int((lens + gap).sum())
    rows = np.zeros(Tc, dtype=bool)
    for k in range(len(order)):
        rows[off[k]: off[k] + lens[k]] = True

    def cat(a, fill):
        if a is None:
            return None
        out = np.full((Tc, 1) + a.shape[2:], fill, dtype=a.dtype)
        for k, j in enumerate(order):
            out[off[k]: off[k] + lens[k], 0] = a[: lens[k], j]
        return out

    def cat_avail(a):
        if a is None:
            return None
        big = NEVER // 2
        out = np.zeros((Tc, 1) + a.shape[2:], dtype=np.int64)
        for k, j in enumerate(order):
            col = a[: lens[k], j].astype(np.int64)
            out[off[k]: off[k] + lens[k], 0] = np.where(col >= big, NEVER, np.minimum(col + off[k], Tc - 1))
        gi = np.arange(Tc)[~rows]
        out[~rows, 0] = gi.reshape((-1,) + (1,) * (out.ndim - 2)) if out.ndim > 2 else gi
        return out

    meta = dict(panel.meta)
    meta.update({"layout": f"calendar: {len(order)} windows (trimmed lengths {int(lens.min())}..{int(lens.max())} bars), {gap}-bar empty gaps, single series",
                 "assets_time_aligned": True, "window_offset": off, "window_order": order, "window_bars": lens, "gap_bars": gap,
                 "stacked_assets": list(np.asarray(panel.assets)[order]), "split_axis": "time"})
    q = Panel(times=np.arange(Tc, dtype=np.int64), assets=[name], mid=cat(panel.mid, np.nan), price_feat=cat(panel.price_feat, np.float32(np.nan)),
              liq=cat(panel.liq, np.float32(np.nan)), tradable=cat(panel.tradable, False), labels=cat(panel.labels, panel.labels.dtype.type(0)),
              bars_to_peak=cat(panel.bars_to_peak, np.nan), lob=cat(panel.lob, np.float32(np.nan)) if panel.lob is not None else None,
              text=cat(panel.text, np.float32(0)), text_mask=cat(panel.text_mask, False), text_age=cat(panel.text_age, np.float32(0)),
              graph=cat(panel.graph, np.float32(0)), graph_mask=cat(panel.graph_mask, False), meta=meta)
    q.meta["pit_fields"] = []
    uni = get_avail(panel, "universe")
    return finish_panel(q, avail_label=cat_avail(get_avail(panel, "avail_label")), avail_text=cat_avail(get_avail(panel, "avail_text")),
                        avail_graph=cat_avail(get_avail(panel, "avail_graph")),
                        universe=(cat(uni, False)[:, :1] if uni is not None else None))

"""D2 limit-order-book panel: FI-2010 (free; Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis, J. Forecasting 2018,
arXiv:1705.03233) with the LOBSTER-licensed variant (Huang & Polak, https://lobsterdata.com) behind cfg variant='lobster'.
Registered as dataset "cg_fi2010".

FI-2010 source and licence: public benchmark, https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
(mirror used by DeepLOB: "BenchmarkDatasets.zip"). 5 NASDAQ-Nordic (Helsinki) stocks, 10 trading days (June 2010),
10 LOB levels, events SAMPLED EVERY 10 ORDER-BOOK UPDATES, no timestamps, no stock identifiers in the txt files.

Required raw layout under <root>/fi2010 (any sub-folder depth; the public zip unpacks to
BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_7.txt etc.):
    *Train_Dst_<Auction>_<Norm>_CF_7.txt        149 x n matrix, whitespace separated: days 1-7
    *Test_Dst_<Auction>_<Norm>_CF_{7,8,9}.txt   149 x n each: days 8, 9, 10
  <Norm> in {Raw, DecPre, ZScore, MinMax}. Only *Raw* (not shipped publicly; a user export in the same layout) or
  *DecPre* (decimal-precision: each feature divided by a constant power of ten, so returns / ratios are exact) qualify
  for RAW features; ZScore (which uses the PRIOR day's statistics -- a built-in PIT example) and MinMax are refused for
  price features because a normalised variant would hide the L1 channel. meta['pit_notes']['normalisation'] records which
  variant was loaded.
Matrix layout (rows; events are columns):
    0..39     LOB, per level i=1..10 in this order: ask_px_i, ask_sz_i, bid_px_i, bid_sz_i   (FI-2010 feature set u1)
    40..143   hand-crafted time-insensitive / time-sensitive features (ignored here; kept raw in meta if `keep_handcrafted`)
    144..148  mid-price trend labels for k = 10, 20, 30, 50, 100 events; values 1 = up, 2 = stationary, 3 = down
Assumption stated: the txt files carry no stock / day identifiers and the ordering (stock-major vs day-major) is
undocumented. Assets are therefore SEGMENTS of the concatenated event stream split where the level-1 mid price jumps by
more than `jump_thresh` (default 0.25 in log terms) between consecutive events (cfg segmentation='auto'), or explicit
cfg `segments: [[start, end, name], ...]` (segmentation='list'), or one stream (segmentation='none'). Two stocks at
similar price levels may end up in one segment; meta['segments'] records the boundaries and meta['warnings'] the count.
Segments are stacked at row 0 (asset columns are NOT time-aligned across segments: event clocks differ); the default
layout: calendar concatenates them in stream order into one series (common.stack_to_calendar), layout: stacked keeps columns.

Panel (T events x N segments):
    mid          (ask1 + bid1) / 2
    lob          (T, N, 40) RAW prices/sizes reordered to the schema layout: per level [bid_px, bid_sz, ask_px, ask_sz]
    price_feat   [log_ret(mid), log_depth_l1, log_spread_bps, ofi_l1 (Cont-Kukanov-Stoikov 2014 level-1 order-flow
                  imbalance from t-1 -> t), imbalance_l1, imbalance_l10]  -- all from bars <= t
    liq          [relative half-spread, depth_l1 / causal rolling-median depth, 0.02 (no ADV in an event stream)]
    tradable     spread <= `spread_ticks_max` (20) ticks AND both level-1 quotes present with positive size; tick size
                 = cfg tick_size or the smallest positive level-1 price increment in the segment (DecPre units)
    labels       abnormal_move_labels on mid with depth_l1 as the volume proxy (no trades in the LOB rows); the native
                 FI-2010 trend labels are kept in meta['fi2010_trend_labels'] (T, N, 5) int8 with horizons
                 meta['fi2010_trend_horizons'] = [10, 20, 30, 50, 100] (avail = t + k)
Point-in-time rules (meta['pit_notes']):
    avail_feat   = t (event index): a snapshot is knowable when it is observed
    avail_label  = t + max(label horizon + lead, max(horizons)) for the rule label
    universe     = segment has an event at row t
    fills        same-event fills are forbidden by the protocol (latency >= 1 event); see docs/causalgate_research_plan.md
bars_per_year  = 252 * 8000 (approximate: ~8k sampled events per stock-day in the public txt files)

LOBSTER variant (cfg variant='lobster'; licensed, academic): files <TICKER>_<YYYY-MM-DD>_<start>_<end>_orderbook_10.csv
(40 columns: ask_px1, ask_sz1, bid_px1, bid_sz1, ... prices in 1e-4 USD) and the matching *_message_10.csv (time in
seconds after midnight, event type, order id, size, price, direction). One asset per (ticker, day) file, sampled to
`bar_ms` buckets (last snapshot per bucket) or every `stride` events; tick 0.01 USD. meta['event_times_s'] keeps the
per-asset clock. Same features / PIT rules as above.
"""
from __future__ import annotations
import glob
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from .common import register_dataset, require, rule_labels_and_peaks, finish_panel

SOURCE_URL = "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649 (Ntakaris et al. 2018, arXiv:1705.03233)"
LICENCE = "FI-2010: public benchmark dataset (Ntakaris et al. 2018); LOBSTER variant: academic licence, https://lobsterdata.com"
RECIPE = ("# FI-2010 (free): download BenchmarkDatasets.zip from the Fairdata record (arXiv:1705.03233) and unpack it under\n"
          "#   {root}/fi2010/   so that *Train_Dst_NoAuction_DecPre_CF_7.txt and *Test_Dst_NoAuction_DecPre_CF_{{7,8,9}}.txt exist\n"
          "# (the ZScore / MinMax variants are not sufficient: raw or DecPre prices are required).\n"
          "# LOBSTER (licensed): place <TICKER>_<date>_*_orderbook_10.csv and *_message_10.csv under {root}/lobster/ and set variant: lobster")
TREND_HORIZONS = [10, 20, 30, 50, 100]
N_LEVELS = 10


# ------------------------------------------------------------------------------------------------ parsing
def read_fi2010_matrix(path: str) -> np.ndarray:
    """Whitespace-separated numeric matrix with one feature per LINE (149 lines in the public files) -> (rows, n)."""
    with open(path, "r") as f:
        first = f.readline()
    n_cols = len(first.split())
    flat = np.fromfile(path, dtype=np.float64, sep=" ")
    if n_cols == 0 or flat.size % n_cols != 0:
        mat = np.loadtxt(path, dtype=np.float64)
    else:
        mat = flat.reshape(-1, n_cols)
    if mat.ndim != 2 or mat.shape[0] < 4 * N_LEVELS:
        raise ValueError(f"{path}: expected >= {4 * N_LEVELS} feature rows (FI-2010 layout), got shape {mat.shape}")
    return mat


def discover_fi2010_files(base: str, auction: str = "NoAuction", prefer: Sequence[str] = ("raw", "decpre")) -> Tuple[List[str], str]:
    """(ordered file list [train CF_7, test CF_7, CF_8, CF_9], normalisation token) for the first available variant."""
    files = glob.glob(os.path.join(base, "**", "*.txt"), recursive=True)
    for norm in prefer:
        cand = [f for f in files if norm in os.path.basename(f).lower() and auction.lower() in os.path.basename(f).lower()]
        train = sorted(f for f in cand if os.path.basename(f).lower().startswith("train"))
        test = sorted(f for f in cand if os.path.basename(f).lower().startswith("test"))
        if train:
            train = [f for f in train if re.search(r"cf_7\b", os.path.basename(f).lower())] or train[:1]
            test = sorted(test, key=lambda p: int(re.findall(r"cf_(\d+)", os.path.basename(p).lower())[0]) if re.findall(r"cf_(\d+)", os.path.basename(p).lower()) else 0)
            return train[:1] + test, norm
    return [], ""


def split_lob_rows(lob40: np.ndarray) -> Dict[str, np.ndarray]:
    """(n, 40) in FI-2010 / LOBSTER order [ask_px, ask_sz, bid_px, bid_sz] per level -> dict of (n, 10) arrays."""
    return {"ask_px": lob40[:, 0::4], "ask_sz": lob40[:, 1::4], "bid_px": lob40[:, 2::4], "bid_sz": lob40[:, 3::4]}


def to_schema_lob(parts: Dict[str, np.ndarray]) -> np.ndarray:
    """-> (n, 40) in the schema layout per level [bid_px, bid_sz, ask_px, ask_sz]."""
    return np.stack([parts["bid_px"], parts["bid_sz"], parts["ask_px"], parts["ask_sz"]], -1).reshape(parts["bid_px"].shape[0], -1)


def infer_tick(px: np.ndarray) -> float:
    """Smallest positive increment among the sorted unique finite positive prices (DecPre / raw units)."""
    u = np.unique(px[np.isfinite(px) & (px > 0)])
    if u.size < 2:
        return float("nan")
    d = np.diff(u)
    d = d[d > 0]
    return float(d.min()) if d.size else float("nan")


def segment_stream(mid: np.ndarray, jump_thresh: float) -> List[Tuple[int, int]]:
    """[(start, end)) segments of a 1-D mid series split at |log jump| > jump_thresh."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lj = np.abs(np.diff(np.log(mid)))
    cuts = np.nonzero(np.nan_to_num(lj, nan=0.0) > jump_thresh)[0] + 1
    bounds = [0] + cuts.tolist() + [len(mid)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1) if bounds[k + 1] > bounds[k]]


# ------------------------------------------------------------------------------------------------ features
def lob_feature_block(parts: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Causal features from (n, 10) level arrays of ONE stream. Returns (feat (n, 6), names, mid (n,), spread (n,))."""
    a1, as1, b1, bs1 = parts["ask_px"][:, 0], parts["ask_sz"][:, 0], parts["bid_px"][:, 0], parts["bid_sz"][:, 0]
    mid = 0.5 * (a1 + b1)
    spread = a1 - b1
    n = len(mid)
    lr = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr[1:] = np.log(mid[1:]) - np.log(mid[:-1])
        depth1 = bs1 + as1
        log_depth = np.log1p(np.where(depth1 > 0, depth1, np.nan))
        log_spread = np.log(np.clip(spread / mid, 1e-6, None) * 1e4)
        imb1 = (bs1 - as1) / np.where(depth1 > 0, depth1, np.nan)
        tb, ta = parts["bid_sz"].sum(1), parts["ask_sz"].sum(1)
        imb10 = (tb - ta) / np.where(tb + ta > 0, tb + ta, np.nan)
    ofi = np.full(n, np.nan)
    if n > 1:
        bp, bq, ap, aq = b1, bs1, a1, as1
        e = ((bp[1:] >= bp[:-1]) * bq[1:] - (bp[1:] <= bp[:-1]) * bq[:-1]
             - (ap[1:] <= ap[:-1]) * aq[1:] + (ap[1:] >= ap[:-1]) * aq[:-1])
        ofi[1:] = e
    feat = np.stack([lr, log_depth, log_spread, ofi, imb1, imb10], -1)
    return feat, ["log_ret", "log_depth_l1", "log_spread_bps", "ofi_l1", "imbalance_l1", "imbalance_l10"], mid, spread


def causal_rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=max(5, window // 10)).median().to_numpy()


# ------------------------------------------------------------------------------------------------ builder
@register_dataset("cg_fi2010")
class FI2010CausalBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_fi2010", "lob", 252 * 8000, (10, 20, 30, 50, 100)

    # ---- raw streams: list of dicts {name, lob40 (n, 40) in source order, trend (n, 5) or None, times_s or None}
    def _streams_fi2010(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        c = self.cfg
        sub = c.get("subdir", "fi2010")
        base = os.path.join(self.root, sub)
        recipe = RECIPE.format(root=self.root)
        require(self.root, [sub], recipe)
        if c.get("files"):
            files, norm = [os.path.join(base, f) for f in c["files"]], str(c.get("normalisation", "decpre"))
            require(base, list(c["files"]), recipe)
        else:
            files, norm = discover_fi2010_files(base, c.get("auction", "NoAuction"), tuple(c.get("prefer", ("raw", "decpre"))))
            if not files:
                raise FileNotFoundError(f"No *Raw*/*DecPre* FI-2010 txt files under {base} (ZScore/MinMax cannot give raw prices).\n"
                                        f"Download recipe:\n{recipe}\n(see scripts/cg_prepare_data.py)")
        mats = [read_fi2010_matrix(f) for f in files]
        rows = mats[0].shape[0]
        mat = np.concatenate(mats, axis=1)
        lob40 = mat[: 4 * N_LEVELS].T.copy()                                      # (n, 40)
        trend = mat[-5:].T.astype(np.int8).copy() if rows >= 4 * N_LEVELS + 5 else None
        if c.get("max_events"):
            lob40, trend = lob40[: int(c["max_events"])], (trend[: int(c["max_events"])] if trend is not None else None)
        parts = split_lob_rows(lob40)
        warnings: List[str] = []
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.nanmedian(parts["ask_px"][:, 0] / parts["bid_px"][:, 0])
        if np.isfinite(ratio) and not (0.5 < ratio < 2.0):                    # DecPre scaled bid and ask by different powers of ten
            k = int(np.round(np.log10(ratio)))
            lob40 = lob40.copy(); lob40[:, 2::4] *= 10.0 ** k
            parts = split_lob_rows(lob40)
            warnings.append(f"bid prices rescaled by 1e{k} to match ask prices (median ask/bid ratio {ratio:.3g}: per-feature DecPre constants differed)")
        mid = 0.5 * (parts["ask_px"][:, 0] + parts["bid_px"][:, 0])
        segm = str(c.get("segmentation", "auto"))
        if segm == "auto":
            segs = segment_stream(mid, float(c.get("jump_thresh", 0.25)))
            names = [f"FI_s{k:02d}" for k in range(len(segs))]
        elif segm == "list":
            segs = [(int(s[0]), int(s[1])) for s in c["segments"]]
            names = [str(s[2]) if len(s) > 2 else f"FI_s{k:02d}" for k, s in enumerate(c["segments"])]
        else:
            segs, names = [(0, len(mid))], ["FI_all"]
        if len(segs) != int(c.get("n_assets_expected", 5)):
            warnings.append(f"{len(segs)} segment(s) found, {c.get('n_assets_expected', 5)} stocks expected: segments may merge stocks at similar price levels (no identifiers in the txt files)")
        streams = [{"name": nm, "lob40": lob40[a:b], "trend": (trend[a:b] if trend is not None else None), "times_s": None}
                   for nm, (a, b) in zip(names, segs)]
        info = {"files": [os.path.relpath(f, self.root) for f in files], "normalisation": norm, "segments": [[int(a), int(b), nm] for nm, (a, b) in zip(names, segs)],
                "warnings": warnings, "n_events_total": int(len(mid)), "variant": "fi2010"}
        return streams, info

    def _streams_lobster(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        c = self.cfg
        sub = c.get("subdir", "lobster")
        base = os.path.join(self.root, sub)
        recipe = RECIPE.format(root=self.root)
        require(self.root, [sub], recipe)
        ob_files = sorted(glob.glob(os.path.join(base, "**", "*_orderbook_10.csv"), recursive=True))
        if c.get("tickers"):
            ob_files = [f for f in ob_files if os.path.basename(f).split("_")[0] in set(c["tickers"])]
        if not ob_files:
            raise FileNotFoundError(f"No *_orderbook_10.csv under {base}.\nDownload recipe:\n{recipe}\n(see scripts/cg_prepare_data.py)")
        streams, warnings = [], []
        bar_ms, stride = c.get("bar_ms"), int(c.get("stride", 1))
        for ob in ob_files[: int(c.get("max_files", 10 ** 9))]:
            lob = pd.read_csv(ob, header=None).to_numpy(dtype=np.float64)
            if lob.shape[1] < 4 * N_LEVELS:
                warnings.append(f"{os.path.basename(ob)}: {lob.shape[1]} columns < 40, skipped"); continue
            lob = lob[:, : 4 * N_LEVELS]
            lob[:, 0::2] = lob[:, 0::2] / 1e4                                 # prices in 1e-4 USD -> USD (even cols are prices)
            lob[:, 0::2] = np.where(np.abs(lob[:, 0::2]) > 1e5, np.nan, lob[:, 0::2])   # LOBSTER empty-level sentinels (+-9999999999)
            msg = ob.replace("_orderbook_10.csv", "_message_10.csv")
            times = pd.read_csv(msg, header=None).iloc[:, 0].to_numpy(dtype=np.float64) if os.path.exists(msg) else None
            if times is not None and len(times) != len(lob):
                warnings.append(f"{os.path.basename(ob)}: message/orderbook length mismatch; clock dropped"); times = None
            if bar_ms and times is not None:
                bucket = np.floor(times * 1000.0 / float(bar_ms)).astype(np.int64)
                last = np.nonzero(np.r_[bucket[1:] != bucket[:-1], True])[0]       # last snapshot of each bucket
                lob, times = lob[last], times[last]
            elif stride > 1:
                lob, times = lob[::stride], (times[::stride] if times is not None else None)
            name = os.path.basename(ob).replace("_orderbook_10.csv", "")
            streams.append({"name": name, "lob40": lob, "trend": None, "times_s": times})
        info = {"files": [os.path.relpath(f, self.root) for f in ob_files], "normalisation": "raw (USD; LOBSTER prices / 1e4)",
                "segments": [[0, int(len(s["lob40"])), s["name"]] for s in streams], "warnings": warnings,
                "n_events_total": int(sum(len(s["lob40"]) for s in streams)), "variant": "lobster", "bar_ms": bar_ms, "stride": stride}
        return streams, info

    def build_panel(self) -> Panel:
        c = self.cfg
        variant = str(c.get("variant", "fi2010"))
        streams, info = self._streams_lobster() if variant == "lobster" else self._streams_fi2010()
        horizons = tuple(c.get("horizons", self.horizons))
        N = len(streams)
        T = max(len(s["lob40"]) for s in streams)
        mid = np.full((T, N), np.nan)
        feat = np.full((T, N, 6), np.nan, dtype=np.float32)
        lob = np.full((T, N, 4 * N_LEVELS), np.nan, dtype=np.float32)
        liq = np.zeros((T, N, 3), dtype=np.float32)
        tradable = np.zeros((T, N), dtype=bool)
        depth1 = np.full((T, N), np.nan)
        trend = np.zeros((T, N, 5), dtype=np.int8)
        ticks, names = [], []
        tick_cfg = c.get("tick_size", 0.01 if variant == "lobster" else None)
        med_win = int(c.get("depth_median_window", 1000))
        for j, s in enumerate(streams):
            n = len(s["lob40"])
            parts = split_lob_rows(s["lob40"])
            f, names, m, spread = lob_feature_block(parts)
            a1, b1, as1, bs1 = parts["ask_px"][:, 0], parts["bid_px"][:, 0], parts["ask_sz"][:, 0], parts["bid_sz"][:, 0]
            tick = float(tick_cfg) if tick_cfg is not None else infer_tick(np.concatenate([a1, b1]))
            ticks.append(tick)
            quotes_ok = np.isfinite(a1) & np.isfinite(b1) & (a1 > 0) & (b1 > 0) & (as1 > 0) & (bs1 > 0) & (spread >= 0)
            wide = (spread > float(c.get("spread_ticks_max", 20)) * tick) if np.isfinite(tick) else np.zeros(n, bool)
            mid[:n, j] = m
            feat[:n, j] = f.astype(np.float32)
            lob[:n, j] = to_schema_lob(parts).astype(np.float32)
            d1 = bs1 + as1
            depth1[:n, j] = d1
            with np.errstate(invalid="ignore", divide="ignore"):
                rel_hs = np.clip(np.nan_to_num(0.5 * spread / m, nan=5e-4), 1e-5, 0.05)
                rel_depth = d1 / causal_rolling_median(d1, med_win)
            liq[:n, j, 0] = rel_hs
            liq[:n, j, 1] = np.nan_to_num(rel_depth, nan=1.0)
            liq[:n, j, 2] = 0.02
            tradable[:n, j] = quotes_ok & ~wide
            if s["trend"] is not None:
                trend[:n, j] = s["trend"]
        lab = c.get("labels", {})
        labels, btp, avail_label = rule_labels_and_peaks(mid, depth1, horizons, horizon=int(lab.get("horizon", 50)),
                                                         k_ret=float(lab.get("k_ret", 3.0)), k_vol=float(lab.get("k_vol", 3.0)),
                                                         window=int(lab.get("window", 600)), lead=int(lab.get("lead", 10)))
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": int(c.get("bars_per_year", self.bars_per_year)), "horizons": list(horizons),
            "raw_features": True, "ret_col": 0, "feat_names": names, "source": SOURCE_URL, "licence": LICENCE, "dataset": "fi2010",
            "variant": variant, "root": os.path.join(self.root, c.get("subdir", "lobster" if variant == "lobster" else "fi2010")),
            "n_assets": N, "assets_time_aligned": False, "tick_size": ticks, "lob_layout": {"levels": N_LEVELS, "per_level": ["bid_px", "bid_sz", "ask_px", "ask_sz"],
                                                        "source_row_order": "ask_px_i, ask_sz_i, bid_px_i, bid_sz_i (FI-2010 rows 0-39 / LOBSTER orderbook_10 columns)",
                                                        "units": info["normalisation"]},
            "fi2010_trend_horizons": TREND_HORIZONS, "fi2010_trend_labels": trend, "fi2010_trend_values": {"1": "up", "2": "stationary", "3": "down", "0": "padding"},
            "label_rule": {"type": "abnormal_move_labels", "volume_proxy": "level-1 depth (bid_sz1 + ask_sz1)", **{k: lab.get(k, v) for k, v in dict(horizon=50, k_ret=3.0, k_vol=3.0, window=600, lead=10).items()}},
            "required_layout": ["fi2010/**/Train_Dst_NoAuction_{Raw|DecPre}_CF_7.txt", "fi2010/**/Test_Dst_NoAuction_{Raw|DecPre}_CF_{7,8,9}.txt",
                                "lobster/<TICKER>_<date>_<start>_<end>_orderbook_10.csv + _message_10.csv (variant lobster)"],
            "pit_notes": {
                "normalisation": f"loaded the '{info['normalisation']}' variant; ZScore (prior-day statistics) and MinMax variants are refused for price features",
                "avail_feat": "t (event index): a snapshot is knowable when observed; FI-2010 has no wall-clock timestamps",
                "avail_label": "t + max(label horizon + lead, max(horizons)) for the rule label; native trend label k is knowable at t + k",
                "universe": "segment has an event at row t (segments stacked at row 0; columns are not time-aligned)",
                "tradable": f"spread <= {c.get('spread_ticks_max', 20)} ticks and both level-1 quotes present with positive size",
                "fills": "same-event fills forbidden: latency >= 1 event",
            },
            "times": "event index within the segment (no timestamps in FI-2010)",
            **{k: v for k, v in info.items() if k != "normalisation"},
        }
        if variant == "lobster":
            meta["event_times_s"] = [s["times_s"] for s in streams]
        panel = Panel(times=np.arange(T, dtype=np.int64), assets=[s["name"] for s in streams], mid=mid, price_feat=feat, liq=liq,
                      tradable=tradable, labels=labels, bars_to_peak=btp, lob=lob, meta=meta)
        panel = finish_panel(panel, avail_label=avail_label, universe=np.isfinite(mid))
        if str(c.get("layout", "calendar")) == "calendar":
            # segments are in stream (file) order = calendar order; see common.stack_to_calendar for why
            from .common import stack_to_calendar
            panel = stack_to_calendar(panel, None, int(c.get("gap_bars", 200)), "FI2010_STREAM")
        return panel

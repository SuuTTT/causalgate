"""D8 Microsoft Qlib daily bundles (US: S&P 500 universe; CN: CSI 300 universe) with Alpha158-style RAW features.
Registered as dataset "cg_qlib_alpha158". Source: https://github.com/microsoft/qlib (data bundle built by Qlib's
Yahoo Finance collector; research use, Yahoo-derived -- no redistribution claim is made here).

Required raw layout under <root>/<subdir> (subdir default "qlib_us" for region us, "qlib_cn" for region cn; this is
exactly what `python -m qlib.run.get_data qlib_data --target_dir <subdir> --region us|cn` writes):
    calendars/day.txt                 one trading date per line (YYYY-MM-DD)
    instruments/all.txt               <INSTRUMENT>\t<start>\t<end> listing ranges
    instruments/sp500.txt | csi300.txt  membership ranges (several rows per instrument allowed) -> as-of universe
    features/<instrument lower-case>/<field>.day.bin   float32: [start_index, v_0, v_1, ...] where v_k belongs to
                                      calendar[start_index + k]; fields open, high, low, close, volume, factor, change
                                      (the bundle stores split/dividend-ADJUSTED prices; raw price = price / factor;
                                      volume is factor-adjusted share volume)
Qlib itself is NOT required (the binary layout is read directly) and Qlib's processors (ZScoreNorm, CSRankNorm,
fit_start/fit_end) are deliberately NOT applied: that fit range is the L1 channel the audit switches downstream.

Panel (T trading days x N instruments):
    mid          $close (adjusted, as stored)
    price_feat   [log_ret, log_volume, hl_range] + Alpha158 expressions computed causally from the raw fields
                 (cfg alpha158: 'lite' (default) = KMID, KLEN, KUP, KLOW and ROC/MA/STD/MAX/MIN/VMA/VSTD over
                 windows (5, 10, 20, 60) = 32 columns with Qlib's names; 'none' = the three raw columns only).
                 The full 158-expression set needs Qlib's Alpha158 handler (hook: cfg alpha158: 'full' raises
                 ImportError when `qlib` is missing; never emulated).
    liq          [half-spread proxy from the daily range, 1, ADV fraction from close * volume]
    tradable     close finite and volume > 0, and NOT limit-hit under cfg `limit_rule`:
                 'none' (default for us) | 'cn_10pct' (default for cn): |change| >= limit_pct - 0.005 with close == high
                 (limit-up) or close == low (limit-down). The bundle ships NO suspension / limit flags: suspensions
                 appear as missing rows (-> untradable); ST (5 %) and ChiNext/STAR (20 %) boards are NOT special-cased.
    labels       abnormal_move_labels(horizon 5, k_ret 3, k_vol 3, window 60, lead 3) ("tabular outlier" setting)
Point-in-time rules (meta['pit_notes']):
    avail_feat   = t (session close: 16:00 America/New_York for us, 15:00 Asia/Shanghai for cn)
    avail_label  = t + max(label horizon + lead, max(horizons))
    universe     = as-of membership from instruments/<universe>.txt: universe[t, i] = True iff date_t lies in one of
                   the instrument's [start, end] ranges. Qlib's range files are built from historical index change
                   records keyed by EFFECTIVE date (the announcement precedes it), so membership known at t is
                   not back-filled; delisted names stay in the panel until their last range end (L6).
    tradable     see above (L7)
Assumptions stated: instrument directory names are lower-case (fallback: exact spelling); instruments without
feature files stay as all-NaN columns (counted in meta['warnings']); the universe file is required (no 'all' fallback,
because 'all' is not an index membership).
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from .common import register_dataset, require, session_close_ns, raw_price_features, liquidity_fields, rule_labels_and_peaks, finish_panel

SOURCE_URL = "https://github.com/microsoft/qlib (qlib_data bundle; Yahoo Finance-derived)"
LICENCE = "Qlib code MIT; the data bundle is Yahoo-derived and for research use only (no redistribution)"
RECIPE = ("python3 -m pip install pyqlib   # only for downloading the bundle; the loader reads the .bin files directly\n"
          "python3 -m qlib.run.get_data qlib_data --target_dir {root}/qlib_us --region us --interval 1d\n"
          "python3 -m qlib.run.get_data qlib_data --target_dir {root}/qlib_cn --region cn --interval 1d")
REGION_DEFAULTS = {"us": {"subdir": "qlib_us", "universe": "sp500", "close_local": "16:00", "tz": "America/New_York", "limit_rule": "none"},
                   "cn": {"subdir": "qlib_cn", "universe": "csi300", "close_local": "15:00", "tz": "Asia/Shanghai", "limit_rule": "cn_10pct"}}
FIELDS = ("open", "high", "low", "close", "volume", "factor", "change")
ALPHA_WINDOWS = (5, 10, 20, 60)


# ------------------------------------------------------------------------------------------------ bundle readers
def read_calendar(path: str) -> pd.DatetimeIndex:
    with open(path, "r") as f:
        days = [ln.strip() for ln in f if ln.strip()]
    return pd.DatetimeIndex(pd.to_datetime(days))


def read_instruments(path: str) -> pd.DataFrame:
    """instruments/*.txt -> DataFrame[inst, start, end] (tab separated; whitespace fallback)."""
    rows = []
    with open(path, "r") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t") if "\t" in ln else ln.split()
            if len(parts) >= 3:
                rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
            elif len(parts) == 1 and parts[0].strip():
                rows.append((parts[0].strip(), "1900-01-01", "2999-12-31"))
    df = pd.DataFrame(rows, columns=["inst", "start", "end"])
    df["start"], df["end"] = pd.to_datetime(df["start"], errors="coerce"), pd.to_datetime(df["end"], errors="coerce")
    return df


def read_bin(path: str, n_cal: int) -> np.ndarray:
    """<field>.day.bin -> (n_cal,) float64, NaN outside the stored range."""
    arr = np.fromfile(path, dtype="<f4")
    out = np.full(n_cal, np.nan)
    if arr.size == 0:
        return out
    start = int(arr[0])
    vals = arr[1:].astype(np.float64)
    if start < 0:
        vals, start = vals[-start:], 0
    n = min(len(vals), n_cal - start)
    if n > 0:
        out[start : start + n] = vals[:n]
    return out


def instrument_dir(fdir: str, inst: str) -> Optional[str]:
    for cand in (inst.lower(), inst, inst.upper()):
        p = os.path.join(fdir, cand)
        if os.path.isdir(p):
            return p
    return None


def universe_from_ranges(inst_df: pd.DataFrame, assets: Sequence[str], dates: pd.DatetimeIndex) -> np.ndarray:
    T, N = len(dates), len(assets)
    u = np.zeros((T, N), dtype=bool)
    d = dates.values.astype("datetime64[ns]")
    for j, a in enumerate(assets):
        for r in inst_df[inst_df["inst"] == a].itertuples(index=False):
            if pd.isna(r.start) or pd.isna(r.end):
                continue
            u[:, j] |= (d >= np.datetime64(r.start)) & (d <= np.datetime64(r.end))
    return u


# ------------------------------------------------------------------------------------------------ Alpha158-lite
def alpha158_lite(opn: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                  windows: Sequence[int] = ALPHA_WINDOWS) -> Tuple[np.ndarray, List[str]]:
    """Causal Alpha158 expressions (Qlib names) on (T, N) arrays; rolling windows look back only."""
    o, h, l, c, v = (pd.DataFrame(x) for x in (opn, high, low, close, volume))
    feats, names = [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        feats += [((c - o) / o), ((h - l) / o), ((h - np.maximum(o, c)) / o), ((np.minimum(o, c) - l) / o)]
        names += ["KMID", "KLEN", "KUP", "KLOW"]
        for n in windows:
            feats += [c.shift(n) / c, c.rolling(n).mean() / c, c.rolling(n).std() / c, h.rolling(n).max() / c, l.rolling(n).min() / c,
                      v.rolling(n).mean() / (v + 1e-12), v.rolling(n).std() / (v + 1e-12)]
            names += [f"ROC{n}", f"MA{n}", f"STD{n}", f"MAX{n}", f"MIN{n}", f"VMA{n}", f"VSTD{n}"]
    arr = np.stack([f.to_numpy(dtype=np.float64) for f in feats], -1)
    return arr.astype(np.float32), names


# ------------------------------------------------------------------------------------------------ builder
@register_dataset("cg_qlib_alpha158")
class QlibAlpha158CausalBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_qlib_alpha158", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        c = self.cfg
        region = str(c.get("region", "us")).lower()
        if region not in REGION_DEFAULTS:
            raise ValueError(f"region must be us or cn, got {region!r}")
        rd = REGION_DEFAULTS[region]
        sub = c.get("subdir") or rd["subdir"]
        universe_name = str(c.get("universe") or rd["universe"])
        root = self.root
        base = os.path.join(root, sub)
        recipe = RECIPE.format(root=root)
        require(root, [f"{sub}/calendars/day.txt", f"{sub}/instruments/{universe_name}.txt", f"{sub}/features"], recipe)
        cal = read_calendar(os.path.join(base, "calendars", "day.txt"))
        start, end = c.get("start"), c.get("end")
        lo = int(np.searchsorted(cal.values, np.datetime64(pd.Timestamp(start)))) if start else 0
        hi = int(np.searchsorted(cal.values, np.datetime64(pd.Timestamp(end)), side="right")) if end else len(cal)
        dates = cal[lo:hi]
        if len(dates) < 2:
            raise ValueError(f"Qlib {region}: fewer than 2 calendar days in [{start}, {end}]")
        inst_df = read_instruments(os.path.join(base, "instruments", f"{universe_name}.txt"))
        inst_df = inst_df[inst_df["inst"].astype(str).str.len() > 0]
        if c.get("instruments"):
            assets = [str(a) for a in c["instruments"]]
        else:
            order = inst_df.sort_values(["start", "inst"]).drop_duplicates("inst")
            assets = order["inst"].tolist()[: int(c.get("max_assets", 500))]
        T, N = len(dates), len(assets)
        fdir = os.path.join(base, "features")
        data = {f: np.full((T, N), np.nan) for f in FIELDS}
        warnings: List[str] = []
        for j, a in enumerate(assets):
            d = instrument_dir(fdir, a)
            if d is None:
                warnings.append(f"{a}: no feature directory"); continue
            for f in FIELDS:
                p = os.path.join(d, f"{f}.day.bin")
                if os.path.exists(p):
                    data[f][:, j] = read_bin(p, len(cal))[lo:hi]
        if not np.isfinite(data["close"]).any():
            raise FileNotFoundError(f"No close.day.bin found for {assets[:5]}... under {fdir}.\nDownload recipe:\n{recipe}")
        close, opn, high, low, vol, factor = (data[k] for k in ("close", "open", "high", "low", "volume", "factor"))
        change = data["change"]
        with np.errstate(invalid="ignore", divide="ignore"):
            if not np.isfinite(change).any():
                change[1:] = close[1:] / close[:-1] - 1.0
        horizons = tuple(c.get("horizons", self.horizons))
        feats, names = raw_price_features(close, vol, high, low)
        mode = str(c.get("alpha158", "lite"))
        if mode == "lite":
            extra, en = alpha158_lite(opn, high, low, close, vol, tuple(c.get("alpha_windows", ALPHA_WINDOWS)))
            feats, names = np.concatenate([feats, extra], -1), names + en
        elif mode == "full":
            try:
                import qlib  # noqa: F401
            except ImportError as e:
                raise ImportError("alpha158: 'full' needs `pip install pyqlib` and Qlib's Alpha158 handler; not installed. Use 'lite' or 'none'.") from e
            raise NotImplementedError("alpha158 'full' hook: run Qlib's Alpha158 handler without processors and pass the frame in; not wired in this session")
        elif mode != "none":
            raise ValueError(f"alpha158 must be lite | none | full, got {mode!r}")
        liq = liquidity_fields(close, vol, high, low)
        tradable = np.isfinite(close) & (np.nan_to_num(vol) > 0)
        limit_rule = str(c.get("limit_rule") or rd["limit_rule"])
        limit_hit = np.zeros((T, N), dtype=bool)
        if limit_rule == "cn_10pct":
            pct = float(c.get("limit_pct", 0.10))
            with np.errstate(invalid="ignore"):
                up = (change >= pct - 0.005) & np.isclose(close, high, rtol=1e-4, atol=0)
                dn = (change <= -(pct - 0.005)) & np.isclose(close, low, rtol=1e-4, atol=0)
            limit_hit = np.nan_to_num(up | dn, nan=False).astype(bool)
            tradable &= ~limit_hit
        elif limit_rule != "none":
            raise ValueError(f"limit_rule must be none | cn_10pct, got {limit_rule!r}")
        lab = c.get("labels", {})
        labels, btp, avail_label = rule_labels_and_peaks(close, vol, horizons, horizon=int(lab.get("horizon", 5)),
                                                         k_ret=float(lab.get("k_ret", 3.0)), k_vol=float(lab.get("k_vol", 3.0)),
                                                         window=int(lab.get("window", 60)), lead=int(lab.get("lead", 3)))
        universe = universe_from_ranges(inst_df, assets, dates)
        close_ns = session_close_ns(dates, c.get("close_local", rd["close_local"]), c.get("tz", rd["tz"]))
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(horizons), "raw_features": True, "ret_col": 0,
            "feat_names": names, "source": SOURCE_URL, "licence": LICENCE, "dataset": "qlib_alpha158", "region": region, "root": base,
            "universe_file": f"instruments/{universe_name}.txt", "n_assets": N, "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "mid": "$close as stored in the bundle (split/dividend adjusted; raw = close / factor)", "alpha158": mode,
            "processors": "none (Qlib ZScoreNorm / CSRankNorm / fit range deliberately not applied: L1 is switched downstream)",
            "limit_rule": limit_rule, "n_limit_hit_bars": int(limit_hit.sum()), "warnings": warnings,
            "label_rule": {"type": "abnormal_move_labels", **{k: lab.get(k, v) for k, v in dict(horizon=5, k_ret=3.0, k_vol=3.0, window=60, lead=3).items()}},
            "required_layout": ["calendars/day.txt", f"instruments/{universe_name}.txt (<INST>\\t<start>\\t<end>)", "instruments/all.txt",
                                "features/<inst lower-case>/{open,high,low,close,volume,factor,change}.day.bin"],
            "pit_notes": {
                "avail_feat": f"t: session close {rd['close_local']} {rd['tz']}",
                "avail_label": "t + max(label horizon + lead, max(horizons)): rule label knowable once the horizon elapsed",
                "universe": f"as-of membership from instruments/{universe_name}.txt effective-date ranges (not back-filled; delisted names kept until range end)",
                "tradable": f"close finite and volume > 0; limit rule '{limit_rule}' (bundle has no suspension / limit flags; suspensions = missing rows)",
            },
            "times": f"int64 UTC ns of the session close ({rd['close_local']} {rd['tz']})",
        }
        panel = Panel(times=close_ns, assets=assets, mid=close, price_feat=feats, liq=liq, tradable=tradable, labels=labels, bars_to_peak=btp, meta=meta)
        return finish_panel(panel, avail_label=avail_label, universe=universe)

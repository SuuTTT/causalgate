"""D6 CSI 300 / A-share daily bars with limit-up / limit-down events (CausalGate PIT loader). CONSTRUCTED dataset.

Registered as "cg_csi300_limit". Channels exercised: L6 (as-of index membership from the instruments date ranges),
L7 (limit-hit tradability: you cannot buy at limit-up nor sell at limit-down) and L1 (raw features, no global scaler).
Nothing is fabricated: missing raw files raise FileNotFoundError with the download recipe (RECIPES_D4_D10.md#d6).

Sources
    Qlib CN daily bundle (Microsoft Qlib; research use, see https://github.com/microsoft/qlib "Data" section),
        python -m qlib.run.get_data qlib_data --target_dir data_raw/csi300_limit/qlib_cn --region cn
    or a Tushare (https://tushare.pro, token required) / AkShare (https://akshare.akfamily.xyz) CSV export.
    ST status and the exact up_limit / down_limit prices are not in the Qlib bundle: optional side files below.

Raw file layout expected under cfg['root'] (default data_raw/csi300_limit) -- ONE of the two layouts
    (A) Qlib bundle  qlib_cn/calendars/day.txt               one YYYY-MM-DD per line
                     qlib_cn/instruments/csi300.txt          "<INST>\\t<start>\\t<end>" rows (cfg 'instruments' picks the file;
                                                             several rows per instrument = discontinuous membership)
                     qlib_cn/features/<inst lower-case>/{open,high,low,close,volume,factor}.day.bin
                       .bin = little-endian float32 array whose FIRST element is the calendar index of the first value
                       (qlib FileFeatureStorage layout). ASSUMPTION (verify on your bundle): $close is the adjusted
                       price and $factor the adjustment factor so that the traded price = $close / $factor; set cfg
                       qlib_prices: raw if your bundle stores unadjusted prices.
    (B) CSV export   bars.csv or csv/<code>.csv with columns (case-insensitive)
                       ts_code | code | symbol, trade_date | date (YYYYMMDD or YYYY-MM-DD), open, high, low, close,
                       vol | volume, amount (optional), pre_close (optional), up_limit / down_limit (optional, exact
                       Tushare stk_limit prices), adj_factor (optional, Tushare adj_factor -> continuous mid)
                     members.csv (optional)  ts_code | code, in_date | start, out_date | end (blank end = still a member)
    both             st_flags.csv (optional)  code, start, end : periods during which the stock was ST / *ST (5 % limit)

Limit rule (close at the limit; cfg 'limit_mode' = auto | exact | ratio | columns)
    daily price limit by code prefix: 688/689 (STAR) 20 %; 300/301 (ChiNext) 20 % from 2020-08-24 else 10 %;
    8xx / 4xx (BSE) 30 %; ST / *ST 5 % (except STAR / ChiNext); everything else 10 %.
    columns : |close - up_limit| <= half a tick                                          (needs up_limit / down_limit)
    exact   : limit = round_half_up(prev_close * (1 +- pct), 2) on UNADJUSTED prices, hit iff |close - limit| <= 0.0051
    ratio   : |close / prev_close - (1 +- pct)| <= cfg limit_ratio_tol (default 0.003; ASSUMPTION: covers the 0.01 tick
              rounding for prices above ~1.7 CNY) -- used when only adjusted prices are available
    First listing day (no prev_close) is never a limit hit. Without st_flags.csv the 5 % ST limit is not applied
    (documented false negatives; CSI 300 constituents are rarely ST).

Panel construction
    time axis      the bundle calendar (A) or the union of trade dates (B), restricted to cfg start / end; bar close =
                   15:00 Asia/Shanghai
    mid            adjusted close when available (continuous returns), traded price otherwise
    price_feat     RAW [log_ret, log_volume, hl_range] (column 0 = log return, meta['ret_col'] = 0)
    labels         1 on [t_hit - lead, t_peak] where t_hit = first bar of a run of consecutive same-direction limit closes
                   and t_peak = last bar of the run (= t_hit for the usual single-day event); bars_to_peak = t_peak - t
    tradable       False on EVERY limit-hit bar (conservative whole-bar reading of "cannot trade in the limit
                   direction"); the direction is in meta['limit_direction'] (T, N) int8 (+1 limit-up, -1 limit-down)
    universe       as-of membership from the instruments date ranges (A) / members.csv (B)

PIT fields
    avail_feat     t
    avail_label    rows inside an event: the first bar at which the run is observed to have ended = last limit bar + 1
                   (t_hit + 1 for single-bar events); NEVER when the run is still open at the last bar of the panel;
                   elsewhere t + max(horizon)
    universe       as-of: a constituent change becomes effective -- and knowable -- at its start date (L6). The leak this
                   guards against is applying today's membership backwards.
"""
from __future__ import annotations
import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from .common import (register_dataset, require, session_close_ns, raw_price_features, liquidity_fields, event_table_labels,
                     finish_panel)

RECIPE = """D6 CSI 300 limit-hit (see causalgate/data/loaders/RECIPES_D4_D10.md#d6):
  (A) python -m qlib.run.get_data qlib_data --target_dir data_raw/csi300_limit/qlib_cn --region cn
      (needs `pip install pyqlib` on a machine with network access; the bundle is ~1 GB)
  (B) or export Tushare `daily` (+ `stk_limit`, `index_weight`/`index_member` for membership, `namechange` for ST)
      or AkShare `stock_zh_a_hist` to data_raw/csi300_limit/bars.csv [+ members.csv, st_flags.csv]"""

_CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _code6(code: str) -> str:
    m = re.search(r"(\d{6})", str(code))
    return m.group(1) if m else str(code)


def _to_date(x) -> pd.Series:
    s = pd.Series(x).astype(str).str.strip()
    if s.str.fullmatch(r"\d{8}").fillna(False).all():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()


def _round_half_up_2(x: np.ndarray) -> np.ndarray:
    return np.floor(x * 100.0 + 0.5 + 1e-9) / 100.0


# ------------------------------------------------------------------------------------------------ readers
def _read_qlib_bin(path: str) -> Tuple[int, np.ndarray]:
    arr = np.fromfile(path, dtype="<f4")
    if arr.size == 0:
        return 0, arr
    return int(arr[0]), arr[1:].astype(np.float64)


def _read_qlib(root: str, cfg: Dict[str, Any]):
    q = os.path.join(root, "qlib_cn")
    inst_name = str(cfg.get("instruments", "csi300"))
    require(root, ["qlib_cn/calendars/day.txt", f"qlib_cn/instruments/{inst_name}.txt", "qlib_cn/features"], RECIPE)
    with open(os.path.join(q, "calendars", "day.txt")) as fh:
        cal_full = pd.DatetimeIndex([ln.strip() for ln in fh if ln.strip()])
    rows = []
    with open(os.path.join(q, "instruments", f"{inst_name}.txt")) as fh:
        for ln in fh:
            parts = [p for p in re.split(r"[\t ]+", ln.strip()) if p]
            if len(parts) >= 3:
                rows.append((parts[0].upper(), pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
            elif len(parts) == 1:
                rows.append((parts[0].upper(), cal_full[0], cal_full[-1]))
    members = pd.DataFrame(rows, columns=["code", "start", "end"])
    insts = sorted(members["code"].unique())
    if cfg.get("assets"):
        insts = [i for i in insts if i in {a.upper() for a in cfg["assets"]}]
    if cfg.get("max_assets"):
        insts = insts[: int(cfg["max_assets"])]
    Tf, N = len(cal_full), len(insts)
    fields = ("open", "high", "low", "close", "volume", "factor")
    A = {k: np.full((Tf, N), np.nan) for k in fields}
    for i, inst in enumerate(insts):
        d = None
        for cand in (inst.lower(), inst, inst.upper()):
            if os.path.isdir(os.path.join(q, "features", cand)):
                d = os.path.join(q, "features", cand)
                break
        if d is None:
            continue
        for k in fields:
            p = os.path.join(d, f"{k}.day.bin")
            if not os.path.exists(p):
                continue
            s0, vals = _read_qlib_bin(p)
            hi = min(Tf, s0 + len(vals))
            if hi > s0 >= 0:
                A[k][s0:hi, i] = vals[: hi - s0]
    has_factor = np.isfinite(A["factor"]).any()
    adjusted = str(cfg.get("qlib_prices", "adjusted")) == "adjusted"
    if has_factor and adjusted:
        raw = {k: A[k] / A["factor"] for k in ("open", "high", "low", "close")}      # traded prices
        mid, exact_ok = A["close"], True
    else:
        raw = {k: A[k] for k in ("open", "high", "low", "close")}
        mid, exact_ok = A["close"], not adjusted                                      # adjusted-only bundle -> ratio mode
    return cal_full, insts, raw, A["volume"], None, mid, members, None, None, exact_ok


def _read_csv_export(root: str, cfg: Dict[str, Any]):
    single, cdir = os.path.join(root, "bars.csv"), os.path.join(root, "csv")
    if os.path.exists(single):
        df = pd.read_csv(single)
    else:
        files = sorted(glob.glob(os.path.join(cdir, "*.csv")))
        require(root, ["bars.csv or csv/*.csv"] if not files else [], RECIPE)
        parts = []
        for f in files:
            g = pd.read_csv(f)
            if _col(g, "ts_code", "code", "symbol") is None:
                g["code"] = os.path.splitext(os.path.basename(f))[0]
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
    cc, dc = _col(df, "ts_code", "code", "symbol"), _col(df, "trade_date", "date", "datetime")
    if cc is None or dc is None:
        raise ValueError("CSV export needs ts_code|code|symbol and trade_date|date columns")
    df["_code"] = df[cc].astype(str).str.upper().str.strip()
    df["_date"] = _to_date(df[dc])
    df = df.dropna(subset=["_date"]).sort_values(["_code", "_date"])
    codes = sorted(df["_code"].unique())
    if cfg.get("assets"):
        codes = [k for k in codes if k in {a.upper() for a in cfg["assets"]}]
    if cfg.get("max_assets"):
        codes = codes[: int(cfg["max_assets"])]
    cal = pd.DatetimeIndex(sorted(df["_date"].unique()))
    T, N = len(cal), len(codes)
    pos = pd.Series(np.arange(T), index=cal)
    cpos = {k: i for i, k in enumerate(codes)}
    want = {"open": ("open",), "high": ("high",), "low": ("low",), "close": ("close",), "volume": ("vol", "volume"),
            "amount": ("amount",), "pre_close": ("pre_close", "preclose"), "up_limit": ("up_limit",), "down_limit": ("down_limit",),
            "adj_factor": ("adj_factor", "factor")}
    A: Dict[str, Optional[np.ndarray]] = {}
    df = df[df["_code"].isin(cpos)]
    rr, ii = pos.reindex(df["_date"]).to_numpy(), df["_code"].map(cpos).to_numpy()
    for k, names in want.items():
        c = _col(df, *names)
        if c is None:
            A[k] = None
            continue
        arr = np.full((T, N), np.nan)
        arr[rr, ii] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        A[k] = arr
    raw = {k: A[k] for k in ("open", "high", "low", "close")}
    if raw["close"] is None:
        raise ValueError("CSV export needs a close column")
    mid = raw["close"] * A["adj_factor"] if A["adj_factor"] is not None else raw["close"]
    members = None
    mp = os.path.join(root, "members.csv")
    if os.path.exists(mp):
        m = pd.read_csv(mp)
        mc, sc, ec = _col(m, "ts_code", "code", "symbol"), _col(m, "in_date", "start", "start_date"), _col(m, "out_date", "end", "end_date")
        members = pd.DataFrame({"code": m[mc].astype(str).str.upper().str.strip(), "start": _to_date(m[sc]),
                                "end": _to_date(m[ec]) if ec else pd.NaT})
        members["end"] = members["end"].fillna(cal[-1])
    limits = (A["up_limit"], A["down_limit"]) if A["up_limit"] is not None and A["down_limit"] is not None else None
    return cal, codes, raw, A["volume"], A["amount"], mid, members, limits, A["pre_close"], True   # exports are unadjusted


# ------------------------------------------------------------------------------------------------ limit logic
def _limit_pct(code: str, dates: pd.DatetimeIndex, st_mask: np.ndarray) -> np.ndarray:
    c6 = _code6(code)
    pct = np.full(len(dates), 0.10)
    if c6.startswith(("688", "689")):
        pct[:] = 0.20
    elif c6.startswith(("300", "301")):
        pct[:] = np.where(dates >= _CHINEXT_20PCT_FROM, 0.20, 0.10)
    elif c6.startswith(("8", "4")):
        pct[:] = 0.30
    if c6.startswith(("688", "689", "300", "301")):
        return pct
    return np.where(st_mask, 0.05, pct)


def _st_mask(root: str, codes: List[str], cal: pd.DatetimeIndex) -> np.ndarray:
    T, N = len(cal), len(codes)
    out = np.zeros((T, N), dtype=bool)
    p = os.path.join(root, "st_flags.csv")
    if not os.path.exists(p):
        return out
    s = pd.read_csv(p)
    cc, sc, ec = _col(s, "code", "ts_code", "symbol"), _col(s, "start", "start_date"), _col(s, "end", "end_date")
    c6 = {_code6(k): i for i, k in enumerate(codes)}
    for code, a, b in zip(s[cc].astype(str), _to_date(s[sc]), _to_date(s[ec]) if ec else [pd.NaT] * len(s)):
        i = c6.get(_code6(code))
        if i is None or pd.isna(a):
            continue
        b = cal[-1] if pd.isna(b) else b
        out[(cal >= a) & (cal <= b), i] = True
    return out


def _detect_hits(raw_close: np.ndarray, prev_close: np.ndarray, pct: np.ndarray, mode: str, limits, ratio_tol: float) -> np.ndarray:
    """(T, N) int8 direction of the limit close: +1 up, -1 down, 0 none."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "columns":
            up, dn = limits
            hit_up, hit_dn = np.abs(raw_close - up) <= 0.0051, np.abs(raw_close - dn) <= 0.0051
        elif mode == "exact":
            up, dn = _round_half_up_2(prev_close * (1.0 + pct)), _round_half_up_2(prev_close * (1.0 - pct))
            hit_up, hit_dn = np.abs(raw_close - up) <= 0.0051, np.abs(raw_close - dn) <= 0.0051
        else:
            ratio = raw_close / prev_close
            hit_up, hit_dn = np.abs(ratio - (1.0 + pct)) <= ratio_tol, np.abs(ratio - (1.0 - pct)) <= ratio_tol
    hit_up = np.nan_to_num(hit_up, nan=False).astype(bool) & np.isfinite(prev_close) & np.isfinite(raw_close)
    hit_dn = np.nan_to_num(hit_dn, nan=False).astype(bool) & np.isfinite(prev_close) & np.isfinite(raw_close)
    return (hit_up.astype(np.int8) - hit_dn.astype(np.int8)).astype(np.int8)


def _runs_to_events(direction: np.ndarray) -> pd.DataFrame:
    """Maximal runs of consecutive same-direction limit closes -> event rows (asset, onset_t, peak_t, end_t, direction)."""
    T, N = direction.shape
    rows = []
    for i in range(N):
        d = direction[:, i]
        idx = np.flatnonzero(d != 0)
        if idx.size == 0:
            continue
        # a new run starts where the index is not consecutive or the sign flips
        brk = np.r_[True, (np.diff(idx) != 1) | (d[idx[1:]] != d[idx[:-1]])]
        starts = idx[brk]
        ends = np.r_[idx[np.flatnonzero(brk)[1:] - 1], idx[-1]]
        for t0, t1 in zip(starts, ends):
            # the run's END (hence the peak / bars_to_peak) is first observable at t1 + 1, when bar t1+1 is not a
            # limit close; a run still open at the last bar ends past the panel (event_table_labels -> NEVER).
            rows.append({"asset": i, "onset_t": int(t0), "peak_t": int(t1), "end_t": int(t1) + 1, "direction": int(d[t0])})
    return pd.DataFrame(rows, columns=["asset", "onset_t", "peak_t", "end_t", "direction"])


# ------------------------------------------------------------------------------------------------ builder
@register_dataset("cg_csi300_limit")
class CSI300LimitPITBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_csi300_limit", "daily", 243, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        c, root = self.cfg, self.root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{root} does not exist.\nDownload recipe:\n{RECIPE}")
        horizons = [int(h) for h in c.get("horizons", self.horizons)]
        lead = int(c.get("lead", 3))
        layout = str(c.get("layout", "auto"))
        if layout == "auto":
            layout = "qlib" if os.path.isdir(os.path.join(root, "qlib_cn")) else "csv"
        if layout == "qlib":
            cal, codes, raw, volume, amount, mid, members, limits, pre_close, exact_ok = _read_qlib(root, c)
        elif layout == "csv":
            if not (os.path.exists(os.path.join(root, "bars.csv")) or glob.glob(os.path.join(root, "csv", "*.csv"))):
                raise FileNotFoundError(f"Missing qlib_cn/ or bars.csv / csv/*.csv under {root}.\nDownload recipe:\n{RECIPE}")
            cal, codes, raw, volume, amount, mid, members, limits, pre_close, exact_ok = _read_csv_export(root, c)
        else:
            raise ValueError("layout must be auto | qlib | csv")
        # ---- date window
        keep = np.ones(len(cal), dtype=bool)
        if c.get("start"):
            keep &= np.asarray(cal >= pd.Timestamp(c["start"]), dtype=bool)
        if c.get("end"):
            keep &= np.asarray(cal <= pd.Timestamp(c["end"]), dtype=bool)
        # keep one extra bar before the window so that prev_close of the first bar exists
        first = int(np.argmax(keep)) if keep.any() else 0
        if first > 0:
            keep[first - 1] = True
        cal = cal[keep]
        raw = {k: (v[keep] if v is not None else None) for k, v in raw.items()}
        mid = mid[keep]
        volume = volume[keep] if volume is not None else None
        amount = amount[keep] if amount is not None else None
        limits = (limits[0][keep], limits[1][keep]) if limits is not None else None
        pre_close = pre_close[keep] if pre_close is not None else None
        T, N = len(cal), len(codes)
        if T < 3:
            raise ValueError("fewer than 3 bars after the start/end window")
        if volume is None:
            volume = np.full((T, N), np.nan)

        # ---- limit hits
        if pre_close is None:
            pre_close = np.full((T, N), np.nan)
            pre_close[1:] = raw["close"][:-1]
        st = _st_mask(root, codes, cal)
        pct = np.stack([_limit_pct(k, cal, st[:, i]) for i, k in enumerate(codes)], 1)
        mode = str(c.get("limit_mode", "auto"))
        if mode == "auto":
            if limits is not None:
                mode = "columns"
            elif exact_ok:
                mode = "exact"
            else:
                mode = "ratio"
        direction = _detect_hits(raw["close"], pre_close, pct, mode, limits, float(c.get("limit_ratio_tol", 0.003)))
        events = _runs_to_events(direction)
        labels, btp, avail_label = event_table_labels(T, N, events, lead, horizons)
        tradable = np.isfinite(mid) & (direction == 0)

        # ---- universe as-of from membership ranges
        has = np.isfinite(mid)
        if members is not None and len(members):
            universe = np.zeros((T, N), dtype=bool)
            cpos = {k: i for i, k in enumerate(codes)}
            c6pos = {_code6(k): i for i, k in enumerate(codes)}
            for r in members.itertuples(index=False):
                i = cpos.get(str(r.code).upper(), c6pos.get(_code6(r.code)))
                if i is None or pd.isna(r.start):
                    continue
                e = cal[-1] if pd.isna(r.end) else r.end
                universe[(cal >= r.start) & (cal <= e), i] = True
            universe &= has
            universe_source = "instruments date ranges (as-of; change effective at its start date)"
        else:
            universe = has.copy()
            universe_source = "no membership file: universe = has a price (NOT as-of; add members.csv)"

        # ---- features
        price_feat, feat_names = raw_price_features(mid, volume, raw["high"], raw["low"]) if raw["high"] is not None and raw["low"] is not None \
            else raw_price_features(mid, volume)
        if raw["high"] is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                hl = (raw["high"] - raw["low"]) / raw["close"]
            price_feat[..., feat_names.index("hl_range")] = hl.astype(np.float32)      # range on traded prices
        liq = liquidity_fields(mid, volume, raw["high"], raw["low"])
        bar_ns = session_close_ns(cal, "15:00", "Asia/Shanghai")

        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": horizons, "raw_features": True, "ret_col": 0,
            "feat_names": feat_names, "assets": codes, "layout": layout, "limit_mode": mode, "lead": lead,
            "limit_direction": direction, "n_limit_events": int(len(events)), "events": events, "universe_source": universe_source,
            "st_flags_used": bool(st.any()),
            "source": {"qlib": "https://github.com/microsoft/qlib (CN daily bundle)", "tushare": "https://tushare.pro",
                       "akshare": "https://akshare.akfamily.xyz"},
            "licence": "Qlib CN bundle: research use per the Qlib repository terms. Tushare / AkShare exports: per their terms "
                       "(Tushare needs an account token). Constructed labels: ours (rule in the module docstring).",
            "layout_doc": "qlib_cn/{calendars/day.txt, instruments/csi300.txt, features/<inst>/<field>.day.bin}  OR  "
                          "bars.csv | csv/<code>.csv [+ members.csv] ; optional st_flags.csv",
            "pit_notes": {
                "avail_feat": "t: the daily bar (15:00 Asia/Shanghai close) is knowable at its bar",
                "avail_label": "event rows: last limit bar of the run + 1 (the first bar at which the run is seen to have "
                               "ended; NEVER if still open at the panel end); other rows: t + max(horizon)",
                "universe": "as-of membership from the instruments date ranges / members.csv; a change is knowable at its start "
                            "date (L6); without a membership file the universe degrades to 'has a price' (flagged in meta)",
                "tradable": "False on every limit-hit bar (conservative whole-bar reading); direction in meta['limit_direction']",
            },
            "assumptions": ["Qlib $close adjusted, $factor adjustment factor (traded price = close / factor)",
                            "ratio mode tolerance 0.003 absorbs 0.01 tick rounding; ST 5 % limit needs st_flags.csv",
                            "ChiNext 20 % limit from 2020-08-24; STAR 20 %; BSE 30 %"],
        }
        panel = Panel(times=bar_ns, assets=codes, mid=mid, price_feat=price_feat, liq=liq, tradable=tradable, labels=labels,
                      bars_to_peak=btp, meta=meta)
        return finish_panel(panel, avail_label=avail_label, universe=universe)

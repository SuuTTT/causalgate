"""D7 S&P 500 1-minute bars + LULD halt calendar + market-wide circuit breakers (CausalGate PIT loader). CONSTRUCTED.

Registered as "cg_sp500_tick". Channels exercised: L7 (halts: nothing can be entered or exited inside a LULD pause
and for 5 minutes after it), L6 (as-of index membership from a constituents file with add / remove dates) and L4
(the mini-flash-crash label is *future-defined*: it exists only once the reversion has happened). Nothing is fabricated:
missing raw files raise FileNotFoundError with the recipe (RECIPES_D4_D10.md#d7).

Sources
    bars     Polygon.io aggregates (https://polygon.io/docs/stocks, paid plan for full history; licence: Polygon terms)
             or IEX Cloud / IEX HIST (https://iexcloud.io, terms apply). NYSE TAQ (WRDS) is LICENSED and only the
             user's own export can be read here; the paper reports TAQ as "licensed, free substitutes reported".
    halts    LULD trading pauses: Nasdaq Trader "Trading Halts" (https://www.nasdaqtrader.com/trader.aspx?id=TradeHalts,
             current only) or the NYSE / CTA historical halt files; export to halts.csv.
    breakers market-wide circuit breakers (e.g. 2020-03-09, -12, -16, -18 level-1 halts): circuit_breakers.csv.
    members  S&P 500 constituents with add / remove dates (e.g. a Wikipedia "changes" scrape or a vendor file).
    labels   mini flash crashes. Johnson et al. 2013 (Nature Sci. Rep. 3:2627) define them on ticks (>= 10 ticks in one
             direction within 1.5 s, >= 0.8 % move, reverted). ADAPTATION / ASSUMPTION for minute bars: >= cfg mfc_move
             (0.8 %) monotone move within <= cfg mfc_max_bars (3) bars that reverts >= cfg mfc_revert_frac (50 %) within
             cfg mfc_revert_window (10) bars. Events are non-overlapping (earliest onset wins).

Raw file layout expected under cfg['root'] (default data_raw/sp500_tick)
    bars/<TICKER>.csv        1-minute bars. Columns (case-insensitive; Polygon short names accepted):
                             timestamp | t | time | datetime | date, open | o, high | h, low | l, close | c,
                             volume | v | vol, vwap | vw (optional), trades | n | transactions (optional)
                             timestamps: epoch (s / ms / us / ns by magnitude) or ISO strings; naive strings are read in
                             cfg 'tz' (default America/New_York). ASSUMPTION: the timestamp is the bar START (Polygon
                             convention); bar close = start + cfg 'bar_minutes' (1). Set timestamp_is: close otherwise.
    bars/<TICKER>.parquet    NOT readable here (pyarrow is not installed): the loader raises with a hint to export csv.
    halts.csv                ticker | symbol, halt_start | start, halt_end | end  (timestamps as above)
    circuit_breakers.csv     start, end  (market-wide; a date-only row marks the WHOLE session untradable -- conservative)
    constituents.csv         ticker | symbol, add_date | start | added, remove_date | end | removed (blank = current)
                             (optional; without it universe = has a bar, flagged in meta)

Panel construction
    time axis      union of bar-close timestamps across tickers, regular hours only ((09:30, 16:00] ET close times;
                   cfg regular_hours_only) ; gaps inside a session are forward-filled with volume 0 (cfg ffill_within_session)
    mid            close
    price_feat     RAW [log_ret, log_volume, hl_range] (column 0 = log return, meta['ret_col'] = 0)
    labels         1 on [onset - lead, peak] where onset = last close before the move, peak = extreme close;
                   bars_to_peak = peak - t (exact); meta['events'] has (asset, onset_t, peak_t, end_t, direction, move)
    tradable       False inside each halt and for post_halt_minutes (5) after it, inside circuit breakers, and where
                   there is no real bar (forward-filled gap minutes are untradable); a halt with a blank end is treated
                   as open through the end of its session, one with an unparseable start is skipped (counted in meta)
    universe       as-of membership from constituents.csv (a change becomes effective and knowable at its add date)

PIT fields
    avail_feat     t
    avail_label    event rows: the reversion bar end_t (the label does not exist before the reversion, L4); other rows:
                   t + max(horizon)
    universe       as-of constituents; the leak guarded against is applying today's list backwards
"""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from .common import register_dataset, require, raw_price_features, liquidity_fields, event_table_labels, finish_panel

RECIPE = """D7 S&P 500 minute bars (see causalgate/data/loaders/RECIPES_D4_D10.md#d7):
  1. export 1-minute aggregates per ticker (Polygon /v2/aggs/ticker/{T}/range/1/minute/..., or IEX) to
     data_raw/sp500_tick/bars/<TICKER>.csv  (csv, not parquet: pyarrow is not available here)
  2. LULD halt calendar -> data_raw/sp500_tick/halts.csv (ticker, halt_start, halt_end)
  3. market-wide circuit breakers -> data_raw/sp500_tick/circuit_breakers.csv (start, end)
  4. constituents with add/remove dates -> data_raw/sp500_tick/constituents.csv (ticker, add_date, remove_date)"""

_ET = "America/New_York"


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _parse_ts(x, assume_tz: str = _ET) -> np.ndarray:
    """-> int64 UTC ns (NaT -> int64 min). Numeric: epoch s / ms / us / ns by magnitude, or integer YYYYMMDD /
    YYYYMMDDHHMMSS. Strings: per-element formats may be mixed (ISO, YYYYMMDD, YYYYMMDDHHMMSS, with or without offset);
    naive values are localised to `assume_tz`, tz-aware ones converted."""
    s = pd.Series(x).reset_index(drop=True)
    nat = np.iinfo(np.int64).min
    out = np.full(len(s), nat, dtype=np.int64)

    def _loc(ts: pd.Series) -> np.ndarray:
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(assume_tz, ambiguous="NaT", nonexistent="shift_forward")
        return ts.dt.tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)

    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
        fin = v.to_numpy()[np.isfinite(v.to_numpy())]
        mx = float(np.abs(fin).max()) if fin.size else 0.0
        if fin.size and np.all((fin >= 19000101) & (fin <= 21001231)):            # integer YYYYMMDD (not an epoch)
            return _loc(pd.to_datetime(v.map(lambda q: f"{int(q):08d}" if np.isfinite(q) else None), format="%Y%m%d", errors="coerce"))
        if fin.size and np.all((fin >= 19000101000000) & (fin <= 21001231235959)):   # integer YYYYMMDDHHMMSS
            return _loc(pd.to_datetime(v.map(lambda q: f"{int(q):014d}" if np.isfinite(q) else None), format="%Y%m%d%H%M%S", errors="coerce"))
        unit = "s" if mx < 1e11 else ("ms" if mx < 1e14 else ("us" if mx < 1e17 else "ns"))
        return pd.to_datetime(v, unit=unit, utc=True, errors="coerce").values.astype("datetime64[ns]").astype(np.int64)
    st = s.astype(str).str.strip()
    m14 = st.str.fullmatch(r"\d{14}").fillna(False).to_numpy()
    m8 = st.str.fullmatch(r"\d{8}").fillna(False).to_numpy()
    aware = st.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).fillna(False).to_numpy() & ~m14 & ~m8
    rest = ~(m14 | m8 | aware)
    if m14.any():
        out[m14] = _loc(pd.to_datetime(st[m14], format="%Y%m%d%H%M%S", errors="coerce"))
    if m8.any():
        out[m8] = _loc(pd.to_datetime(st[m8], format="%Y%m%d", errors="coerce"))
    if aware.any():
        out[aware] = pd.to_datetime(st[aware], errors="coerce", utc=True, format="mixed").values.astype("datetime64[ns]").astype(np.int64)
    if rest.any():
        out[rest] = _loc(pd.to_datetime(st[rest], errors="coerce", format="mixed"))
    return out


def _is_date_only(x) -> np.ndarray:
    s = pd.Series(x).astype(str).str.strip()
    return s.str.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}|\d{4}/\d{2}/\d{2}").fillna(False).to_numpy()


# ------------------------------------------------------------------------------------------------ bars
def _load_bars(bdir: str, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    csvs = sorted(glob.glob(os.path.join(bdir, "*.csv")))
    pqs = sorted(glob.glob(os.path.join(bdir, "*.parquet")))
    if not csvs and pqs:
        raise FileNotFoundError(f"{len(pqs)} parquet file(s) under {bdir} but pyarrow is not installed here: export csv "
                                f"(e.g. `pd.read_parquet(f).to_csv(f[:-8] + '.csv', index=False)` on a machine with pyarrow).\n{RECIPE}")
    assets = {a.upper() for a in cfg.get("assets", [])} or None
    tz, bar_min = str(cfg.get("tz", _ET)), int(cfg.get("bar_minutes", 1))
    ts_is_close = str(cfg.get("timestamp_is", "start")) == "close"
    out: Dict[str, pd.DataFrame] = {}
    for f in csvs:
        tk = os.path.splitext(os.path.basename(f))[0].upper()
        if assets and tk not in assets:
            continue
        df = pd.read_csv(f)
        tc = _col(df, "timestamp", "t", "time", "datetime", "date", "ts")
        cc = _col(df, "close", "c")
        if tc is None or cc is None:
            continue
        ns = _parse_ts(df[tc], tz)
        if not ts_is_close:
            ns = ns + int(bar_min * 60 * 1e9)
        g = pd.DataFrame({"close_ns": ns, "close": pd.to_numeric(df[cc], errors="coerce")})
        for k, names in (("open", ("open", "o")), ("high", ("high", "h")), ("low", ("low", "l")), ("volume", ("volume", "v", "vol")),
                         ("vwap", ("vwap", "vw")), ("trades", ("trades", "n", "transactions"))):
            c = _col(df, *names)
            g[k] = pd.to_numeric(df[c], errors="coerce") if c else np.nan
        g = g.dropna(subset=["close"]).drop_duplicates("close_ns").sort_values("close_ns")
        if bool(cfg.get("regular_hours_only", True)):
            et = pd.to_datetime(g["close_ns"], utc=True).dt.tz_convert(_ET)
            mins = et.dt.hour * 60 + et.dt.minute
            g = g[(mins > 9 * 60 + 30) & (mins <= 16 * 60)]
        if cfg.get("start"):
            g = g[g["close_ns"] >= _parse_ts([cfg["start"]], tz)[0]]
        if cfg.get("end"):
            g = g[g["close_ns"] <= _parse_ts([cfg["end"]], tz)[0]]
        if len(g):
            out[tk] = g.reset_index(drop=True)
    return out


def _session_id(close_ns: np.ndarray) -> np.ndarray:
    et = pd.to_datetime(pd.Series(close_ns), utc=True).dt.tz_convert(_ET)
    return et.dt.strftime("%Y%m%d").to_numpy()


# ------------------------------------------------------------------------------------------------ labels
def _mini_flash_crashes(close_col: np.ndarray, move: float, kmax: int, rev_frac: float, rev_win: int) -> List[Tuple[int, int, int, int, float]]:
    """Minute-bar adaptation of Johnson et al. 2013 -> [(onset_t, peak_t, end_t, direction, move)] non-overlapping."""
    T = len(close_col)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(close_col)
    r1 = lp[1:] - lp[:-1]
    cands: List[Tuple[int, int, int, int, float]] = []
    for k in range(1, kmax + 1):
        if T <= k:
            break
        mv = lp[k:] - lp[:-k]                                     # mv[t] = lp[t+k] - lp[t], t in [0, T-k)
        sgn = np.sign(mv)
        mono = sgn != 0
        for d in range(k):
            mono &= np.sign(r1[d : d + T - k]) == sgn
        ok = np.nan_to_num(np.abs(mv) >= move, nan=False) & mono
        for t in np.nonzero(ok)[0]:
            peak = int(t + k)
            denom = lp[t] - lp[peak]
            end = -1
            for j in range(1, rev_win + 1):
                if peak + j >= T:
                    break
                if np.isfinite(lp[peak + j]) and (lp[peak + j] - lp[peak]) / denom >= rev_frac:
                    end = peak + j
                    break
            if end > 0:
                cands.append((int(t), peak, end, int(sgn[t]), float(mv[t])))
    cands.sort(key=lambda e: (e[0], -abs(e[4])))          # same onset: the extreme (largest move) is the peak
    out, last_end = [], -1
    for e in cands:
        if e[0] > last_end:
            out.append(e)
            last_end = e[2]
    return out


# ------------------------------------------------------------------------------------------------ builder
@register_dataset("cg_sp500_tick")
class SP500TickPITBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_sp500_tick", "1min", 252 * 390, (1, 2, 5, 10, 30, 60)

    def build_panel(self) -> Panel:
        c, root = self.cfg, self.root
        require(root, ["bars"], RECIPE)
        horizons = [int(h) for h in c.get("horizons", self.horizons)]
        lead = int(c.get("lead", 3))
        tz = str(c.get("tz", _ET))
        bars = _load_bars(os.path.join(root, "bars"), c)
        if not bars:
            raise FileNotFoundError(f"no usable minute-bar csv under {os.path.join(root, 'bars')}\n{RECIPE}")
        assets = sorted(bars)
        axis = np.array(sorted(set().union(*[set(bars[a]["close_ns"].tolist()) for a in assets])), dtype=np.int64)
        T, N = len(axis), len(assets)
        pos = pd.Series(np.arange(T), index=axis)
        A = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close", "volume")}
        present = np.zeros((T, N), dtype=bool)
        for i, a in enumerate(assets):
            g = bars[a]
            rows = pos.reindex(g["close_ns"].to_numpy()).to_numpy().astype(int)
            for k in A:
                A[k][rows, i] = g[k].to_numpy(dtype=float)
            present[rows, i] = True
        sess = _session_id(axis)
        if bool(c.get("ffill_within_session", True)):
            # forward-fill the close inside a session (gap bars get volume 0 and open/high/low = close)
            filled = pd.DataFrame(A["close"]).groupby(sess, sort=False).ffill().to_numpy()
            gap = np.isnan(A["close"]) & np.isfinite(filled)
            A["close"] = filled
            for k in ("open", "high", "low"):
                A[k] = np.where(gap, filled, A[k])
            A["volume"] = np.where(gap, 0.0, A["volume"])
        has = np.isfinite(A["close"])

        # ---- mini-flash-crash labels (future-defined: knowable at the reversion bar)
        rows = []
        for i in range(N):
            for (t0, pk, en, d, mv) in _mini_flash_crashes(A["close"][:, i], float(c.get("mfc_move", 0.008)), int(c.get("mfc_max_bars", 3)),
                                                            float(c.get("mfc_revert_frac", 0.5)), int(c.get("mfc_revert_window", 10))):
                rows.append({"asset": i, "onset_t": t0, "peak_t": pk, "end_t": en, "direction": d, "move": mv})
        events = pd.DataFrame(rows, columns=["asset", "onset_t", "peak_t", "end_t", "direction", "move"])
        labels, btp, avail_label = event_table_labels(T, N, events, lead, horizons)

        # ---- tradability: halts (+ post window) and circuit breakers
        # only REAL bars are tradable: forward-filled gap minutes keep a stale close for mid / features only
        tradable = has & present
        n_gap_rows = int((has & ~present).sum())
        n_halts_bad_start = n_halts_open_end = 0
        post_ns = int(float(c.get("post_halt_minutes", 5)) * 60 * 1e9)
        hp = os.path.join(root, "halts.csv")
        n_halts = 0
        if os.path.exists(hp):
            h = pd.read_csv(hp)
            tc, sc, ec = _col(h, "ticker", "symbol"), _col(h, "halt_start", "start", "halt_time"), _col(h, "halt_end", "end", "resume_time")
            s_ns, e_ns = _parse_ts(h[sc], tz), _parse_ts(h[ec], tz)
            apos = {a: i for i, a in enumerate(assets)}
            nat = np.iinfo(np.int64).min
            for tk, s0, e0 in zip(h[tc].astype(str).str.upper().str.strip(), s_ns, e_ns):
                i = apos.get(tk)
                if i is None:
                    continue
                if s0 == nat:                                    # unparseable start: cannot place the halt -> skip, count
                    n_halts_bad_start += 1
                    continue
                if e0 == nat or e0 < s0:                         # blank / bad resume time: open-ended through s0's session
                    j = int(np.searchsorted(axis, s0, side="left"))
                    e0 = int(axis[sess == sess[j]].max()) if j < T else s0
                    n_halts_open_end += 1
                tradable[(axis >= s0) & (axis <= e0 + post_ns), i] = False
                n_halts += 1
        cp = os.path.join(root, "circuit_breakers.csv")
        n_cb = 0
        if os.path.exists(cp):
            cb = pd.read_csv(cp)
            sc, ec = _col(cb, "start", "halt_start", "date"), _col(cb, "end", "halt_end")
            s_raw = cb[sc]
            s_ns = _parse_ts(s_raw, tz)
            e_ns = _parse_ts(cb[ec], tz) if ec else np.full(len(cb), -1, dtype=np.int64)
            donly = _is_date_only(s_raw)
            for s0, e0, d0 in zip(s_ns, e_ns, donly):
                if d0 or e0 < 0:
                    e0 = s0 + int(24 * 3600 * 1e9) - 1          # whole session (conservative)
                tradable[(axis >= s0) & (axis <= e0 + post_ns), :] = False
                n_cb += 1

        # ---- universe as-of from constituents
        kp = os.path.join(root, "constituents.csv")
        if os.path.exists(kp):
            k = pd.read_csv(kp)
            tc, ac, rc = _col(k, "ticker", "symbol"), _col(k, "add_date", "start", "added", "date_added"), _col(k, "remove_date", "end", "removed", "date_removed")
            universe = np.zeros((T, N), dtype=bool)
            apos = {a: i for i, a in enumerate(assets)}
            a_ns = _parse_ts(k[ac], tz)
            r_ns = _parse_ts(k[rc], tz) if rc else np.full(len(k), np.iinfo(np.int64).max, dtype=np.int64)
            r_blank = pd.to_datetime(k[rc], errors="coerce").isna().to_numpy() if rc else np.ones(len(k), dtype=bool)
            for tk, a0, r0, blank in zip(k[tc].astype(str).str.upper().str.strip(), a_ns, r_ns, r_blank):
                i = apos.get(tk)
                if i is None:
                    continue
                r_eff = np.iinfo(np.int64).max if blank else r0 + int(24 * 3600 * 1e9)   # removal date-only -> through that session
                universe[(axis >= a0) & (axis < r_eff), i] = True
            universe &= has
            universe_source = "constituents.csv (as-of; a change effective and knowable at its add date)"
        else:
            universe = has.copy()
            universe_source = "no constituents.csv: universe = has a bar (NOT as-of)"

        price_feat, feat_names = raw_price_features(A["close"], A["volume"], A["high"], A["low"])
        liq = liquidity_fields(A["close"], A["volume"], A["high"], A["low"], adv_window=int(c.get("adv_window", 390)))
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": horizons, "raw_features": True, "ret_col": 0,
            "feat_names": feat_names, "assets": assets, "lead": lead, "events": events, "n_events": int(len(events)),
            "n_halts": n_halts, "n_halts_open_end": n_halts_open_end, "n_halts_bad_start": n_halts_bad_start,
            "n_gap_rows_untradable": n_gap_rows, "n_circuit_breakers": n_cb, "universe_source": universe_source, "sessions": sess,
            "mfc_rule": {"move": float(c.get("mfc_move", 0.008)), "max_bars": int(c.get("mfc_max_bars", 3)),
                         "revert_frac": float(c.get("mfc_revert_frac", 0.5)), "revert_window": int(c.get("mfc_revert_window", 10)),
                         "note": "minute-bar ADAPTATION of Johnson et al. 2013 (tick rule: >=10 ticks in one direction within 1.5 s, >=0.8 %)"},
            "source": {"bars": "Polygon.io aggregates / IEX Cloud; NYSE TAQ (WRDS) licensed", "halts": "Nasdaq Trader / NYSE halt files",
                       "constituents": "user-supplied add/remove table"},
            "licence": "Polygon / IEX: per their terms of service (redistribution of bars is not permitted; keep raw files local). "
                       "TAQ: LICENSED (WRDS). Halt calendars: public exchange notices. Labels: constructed rule (ours).",
            "layout": "bars/<TICKER>.csv, halts.csv, circuit_breakers.csv, constituents.csv",
            "pit_notes": {
                "avail_feat": "t: a minute bar is knowable at its close",
                "avail_label": "event rows: the reversion bar (future-defined mini-flash-crash label, L4); other rows: t + max(horizon)",
                "universe": "as-of constituents (effective at the add date); without constituents.csv it degrades to 'has a bar'",
                "tradable": "False inside LULD halts + post_halt_minutes after, inside circuit breakers, and on missing bars "
                            "(forward-filled gap minutes are NOT tradable); a halt with a blank/unparseable end is open-ended "
                            "through the end of its session (n_halts_open_end); one with an unparseable start is skipped "
                            "(n_halts_bad_start)",
            },
            "assumptions": ["timestamp = bar start (Polygon); close = start + bar_minutes", "regular hours only, (09:30, 16:00] ET closes",
                            "session gaps forward-filled with volume 0", "date-only circuit-breaker rows mark the whole session untradable"],
        }
        panel = Panel(times=axis, assets=assets, mid=A["close"], price_feat=price_feat, liq=liq, tradable=tradable, labels=labels,
                      bars_to_peak=btp, meta=meta)
        return finish_panel(panel, avail_label=avail_label, universe=universe)

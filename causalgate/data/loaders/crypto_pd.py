"""D5 Telegram crypto pump-and-dump (La Morgia, Mei, Sassi, Stefa; ACM TOIT 2023, arXiv:2105.00733).
Registered as dataset "cg_crypto_pd". Source: https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset
Licence: as stated in that repository (check the LICENSE file; cite the ACM TOIT 2023 article / arXiv:2105.00733).

Required raw layout under <root>/<subdir> (subdir default "crypto_pd"):
    pump_telegram.csv            the curated pump table (one row per announced pump). Columns are matched
                                 case-insensitively with these aliases (assumption stated: the public table carries
                                 symbol / group / date / hour / exchange; peak columns are OPTIONAL):
                                     symbol   : symbol | coin | pair | ticker
                                     exchange : exchange | market
                                     date     : date | day            (YYYY-MM-DD)
                                     hour     : hour | time | hour_utc (HH:MM or HH:MM:SS); or one datetime|timestamp column
                                     channel  : channel | group | telegram | channel_name | group_name
                                     peak     : peak_time | peak | peak_hour | peak_datetime | peak_ts   (optional)
                                 Times are interpreted in cfg `announce_tz` (default UTC; the authors report UTC).
    klines/<SYMBOL><QUOTE>.csv   Binance 1-minute klines for every pumped pair (QUOTE default BTC), either the
                                 12-column header-less data.binance.vision layout [open_time, open, high, low, close,
                                 volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume,
                                 ignore] (open_time in ms, or us for files after 2025-01-01) or a headed csv with
                                 open_time|timestamp|time|date + open/high/low/close/volume[/count]. Also accepted:
                                 klines/<SYMBOL><QUOTE>-1m-*.csv (monthly files, concatenated), klines/<SYMBOL>.csv.
    messages.csv                 OPTIONAL: Telegram message text the user collected (the public table has NO text):
                                 columns channel, timestamp|date+hour, text [, symbol]. Needed only for --embed.
    cache/crypto_pd_docs.npz     OPTIONAL embedding cache; doc_asset = pump key "<SYMBOL><QUOTE>@<YYYY-MM-DDTHH:MMZ>".

Panel layout = STACKED EVENT WINDOWS: one asset column per (exchange-filtered) pump, sorted by announcement time;
T = pre_minutes + post_minutes + 1 one-minute bars, row `pre_minutes` = the bar containing the announcement.
Columns are not wall-clock aligned (meta['bar_close_ns'] (T, N) keeps each pump's clock); walk-forward splits for this
dataset should be chronological over ASSETS (meta['split_axis'] = 'assets', meta['announce_ns']).
    mid          kline close; price_feat [log_ret, log_volume, hl_range, log_trades (when the count column exists)]
    liq          [half-spread proxy 0.1*0.5*(high-low)/close (no quotes in klines; Roll estimator not applied), 1, ADV frac]
    tradable     volume > 0 and a kline exists (zero-volume minutes are untradable)
    labels       1 on [announcement - lead, peak]; bars_to_peak = peak - t exact (negative after the peak) on
                 [announcement - lead, end]; avail_label = end = peak + end_minutes_after_peak on those rows, else t + max(h)
    peak         curated (peak column) when present, else DERIVED = argmax of the kline high in
                 [announcement, announcement + peak_window_minutes]; meta['peak_source'] records which, per pump.
Point-in-time rules (meta['pit_notes']):
    avail_feat   = t (bar close)
    avail_text   = bar containing the Telegram message (first bar close >= message time) -- the announcement itself is
                   knowable at row `pre_minutes`; pre-announcement channel messages at their own bar
    avail_label  = end bar of the event (the peak, hence the label, is only knowable after the dump has played out);
                   detection must use pre-peak information only (the plan's L2/L7 note)
    universe     = a kline exists at that bar
"""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from .common import register_dataset, require, next_bar_index, raw_price_features, liquidity_fields, event_table_labels, finish_panel
from .tables import pick_col, find_first
from .doc_cache import cache_path, read_doc_cache, fill_text_slots

SOURCE_URL = "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset (La Morgia et al., ACM TOIT 2023, arXiv:2105.00733)"
LICENCE = "see the repository LICENSE; cite La Morgia, Mei, Sassi, Stefa, ACM Trans. Internet Technol. 2023 (arXiv:2105.00733)"
RECIPE = ("git clone https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset {root}/crypto_pd\n"
          "# then fetch Binance 1-minute klines for every pumped <SYMBOL>BTC pair into {root}/crypto_pd/klines/ (e.g. from\n"
          "# https://data.binance.vision/?prefix=data/spot/monthly/klines/<SYMBOL>BTC/1m/ ; unzip to <SYMBOL>BTC-1m-YYYY-MM.csv)\n"
          "# optional: {root}/crypto_pd/messages.csv (channel, timestamp, text) for --embed; the public table has no message text")
PUMP_COLS = {"symbol": ("symbol", "coin", "pair", "ticker"), "exchange": ("exchange", "market"), "date": ("date", "day"),
             "hour": ("hour", "time", "hour_utc", "time_utc"), "datetime": ("datetime", "timestamp", "announce_time", "pump_time"),
             "channel": ("channel", "group", "telegram", "channel_name", "group_name", "telegram_group"),
             "peak": ("peak_time", "peak", "peak_hour", "peak_datetime", "peak_ts")}
BINANCE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KLINE_COLS = {"open_time": ("open_time", "open time", "timestamp", "time", "date", "datetime"), "open": ("open",), "high": ("high",),
              "low": ("low",), "close": ("close",), "volume": ("volume", "vol"), "count": ("count", "trades", "number_of_trades", "n_trades")}
MIN_NS = 60 * 10 ** 9


# ------------------------------------------------------------------------------------------------ parsing
def _epoch_to_ns(v: pd.Series) -> np.ndarray:
    """Numeric epoch (s / ms / us, inferred from magnitude) or datetime strings -> int64 UTC ns."""
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() > 0.5:
        x = num.to_numpy(dtype=np.float64)
        m = np.nanmedian(x)
        unit = 10 ** 3 if m >= 1e14 else (10 ** 6 if m >= 1e11 else 10 ** 9)
        return np.where(np.isfinite(x), x * unit, np.nan).astype(np.float64)
    ts = pd.to_datetime(v, utc=True, errors="coerce")
    return ts.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)


def read_klines(paths: Sequence[str]) -> pd.DataFrame:
    """Binance kline csv(s) -> DataFrame indexed by open_time (int64 UTC ns) with open/high/low/close/volume[/count]."""
    frames = []
    for p in paths:
        with open(p, "r") as f:
            first = f.readline().strip()
        tok = first.split(",")[0].strip()
        headed = not tok.replace(".", "", 1).replace("-", "", 1).isdigit()
        df = pd.read_csv(p, header=0 if headed else None)
        if not headed:
            df.columns = BINANCE_COLS[: df.shape[1]] + [f"extra{k}" for k in range(max(0, df.shape[1] - len(BINANCE_COLS)))]
        cols = list(df.columns)
        out = pd.DataFrame()
        tcol = pick_col(cols, KLINE_COLS["open_time"])
        if tcol is None:
            raise ValueError(f"{p}: no open_time/timestamp column in {cols}")
        out["open_time"] = _epoch_to_ns(df[tcol])
        for k in ("open", "high", "low", "close", "volume", "count"):
            c = pick_col(cols, KLINE_COLS[k])
            if c is not None:
                out[k] = pd.to_numeric(df[c], errors="coerce").to_numpy()
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["open_time"])]
    df["open_time"] = (df["open_time"].to_numpy() // MIN_NS * MIN_NS).astype(np.int64)      # snap to the minute grid
    df = df.drop_duplicates("open_time", keep="last").set_index("open_time").sort_index()
    for k in ("open", "high", "low", "close", "volume"):
        if k not in df:
            raise ValueError(f"{paths[0]}: kline column {k} missing")
    return df


def find_kline_files(kdir: str, symbol: str, quote: str) -> List[str]:
    pair = f"{symbol}{quote}"
    for pattern in (f"{pair}.csv", f"{pair}-1m*.csv", f"{pair}_1m*.csv", f"{symbol}_{quote}.csv", f"{symbol}-{quote}.csv", f"{symbol}.csv"):
        hits = sorted(glob.glob(os.path.join(kdir, pattern))) + sorted(glob.glob(os.path.join(kdir, pattern.lower())))
        if hits:
            return sorted(set(hits))
    return []


def read_pump_table(path: str, announce_tz: str = "UTC") -> pd.DataFrame:
    """-> DataFrame[symbol, exchange, channel, announce_ns, peak_ns (NaN when absent)] sorted by announce_ns."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    get = lambda key: pick_col(cols, PUMP_COLS[key])
    scol, ecol, ccol = get("symbol"), get("exchange"), get("channel")
    if scol is None:
        raise ValueError(f"{path}: no symbol column among {cols}")
    dt = get("datetime")
    if dt is not None:
        ts = pd.to_datetime(df[dt].astype(str), errors="coerce")
    else:
        dcol, hcol = get("date"), get("hour")
        if dcol is None:
            raise ValueError(f"{path}: no date/datetime column among {cols}")
        s = df[dcol].astype(str).str.strip() + (" " + df[hcol].astype(str).str.strip() if hcol is not None else "")
        ts = pd.to_datetime(s, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(announce_tz)
    ts = ts.dt.tz_convert("UTC")
    out = pd.DataFrame({"symbol": df[scol].astype(str).str.strip().str.upper(),
                        "exchange": df[ecol].astype(str).str.strip() if ecol is not None else "unknown",
                        "channel": df[ccol].astype(str).str.strip() if ccol is not None else "unknown",
                        "announce_ns": ts.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)})
    out.loc[ts.isna().to_numpy(), "announce_ns"] = np.nan
    pcol = get("peak")
    peak = np.full(len(df), np.nan)
    if pcol is not None:
        raw = df[pcol].astype(str).str.strip()
        bare = raw.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$").to_numpy()                        # "HH:MM[:SS]" -> announcement day
        day = ts.dt.tz_convert(announce_tz).dt.strftime("%Y-%m-%d").astype(str)
        combo = pd.Series(np.where(bare, day.to_numpy() + " " + raw.to_numpy(), raw.to_numpy()), index=df.index)
        p = pd.to_datetime(combo, errors="coerce")
        if p.notna().any():
            if getattr(p.dt, "tz", None) is None:
                p = p.dt.tz_localize(announce_tz)
            peak = p.dt.tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
            peak[p.isna().to_numpy()] = np.nan
            wrap = bare & np.isfinite(peak) & (peak < out["announce_ns"].to_numpy())          # bare time before the announce time -> next day
            peak[wrap] += 86_400 * 10 ** 9
    out["peak_ns"] = peak
    out = out[np.isfinite(out["announce_ns"])].sort_values("announce_ns").reset_index(drop=True)
    return out


def pump_key(symbol: str, quote: str, announce_ns: int) -> str:
    return f"{symbol}{quote}@{pd.Timestamp(int(announce_ns), tz='UTC').strftime('%Y-%m-%dT%H:%MZ')}"


def iter_documents(root: str, cfg: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Telegram messages (user-collected messages.csv) matched to pumps by channel (or symbol) within +-match_hours.
    Yields nothing -- and says so in the caller -- when messages.csv is absent: the public table has no message text."""
    cfg = cfg or {}
    base = os.path.join(root, cfg.get("subdir", "crypto_pd"))
    mpath = find_first(base, ("messages.csv", "telegram_messages.csv"))
    if mpath is None:
        return
    quote = str(cfg.get("quote", "BTC"))
    pumps = read_pump_table(find_first(base, ("pump_telegram.csv", "pumps.csv", "pump_and_dump.csv")), cfg.get("announce_tz", "UTC"))
    msgs = pd.read_csv(mpath)
    cols = list(msgs.columns)
    ccol, scol, txt = pick_col(cols, PUMP_COLS["channel"]), pick_col(cols, PUMP_COLS["symbol"]), pick_col(cols, ("text", "message", "content"))
    dt = pick_col(cols, PUMP_COLS["datetime"])
    if dt is not None:
        ts = pd.to_datetime(msgs[dt].astype(str), errors="coerce")
    else:
        dcol, hcol = pick_col(cols, PUMP_COLS["date"]), pick_col(cols, PUMP_COLS["hour"])
        ts = pd.to_datetime(msgs[dcol].astype(str).str.strip() + (" " + msgs[hcol].astype(str).str.strip() if hcol else ""), errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(cfg.get("announce_tz", "UTC"))
    ts_ns = ts.dt.tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
    win = float(cfg.get("match_hours", 12.0)) * 3600 * 10 ** 9
    for k in range(len(msgs)):
        if pd.isna(ts.iloc[k]) or txt is None:
            continue
        cand = pumps
        if ccol is not None:
            cand = cand[cand["channel"].str.lower() == str(msgs.at[k, ccol]).strip().lower()]
        if scol is not None and str(msgs.at[k, scol]).strip():
            cand = cand[cand["symbol"] == str(msgs.at[k, scol]).strip().upper()]
        if len(cand) == 0:
            continue
        d = np.abs(cand["announce_ns"].to_numpy() - ts_ns[k])
        j = int(np.argmin(d))
        if d[j] > win:
            continue
        r = cand.iloc[j]
        yield {"asset": pump_key(r["symbol"], quote, int(r["announce_ns"])), "content_ns": int(ts_ns[k]), "publish_ns": int(ts_ns[k]),
               "text": str(msgs.at[k, txt]), "doc_id": f"msg/{k}"}


# ------------------------------------------------------------------------------------------------ builder
@register_dataset("cg_crypto_pd")
class CryptoPumpDumpCausalBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_crypto_pd", "1min", 365 * 24 * 60, (1, 2, 5, 10, 30, 60)

    def build_panel(self) -> Panel:
        c = self.cfg
        root, sub = self.root, c.get("subdir", "crypto_pd")
        base = os.path.join(root, sub)
        recipe = RECIPE.format(root=root)
        ptab = find_first(base, ("pump_telegram.csv", "pumps.csv", "pump_and_dump.csv"))
        kdir = find_first(base, (c.get("klines_dir", "klines"), "binance", "data"))
        if ptab is None or kdir is None:
            require(root, [f"{sub}/pump_telegram.csv", f"{sub}/{c.get('klines_dir', 'klines')}"], recipe)
        quote = str(c.get("quote", "BTC"))
        pre, post = int(c.get("pre_minutes", 360)), int(c.get("post_minutes", 360))
        lead, peak_win, end_after = int(c.get("lead", 3)), int(c.get("peak_window_minutes", 60)), int(c.get("end_minutes_after_peak", 60))
        horizons = tuple(c.get("horizons", self.horizons))
        exch = str(c.get("exchange", "Binance")).lower()
        pumps = read_pump_table(ptab, str(c.get("announce_tz", "UTC")))
        skipped: Dict[str, int] = {"other_exchange": 0, "no_klines": 0, "no_bars_in_window": 0, "no_peak": 0}
        T = pre + post + 1
        cols: List[Dict[str, Any]] = []
        kcache: Dict[str, Optional[pd.DataFrame]] = {}
        for r in pumps.itertuples(index=False):
            if exch and exch != "all" and str(r.exchange).lower() != exch:
                skipped["other_exchange"] += 1; continue
            if r.symbol not in kcache:
                files = find_kline_files(kdir, r.symbol, quote)
                kcache[r.symbol] = read_klines(files) if files else None
            kl = kcache[r.symbol]
            if kl is None:
                skipped["no_klines"] += 1; continue
            a0 = (int(r.announce_ns) // MIN_NS) * MIN_NS
            grid = a0 + (np.arange(T, dtype=np.int64) - pre) * MIN_NS
            w = kl.reindex(grid)
            if w["close"].notna().sum() < max(10, (T // 10)):
                skipped["no_bars_in_window"] += 1; continue
            close_ns = grid + MIN_NS
            if np.isfinite(r.peak_ns):
                peak_t = int(np.clip(next_bar_index(close_ns, np.array([int(r.peak_ns)]))[0], 0, T - 1)); src = "curated"
            else:
                hi = w["high"].to_numpy()[pre : min(T, pre + peak_win + 1)]
                if not np.isfinite(hi).any():
                    skipped["no_peak"] += 1; continue
                peak_t = pre + int(np.nanargmax(hi)); src = "derived_argmax_high"
            cols.append({"key": pump_key(r.symbol, quote, int(r.announce_ns)), "symbol": r.symbol, "channel": r.channel, "announce_ns": int(r.announce_ns),
                         "peak_ns": (int(r.peak_ns) if np.isfinite(r.peak_ns) else int(close_ns[peak_t])), "peak_t": peak_t, "peak_source": src,
                         "close_ns": close_ns, "w": w})
            if c.get("max_pumps") and len(cols) >= int(c["max_pumps"]):
                break
        if not cols:
            raise FileNotFoundError(f"No pump with 1-minute klines under {kdir} (skipped: {skipped}).\nDownload recipe:\n{recipe}\n(see scripts/cg_prepare_data.py)")
        N = len(cols)
        get = lambda k: np.stack([col["w"][k].to_numpy(dtype=np.float64) if k in col["w"] else np.full(T, np.nan) for col in cols], 1)
        close, high, low, vol = get("close"), get("high"), get("low"), get("volume")
        has_count = all("count" in col["w"] for col in cols)
        feats, names = raw_price_features(close, vol, high, low)
        if has_count:
            cnt = get("count")
            with np.errstate(invalid="ignore"):
                feats = np.concatenate([feats, np.log1p(np.where(cnt >= 0, cnt, np.nan))[..., None].astype(np.float32)], -1)
            names = names + ["log_trades"]
        liq = liquidity_fields(close, vol, high, low)
        tradable = np.isfinite(close) & (np.nan_to_num(vol) > 0)
        events = pd.DataFrame({"asset": np.arange(N), "onset_t": pre, "peak_t": [col["peak_t"] for col in cols],
                               "end_t": [min(T - 1, col["peak_t"] + end_after) for col in cols]})
        labels, btp, avail_label = event_table_labels(T, N, events, lead, horizons)
        bar_close_ns = np.stack([col["close_ns"] for col in cols], 1)
        keys = [col["key"] for col in cols]
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(horizons), "raw_features": True, "ret_col": 0,
            "feat_names": names, "source": SOURCE_URL, "licence": LICENCE, "dataset": "crypto_pd", "root": base, "n_assets": N,
            "layout": "stacked event windows (one column per pump; row pre_minutes = announcement bar)", "split_axis": "assets",
            "assets_time_aligned": False, "pre_minutes": pre, "post_minutes": post, "lead": lead, "end_minutes_after_peak": end_after,
            "announce_ns": np.array([col["announce_ns"] for col in cols], dtype=np.int64), "peak_row": np.array([col["peak_t"] for col in cols]),
            "peak_source": [col["peak_source"] for col in cols], "symbols": [col["symbol"] for col in cols], "channels": [col["channel"] for col in cols],
            "bar_close_ns": bar_close_ns, "bar_offset_min": np.arange(T) - pre, "skipped": skipped, "n_pumps_in_table": int(len(pumps)),
            "label_rule": {"type": "event_table", "label_on": "[announcement - lead, peak]", "bars_to_peak": "peak - t (exact, negative after the peak)",
                           "peak": "curated column when present else argmax high within peak_window_minutes after the announcement"},
            "required_layout": ["pump_telegram.csv (symbol, exchange, date, hour, channel [, peak_time])", f"klines/<SYMBOL>{quote}.csv or <SYMBOL>{quote}-1m-*.csv (Binance 1m klines)",
                                "messages.csv (optional, for --embed)", "cache/crypto_pd_docs.npz (optional)"],
            "pit_notes": {
                "avail_feat": "t: the 1-minute bar is knowable at its close",
                "avail_text": "first bar close >= Telegram message time (announcement at row pre_minutes; earlier channel messages at their own bar)",
                "avail_label": "event end bar (peak + end_minutes_after_peak): the peak and hence the label are knowable only after the dump; else t + max(horizons)",
                "universe": "a kline exists for that bar", "tradable": "volume > 0 and a kline exists (zero-volume minutes untradable)",
            },
            "times": "bar index; per-pump UTC close times in meta['bar_close_ns'] (T, N)",
        }
        cp = c.get("doc_cache") or cache_path(root, "crypto_pd")
        cache = read_doc_cache(cp)
        text = text_mask = text_age = avail_text = None
        if cache is None:
            meta["text"] = "missing cache"
            meta["text_recipe"] = "python3 scripts/cg_prepare_data.py --dataset crypto_pd --embed --embedder hash_tfidf --cutoff <YYYY-MM-DD> (needs messages.csv)"
        else:
            idx = {k: i for i, k in enumerate(keys)}
            asset = np.array([idx.get(str(a), -1) for a in cache.doc_asset], dtype=np.int64)
            rows = np.full(cache.n, -1, dtype=np.int64)
            for j in range(cache.n):
                if asset[j] >= 0:
                    rows[j] = int(next_bar_index(bar_close_ns[:, asset[j]], np.array([cache.doc_publish_ns[j]]))[0])
            fs = fill_text_slots(T, N, int(c.get("text_slots", 4)), cache.emb, rows, rows, asset, cache.doc_publish_ns)
            text, text_mask, text_age, avail_text = fs["text"], fs["text_mask"], fs["text_age"], fs["avail_text"]
            meta["text"], meta["text_placement"] = dict(cache.meta), fs["stats"]
        layout = str(c.get("layout", "calendar"))
        if layout == "calendar":
            # One calendar-ordered series: pump windows concatenated in announcement order, separated by `gap_bars` empty
            # rows (NaN mid -> no decision point, no return, no impact across a boundary). Decision order = calendar order,
            # so gates / normalisers never see a later pump before an earlier one, and the purged walk-forward split over
            # bars is chronological over pumps. The stacked (T, N_pumps) layout is NOT time-aligned across columns: any
            # cross-column statistic (e.g. the per-bar cross-asset median in liquidity_fields) would mix later pumps into
            # earlier decisions, so it is only kept as an explicit option (layout: stacked).
            order = np.argsort(meta["announce_ns"], kind="stable")
            G = int(c.get("gap_bars", 200))
            Tc = N * (T + G)
            def cat(a, fill):
                out = np.full((Tc, 1) + a.shape[2:], fill, dtype=a.dtype)
                for k, j in enumerate(order):
                    out[k * (T + G): k * (T + G) + T, 0] = a[:, j]
                return out
            off = np.array([k * (T + G) for k in range(N)])
            al = avail_label.astype(np.int64).copy()
            NEVER_ = np.iinfo(np.int64).max // 4
            al_cat = np.full((Tc, 1), NEVER_, dtype=np.int64)
            for k, j in enumerate(order):
                col = al[:, j]
                al_cat[off[k]: off[k] + T, 0] = np.where(col >= NEVER_ // 2, NEVER_, np.minimum(col + off[k], Tc - 1))
            # gap rows: labels knowable immediately (they are 0 and never decision points)
            gap = np.ones(Tc, dtype=bool)
            for k in range(N):
                gap[off[k]: off[k] + T] = False
            al_cat[gap, 0] = np.arange(Tc)[gap]
            close_c = cat(close, np.nan); feats_c = cat(feats, np.float32(np.nan)); tradable_c = cat(tradable, False)
            labels_c = cat(labels, labels.dtype.type(0)); btp_c = cat(btp, np.nan)
            # liquidity recomputed on the single series (trailing statistics only, no cross-pump median)
            liq_c = liquidity_fields(close_c, cat(vol, np.nan), cat(high, np.nan), cat(low, np.nan))
            meta.update({"layout": f"calendar: {N} pump windows of {T} bars in announcement order, {G}-bar NaN gaps between them",
                         "split_axis": "time", "assets_time_aligned": True, "n_pumps": N, "window_bars": T, "gap_bars": G,
                         "window_offset": off, "window_order": order, "bar_close_ns": None,
                         "symbols": [meta["symbols"][j] for j in order], "peak_row": meta["peak_row"][order]})
            meta["pit_notes"]["liquidity"] = "trailing dollar-volume statistics on the single calendar series (no cross-pump statistic)"
            if text is not None:
                text = cat(text, np.float32(0)); text_mask = cat(text_mask, False); text_age = cat(text_age, np.float32(0))
                at = np.full(text_mask.shape, 0, dtype=np.int64)
                for k, j in enumerate(order):
                    at[off[k]: off[k] + T, 0] = avail_text[:, j] + off[k]
                at[gap, 0] = np.arange(Tc)[gap][:, None]
                avail_text = at
            panel = Panel(times=np.arange(Tc, dtype=np.int64), assets=["PUMPS_CALENDAR"], mid=close_c, price_feat=feats_c, liq=liq_c,
                          tradable=tradable_c, labels=labels_c, bars_to_peak=btp_c, text=text, text_mask=text_mask, text_age=text_age, meta=meta)
            return finish_panel(panel, avail_label=al_cat, avail_text=avail_text, universe=np.isfinite(close_c))
        panel = Panel(times=np.arange(T, dtype=np.int64), assets=keys, mid=close, price_feat=feats, liq=liq, tradable=tradable, labels=labels,
                      bars_to_peak=btp, text=text, text_mask=text_mask, text_age=text_age, meta=meta)
        return finish_panel(panel, avail_label=avail_label, avail_text=avail_text, universe=np.isfinite(close))

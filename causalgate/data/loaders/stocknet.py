"""D1 StockNet (Xu & Cohen, ACL 2018) -- daily prices + tweets for 88 US stocks, 2014-01-01 .. 2016-01-01.
Registered as dataset "cg_stocknet". Source: https://github.com/yumoxu/stocknet-dataset (MIT licence).

Required raw layout under <root>/<subdir> (subdir default "stocknet"; the repo's own tree):
    price/raw/<TICKER>.csv                  Date,Open,High,Low,Close,Adj Close,Volume   (Yahoo Finance export)
    tweet/preprocessed/<TICKER>/<YYYY-MM-DD> JSON lines; fields text (token list or string), user_id_str, created_at
                                            ("Thu Jan 02 14:26:36 +0000 2014", Twitter format, UTC)
    tweet/raw/<TICKER>/<YYYY-MM-DD>         optional (raw Twitter objects; same created_at format; text is a string)
    StockTable                              optional markdown table (sector per ticker; not used yet)
    cache/stocknet_docs.npz                 OPTIONAL embedding cache written by scripts/cg_prepare_data.py --embed
                                            (see doc_cache.py). Absent -> text=None, meta['text']='missing cache'.

Panel (T trading days x N tickers), everything RAW (no full-sample statistics; L1 is switched downstream):
    mid          Adj Close (dividend/split adjusted; falls back to Close when the column is absent)
    price_feat   [log_ret(mid), log_volume, hl_range = (High-Low)/Close, gap = log(Open_t / Close_{t-1})]   meta['feat_names']
    liq          [half-spread proxy 0.1 * 0.5 * (High-Low)/Close, depth 1, ADV fraction]  (no quotes in the dataset)
    tradable     Volume > 0 and a price row exists
    labels       abnormal_move_labels(horizon 5, k_ret 2.5, k_vol 2.0, window 60, lead 3) (no curated event table exists)
    bars_to_peak argmax-|impact| horizon for labelled bars (labels.py); avail_label = t + max(horizon + lead, max(horizons))
    text         from the cache, M = text_slots most recent tweets per (content date, ticker)

Point-in-time rules (meta['pit_notes']):
    avail_feat   = t: a daily bar is knowable at its 16:00 America/New_York close
    avail_text   = first NYSE close (16:00 ET) at or after the tweet's created_at (UTC). The document row is the tweet
                   directory date (the naive join key used by the original StockNet protocol); an after-close or weekend
                   tweet therefore has avail_text > row.
    avail_label  = t + max(horizon + lead, max(horizons)): a forward-defined rule label is knowable once the horizon elapsed
    universe     = a price row exists. The 88 names were selected in 2016 by market cap: survivorship is a dataset-level
                   bias (L6) that no as-of file can undo; recorded, not fixed.

Assumptions stated: created_at is parsed as UTC Twitter format with a generic fallback; a `text` list is joined with
spaces; tickers come from price/raw/*.csv file names; the trading calendar is the union of the price files' dates.
"""
from __future__ import annotations
import glob
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from .common import (register_dataset, require, session_close_ns, raw_price_features, liquidity_fields,
                     rule_labels_and_peaks, finish_panel)
from .tables import read_ohlcv_csv, align_frames
from .doc_cache import cache_path, read_doc_cache, text_fields_from_cache, dates_to_ns

SOURCE_URL = "https://github.com/yumoxu/stocknet-dataset"
LICENCE = "MIT (stocknet-dataset repository, Xu & Cohen ACL 2018); tweets are redistributed by the dataset authors"
RECIPE = ("git clone https://github.com/yumoxu/stocknet-dataset {root}/stocknet\n"
          "python3 scripts/cg_prepare_data.py --dataset stocknet --root {root} --embed --embedder hash_tfidf --cutoff 2015-01-01")
TWITTER_FMT = "%a %b %d %H:%M:%S %z %Y"


def parse_twitter_time(values: Sequence[str]) -> np.ndarray:
    """created_at strings -> int64 UTC ns (NaT -> min int64 sentinel is avoided: rows with unparsable times are dropped
    by the caller via the returned mask)."""
    s = pd.Series(list(values), dtype="object")
    ts = pd.to_datetime(s, format=TWITTER_FMT, utc=True, errors="coerce")
    bad = ts.isna()
    if bad.any():
        ts[bad] = pd.to_datetime(s[bad], utc=True, errors="coerce")
    return ts


def iter_documents(root: str, cfg: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Yield {asset, content_ns, publish_ns, text, doc_id} for every tweet (used by scripts/cg_prepare_data.py --embed).
    content_ns = the tweet directory date (UTC midnight), publish_ns = created_at."""
    cfg = cfg or {}
    base = os.path.join(root, cfg.get("subdir", "stocknet"))
    tdir = os.path.join(base, "tweet", cfg.get("tweet_variant", "preprocessed"))
    tickers = cfg.get("tickers") or sorted(os.listdir(tdir))
    for tk in tickers:
        d = os.path.join(tdir, tk)
        if not os.path.isdir(d):
            continue
        for day_file in sorted(os.listdir(d)):
            try:
                content_ns = int(dates_to_ns([day_file])[0])
            except Exception:
                continue
            with open(os.path.join(d, day_file), "r", encoding="utf-8", errors="replace") as f:
                lines = [ln for ln in f if ln.strip()]
            recs = []
            for ln in lines:
                try:
                    recs.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
            if not recs:
                continue
            ts = parse_twitter_time([r.get("created_at", "") for r in recs])
            for k, (r, t) in enumerate(zip(recs, ts)):
                if pd.isna(t):
                    continue
                txt = r.get("text", "")
                if isinstance(txt, (list, tuple)):
                    txt = " ".join(str(w) for w in txt)
                yield {"asset": tk, "content_ns": content_ns, "publish_ns": int(t.value), "text": str(txt),
                       "doc_id": f"{tk}/{day_file}/{k}"}


@register_dataset("cg_stocknet")
class StockNetCausalBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_stocknet", "daily", 252, (1, 2, 3, 5, 10)

    def build_panel(self) -> Panel:
        c = self.cfg
        root, sub = self.root, c.get("subdir", "stocknet")
        recipe = RECIPE.format(root=root)
        require(root, [f"{sub}/price/raw", f"{sub}/tweet/preprocessed"], recipe)
        base = os.path.join(root, sub)
        pdir = os.path.join(base, "price", "raw")
        tickers = list(c.get("tickers") or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(pdir, "*.csv"))))
        if not tickers:
            raise FileNotFoundError(f"No price csv under {pdir}.\nDownload recipe:\n{recipe}")
        horizons = tuple(c.get("horizons", self.horizons))
        start, end = str(c.get("start", "2014-01-01")), str(c.get("end", "2016-01-01"))
        frames: Dict[str, pd.DataFrame] = {}
        for tk in tickers:
            path = os.path.join(pdir, f"{tk}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}.\nDownload recipe:\n{recipe}")
            df = read_ohlcv_csv(path)
            frames[tk] = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in frames.values()])))
        if len(dates) < 2:
            raise ValueError(f"StockNet: fewer than 2 trading days in [{start}, {end}]")
        T, N = len(dates), len(tickers)
        close = align_frames(frames, dates, "close")
        adj = align_frames(frames, dates, "adj_close")
        high, low = align_frames(frames, dates, "high"), align_frames(frames, dates, "low")
        opn, vol = align_frames(frames, dates, "open"), align_frames(frames, dates, "volume")
        mid = np.where(np.isfinite(adj), adj, close)
        feats, names = raw_price_features(mid, vol, high, low)
        gap = np.full((T, N), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap[1:] = np.log(opn[1:]) - np.log(close[:-1])
        price_feat = np.concatenate([feats, gap[..., None].astype(np.float32)], -1)
        names = names + ["gap"]
        liq = liquidity_fields(close, vol, high, low)
        tradable = np.isfinite(close) & (np.nan_to_num(vol) > 0)
        lab = c.get("labels", {})
        labels, btp, avail_label = rule_labels_and_peaks(mid, vol, horizons, horizon=int(lab.get("horizon", 5)),
                                                         k_ret=float(lab.get("k_ret", 2.5)), k_vol=float(lab.get("k_vol", 2.0)),
                                                         window=int(lab.get("window", 60)), lead=int(lab.get("lead", 3)))
        close_ns = session_close_ns(dates, c.get("close_local", "16:00"), c.get("tz", "America/New_York"))
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(horizons), "raw_features": True, "ret_col": 0,
            "feat_names": names, "source": SOURCE_URL, "licence": LICENCE, "dataset": "stocknet", "root": base,
            "n_assets": N, "date_range": [str(dates[0].date()), str(dates[-1].date())], "mid": "adj_close (fallback close)",
            "label_rule": {"type": "abnormal_move_labels", **{k: lab.get(k, v) for k, v in
                           dict(horizon=5, k_ret=2.5, k_vol=2.0, window=60, lead=3).items()}},
            "required_layout": ["price/raw/<TICKER>.csv (Date,Open,High,Low,Close,Adj Close,Volume)",
                                "tweet/preprocessed/<TICKER>/<YYYY-MM-DD> (JSON lines: text, user_id_str, created_at)",
                                "cache/stocknet_docs.npz (optional, cg_prepare_data.py --embed)"],
            "pit_notes": {
                "avail_feat": "t: the daily bar is knowable at its 16:00 America/New_York close",
                "avail_text": "first NYSE close >= tweet created_at (UTC); row = tweet directory date (naive join key); after-close / weekend tweets have avail_text > row",
                "avail_label": "t + max(label horizon + lead, max(horizons)): forward-defined rule label knowable once the horizon elapsed",
                "universe": "price row exists; the 88-name list was fixed ex post (survivorship, L6) and cannot be made as-of",
                "tradable": "Volume > 0 and a price row exists",
            },
            "times": "int64 UTC ns of the session close (16:00 America/New_York)",
        }
        cp = c.get("doc_cache") or cache_path(root, "stocknet")
        cache = read_doc_cache(cp)
        tx = text_fields_from_cache(cache, tickers, dates, close_ns, int(c.get("text_slots", 16)), meta, "stocknet")
        panel = Panel(times=close_ns, assets=tickers, mid=mid, price_feat=price_feat, liq=liq, tradable=tradable, labels=labels,
                      bars_to_peak=btp, text=tx["text"], text_mask=tx["text_mask"], text_age=tx["text_age"], meta=meta)
        return finish_panel(panel, avail_label=avail_label, avail_text=tx["avail_text"], universe=np.isfinite(mid))

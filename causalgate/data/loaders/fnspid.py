"""D3 FNSPID (Dong, Fan, Mao et al., KDD 2024, arXiv:2402.06698) -- daily prices + timestamped financial news.
Registered as dataset "cg_fnspid". Source: https://huggingface.co/datasets/Zihan1004/FNSPID
Licence: read the `license` field of the dataset card before publication (not verified offline in this session; the
GitHub README of the authors states a non-commercial licence -- confirm).

Required raw layout under <root>/<subdir> (subdir default "fnspid"; the HF repository tree, either flat or as-is):
    Stock_news/nasdaq_exteral_data.csv  (sic, HF spelling)  columns: Date, Article_title, Stock_symbol, Url, Publisher,
        Author, Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary   (~22 GB; read in chunks by the
        prepare script only; the loader itself never opens it)
    Stock_price/full_history/<TICKER>.csv   date,open,high,low,close,adj close,volume   (column names matched
        case-insensitively; `adj close` falls back to `close`)
    cache/fnspid_docs.npz                   OPTIONAL embedding cache (scripts/cg_prepare_data.py --dataset fnspid --embed)
  Alternative flat layout accepted: <subdir>/nasdaq_exteral_data.csv and <subdir>/full_history/<TICKER>.csv.

Panel (T trading days x N tickers), RAW features:
    mid          adj close (fallback close); price_feat [log_ret, log_volume, hl_range, gap]; liq as in StockNet
    labels       abnormal_move_labels(horizon 3, k_ret 3, k_vol 3, window 60, lead 3) -- the earnings-event table of the
                 Warn2Trade recipe is not part of FNSPID and is not fabricated here
    text         M = text_slots most recent articles per (article date, ticker) from the cache
Point-in-time rules (meta['pit_notes']):
    avail_feat   = t (16:00 America/New_York close)
    avail_text   = first NYSE close >= the article's `Date` (parsed as UTC when the string carries no zone -- the FNSPID
                   timestamps are stored as "YYYY-MM-DD HH:MM:SS UTC"); row = calendar date of `Date` (naive join key).
                   After-close articles (>= 16:00 ET) -> next session, hence avail_text > row.
    avail_label  = t + max(label horizon + lead, max(horizons))
    universe     = price row exists; FNSPID covers 4,775 tickers incl. delisted names (full_history keeps them) but the
                   universe is still "has data", not an index membership
Assumptions stated: ticker selection = cfg `tickers` or the first `max_assets` price files alphabetically (an
article-count ranking would need a full pass over the 22 GB csv); article text for embedding = Article_title + the
first available summary column (cfg `text_fields`).
"""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from .common import (register_dataset, require, session_close_ns, raw_price_features, liquidity_fields,
                     rule_labels_and_peaks, finish_panel)
from .tables import read_ohlcv_csv, align_frames, find_first, pick_col
from .doc_cache import cache_path, read_doc_cache, text_fields_from_cache, utc_date_ns

SOURCE_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID"
LICENCE = "check the HF dataset card `license` field (Zihan1004/FNSPID) before use; not verified offline"
RECIPE = ("huggingface-cli download Zihan1004/FNSPID --repo-type dataset --local-dir {root}/fnspid "
          "--include 'Stock_price/full_history.zip' 'Stock_news/nasdaq_exteral_data.csv' && unzip -q data_raw/fnspid/Stock_price/full_history.zip -d data_raw/fnspid/Stock_price/\n"
          "python3 scripts/cg_prepare_data.py --dataset fnspid --root {root} --embed --embedder hash_tfidf --cutoff 2018-01-01")
NEWS_CANDIDATES = ("Stock_news/nasdaq_exteral_data.csv", "nasdaq_exteral_data.csv", "Stock_news/nasdaq_external_data.csv", "nasdaq_external_data.csv")
PRICE_CANDIDATES = ("Stock_price/full_history", "full_history")
NEWS_COLS = {"date": ("Date", "date", "datetime", "timestamp"), "title": ("Article_title", "title", "article_title"),
             "symbol": ("Stock_symbol", "symbol", "ticker", "stock_symbol"),
             "summary": ("Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary", "summary"), "article": ("Article", "article", "text")}


def _price_tickers(pdir: str) -> List[str]:
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(pdir, "*.csv")))


def iter_documents(root: str, cfg: Optional[Dict[str, Any]] = None, chunksize: int = 200_000) -> Iterator[Dict[str, Any]]:
    """Yield {asset, content_ns, publish_ns, text, doc_id} from the news csv, restricted to cfg tickers / date range
    (used by scripts/cg_prepare_data.py --embed). Reads in chunks; never loads the 22 GB file at once."""
    cfg = cfg or {}
    base = os.path.join(root, cfg.get("subdir", "fnspid"))
    news = find_first(base, NEWS_CANDIDATES)
    if news is None:
        raise FileNotFoundError(f"Missing {NEWS_CANDIDATES[0]} under {base}.\nDownload recipe:\n{RECIPE.format(root=root)}")
    head = pd.read_csv(news, nrows=0)
    cols = list(head.columns)
    dcol, tcol, scol = pick_col(cols, NEWS_COLS["date"]), pick_col(cols, NEWS_COLS["title"]), pick_col(cols, NEWS_COLS["symbol"])
    if dcol is None or scol is None:
        raise ValueError(f"{news}: need Date and Stock_symbol columns, got {cols}")
    text_fields = list(cfg.get("text_fields") or [c for c in (tcol, pick_col(cols, NEWS_COLS["summary"])) if c])
    usecols = [c for c in {dcol, scol, *text_fields} if c in cols]
    tickers = set(cfg.get("tickers") or [])
    if not tickers:
        pdir = find_first(base, PRICE_CANDIDATES)
        if pdir is not None:
            tickers = set(_price_tickers(pdir)[: int(cfg.get("max_assets", 100))])
    start = pd.Timestamp(cfg.get("start", "2010-01-01"), tz="UTC")
    end = pd.Timestamp(cfg.get("end", "2023-12-31"), tz="UTC") + pd.Timedelta(days=1)
    k = 0
    for chunk in pd.read_csv(news, usecols=usecols, chunksize=chunksize, dtype=str, keep_default_na=False):
        sym = chunk[scol].astype(str).str.strip().str.upper()
        if tickers:
            chunk = chunk[sym.isin(tickers)]
            sym = sym[chunk.index]
        if len(chunk) == 0:
            continue
        ts = pd.to_datetime(chunk[dcol], utc=True, errors="coerce")
        ok = ts.notna() & (ts >= start) & (ts < end)
        for idx in chunk.index[ok]:
            txt = " ".join(str(chunk.at[idx, f]) for f in text_fields if f in chunk and str(chunk.at[idx, f]).strip())
            t_ns = int(ts[idx].value)
            yield {"asset": str(sym[idx]), "content_ns": int(utc_date_ns(np.array([t_ns]))[0]), "publish_ns": t_ns,
                   "text": txt, "doc_id": f"{sym[idx]}/{k}"}
            k += 1


@register_dataset("cg_fnspid")
class FNSPIDCausalBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_fnspid", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        c = self.cfg
        root, sub = self.root, c.get("subdir", "fnspid")
        base = os.path.join(root, sub)
        recipe = RECIPE.format(root=root)
        pdir = find_first(base, PRICE_CANDIDATES)
        if pdir is None:
            require(root, [f"{sub}/{PRICE_CANDIDATES[0]}"], recipe)
        all_tickers = _price_tickers(pdir)
        tickers = [t.upper() for t in (c.get("tickers") or all_tickers[: int(c.get("max_assets", 100))])]
        if not tickers:
            raise FileNotFoundError(f"No price csv under {pdir}.\nDownload recipe:\n{recipe}")
        by_upper = {t.upper(): t for t in all_tickers}
        horizons = tuple(c.get("horizons", self.horizons))
        start, end = str(c.get("start", "2010-01-01")), str(c.get("end", "2023-12-31"))
        frames: Dict[str, pd.DataFrame] = {}
        for tk in tickers:
            fn = by_upper.get(tk)
            if fn is None:
                raise FileNotFoundError(f"Missing {os.path.join(pdir, tk + '.csv')}.\nDownload recipe:\n{recipe}")
            df = read_ohlcv_csv(os.path.join(pdir, fn + ".csv"))
            frames[tk] = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in frames.values()])))
        if len(dates) < 2:
            raise ValueError(f"FNSPID: fewer than 2 trading days in [{start}, {end}] for {tickers}")
        T, N = len(dates), len(tickers)
        close, adj = align_frames(frames, dates, "close"), align_frames(frames, dates, "adj_close")
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
        labels, btp, avail_label = rule_labels_and_peaks(mid, vol, horizons, horizon=int(lab.get("horizon", 3)),
                                                         k_ret=float(lab.get("k_ret", 3.0)), k_vol=float(lab.get("k_vol", 3.0)),
                                                         window=int(lab.get("window", 60)), lead=int(lab.get("lead", 3)))
        close_ns = session_close_ns(dates, c.get("close_local", "16:00"), c.get("tz", "America/New_York"))
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(horizons), "raw_features": True, "ret_col": 0,
            "feat_names": names, "source": SOURCE_URL, "licence": LICENCE, "dataset": "fnspid", "root": base, "n_assets": N,
            "date_range": [str(dates[0].date()), str(dates[-1].date())], "mid": "adj_close (fallback close)",
            "label_rule": {"type": "abnormal_move_labels", **{k: lab.get(k, v) for k, v in dict(horizon=3, k_ret=3.0, k_vol=3.0, window=60, lead=3).items()}},
            "required_layout": ["Stock_price/full_history/<TICKER>.csv (date,open,high,low,close,adj close,volume)",
                                "Stock_news/nasdaq_exteral_data.csv (Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary) -- prepare script only",
                                "cache/fnspid_docs.npz (optional, cg_prepare_data.py --embed)"],
            "pit_notes": {
                "avail_feat": "t: daily bar knowable at its 16:00 America/New_York close",
                "avail_text": "first NYSE close >= article Date (UTC); row = article calendar date (naive join key); after-close articles have avail_text > row",
                "avail_label": "t + max(label horizon + lead, max(horizons)): rule label knowable once the horizon elapsed",
                "universe": "price row exists (full_history keeps delisted names; not an index membership)",
                "tradable": "volume > 0 and a price row exists",
            },
            "times": "int64 UTC ns of the session close (16:00 America/New_York)",
            "ticker_selection": "cfg tickers" if c.get("tickers") else f"first {int(c.get('max_assets', 100))} price files alphabetically",
        }
        cp = c.get("doc_cache") or cache_path(root, "fnspid")
        cache = read_doc_cache(cp)
        tx = text_fields_from_cache(cache, tickers, dates, close_ns, int(c.get("text_slots", 8)), meta, "fnspid")
        panel = Panel(times=close_ns, assets=tickers, mid=mid, price_feat=price_feat, liq=liq, tradable=tradable, labels=labels,
                      bars_to_peak=btp, text=tx["text"], text_mask=tx["text_mask"], text_age=tx["text_age"], meta=meta)
        return finish_panel(panel, avail_label=avail_label, avail_text=tx["avail_text"], universe=np.isfinite(mid))

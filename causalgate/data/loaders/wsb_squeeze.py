"""D9 WallStreetBets short-squeeze panel: Reddit posts + FINRA short interest + Yahoo daily prices (CausalGate PIT loader).

Registered as "cg_wsb_squeeze". Channels exercised: L5 (post timestamps -> next session close; FINRA short interest
is public only on its *publication* date, about two weeks after the settlement date it describes -- the dataset's
L5 example for a *feature* column; Pushshift retro-collection), L6/L7 (broker-restricted days are untradable).
Nothing is fabricated: missing raw files raise FileNotFoundError with the recipe (RECIPES_D4_D10.md#d9).

Sources
    posts    Kaggle "Reddit WallStreetBets Posts" (Gabriel Preda), columns title, score, id, url, comms_num, created,
             body, timestamp. licence: check Kaggle page (the dataset page states its licence; Reddit content terms
             apply on top). Pushshift dumps (https://files.pushshift.io, availability varies) are an alternative.
             PIT note: both are RETRO-COLLECTED snapshots: posts deleted / edited before the crawl are missing or
             changed, and `score` / `comms_num` are crawl-time values, NOT values known at publish time (recorded
             in meta['pit_notes']['text_metadata']).
    short    FINRA equity short interest (public, https://www.finra.org/finra-data/browse-catalog/equity-short-interest;
             bi-monthly settlement dates, disseminated about two weeks later per FINRA's published schedule).
    float    shares float (proxy): a column in short_interest.csv or float.csv (e.g. from Yahoo "sharesOutstanding"
             / "floatShares"; treated as static per symbol).
    prices   Yahoo Finance daily CSV per ticker (public, personal/research use per Yahoo terms).

Raw file layout expected under cfg['root'] (default data_raw/wsb_squeeze)
    reddit_wsb.csv        the Kaggle csv (title, score, id, url, comms_num, created [unix s], body, timestamp)
    embeddings.npy        optional float32 (n_posts, D), row-aligned with reddit_wsb.csv. Without it the text block is
                          built from publish-time post METADATA [log1p title length, has_body]
                          (meta['text_source'] = 'metadata_pit'); crawl-time score / comms_num only with cfg
                          text_metadata='crawl', whose slots are stamped avail_text = NEVER. Nothing is synthesised.
    short_interest.csv    FINRA export; columns resolved case-insensitively by substring:
                            settlement date   : a column containing 'settlement'            (e.g. settlementDate)
                            symbol            : symbolCode | symbol | ticker
                            short interest    : a column containing 'short' and 'position' or 'interest',
                                                preferring one containing 'current'       (e.g. currentShortPositionQuantity)
                            publication date  : optional, a column containing 'publication' | 'dissemination' | 'publish'
                                                else settlement + cfg si_publication_lag_days (default 14, ASSUMPTION;
                                                export FINRA's exact dissemination dates to be precise)
                            float             : optional, a column containing 'float'
    float.csv             optional: symbol, float | float_shares | shares_float
    prices/<TICKER>.csv   Yahoo daily: Date, Open, High, Low, Close, Adj Close, Volume ; the tickers present = assets

Panel construction
    time axis      trading days (union of price dates); bar close 16:00 ET
    mid            Close
    price_feat     RAW [log_ret, log_volume, hl_range, si_ratio] (column 0 = log return, meta['ret_col'] = 0);
                   si_ratio = short interest / float, step function
                     cfg si_join = publication (default, causal): the value enters at its publication bar
                     cfg si_join = settlement  (the leaky convention): the value enters at its settlement bar and the
                       ROW's avail_feat is stamped with the publication bar (the Panel contract has one avail_feat per
                       row, so the stamp masks the whole row under PITView: this is the honest cost of the leaky join)
                   meta['si_content_dated'] (T, N) and meta['si_avail'] (T, N) always carry the settlement-dated series
                   and its publication bars so that the L5 switch can be reproduced downstream.
    labels         squeeze events: settlement-dated si_ratio > cfg squeeze_si_ratio (0.40) AND trailing 5-day return >
                   cfg squeeze_ret_5d (0.50); onset = first bar of a run, peak = argmax close over
                   [onset, onset + peak_window (10)], end = onset + peak_window; label on [onset - lead (5), peak];
                   bars_to_peak = peak - t (exact, from the realised peak close)
    text           posts placed by their creation time -> first session close at or after it (docs_to_slots, M = text_slots 16);
                   tickers extracted by cashtag ($GME) or bare upper-case symbol present in the asset list (heuristic)
    tradable       False on 2021-01-28 .. 2021-02-04 for the broker-restricted list (cfg 'restricted'); default list
                   GME, AMC, BB, KOSS, EXPR, NOK, NAKD, TR, TRVG, BBBY, AAL, SNDL -- source: public reporting at the time
                   (Robinhood / Interactive Brokers / Webull notices), TO BE RE-VERIFIED before publication.

PIT fields
    avail_feat     t under si_join = publication; publication bar of the newest settlement-dated SI value under
                   si_join = settlement
    avail_text     first session close at or after the post's creation timestamp
    avail_label    event rows: onset + peak_window (the realised peak is fixed once the search window has elapsed);
                   other rows: t + max(horizon)
    universe       has a price path at t (no constituent list; the L6 content of this slot is the restriction mask)
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
from .common import (register_dataset, require, next_bar_index, session_close_ns, raw_price_features, liquidity_fields,
                     event_table_labels, docs_to_slots, finish_panel)
from ..pit import NEVER

RECIPE = """D9 WSB squeeze (see causalgate/data/loaders/RECIPES_D4_D10.md#d9):
  1. Kaggle "Reddit WallStreetBets Posts" (gpreda/reddit-wallstreetsbets-posts) -> data_raw/wsb_squeeze/reddit_wsb.csv
     (licence: check the Kaggle page)
  2. FINRA equity short interest files -> data_raw/wsb_squeeze/short_interest.csv (+ float.csv with shares float)
  3. Yahoo daily CSV per ticker -> data_raw/wsb_squeeze/prices/<TICKER>.csv
  4. optional embeddings.npy (post embeddings, row-aligned with reddit_wsb.csv)"""

RESTRICTED_DEFAULT = ["GME", "AMC", "BB", "KOSS", "EXPR", "NOK", "NAKD", "TR", "TRVG", "BBBY", "AAL", "SNDL"]
RESTRICTED_WINDOW = ("2021-01-28", "2021-02-04")
_ET = "America/New_York"


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _col_containing(df: pd.DataFrame, *subs: str, prefer: Optional[str] = None) -> Optional[str]:
    cands = [c for c in df.columns if all(s in c.lower() for s in subs)]
    if prefer:
        pref = [c for c in cands if prefer in c.lower()]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _parse_ts(x, assume_tz: str = "UTC") -> np.ndarray:
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


def _load_yahoo(pdir: str, assets: Optional[List[str]], start, end) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for f in sorted(glob.glob(os.path.join(pdir, "*.csv"))):
        tk = os.path.splitext(os.path.basename(f))[0].upper()
        if assets and tk not in {a.upper() for a in assets}:
            continue
        df = pd.read_csv(f)
        dc, cc = _col(df, "Date", "date"), _col(df, "Close", "close")
        if dc is None or cc is None:
            continue
        g = pd.DataFrame({"date": pd.to_datetime(df[dc], errors="coerce").dt.tz_localize(None).dt.normalize(),
                          "close": pd.to_numeric(df[cc], errors="coerce")})
        for k, names in (("open", ("Open",)), ("high", ("High",)), ("low", ("Low",)), ("volume", ("Volume", "vol"))):
            c = _col(df, *names)
            g[k] = pd.to_numeric(df[c], errors="coerce") if c else np.nan
        g = g.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
        if start:
            g = g[g["date"] >= pd.Timestamp(start)]
        if end:
            g = g[g["date"] <= pd.Timestamp(end)]
        if len(g):
            out[tk] = g.reset_index(drop=True)
    return out


def _extract_tickers(text: str, universe: set) -> List[str]:
    found = {m.upper() for m in re.findall(r"\$([A-Za-z]{1,5})\b", text or "")}
    found |= {m for m in re.findall(r"\b([A-Z]{2,5})\b", text or "") if m in universe}
    return sorted(t for t in found if t in universe)


# ------------------------------------------------------------------------------------------------ short interest
def _short_interest(root: str, cfg: Dict[str, Any], assets: List[str], cal: pd.DatetimeIndex, bar_ns: np.ndarray) -> pd.DataFrame:
    """-> DataFrame[asset, settle_bar, pub_bar, ratio] sorted by settlement date (rows beyond the panel dropped)."""
    si = pd.read_csv(os.path.join(root, "short_interest.csv"))
    set_c = _col_containing(si, "settlement") or _col(si, "date")
    sym_c = _col(si, "symbolCode", "symbol", "ticker")
    qty_c = _col_containing(si, "short", "position", prefer="current") or _col_containing(si, "short", "interest", prefer="current") \
        or _col(si, "short_interest", "shortinterest")
    pub_c = _col_containing(si, "publication") or _col_containing(si, "dissemination") or _col_containing(si, "publish")
    flt_c = _col_containing(si, "float")
    if set_c is None or sym_c is None or qty_c is None:
        raise ValueError("short_interest.csv needs settlement date, symbol and short-interest columns (see module docstring)")
    df = pd.DataFrame({"symbol": si[sym_c].astype(str).str.upper().str.strip(), "settle": pd.to_datetime(si[set_c], errors="coerce").dt.tz_localize(None).dt.normalize(),
                       "qty": pd.to_numeric(si[qty_c], errors="coerce")})
    # float proxy: column in the SI file, else float.csv (static per symbol)
    if flt_c is not None:
        df["float"] = pd.to_numeric(si[flt_c], errors="coerce")
    else:
        fp = os.path.join(root, "float.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"short_interest.csv has no float column and {fp} is missing (symbol, float)\n{RECIPE}")
        f = pd.read_csv(fp)
        fs, fv = _col(f, "symbol", "ticker"), _col(f, "float", "float_shares", "shares_float", "floatShares")
        mp = dict(zip(f[fs].astype(str).str.upper().str.strip(), pd.to_numeric(f[fv], errors="coerce")))
        df["float"] = df["symbol"].map(mp)
    if pub_c is not None:
        df["pub"] = pd.to_datetime(si[pub_c], errors="coerce").dt.tz_localize(None).dt.normalize()
        # a blank / unparseable publication date falls back to settlement + lag (never to the settlement date itself)
        df["pub"] = df["pub"].fillna(df["settle"] + pd.Timedelta(days=int(cfg.get("si_publication_lag_days", 14))))
    else:
        df["pub"] = df["settle"] + pd.Timedelta(days=int(cfg.get("si_publication_lag_days", 14)))
    df = df.dropna(subset=["settle", "qty", "float"])
    df = df[(df["float"] > 0) & df["symbol"].isin(assets)].copy()
    df["ratio"] = df["qty"] / df["float"]
    apos = {a: i for i, a in enumerate(assets)}
    df["asset"] = df["symbol"].map(apos).astype(int)
    # content bar = the settlement date's session close (state as of that close); publication date-only -> knowable at the
    # next session close AFTER that date (conservative: FINRA's intraday dissemination time is not modelled)
    df["settle_bar"] = next_bar_index(bar_ns, session_close_ns(df["settle"], "16:00", _ET))
    pub_after = session_close_ns(df["pub"], "16:00", _ET) + 1
    df["pub_bar"] = next_bar_index(bar_ns, pub_after)
    df["pub_bar"] = np.maximum(df["pub_bar"], df["settle_bar"])
    df = df[df["settle_bar"] < len(cal)].sort_values(["asset", "settle"]).reset_index(drop=True)
    return df[["asset", "settle_bar", "pub_bar", "ratio", "settle", "pub"]]


def _step_join(T: int, N: int, si: pd.DataFrame, by: str) -> Tuple[np.ndarray, np.ndarray]:
    """Step-function join of the SI ratio on bar `by` (settle_bar | pub_bar). Returns (ratio (T,N), pub_bar_of_value (T,N))."""
    ratio, pub = np.full((T, N), np.nan), np.full((T, N), -1, dtype=np.int64)
    for i in range(N):
        g = si[si["asset"] == i].sort_values(by)
        for r in g.itertuples(index=False):
            b = int(getattr(r, by))
            if b < T:
                ratio[b:, i] = r.ratio
                pub[b:, i] = int(r.pub_bar)
    return ratio, pub


# ------------------------------------------------------------------------------------------------ builder
@register_dataset("cg_wsb_squeeze")
class WSBSqueezePITBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_wsb_squeeze", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        c, root = self.cfg, self.root
        require(root, ["reddit_wsb.csv", "short_interest.csv", "prices"], RECIPE)
        horizons = [int(h) for h in c.get("horizons", self.horizons)]
        hmax = max(horizons)
        lead, peak_win = int(c.get("lead", 5)), int(c.get("peak_window", 10))
        px = _load_yahoo(os.path.join(root, "prices"), c.get("assets"), c.get("start"), c.get("end"))
        if not px:
            raise FileNotFoundError(f"no usable Yahoo CSV under {os.path.join(root, 'prices')}\n{RECIPE}")
        assets = sorted(px)
        cal = pd.DatetimeIndex(sorted(set().union(*[set(px[a]["date"]) for a in assets])))
        T, N = len(cal), len(assets)
        pos = pd.Series(np.arange(T), index=cal)
        A = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close", "volume")}
        for i, a in enumerate(assets):
            rows = pos.reindex(px[a]["date"]).to_numpy().astype(int)
            for k in A:
                A[k][rows, i] = px[a][k].to_numpy(dtype=float)
        has = np.isfinite(A["close"])
        bar_ns = session_close_ns(cal, "16:00", _ET)
        tt = np.arange(T, dtype=np.int64)[:, None]

        # ---- short interest: causal (publication) vs leaky (settlement) join
        si = _short_interest(root, c, assets, cal, bar_ns)
        si_content, si_avail = _step_join(T, N, si, "settle_bar")          # content-dated series + its publication bars
        si_public, _ = _step_join(T, N, si, "pub_bar")                      # what was public at t
        join = str(c.get("si_join", "publication"))
        if join == "publication":
            si_col, avail_feat = si_public, np.broadcast_to(tt, (T, N)).copy()
        elif join == "settlement":
            si_col = si_content
            avail_feat = np.where(si_avail > tt, si_avail, tt).astype(np.int64)
            avail_feat = np.maximum(avail_feat, tt)
        else:
            raise ValueError("si_join must be publication | settlement")

        # ---- squeeze events (label rule uses the settlement-dated ratio: the true state of the world)
        with np.errstate(invalid="ignore", divide="ignore"):
            ret5 = np.full((T, N), np.nan)
            ret5[5:] = A["close"][5:] / A["close"][:-5] - 1.0
        cond = np.nan_to_num(si_content > float(c.get("squeeze_si_ratio", 0.40)), nan=False) & np.nan_to_num(ret5 > float(c.get("squeeze_ret_5d", 0.50)), nan=False)
        rows = []
        for i in range(N):
            t, last_end = 0, -1
            while t < T:
                if cond[t, i] and t > last_end:
                    hi = min(T - 1, t + peak_win)
                    seg = A["close"][t : hi + 1, i]
                    pk = t + int(np.nanargmax(np.where(np.isfinite(seg), seg, -np.inf))) if np.isfinite(seg).any() else t
                    rows.append({"asset": i, "onset_t": t, "peak_t": pk, "end_t": hi})
                    last_end = hi
                    t = hi + 1
                else:
                    t += 1
        events = pd.DataFrame(rows, columns=["asset", "onset_t", "peak_t", "end_t"])
        labels, btp, avail_label = event_table_labels(T, N, events, lead, horizons)

        # ---- broker restrictions
        tradable = has.copy()
        restricted = [r.upper() for r in c.get("restricted", RESTRICTED_DEFAULT)]
        w0, w1 = c.get("restricted_window", RESTRICTED_WINDOW)
        in_win = np.asarray((cal >= pd.Timestamp(w0)) & (cal <= pd.Timestamp(w1)), dtype=bool)
        n_restricted_bars = 0
        for i, a in enumerate(assets):
            if a in restricted:
                tradable[in_win, i] = False
                n_restricted_bars += int(in_win.sum())

        # ---- posts -> text slots by creation time
        posts = pd.read_csv(os.path.join(root, "reddit_wsb.csv"))
        cre_c, ts_c = _col(posts, "created", "created_utc"), _col(posts, "timestamp")
        if cre_c is not None and pd.api.types.is_numeric_dtype(posts[cre_c]):
            post_ns = _parse_ts(posts[cre_c], "UTC")
        elif ts_c is not None:
            post_ns = _parse_ts(posts[ts_c], str(c.get("post_tz", "UTC")))      # ASSUMPTION: Kaggle 'timestamp' is UTC
        else:
            raise ValueError("reddit_wsb.csv needs created (unix s) or timestamp")
        uni = set(assets)
        title_c, body_c = _col(posts, "title"), _col(posts, "body", "selftext")
        texts = posts[title_c].fillna("").astype(str) + " " + (posts[body_c].fillna("").astype(str) if body_c else "")
        text_meta = str(c.get("text_metadata", "pit"))
        if text_meta not in ("pit", "crawl"):
            raise ValueError("text_metadata must be pit | crawl")
        ep = os.path.join(root, "embeddings.npy")
        if os.path.exists(ep):
            E = np.asarray(np.load(ep, mmap_mode="r"), dtype=np.float32)
            if E.shape[0] != len(posts):
                raise ValueError(f"embeddings.npy has {E.shape[0]} rows, reddit_wsb.csv {len(posts)}")
            text_source = "embeddings.npy"
        else:
            # default: ONLY publish-time-knowable metadata (title length, has_body). score / comms_num are crawl-time
            # values (days-months after the post) and are excluded; cfg text_metadata='crawl' adds them as an explicit
            # L5 ablation and then stamps every text slot avail_text = NEVER (never visible under a causal view).
            cols = [np.log1p(posts[title_c].fillna("").astype(str).str.len().to_numpy()),
                    (posts[body_c].fillna("").astype(str).str.len() > 0).to_numpy().astype(float) if body_c else np.zeros(len(posts))]
            text_source = "metadata_pit"
            if text_meta == "crawl":
                sc, cc = _col(posts, "score"), _col(posts, "comms_num", "num_comments")
                cols += [np.log1p(np.clip(pd.to_numeric(posts[sc], errors="coerce").fillna(0).to_numpy(), 0, None)) if sc else np.zeros(len(posts)),
                         np.log1p(np.clip(pd.to_numeric(posts[cc], errors="coerce").fillna(0).to_numpy(), 0, None)) if cc else np.zeros(len(posts))]
                text_source = "metadata_crawl (avail_text = NEVER)"
            E = np.stack(cols, 1).astype(np.float32)
        doc_rows = []
        n_posts_bad_ts = int((post_ns == np.iinfo(np.int64).min).sum())
        for j, (txt, ns) in enumerate(zip(texts, post_ns)):
            if ns == np.iinfo(np.int64).min:               # unparseable creation time: dropped (counted), never bar 0
                continue
            for tk in _extract_tickers(txt, uni):
                doc_rows.append((assets.index(tk), int(ns), j))
        docs = pd.DataFrame(doc_rows, columns=["asset", "publish_ns", "emb_row"])
        M = int(c.get("text_slots", 16))
        slots = docs_to_slots(T, N, M, E.shape[1], docs, bar_ns, E, max_age_bars=int(c.get("text_max_age_bars", 5)))
        avail_text = slots.pop("avail_text")
        if text_source.startswith("metadata_crawl"):
            avail_text = np.where(slots["text_mask"], NEVER, avail_text)

        # ---- features
        pf, feat_names = raw_price_features(A["close"], A["volume"], A["high"], A["low"])
        price_feat = np.concatenate([pf, si_col[..., None].astype(np.float32)], -1)
        feat_names = feat_names + ["si_ratio"]
        liq = liquidity_fields(A["close"], A["volume"], A["high"], A["low"])
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": horizons, "raw_features": True, "ret_col": 0,
            "feat_names": feat_names, "assets": assets, "lead": lead, "peak_window": peak_win, "events": events, "n_events": int(len(events)),
            "si_join": join, "si_table": si, "si_content_dated": si_content, "si_avail": si_avail, "n_docs": int(len(docs)), "n_posts_bad_ts": n_posts_bad_ts,
            "text_source": text_source, "restricted": restricted, "restricted_window": [w0, w1], "n_restricted_bars": n_restricted_bars,
            "source": {"posts": "Kaggle gpreda/reddit-wallstreetsbets-posts ; Pushshift", "short_interest": "FINRA equity short interest (public)",
                       "prices": "Yahoo Finance daily CSV"},
            "licence": "Kaggle WSB dataset: licence: check Kaggle page (Reddit content terms apply). FINRA short interest: public. "
                       "Yahoo prices: personal/research use per Yahoo terms. Post embeddings: per the embedding model's licence.",
            "layout": "reddit_wsb.csv, short_interest.csv [+ float.csv], prices/<TICKER>.csv [, embeddings.npy]",
            "pit_notes": {
                "avail_feat": "si_join=publication: t (the SI column already holds only published values); si_join=settlement: the "
                              "publication bar of the SI value in the row (whole-row stamp, honest cost of the leaky join)",
                "avail_text": "first session close at or after the post's creation timestamp (Kaggle 'created' unix s, UTC); "
                              "posts with an unparseable creation time are dropped (n_posts_bad_ts)",
                "text_metadata": "without embeddings.npy the default text block is publish-time-only metadata [log1p title length, "
                                 "has_body]; score / comms_num are crawl-time values (Pushshift / Kaggle retro-collection) and "
                                 "are included only with cfg text_metadata='crawl', which stamps those slots avail_text = NEVER",
                "avail_label": "event rows: onset + peak_window (realised peak fixed once the window elapsed); else t + max(horizon)",
                "tradable": "False on the broker-restricted days 2021-01-28..2021-02-04 for the restricted list (source: public "
                            "reporting at the time, to be re-verified)",
                "universe": "has a price path at t",
            },
            "assumptions": ["FINRA publication = settlement + 14 calendar days when no publication column is given",
                            "Kaggle 'timestamp' column read as UTC when 'created' is absent", "float treated as static per symbol",
                            "ticker extraction by cashtag or bare upper-case symbol in the asset list"],
        }
        panel = Panel(times=bar_ns, assets=assets, mid=A["close"], price_feat=price_feat, liq=liq, tradable=tradable, labels=labels,
                      bars_to_peak=btp, meta=meta, **slots)
        return finish_panel(panel, avail_label=avail_label, avail_text=avail_text, universe=has, avail_feat=avail_feat)

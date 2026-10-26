"""D4 SEC EDGAR 10-K / 10-Q filings + AAER accounting-fraud labels + Yahoo daily prices (CausalGate PIT loader).

Registered as "cg_edgar". Channels exercised: L5 (the AAER label is released *years* after the fiscal year it
describes) and L4 (the fiscal-year flag is a retrospective label). The loader never fabricates anything: missing raw
files raise FileNotFoundError with the download recipe (also in RECIPES_D4_D10.md).

Sources (all verified public unless marked)
    text      EDGAR full text (https://www.sec.gov/edgar; public domain) or EDGAR-CORPUS (Loukas et al. 2021,
              https://huggingface.co/datasets/eloukas/edgar-corpus). Acceptance datetimes come from the filing header
              (<ACCEPTANCE-DATETIME>) or the sec-edgar daily index; EDGAR-CORPUS itself carries only the filing year.
    labels    AAER misstatement labels of Bao, Ke, Yu, Zhang & Zhang, JAR 2020 (public GitHub release,
              https://github.com/JarFraud/FraudDetection). To the best of our knowledge the public file carries
              fyear / gvkey / p_aaer / new_p_aaer / misstate + 28 raw financial items and NOT the AAER release date;
              the `aaer_release_date` column must be added by joining p_aaer (the AAER number) to the SEC AAER index
              (https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases). Rows with
              misstate = 1 and no release date get avail_label = NEVER (counted in meta['n_fraud_without_release']).
    prices    Yahoo Finance daily CSV per ticker (public, personal/research use per Yahoo terms).
    optional  Audit Analytics restatements (LICENSED): accepted through `restatements.csv`, never required.

Raw file layout expected under cfg['root'] (default data_raw/edgar)
    filings.csv          one row per filing. Columns (case-insensitive, first match wins):
                           ticker | cik (cik is mapped through tickers.csv)
                           form | form_type                       kept if it starts with 10-K or 10-Q (cfg 'forms')
                           fyear | fiscal_year                     else the year of period | period_of_report | period_end
                           accepted | acceptance_datetime | acceptance   ET wall-clock, e.g. "2015-02-27 16:31:05"
                             else date_filed | filed | filing_date   (ASSUMPTION: a date-only filing is treated as
                             accepted after the close of that day -> available at the next session close)
                           sic (optional), accession (optional), emb_row (optional row into embeddings.npy; default =
                           the filing's own row number in filings.csv)
    tickers.csv          optional cik -> ticker map (columns cik, ticker; e.g. from sec.gov/files/company_tickers.json)
    aaer.csv             label table: ticker | cik | gvkey (gvkey needs gvkey_cik.csv), fyear, misstate (0/1),
                         aaer_release_date | release_date (YYYY-MM-DD; see note above), p_aaer (optional)
    gvkey_cik.csv        optional gvkey -> cik map (Compustat; LICENSED) when aaer.csv is keyed by gvkey only
    embeddings.npy       optional float32 (n_filings, D) or (n_filings, C, D) [C MD&A chunks], row-aligned with
                         filings.csv. Without it the panel has no text block (nothing is synthesised).
    sic.csv              optional ticker | cik -> sic when filings.csv has no sic column
    restatements.csv     optional Audit Analytics export (LICENSED): ticker | cik, fyear | restated_fiscal_year,
                         restatement_date | disclosure_date. Used only when cfg label_source == 'restatement'.
    prices/<TICKER>.csv  Yahoo daily: Date, Open, High, Low, Close, Adj Close, Volume

Panel construction
    time axis      the trading-day calendar (union of price dates), so that impact horizons are exact trading days;
                   decision points are the filing bars t = first session close at/after the acceptance datetime and
                   are marked in meta['decision_mask'] (T, N) bool. Downstream samplers may restrict to that mask.
    mid            Close (not Adj Close: the limit/impact arithmetic needs the traded price; cfg 'price_col' overrides)
    price_feat     RAW [log_ret, log_volume, hl_range] (column 0 = log return, meta['ret_col'] = 0)
    labels         fiscal-year misstate flag placed on the filing bar (+ cfg 'label_span_bars' bars after it, default 0,
                   and cfg 'lead' bars before it, default 0: text is not knowable before the filing)
    bars_to_peak   realised |impact| argmax over the horizon grid on labelled bars (no curated peak for fraud)
    text           filing embeddings placed by *acceptance* time (docs_to_slots, M = cfg text_slots, repeated for
                   text_max_age_bars bars with growing age)
    graph          SIC-2 industry neighbours: graph[t, i, k] = raw feature row of neighbour k at t (K = cfg graph_k)

PIT fields
    avail_feat     t (a session close is knowable at its bar)
    avail_text     filing bar t (acceptance datetime -> first close at/after it)
    avail_label    misstate = 1 -> bar of the AAER release date (max(t, release bar); NEVER if unknown; a date-only
                   release is treated as after that day's close -> next session close, the same rule as filings);
                   filings with an unparseable acceptance/filing time are dropped (meta['n_filings_bad_ts']);
                   misstate = 0 and non-filing rows -> t + max(horizon) (the optimistic "no event yet" convention;
                   cfg 'clean_label_lag_bars' raises it)
    avail_graph    t (industry membership taken from the filing header; treated as static)
    universe       the firm has a price path at t (no survivorship list exists for this slot; delisted paths are kept
                   until their last price)
"""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from warn2trade.data.labels import impact_curves, bars_to_peak_from_labels
from .common import (register_dataset, require, next_bar_index, session_close_ns, raw_price_features, liquidity_fields,
                     docs_to_slots, finish_panel)
from ..pit import NEVER

RECIPE = """D4 EDGAR + AAER (see causalgate/data/loaders/RECIPES_D4_D10.md#d4):
  1. filings index with acceptance datetimes: EDGAR full-text index (https://www.sec.gov/Archives/edgar/full-index/)
     + the <ACCEPTANCE-DATETIME> header of each filing, or EDGAR-CORPUS (https://huggingface.co/datasets/eloukas/edgar-corpus)
     -> data_raw/edgar/filings.csv (ticker|cik, form, fyear, accepted, sic)
  2. AAER labels: https://github.com/JarFraud/FraudDetection (Bao et al., JAR 2020) -> data_raw/edgar/aaer.csv with an
     added aaer_release_date column (join p_aaer to the SEC AAER index)
  3. prices: Yahoo daily CSV per ticker -> data_raw/edgar/prices/<TICKER>.csv
  4. optional: embeddings.npy (MD&A chunk embeddings, row-aligned with filings.csv), tickers.csv, sic.csv"""

_FORM_PREFIXES = ("10-K", "10-Q")


# ------------------------------------------------------------------------------------------------ small helpers
def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


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


def _load_yahoo_prices(pdir: str, assets: Optional[List[str]], start: Optional[str], end: Optional[str],
                       price_col: str = "Close") -> Dict[str, pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(pdir, "*.csv")))
    out: Dict[str, pd.DataFrame] = {}
    for f in files:
        tk = os.path.splitext(os.path.basename(f))[0].upper()
        if assets and tk not in {a.upper() for a in assets}:
            continue
        df = pd.read_csv(f)
        dc, cc = _col(df, "Date", "date", "timestamp"), _col(df, price_col, "Close", "close", "Adj Close")
        if dc is None or cc is None:
            continue
        d = pd.to_datetime(df[dc], errors="coerce").dt.tz_localize(None).dt.normalize()
        g = pd.DataFrame({"date": d, "close": pd.to_numeric(df[cc], errors="coerce")})
        for k, names in (("open", ("Open", "open")), ("high", ("High", "high")), ("low", ("Low", "low")), ("volume", ("Volume", "volume", "vol"))):
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


def _panel_arrays(px: Dict[str, pd.DataFrame]) -> Tuple[List[str], pd.DatetimeIndex, Dict[str, np.ndarray]]:
    assets = sorted(px)
    cal = pd.DatetimeIndex(sorted(set().union(*[set(px[a]["date"]) for a in assets])))
    T, N = len(cal), len(assets)
    arrs = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close", "volume")}
    pos = pd.Series(np.arange(T), index=cal)
    for i, a in enumerate(assets):
        g = px[a]
        rows = pos.reindex(g["date"]).to_numpy()
        for k in arrs:
            arrs[k][rows, i] = g[k].to_numpy(dtype=float)
    return assets, cal, arrs


# ------------------------------------------------------------------------------------------------ label tables
def _key_to_ticker(df: pd.DataFrame, root: str) -> pd.Series:
    """Resolve a ticker column from ticker | cik (tickers.csv) | gvkey (gvkey_cik.csv -> tickers.csv)."""
    tc = _col(df, "ticker", "symbol", "tic")
    if tc is not None:
        return df[tc].astype(str).str.upper().str.strip()
    cik_col = _col(df, "cik")
    if cik_col is None:
        gv = _col(df, "gvkey")
        mp = os.path.join(root, "gvkey_cik.csv")
        if gv is None or not os.path.exists(mp):
            raise ValueError("label table needs ticker, cik (+tickers.csv) or gvkey (+gvkey_cik.csv +tickers.csv)")
        m = pd.read_csv(mp)
        m_g, m_c = _col(m, "gvkey"), _col(m, "cik")
        mapping = dict(zip(pd.to_numeric(m[m_g], errors="coerce"), pd.to_numeric(m[m_c], errors="coerce")))
        cik = pd.to_numeric(df[gv], errors="coerce").map(mapping)
    else:
        cik = pd.to_numeric(df[cik_col], errors="coerce")
    tp = os.path.join(root, "tickers.csv")
    if not os.path.exists(tp):
        raise FileNotFoundError(f"cik-keyed table needs {tp} (columns cik, ticker)")
    t = pd.read_csv(tp)
    t_c, t_t = _col(t, "cik"), _col(t, "ticker", "symbol")
    mapping = dict(zip(pd.to_numeric(t[t_c], errors="coerce"), t[t_t].astype(str).str.upper().str.strip()))
    return cik.map(mapping)


_DATE_ONLY_OFFSET_NS = int((23 * 3600 + 59 * 60) * 1e9)


def _date_only_mask(x) -> np.ndarray:
    """True where a timestamp has no time of day (YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD strings or integer YYYYMMDD)."""
    sr = pd.Series(x).reset_index(drop=True)
    st = sr.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    m = st.str.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}|\d{4}/\d{2}/\d{2}").fillna(False).to_numpy()
    return m & sr.notna().to_numpy()


def _label_table(root: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """-> DataFrame[ticker, fyear, label, release_ns (float, NaN = unknown)]."""
    src = str(cfg.get("label_source", "aaer"))
    if src == "aaer":
        df = pd.read_csv(os.path.join(root, "aaer.csv"))
        lab_c = _col(df, "misstate", "label", "fraud")
        rel_c = _col(df, "aaer_release_date", "release_date", "aaer_date", "release")
    elif src == "restatement":
        p = os.path.join(root, "restatements.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"label_source=restatement needs {p} (Audit Analytics export, LICENSED; never required otherwise)")
        df = pd.read_csv(p)
        lab_c = _col(df, "restated", "label")
        rel_c = _col(df, "restatement_date", "disclosure_date", "release_date", "file_date")
    else:
        raise ValueError(f"label_source must be aaer | restatement, got {src!r}")
    fy_c = _col(df, "fyear", "fiscal_year", "year", "restated_fiscal_year")
    if fy_c is None:
        raise ValueError("label table needs a fiscal-year column (fyear | fiscal_year | year)")
    out = pd.DataFrame({"ticker": _key_to_ticker(df, root), "fyear": pd.to_numeric(df[fy_c], errors="coerce")})
    out["label"] = pd.to_numeric(df[lab_c], errors="coerce").fillna(0).astype(int).clip(0, 1) if lab_c else 1
    out["release_ns"] = _parse_ts(df[rel_c], "America/New_York").astype(float) if rel_c else np.nan
    if rel_c:
        out.loc[pd.to_datetime(df[rel_c], errors="coerce").isna().to_numpy(), "release_ns"] = np.nan
        # same rule as date-only filings: an unknown intraday release time is treated as after that day's close
        # (23:59 ET) -> knowable at the NEXT session close
        donly = _date_only_mask(df[rel_c]) & np.isfinite(out["release_ns"].to_numpy(dtype=float))
        out.loc[donly, "release_ns"] = out.loc[donly, "release_ns"] + float(_DATE_ONLY_OFFSET_NS)
    out = out.dropna(subset=["ticker", "fyear"])
    out["fyear"] = out["fyear"].astype(int)
    # a (ticker, fyear) flagged by any AAER is a fraud year; keep the earliest known release date
    agg = out.groupby(["ticker", "fyear"], as_index=False).agg(label=("label", "max"), release_ns=("release_ns", "min"))
    return agg


# ------------------------------------------------------------------------------------------------ builder
@register_dataset("cg_edgar")
class EdgarPITBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_edgar", "daily", 252, (1, 3, 5, 10, 20, 60)

    def build_panel(self) -> Panel:
        c, root = self.cfg, self.root
        require(root, ["filings.csv", "aaer.csv" if str(c.get("label_source", "aaer")) == "aaer" else "restatements.csv", "prices"], RECIPE)
        horizons = [int(h) for h in c.get("horizons", self.horizons)]
        hmax = max(horizons)
        lead, span = int(c.get("lead", 0)), int(c.get("label_span_bars", 0))
        clean_lag = int(c.get("clean_label_lag_bars", hmax))

        # ---- prices -> trading-day axis
        px = _load_yahoo_prices(os.path.join(root, "prices"), c.get("assets"), c.get("start"), c.get("end"), str(c.get("price_col", "Close")))
        if not px:
            raise FileNotFoundError(f"no usable Yahoo CSV under {os.path.join(root, 'prices')}\n{RECIPE}")
        assets, cal, A = _panel_arrays(px)
        T, N = len(cal), len(assets)
        bar_ns = session_close_ns(cal, "16:00", "America/New_York")
        apos = {a: i for i, a in enumerate(assets)}

        # ---- filings
        fil = pd.read_csv(os.path.join(root, "filings.csv"))
        fil["_row"] = np.arange(len(fil))                     # original row = embedding row when emb_row is absent
        fil["_ticker"] = _key_to_ticker(fil, root)
        fc = _col(fil, "form", "form_type", "type")
        forms = tuple(c.get("forms", _FORM_PREFIXES))
        if fc is not None:
            fil = fil[fil[fc].astype(str).str.upper().str.strip().str.startswith(forms)]
        fy_c = _col(fil, "fyear", "fiscal_year")
        if fy_c is not None:
            fil["_fyear"] = pd.to_numeric(fil[fy_c], errors="coerce")
        else:
            pc = _col(fil, "period", "period_of_report", "period_end")
            if pc is None:
                raise ValueError("filings.csv needs fyear | fiscal_year or period | period_of_report")
            fil["_fyear"] = pd.to_datetime(fil[pc], errors="coerce").dt.year
        acc_c = _col(fil, "accepted", "acceptance_datetime", "acceptance", "accepted_at")
        if acc_c is not None:
            fil["_acc_ns"] = _parse_ts(fil[acc_c], "America/New_York")
            date_only = _date_only_mask(fil[acc_c])
        else:
            dc = _col(fil, "date_filed", "filed", "filing_date", "date")
            if dc is None:
                raise ValueError("filings.csv needs accepted | acceptance_datetime or date_filed | filing_date")
            fil["_acc_ns"] = _parse_ts(fil[dc], "America/New_York")
            date_only = np.ones(len(fil), dtype=bool)
        # ASSUMPTION: a date-only acceptance is treated as after the close of that day (23:59 ET) -> next session close
        # unparseable acceptance / filing times (NaT sentinel) are DROPPED and counted, never mapped to bar 0
        bad_ts = fil["_acc_ns"].to_numpy() == np.iinfo(np.int64).min
        n_bad_ts = int(bad_ts.sum())
        fil = fil[~bad_ts].copy()
        date_only = np.asarray(date_only)[~bad_ts]
        fil.loc[date_only, "_acc_ns"] = fil.loc[date_only, "_acc_ns"] + _DATE_ONLY_OFFSET_NS
        fil = fil[fil["_ticker"].isin(apos) & fil["_fyear"].notna()].copy()
        fil["_asset"] = fil["_ticker"].map(apos).astype(int)
        fil["_t"] = next_bar_index(bar_ns, fil["_acc_ns"].to_numpy())
        fil = fil[(fil["_t"] >= 0) & (fil["_t"] < T)].reset_index(drop=True)
        fil["_fyear"] = fil["_fyear"].astype(int)

        # ---- labels: fiscal-year flag on the filing bar, availability = AAER release bar
        lab = _label_table(root, c)
        key = pd.MultiIndex.from_frame(lab[["ticker", "fyear"]])
        lab_label = pd.Series(lab["label"].to_numpy(), index=key)
        lab_rel = pd.Series(lab["release_ns"].to_numpy(), index=key)
        fkey = pd.MultiIndex.from_arrays([fil["_ticker"], fil["_fyear"]])
        fil["_label"] = lab_label.reindex(fkey).fillna(0).astype(int).to_numpy()
        fil["_release_ns"] = lab_rel.reindex(fkey).to_numpy()
        labels = np.zeros((T, N), dtype=np.int8)
        tt = np.arange(T, dtype=np.int64)[:, None]
        avail_label = np.broadcast_to(tt + clean_lag, (T, N)).copy()
        decision = np.zeros((T, N), dtype=bool)
        n_fraud_no_release = 0
        for t, i, lab_v, rel in zip(fil["_t"].to_numpy(), fil["_asset"].to_numpy(), fil["_label"].to_numpy(), fil["_release_ns"].to_numpy(dtype=float)):
            t, i = int(t), int(i)
            decision[t, i] = True
            lo, hi = max(0, t - lead), min(T - 1, t + span)
            if int(lab_v) == 1:
                labels[lo : hi + 1, i] = 1
                if np.isfinite(rel):
                    rb = int(next_bar_index(bar_ns, np.array([int(rel)]))[0])
                    av = rb if rb < T else NEVER
                else:
                    av, n_fraud_no_release = NEVER, n_fraud_no_release + 1
                rows = np.arange(lo, hi + 1)
                avail_label[rows, i] = np.maximum(rows, av)
        imp = impact_curves(A["close"], horizons)
        btp = bars_to_peak_from_labels(labels, imp, horizons)

        # ---- features / liquidity / tradable
        price_feat, feat_names = raw_price_features(A["close"], A["volume"], A["high"], A["low"])
        liq = liquidity_fields(A["close"], A["volume"], A["high"], A["low"])
        has = ~np.isnan(A["close"])
        tradable = has.copy()

        # ---- text: filing embeddings placed by acceptance time
        text_kw: Dict[str, Any] = {}
        avail_text = None
        ep = os.path.join(root, "embeddings.npy")
        text_source = None
        if os.path.exists(ep) and len(fil):
            emb = np.load(ep, mmap_mode="r")
            emb_row_c = _col(fil, "emb_row")
            rows_all = fil[emb_row_c].to_numpy(dtype=int) if emb_row_c else fil["_row"].to_numpy(dtype=int)
            if rows_all.max() >= emb.shape[0]:
                raise ValueError(f"embeddings.npy has {emb.shape[0]} rows but filings reference row {rows_all.max()}")
            if emb.ndim == 2:
                E = np.asarray(emb, dtype=np.float32)
                docs = pd.DataFrame({"asset": fil["_asset"], "publish_ns": fil["_acc_ns"], "emb_row": rows_all})
                M = int(c.get("text_slots", 2))
            elif emb.ndim == 3:
                n, C, D = emb.shape
                E = np.asarray(emb, dtype=np.float32).reshape(n * C, D)
                docs = pd.DataFrame({"asset": np.repeat(fil["_asset"].to_numpy(), C), "publish_ns": np.repeat(fil["_acc_ns"].to_numpy(), C),
                                     "emb_row": (rows_all[:, None] * C + np.arange(C)[None, :]).ravel()})
                M = int(c.get("text_slots", min(C, 32)))
            else:
                raise ValueError(f"embeddings.npy must be 2-D or 3-D, got shape {emb.shape}")
            slots = docs_to_slots(T, N, M, E.shape[1], docs, bar_ns, E, max_age_bars=int(c.get("text_max_age_bars", 5)))
            avail_text = slots.pop("avail_text")
            text_kw = slots
            text_source = "embeddings.npy"

        # ---- graph: SIC-2 industry neighbours (static membership from the filing header)
        graph_kw: Dict[str, Any] = {}
        avail_graph = None
        sic = self._sic_per_asset(root, fil, assets)
        if bool(c.get("graph", True)) and sic is not None:
            K = int(c.get("graph_k", 8))
            graph = np.zeros((T, N, K, price_feat.shape[-1]), dtype=np.float32)
            gmask = np.zeros((T, N, K), dtype=bool)
            buckets: Dict[str, List[int]] = {}
            for j, s2 in enumerate(sic):
                if s2 is not None:
                    buckets.setdefault(s2, []).append(j)
            for i in range(N):
                if sic[i] is None:
                    continue
                nbrs = [j for j in buckets[sic[i]] if j != i][:K]
                for k, j in enumerate(nbrs):
                    graph[:, i, k] = np.nan_to_num(price_feat[:, j])
                    gmask[:, i, k] = has[:, j]
            graph_kw = {"graph": graph, "graph_mask": gmask}
            avail_graph = np.broadcast_to(tt[:, :, None], (T, N, K)).copy()

        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": horizons, "raw_features": True, "ret_col": 0,
            "feat_names": feat_names, "assets": assets, "n_filings": int(len(fil)), "n_fraud_filings": int((fil["_label"] == 1).sum()),
            "n_fraud_without_release": int(n_fraud_no_release), "n_filings_bad_ts": n_bad_ts, "decision_mask": decision, "lead": lead, "label_span_bars": span,
            "label_source": str(c.get("label_source", "aaer")), "text_source": text_source, "sic2": sic,
            "filings": fil[["_ticker", "_asset", "_fyear", "_t", "_label", "_release_ns"]].rename(columns=lambda s: s.lstrip("_")),
            "source": {"text": "https://www.sec.gov/edgar ; EDGAR-CORPUS https://huggingface.co/datasets/eloukas/edgar-corpus",
                       "labels": "AAER, Bao et al. JAR 2020, https://github.com/JarFraud/FraudDetection",
                       "prices": "Yahoo Finance daily CSV"},
            "licence": "EDGAR text: public domain (US government work). AAER labels: public from the JAR 2020 GitHub release "
                       "(cite Bao, Ke, Yu, Zhang & Zhang 2020). Yahoo prices: personal/research use per Yahoo terms. "
                       "Audit Analytics restatements: LICENSED, optional, never required.",
            "layout": "filings.csv, aaer.csv, prices/<TICKER>.csv [, tickers.csv, embeddings.npy, sic.csv, gvkey_cik.csv, restatements.csv]",
            "pit_notes": {
                "avail_feat": "t: a session close is knowable at its bar",
                "avail_text": "filing bar = first session close at/after the EDGAR acceptance datetime (date-only -> next session)",
                "avail_label": "misstate=1: bar of the AAER release date (years after the fiscal year; date-only release -> next "
                               "session close; NEVER when the release date is missing); misstate=0 / non-filing rows: t + clean_label_lag_bars (default max horizon)",
                "avail_graph": "t: SIC-2 membership read from the filing header, treated as static",
                "universe": "firm has a price path at t; no survivorship list for this slot (delisted paths kept until exit)",
            },
            "assumptions": ["time axis = trading days; decision points = filing bars in meta['decision_mask']",
                            "date-only filings are treated as accepted after the close (next session)",
                            "AAER release date column is user-constructed from the SEC AAER index (not in the JAR 2020 file)"],
        }
        panel = Panel(times=bar_ns, assets=assets, mid=A["close"], price_feat=price_feat, liq=liq, tradable=tradable, labels=labels,
                      bars_to_peak=btp, meta=meta, **text_kw, **graph_kw)
        return finish_panel(panel, avail_label=avail_label, avail_text=avail_text, universe=has, avail_graph=avail_graph)

    @staticmethod
    def _sic_per_asset(root: str, fil: pd.DataFrame, assets: List[str]) -> Optional[List[Optional[str]]]:
        sic_c = _col(fil, "sic")
        table: Dict[str, str] = {}
        if sic_c is not None:
            for tk, v in zip(fil["_ticker"], fil[sic_c]):
                if pd.notna(v) and tk not in table:
                    table[tk] = str(int(float(v))).zfill(4)[:2]
        sp = os.path.join(root, "sic.csv")
        if os.path.exists(sp):
            s = pd.read_csv(sp)
            s["_ticker"] = _key_to_ticker(s, root)
            sc = _col(s, "sic")
            for tk, v in zip(s["_ticker"], s[sc]):
                if pd.notna(v) and tk not in table:
                    table[tk] = str(int(float(v))).zfill(4)[:2]
        if not table:
            return None
        return [table.get(a) for a in assets]

"""D10 Elliptic++ Bitcoin transaction graph aggregated to hourly BTC bars (CausalGate PIT loader).

Registered as "cg_fin_graph". Channels exercised: L4 / L5 (illicit labels are assigned RETROSPECTIVELY by the data
providers; the loader stamps every label with avail_label = t + cfg label_lag_bars, the paper's sensitivity axis,
default 24 * 7 bars = one week) and the graph-feature analogue (label-derived graph snapshot features carry the same
lag; structural ones are knowable at t). Nothing is fabricated: missing raw files raise FileNotFoundError with the
recipe (RECIPES_D4_D10.md#d10).

Sources
    graph    Elliptic++ (Elmougy & Liu, KDD 2023), https://github.com/git-disl/EllipticPlusPlus -- licence: per the
             repository (check the LICENSE file before redistribution); the original Elliptic data set (Weber et al.
             2019, Kaggle "elliptic-data-set") is under the Kaggle page's licence.
    prices   hourly BTC-USD bars from any exchange export (e.g. Kraken / Binance / Bitstamp public OHLCV history);
             licence per the exchange's data terms. Multi-asset variant: one csv per asset under prices/.

Timestamps -- the honest limitation of this slot
    Elliptic / Elliptic++ ship a 1..49 "Time step" per transaction (steps about two weeks apart; each step a connected
    component of transactions "within less than three hours of each other"), NOT block timestamps. Hourly aggregation
    therefore needs ONE of
      (a) ellipticpp/tx_times.csv   txId, timestamp   -- user-constructed (e.g. from a published txId -> tx-hash
                                                           de-anonymisation mapping + block times); NOT shipped with
                                                           Elliptic++
      (b) ellipticpp/timestep_calendar.csv   time_step, start, end  (UTC) -- user-constructed step calendar; every
          transaction of a step is then assigned to the bar of the step END (conservative: the whole step becomes
          knowable only when it is complete; nothing inside the step is interpolated or spread)
    Without either file the loader raises (it refuses to invent a calendar).

Raw file layout expected under cfg['root'] (default data_raw/fin_graph)
    ellipticpp/txs_features.csv    txId, Time step, <local + aggregated features>  (names as shipped; a column whose
                                   lower-case name contains both 'btc' and 'total' is used as the BTC amount if present)
    ellipticpp/txs_classes.csv     txId, class   with 1 = illicit, 2 = licit, 3 | 'unknown' = unknown
    ellipticpp/txs_edgelist.csv    txId1, txId2
    ellipticpp/tx_times.csv  OR  ellipticpp/timestep_calendar.csv   (see above)
    ellipticpp/wallets_*.csv, *_edgelist.csv   optional wallet files, not used by this loader (recorded in meta)
    prices/<ASSET>.csv             hourly bars: timestamp, open, high, low, close, volume (timestamp = bar OPEN time,
                                   epoch s/ms or ISO UTC; ASSUMPTION; set cfg timestamp_is: close otherwise);
                                   default assets ['BTC']; legacy single file btc_1h.csv is also accepted

Panel construction
    time axis      the hourly bar closes of the first asset (others are aligned on it); cfg start / end restrict it
    mid            close ; price_feat RAW [log_ret, log_volume, hl_range] (column 0 = log return, meta['ret_col'] = 0)
    graph          (T, N, K = 2, Dg = 6) per-bar transaction-graph snapshot (the BTC graph is shared by every asset):
                     slot 0 (structural, avail_graph = t):
                       [log1p n_tx, log1p n_internal_edges, log1p n_edges_to_past, mean in-degree, log1p BTC total, 0]
                     slot 1 (label-derived, avail_graph = t + label_lag_bars):
                       [illicit share of class-known txs, share of txs with an illicit neighbour, licit share,
                        log1p n_illicit, log1p n_class_known, 0]
                   graph_mask is True on bars that contain at least one transaction
    labels         1 when the illicit share of class-known transactions in the bar > cfg illicit_share_thr (0.05,
                   ASSUMPTION: above the ~2 % base rate of the data set); bars_to_peak = realised |impact| argmax
    tradable       has a price bar

PIT fields
    avail_feat     t
    avail_label    t + label_lag_bars for EVERY row (a "no illicit activity" label is confirmed only after the lag too);
                   label_lag_bars is in meta (sensitivity axis)
    avail_graph    slot 0: t ; slot 1: t + label_lag_bars
    universe       has a price bar
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
from .common import register_dataset, require, next_bar_index, raw_price_features, liquidity_fields, finish_panel

RECIPE = """D10 Elliptic++ + BTC hourly (see causalgate/data/loaders/RECIPES_D4_D10.md#d10):
  1. git clone https://github.com/git-disl/EllipticPlusPlus (or download its Google-Drive release) ->
     data_raw/fin_graph/ellipticpp/{txs_features.csv, txs_classes.csv, txs_edgelist.csv}
  2. transaction timestamps: data_raw/fin_graph/ellipticpp/tx_times.csv (txId, timestamp) from a txId->hash mapping,
     or data_raw/fin_graph/ellipticpp/timestep_calendar.csv (time_step, start, end)
  3. hourly BTC-USD bars -> data_raw/fin_graph/prices/BTC.csv (timestamp, open, high, low, close, volume)"""

GRAPH_DIM = 6


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


def _load_hourly(root: str, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    pdir = os.path.join(root, "prices")
    files = sorted(glob.glob(os.path.join(pdir, "*.csv")))
    legacy = os.path.join(root, "btc_1h.csv")
    if not files and os.path.exists(legacy):
        files = [legacy]
    assets = {a.upper() for a in cfg.get("assets", [])} or None
    bar_h = float(cfg.get("bar_hours", 1.0))
    ts_is_close = str(cfg.get("timestamp_is", "open")) == "close"
    out: Dict[str, pd.DataFrame] = {}
    for f in files:
        tk = os.path.splitext(os.path.basename(f))[0].upper().replace("_1H", "")
        if assets and tk not in assets:
            continue
        df = pd.read_csv(f)
        tc, cc = _col(df, "timestamp", "time", "datetime", "date", "t", "open_time"), _col(df, "close", "c")
        if tc is None or cc is None:
            continue
        ns = _parse_ts(df[tc], "UTC")
        if not ts_is_close:
            ns = ns + int(bar_h * 3600 * 1e9)
        g = pd.DataFrame({"close_ns": ns, "close": pd.to_numeric(df[cc], errors="coerce")})
        for k, names in (("open", ("open", "o")), ("high", ("high", "h")), ("low", ("low", "l")), ("volume", ("volume", "v", "vol"))):
            c = _col(df, *names)
            g[k] = pd.to_numeric(df[c], errors="coerce") if c else np.nan
        g = g.dropna(subset=["close"]).drop_duplicates("close_ns").sort_values("close_ns")
        if cfg.get("start"):
            g = g[g["close_ns"] >= _parse_ts([cfg["start"]])[0]]
        if cfg.get("end"):
            g = g[g["close_ns"] <= _parse_ts([cfg["end"]])[0]]
        if len(g):
            out[tk] = g.reset_index(drop=True)
    return out


def _tx_times(edir: str, feats: pd.DataFrame, tx_c: str) -> Tuple[np.ndarray, str]:
    """int64 UTC ns per transaction row of txs_features (NaT -> -1): tx_times.csv, else timestep_calendar.csv (step end)."""
    tp, cp = os.path.join(edir, "tx_times.csv"), os.path.join(edir, "timestep_calendar.csv")
    if os.path.exists(tp):
        t = pd.read_csv(tp)
        a, b = _col(t, "txId", "tx_id", "txid"), _col(t, "timestamp", "time", "block_time", "datetime")
        mp = pd.Series(_parse_ts(t[b]), index=pd.to_numeric(t[a], errors="coerce"))
        mp = mp[~mp.index.duplicated()]
        out = mp.reindex(pd.to_numeric(feats[tx_c], errors="coerce")).to_numpy()
        return np.where(np.isfinite(out), out, -1).astype(np.int64), "tx_times.csv"
    if os.path.exists(cp):
        step_c = _col(feats, "Time step", "time_step", "timestep", "ts")
        if step_c is None:
            raise ValueError("txs_features.csv has no 'Time step' column; timestep_calendar.csv cannot be applied")
        cal = pd.read_csv(cp)
        s_c, e_c = _col(cal, "time_step", "step", "Time step"), _col(cal, "end", "end_time")
        if s_c is None or e_c is None:
            raise ValueError("timestep_calendar.csv needs time_step and end columns")
        mp = pd.Series(_parse_ts(cal[e_c]), index=pd.to_numeric(cal[s_c], errors="coerce").astype(int))
        mp = mp[~mp.index.duplicated()]
        out = mp.reindex(pd.to_numeric(feats[step_c], errors="coerce")).to_numpy()
        return np.where(np.isfinite(out), out, -1).astype(np.int64), "timestep_calendar.csv (step end, conservative)"
    raise FileNotFoundError(f"Elliptic++ has no block timestamps: provide {tp} (txId, timestamp) or {cp} (time_step, start, end).\n{RECIPE}")


@register_dataset("cg_fin_graph")
class FinGraphPITBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_fin_graph", "1h", 365 * 24, (1, 2, 4, 8, 24, 48)

    def build_panel(self) -> Panel:
        c, root = self.cfg, self.root
        edir = os.path.join(root, "ellipticpp")
        require(root, ["ellipticpp/txs_features.csv", "ellipticpp/txs_classes.csv", "ellipticpp/txs_edgelist.csv"], RECIPE)
        if not (glob.glob(os.path.join(root, "prices", "*.csv")) or os.path.exists(os.path.join(root, "btc_1h.csv"))):
            raise FileNotFoundError(f"Missing prices/<ASSET>.csv (hourly bars) under {root}.\nDownload recipe:\n{RECIPE}")
        horizons = [int(h) for h in c.get("horizons", self.horizons)]
        lag = int(c.get("label_lag_bars", 24 * 7))
        thr = float(c.get("illicit_share_thr", 0.05))

        # ---- prices -> hourly axis (first asset's closes; others aligned)
        px = _load_hourly(root, c)
        if not px:
            raise FileNotFoundError(f"no usable hourly csv under {os.path.join(root, 'prices')}\n{RECIPE}")
        assets = sorted(px)
        axis = px[assets[0]]["close_ns"].to_numpy(dtype=np.int64)
        T, N = len(axis), len(assets)
        pos = pd.Series(np.arange(T), index=axis)
        A = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close", "volume")}
        for i, a in enumerate(assets):
            g = px[a]
            r = pos.reindex(g["close_ns"].to_numpy())
            ok = r.notna().to_numpy()
            rows = r.to_numpy()[ok].astype(int)
            for k in A:
                A[k][rows, i] = g[k].to_numpy(dtype=float)[ok]
        has = np.isfinite(A["close"])
        tt = np.arange(T, dtype=np.int64)[:, None]

        # ---- transactions -> bars
        feats = pd.read_csv(os.path.join(edir, "txs_features.csv"))
        tx_c = _col(feats, "txId", "tx_id", "txid")
        if tx_c is None:
            raise ValueError("txs_features.csv needs a txId column")
        tx_ns, time_source = _tx_times(edir, feats, tx_c)
        txid = pd.to_numeric(feats[tx_c], errors="coerce").to_numpy()
        bar = np.full(len(feats), -1, dtype=np.int64)
        okt = tx_ns >= 0
        bar[okt] = next_bar_index(axis, tx_ns[okt])                      # first bar close >= tx time
        bar[(bar >= T)] = -1
        cls = pd.read_csv(os.path.join(edir, "txs_classes.csv"))
        cc_id, cc_cls = _col(cls, "txId", "tx_id", "txid"), _col(cls, "class", "label")
        cmap = {"1": 1, "2": 2, "3": 3, "unknown": 3, "illicit": 1, "licit": 2}
        cls_s = pd.Series(cls[cc_cls].astype(str).str.strip().str.lower().map(cmap).fillna(3).astype(int).to_numpy(),
                          index=pd.to_numeric(cls[cc_id], errors="coerce"))
        cls_s = cls_s[~cls_s.index.duplicated()]
        klass = cls_s.reindex(txid).fillna(3).astype(int).to_numpy()
        btc_c = next((k for k in feats.columns if "btc" in k.lower() and "total" in k.lower()), None)
        btc = pd.to_numeric(feats[btc_c], errors="coerce").fillna(0).to_numpy() if btc_c else np.zeros(len(feats))
        edges = pd.read_csv(os.path.join(edir, "txs_edgelist.csv"))
        e1, e2 = _col(edges, "txId1", "src", "source"), _col(edges, "txId2", "dst", "target")
        tx_pos = pd.Series(np.arange(len(feats)), index=txid)
        tx_pos = tx_pos[~tx_pos.index.duplicated()]
        s = tx_pos.reindex(pd.to_numeric(edges[e1], errors="coerce")).to_numpy()
        d = tx_pos.reindex(pd.to_numeric(edges[e2], errors="coerce")).to_numpy()
        oke = np.isfinite(s) & np.isfinite(d)
        s, d = s[oke].astype(int), d[oke].astype(int)
        bs, bd = bar[s], bar[d]
        illicit = klass == 1
        # per-transaction: has an illicit neighbour (either direction)
        nb_ill = np.zeros(len(feats), dtype=bool)
        nb_ill[s[illicit[d]]] = True
        nb_ill[d[illicit[s]]] = True
        in_deg = np.bincount(d, minlength=len(feats)).astype(float)

        def _bincount(mask: np.ndarray, weights=None) -> np.ndarray:
            b = bar[mask]
            w = None if weights is None else weights[mask]
            return np.bincount(b[b >= 0], weights=None if w is None else w[b >= 0], minlength=T)[:T].astype(float)

        inbar = bar >= 0
        n_tx = _bincount(inbar)
        n_known = _bincount(inbar & (klass != 3))
        n_ill = _bincount(inbar & illicit)
        n_lic = _bincount(inbar & (klass == 2))
        n_nb_ill = _bincount(inbar & nb_ill)
        sum_in = _bincount(inbar, in_deg)
        btc_sum = _bincount(inbar, btc)
        e_int = (bs >= 0) & (bs == bd)
        e_past = (bs >= 0) & (bd >= 0) & (bs != bd)
        n_e_int = np.bincount(bs[e_int], minlength=T)[:T].astype(float)
        n_e_past = np.bincount(np.maximum(bs, bd)[e_past], minlength=T)[:T].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ill_share = np.where(n_known > 0, n_ill / np.maximum(n_known, 1), 0.0)
            lic_share = np.where(n_known > 0, n_lic / np.maximum(n_known, 1), 0.0)
            nb_share = np.where(n_tx > 0, n_nb_ill / np.maximum(n_tx, 1), 0.0)
            mean_in = np.where(n_tx > 0, sum_in / np.maximum(n_tx, 1), 0.0)
        struct = np.stack([np.log1p(n_tx), np.log1p(n_e_int), np.log1p(n_e_past), mean_in, np.log1p(btc_sum), np.zeros(T)], 1)
        labd = np.stack([ill_share, nb_share, lic_share, np.log1p(n_ill), np.log1p(n_known), np.zeros(T)], 1)
        K = 2
        graph = np.zeros((T, N, K, GRAPH_DIM), dtype=np.float32)
        graph[:, :, 0] = struct[:, None, :]
        graph[:, :, 1] = labd[:, None, :]
        gmask = np.broadcast_to((n_tx > 0)[:, None, None], (T, N, K)).copy()
        avail_graph = np.zeros((T, N, K), dtype=np.int64)
        avail_graph[:, :, 0] = tt
        avail_graph[:, :, 1] = tt + lag

        # ---- labels: retrospective illicit activity, knowable after the lag
        labels = np.broadcast_to(((ill_share > thr) & (n_tx > 0))[:, None], (T, N)).astype(np.int8).copy()
        imp = impact_curves(A["close"], horizons)
        btp = bars_to_peak_from_labels(labels, imp, horizons)
        avail_label = np.broadcast_to(tt + lag, (T, N)).copy()

        price_feat, feat_names = raw_price_features(A["close"], A["volume"], A["high"], A["low"])
        liq = liquidity_fields(A["close"], A["volume"], A["high"], A["low"], adv_window=int(c.get("adv_window", 24 * 7)))
        wallet_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(edir, "wallets_*.csv")) + glob.glob(os.path.join(edir, "*Addr*.csv")))
        meta: Dict[str, Any] = {
            "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": horizons, "raw_features": True, "ret_col": 0,
            "feat_names": feat_names, "assets": assets, "label_lag_bars": lag, "illicit_share_thr": thr, "time_source": time_source,
            "n_tx": int(inbar.sum()), "n_tx_unplaced": int((~inbar).sum()), "n_edges": int(len(s)), "n_bars_with_tx": int((n_tx > 0).sum()),
            "n_label_bars": int(labels[:, 0].sum()), "graph_feat_names": {"slot0_structural": ["log1p_n_tx", "log1p_n_internal_edges", "log1p_n_edges_to_past", "mean_in_degree", "log1p_btc_total", "pad"],
                                                                        "slot1_label_derived": ["illicit_share", "illicit_neighbour_share", "licit_share", "log1p_n_illicit", "log1p_n_class_known", "pad"]},
            "wallet_files_present_unused": wallet_files,
            "source": {"graph": "Elliptic++ https://github.com/git-disl/EllipticPlusPlus (Elmougy & Liu, KDD 2023)", "prices": "exchange hourly OHLCV export"},
            "licence": "Elliptic++: per its GitHub licence (check the repository LICENSE); original Elliptic (Weber et al. 2019) per the Kaggle page. "
                       "BTC bars: per the exchange's data terms.",
            "layout": "ellipticpp/{txs_features.csv, txs_classes.csv, txs_edgelist.csv, tx_times.csv | timestep_calendar.csv}, prices/<ASSET>.csv",
            "pit_notes": {
                "avail_feat": "t: an hourly bar is knowable at its close",
                "avail_label": "t + label_lag_bars for every row (illicit labels are assigned retrospectively; lag is the sensitivity axis)",
                "avail_graph": "slot 0 structural snapshot: t ; slot 1 label-derived snapshot: t + label_lag_bars",
                "tx_time": "tx_times.csv (block times) or timestep_calendar.csv (all txs of a step land on the step-end bar, conservative)",
                "universe": "has a price bar",
            },
            "assumptions": ["price timestamp = bar open (close = open + bar_hours)", "illicit_share_thr = 0.05 default (above the ~2 % base rate)",
                            "Elliptic++ ships no block timestamps: tx_times.csv / timestep_calendar.csv are user-constructed"],
        }
        panel = Panel(times=axis, assets=assets, mid=A["close"], price_feat=price_feat, liq=liq, tradable=has.copy(), labels=labels,
                      bars_to_peak=btp, graph=graph, graph_mask=gmask, meta=meta)
        return finish_panel(panel, avail_label=avail_label, universe=has, avail_graph=avail_graph)

"""Vendor-agnostic daily-bars builder ("daily_bars") for the enforcement-anchored benchmarks.

Input: one long table (CSV, or parquet if pyarrow is installed) exported from any vendor (Tushare / AkShare / CSMAR for
A-shares; CRSP, Sharadar, EODHD, Polygon for US incl. OTC and delisted names). Required columns (renamable via
`columns`): date, ticker, open, high, low, close, volume. Optional: amount (traded value), adj_factor, suspended (0/1),
is_st (0/1). The universe must be delisting-inclusive and, for SEC pump-and-dump cases, OTC-inclusive: a large-cap panel
cannot contain the manipulated issuers (report the per-class match rate from regulatory.sheet).

Panel fields
    mid         adjusted close
    price_feat  [ret_z, vol_z, hl_z, gap_z, log_hs_bps]  (causal rolling statistics, NaN -> 0 inside windows)
    liq         [half-spread proxy, depth = 1, ADV$_20 / capital_per_slot]
    tradable    traded volume > 0, not suspended, and (market 'cn') not closing at the daily price limit
                (main board 10 %, ST 5 %, ChiNext 300/301 and STAR 688 20 %, BSE 8xxxxx/4xxxxx 30 %, after 2020-08-24
                for ChiNext); limit-hit bars are frozen in both directions (conservative)
    labels      rule labels (abnormal_move_labels): PROXY events only. Evaluation uses the oracle anchor raster.
"""
from __future__ import annotations
import os
from typing import Any, Dict
import numpy as np
import pandas as pd

from ..base import BaseDatasetBuilder
from ..schema import Panel
from ..labels import abnormal_move_labels, impact_curves, bars_to_peak_from_labels
from ...utils.registry import DATASETS

CHINEXT_REFORM = pd.Timestamp("2020-08-24")


def _read(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"ticker": str, "code": str})


def cn_limit_pct(codes: pd.Index, dates: pd.DatetimeIndex, is_st: np.ndarray = None) -> np.ndarray:
    T, N = len(dates), len(codes)
    lim = np.full((T, N), 0.10)
    c = pd.Series(codes.astype(str).str.zfill(6))
    star = c.str.startswith("688").to_numpy()
    chinext = c.str.startswith(("300", "301")).to_numpy()
    bse = c.str.startswith(("8", "4")).to_numpy()
    lim[:, star] = 0.20
    post = (dates >= CHINEXT_REFORM)[:, None]
    lim = np.where(post & chinext[None, :], 0.20, lim)
    lim[:, bse] = 0.30
    if is_st is not None:
        lim = np.where(is_st.astype(bool) & ~star[None, :] & ~(post & chinext[None, :]), 0.05, lim)
    return lim


def _roll(x: pd.DataFrame, w: int, fn: str) -> np.ndarray:
    r = x.rolling(w, min_periods=max(2, w // 2))
    return getattr(r, fn)().to_numpy()


@DATASETS.register("daily_bars")
class DailyBarsBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "daily_bars", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        c: Dict[str, Any] = self.cfg
        path = c["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Export daily bars (date,ticker,open,high,low,close,volume[,amount,adj_factor,"
                                    f"suspended,is_st]) from your vendor; see docs/sparse_label_research_plan.md section 5.")
        df = _read(path).rename(columns=c.get("columns", {}) or {})
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        if c.get("floor"):
            df["date"] = df["date"].dt.floor(c["floor"])
        df["ticker"] = df["ticker"].astype(str)
        if c.get("start"):
            df = df[df["date"] >= pd.Timestamp(c["start"])]
        if c.get("end"):
            df = df[df["date"] <= pd.Timestamp(c["end"])]
        market = c.get("market", "us")
        self.bars_per_year = int(c.get("bars_per_year", {"cn": 243, "us": 252, "crypto": 365}.get(market, 252)))
        piv = lambda col: df.pivot_table(index="date", columns="ticker", values=col, aggfunc="last").sort_index()
        close = piv("close")
        min_obs = int(c.get("min_obs", 250))
        keep = close.notna().sum() >= min_obs
        close = close.loc[:, keep]
        dates, tickers = close.index, close.columns
        get = lambda col: piv(col).reindex(index=dates, columns=tickers) if col in df else None
        o, h, l, v = get("open"), get("high"), get("low"), get("volume")
        adj = get("adj_factor")
        a = adj.ffill() if adj is not None else 1.0
        mid = (close * a).to_numpy(dtype=np.float64)
        raw_close = close.to_numpy(dtype=np.float64)
        vol = v.fillna(0.0).to_numpy(dtype=np.float64)
        amount = get("amount")
        dollar = amount.to_numpy(dtype=np.float64) if amount is not None else raw_close * vol
        logp = np.log(mid)
        ret = np.full_like(mid, np.nan)
        ret[1:] = logp[1:] - logp[:-1]
        w = int(c.get("vol_window", 60))
        sigma = _roll(pd.DataFrame(ret), w, "std")
        lv = pd.DataFrame(np.log1p(vol))
        vol_z = (lv.to_numpy() - _roll(lv, w, "mean")) / (_roll(lv, w, "std") + 1e-8)
        hl = np.log(h.to_numpy(dtype=np.float64) / l.to_numpy(dtype=np.float64))
        prev_close = np.vstack([np.full((1, mid.shape[1]), np.nan), raw_close[:-1]])
        gap = np.log(o.to_numpy(dtype=np.float64) / prev_close)
        hs = np.clip(float(c.get("hs_k", 0.05)) * (h.to_numpy() - l.to_numpy()) / raw_close, float(c.get("hs_min", 5e-4)), float(c.get("hs_max", 0.05)))
        sd = sigma + 1e-8
        price_feat = np.stack([ret / sd, vol_z, hl / sd, gap / sd, np.log(hs * 1e4)], -1)
        price_feat = np.nan_to_num(price_feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        adv = _roll(pd.DataFrame(dollar), 20, "mean")
        cap = float(c.get("capital_per_slot", 1e5))
        liq = np.stack([np.nan_to_num(hs, nan=0.01), np.ones_like(hs), np.nan_to_num(adv / cap, nan=1e-3).clip(1e-3, None)], -1).astype(np.float32)
        tradable = np.isfinite(mid) & (vol > 0)
        susp = get("suspended")
        if susp is not None:
            tradable &= ~susp.fillna(0).to_numpy().astype(bool)
        if market == "cn":
            st = get("is_st")
            lim = cn_limit_pct(tickers, dates, st.fillna(0).to_numpy() if st is not None else None)
            simple = raw_close / prev_close - 1.0
            tradable &= ~(np.abs(simple) >= lim - float(c.get("limit_tol", 0.002)))
        mid_filled = mid.copy()
        labels = abnormal_move_labels(np.where(np.isfinite(mid_filled), mid_filled, np.nan), vol, horizon=int(c.get("rule_horizon", 5)),
                                      k_ret=float(c.get("rule_k_ret", 3.0)), k_vol=float(c.get("rule_k_vol", 3.0)), window=w)
        imp = impact_curves(mid, self.horizons)
        btp = bars_to_peak_from_labels(labels, imp, self.horizons)
        panel = Panel(times=dates.values.astype("datetime64[ns]").astype(np.int64), assets=list(tickers), mid=mid, price_feat=price_feat,
                      liq=liq, tradable=tradable, labels=labels, bars_to_peak=btp,
                      meta={"freq": "daily", "bars_per_year": self.bars_per_year, "horizons": list(self.horizons), "volume": vol,
                            "feat_names": ["ret_z", "vol_z", "hl_z", "gap_z", "log_hs_bps"], "market": market, "source_path": path,
                            "label_kind": "rule_proxy"})
        panel.validate()
        return panel

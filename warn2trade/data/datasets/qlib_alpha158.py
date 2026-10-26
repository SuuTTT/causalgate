"""D8 Qlib Alpha158-lite, US large caps. STATUS: PUBLIC (Microsoft Qlib US 1d bundle).

Source        official Qlib US daily bundle qlib_data_us_1d_latest.zip, GitHub mirror used by qlib.run.get_data since the
              Azure blob was closed: https://github.com/SunsetWolf/qlib_dataset/releases/download/v3/qlib_data_us_1d_latest.zip
              (450,094,816 bytes). Calendar 1999-12-31 .. 2020-11-10 (the bundle is FROZEN at 2020-11-10; no newer US
              Qlib bundle is reachable from c225 -- Yahoo is blocked). Prices come from Yahoo Finance via Qlib's collector;
              Qlib code is MIT, the Yahoo-derived data carry Yahoo's terms (research use only, not redistributable).
              Downloaded with scripts/prep_qlib.py.
Universe      (ex-ante, point in time) S&P 500 constituents as of each date from the bundle's instruments/sp500.txt
              (Qlib's us_index collector: Wikipedia change log; spells starting before 1999 are coded 1999-01-01) that
              traded within the last 20 bars. KNOWN BIAS: the bundle only holds tickers that Yahoo still served in 2020,
              so some historical members (bankrupt / acquired firms) have no price file; they are absent from the panel
              (count in meta["n_members_without_data"]) -- a residual survivorship bias of the SOURCE that we report rather
              than hide. Outside membership mid = NaN (except the 30-bar exit tail / rare bridged gaps, _qlib_prep.py).
Calendar      2000-01-03 .. 2020-11-10 (features warm up on the bars before date_start when available).
mid           adjusted close ($close; the bundle normalises each series to 1.0 on its first day -- returns and ratios are
              unaffected; nothing in this builder depends on the raw price level).
Labels        label_type = "large_move": abnormal_move_labels(horizon=5, k_ret=3, k_vol=3, window=60, lead=3) on mid and
              adjusted volume, restricted to in-universe bars; bars_to_peak = bars_to_peak_from_labels (argmax |impact|
              over the horizon grid). meta["events"] = onset bars (the first bar of each run of onsets with lead = 0),
              t_peak = t_onset + argmax_h |impact|, direction = sign of that impact.
price_feat    the 23 Alpha158-lite features of csi300_limit (dollar volume = adj close * adj volume replaces amount) plus
              8 more Alpha158 operators: ROC60 (ret60 / sigma sqrt 60), SKEW20, IMAX20, IMIN20, CNTP20, VSTD20, WVMA20,
              RSV20 -> 31 features, all trailing, rank-normalised to [-1, 1] per day against that day's S&P 500 members.
liq           Corwin-Schultz (2012) half-spread, dollar volume / trailing 60-bar median, volume / 20-bar ADV (as D6).
Tradability   tradable = volume > 0 with finite OHLC and mid finite; no price limits in US equities, so can_buy = can_sell
              = tradable. shortable = True for all (S&P 500 names are general-collateral, easy to borrow);
              borrow_bps_per_bar = 0.25 -> 0.25 * 252 = 63 bps / year. GC fees for large caps are typically ~25-30 bps / yr
              (lender rebate rate minus GC rate); 63 bps is a deliberately conservative upper-GC figure that also covers
              occasional warm names, so short legs are not flattered.
Smoke         cfg keys max_assets (first k tickers in sorted order -- not outcome based), date_start, date_end.
"""
from __future__ import annotations
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from ..base import BaseDatasetBuilder
from ..schema import Panel
from ..labels import abnormal_move_labels, bars_to_peak_from_labels, impact_curves
from ...utils.registry import DATASETS
from ._common import require_files
from . import _qlib_prep as qp

SOURCE_URLS = ["https://github.com/SunsetWolf/qlib_dataset/releases/download/v3/qlib_data_us_1d_latest.zip",
               "https://github.com/microsoft/qlib"]


@DATASETS.register("qlib_alpha158")
class QlibAlpha158Builder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "qlib_alpha158", "daily", 252, (1, 2, 3, 5, 10, 20)

    def _key(self) -> str:
        c = self.cfg
        keys = ["max_assets", "date_start", "date_end", "lead", "label_horizon", "k_ret", "k_vol", "stale_bars", "tail",
                "guard_lookback", "horizons", "borrow_bps_per_bar", "version"]
        sub = {k: c.get(k) for k in keys}
        default = all(c.get(k) is None for k in ("max_assets", "date_start", "date_end"))
        h = hashlib.md5(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:10]
        return ("panel" if default else "panel_sub") + "_" + h

    def build_panel(self) -> Panel:
        c = self.cfg
        bin_root = os.path.join(self.root, "qlib_us", "bin")
        require_files(self.root, ["qlib_us/bin/calendars/day.txt", "qlib_us/bin/instruments/sp500.txt"],
                      "ssh c225; cd /date/zyy/warn2trade; /date/zyy/venvs/w2t/bin/python scripts/prep_qlib.py --us")
        cache = os.path.join(self.root, self.name, "cache", self._key() + ".npz")
        if os.path.exists(cache) and not c.get("rebuild", False):
            arrays, meta = qp.load_cache(cache)
            p = Panel(assets=list(meta.pop("assets")), meta=meta, **arrays)
            p.validate()
            return p
        t0 = time.time()
        horizons = tuple(c.get("horizons", self.horizons))
        lead = int(c.get("lead", 3))
        cal = qp.read_calendar(bin_root)
        d0 = pd.Timestamp(c.get("date_start") or "2000-01-01")
        d1 = pd.Timestamp(c.get("date_end") or cal[-1])
        i_start = int(np.searchsorted(cal, d0)); i1 = int(np.searchsorted(cal, d1, side="right"))
        i0 = max(0, i_start - int(c.get("warmup_bars", 130)))
        dates_all = cal[i0:i1]
        spells = qp.parse_membership(os.path.join(bin_root, "instruments", "sp500.txt"))
        mem_all = qp.membership_mask(spells, sorted(spells), dates_all)
        codes = [cd for j, cd in enumerate(sorted(spells)) if mem_all[i_start - i0:, j].any()]
        missing = [cd for cd in codes if not qp.has_features(bin_root, cd)]
        codes = [cd for cd in codes if cd not in missing]
        if c.get("max_assets"):
            codes = codes[: int(c["max_assets"])]
        N = len(codes)
        f = qp.load_fields(bin_root, codes, ["open", "high", "low", "close", "volume"], i0, i1, len(cal))
        o, h, l, cl, v = (f[k] for k in ["open", "high", "low", "close", "volume"])
        member = qp.membership_mask(spells, codes, dates_all)
        traded = np.isfinite(cl) & np.isfinite(v) & (v > 0) & np.isfinite(h) & np.isfinite(l) & np.isfinite(o) & (cl > 0)
        cl_ff = qp.ffill(np.where(traded, cl, np.nan))
        dv = cl * v
        feats, names = qp.compute_features(o, h, l, cl, v, dv, traded, extended=True)
        s = i_start - i0
        sl = slice(s, None)
        member, traded = member[sl], traded[sl]
        dates = dates_all[s:]
        T = len(dates)
        last_tr = qp.ffill(np.where(traded, np.arange(T)[:, None].astype(float), np.nan))
        stale = np.arange(T)[:, None] - np.nan_to_num(last_tr, nan=-1e9)
        active = member & (stale <= int(c.get("stale_bars", 20)))
        pf = qp.rank_normalise([x[sl] for x in feats], member)
        liq = qp.build_liq(h, l, cl, v, dv, traded)[sl]
        mid, tail_m, A, n_bridged = qp.assemble_universe(active, cl_ff[sl], horizons, tail=int(c.get("tail", 30)),
                                                         lookback=int(c.get("guard_lookback", 64)))
        pf[tail_m] = np.nan
        tradable = traded & np.isfinite(mid)
        vol = np.where(traded, v[sl], np.nan)

        # ---- labels (forward-looking by construction; training target only)
        kw = dict(horizon=int(c.get("label_horizon", 5)), k_ret=float(c.get("k_ret", 3.0)), k_vol=float(c.get("k_vol", 3.0)),
                  window=60)
        onset = abnormal_move_labels(mid, vol, lead=0, **kw).astype(bool) & A
        labels = onset.copy()
        for d in range(1, lead + 1):
            labels[:-d] |= onset[d:]
        labels &= np.isfinite(mid)
        labels = labels.astype(np.int8)
        impact = impact_curves(mid, horizons)
        btp = bars_to_peak_from_labels(labels, impact, horizons)
        hz = np.asarray(horizons)
        starts = onset & ~np.vstack([np.zeros((1, N), bool), onset[:-1]])
        events = []
        for t, i in zip(*np.where(starts)):
            imp = impact[t, i]
            if np.isnan(imp).all():
                continue
            k = int(np.nanargmax(np.abs(imp)))
            events.append({"asset_index": int(i), "t_onset": int(t), "t_peak": int(t + hz[k]),
                           "direction": int(np.sign(imp[k])), "source_id": f"{codes[i]}:{dates[t].date()}"})
        events.sort(key=lambda e: (e["asset_index"], e["t_onset"]))

        borrow = float(c.get("borrow_bps_per_bar", 0.25))
        meta = {
            "name": self.name, "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(horizons),
            "label_type": "large_move", "label_params": dict(kw, lead=lead),
            "universe_rule": "S&P 500 constituents as of each date (Qlib instruments/sp500.txt, point in time) with a "
                             "price file in the bundle and a trade within the last 20 bars; 30-bar exit tail; short "
                             "re-entry gaps bridged",
            "source_urls": SOURCE_URLS, "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "feat_names": names, "lead": lead, "events": events, "n_events": len(events),
            "can_buy": tradable.copy(), "can_sell": tradable.copy(), "shortable": np.ones(N, dtype=bool),
            "borrow_bps_per_bar": borrow, "borrow_bps_per_year": borrow * self.bars_per_year,
            "universe": A, "index_member": member, "volume": np.nan_to_num(vol).astype(np.float32),
            "n_bridged_bars": n_bridged, "n_members_without_data": len(missing), "missing_codes": missing,
            "spread_estimator": "corwin_schultz_2012", "assets": codes,
        }
        times = dates.values.astype("datetime64[ns]").astype(np.int64)
        p = Panel(times=times, assets=codes, mid=mid, price_feat=pf.astype(np.float32), liq=liq, tradable=tradable,
                  labels=labels, bars_to_peak=btp, meta=meta)
        p.validate()
        meta["build_sec"] = time.time() - t0
        arrays = dict(times=p.times, mid=p.mid, price_feat=p.price_feat, liq=p.liq, tradable=p.tradable,
                      labels=p.labels, bars_to_peak=p.bars_to_peak)
        qp.save_cache(cache, arrays, meta)
        meta.pop("assets")
        return p

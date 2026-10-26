"""D6 CSI 300 / A-share limit-up / limit-down events. STATUS: CONSTRUCTED from a public Qlib CN bundle.

Source        community Qlib CN daily bundle, https://github.com/chenditc/investment_data/releases (release 2026-10-05,
              qlib_bin.tar.gz, sha256 6bc16968...db999, trade data up to 2026-09-30). Fields used: $open $high $low $close
              (adjusted = raw * $factor), $volume (adjusted), $amount (thousand CNY), $factor; membership history
              instruments/csi300.txt. Licence: the repository is Apache-2.0; the underlying data are aggregated from
              free public vendors (see its README) -- research use only, redistribution terms of the upstream vendors
              are NOT verified. Downloaded with scripts/prep_qlib.py.
Universe      (ex-ante, point in time) CSI 300 constituents AS OF EACH DATE from instruments/csi300.txt; additionally an
              asset counts as in-universe only while it has traded within the last `stale_bars` (20) bars. N = every
              stock that was ever a member in [date_start, date_end]; outside membership mid = NaN (except the
              30-bar exit tail and the rare bridged re-entry gaps described in _qlib_prep.py).
Calendar      bundle trading calendar, date_start 2010-01-04 .. 2026-09-30 (features warm up on the 130 bars before).
mid           adjusted close ($close), forward-filled over suspensions (suspended bars: tradable = False).
Price limits  limits apply to RAW prices: raw = $close / $factor. Reference price = previous adjusted close / today's
              factor (= the exchange's ex-rights previous close). Limit price = round_half_up(ref * (1 +- lim), 0.01 CNY);
              a limit hit = raw close (rounded to 0.01) at / beyond that price. lim = 10% main board; 20% ChiNext
              (SZ300xxx / SZ301xxx) from 2020-08-24 (10% before); 20% STAR (SH688xxx / SH689xxx). The first 5 trading
              days after listing are skipped (no / different limits for IPOs). ST stocks (+-5%) are NOT flagged by the
              bundle, so ST is APPROXIMATED as "none" (CSI 300 deletes ST stocks at the next review); meta reports how
              many member closes sit exactly at a +-5% price as an upper bound on the ST error.
Labels        label_type = "large_move" (the event is price-defined: a close at the limit), but taken from the
              limit-hit EVENT TABLE, not abnormal_move_labels: a run = consecutive limit-hit bars in the same direction;
              t_peak = first limit-hit bar of the run, t_onset = t_peak, direction +1 (limit-up) / -1 (limit-down).
              Only runs whose t_peak is an in-universe bar count. labels = 1 on [t_peak - lead, t_peak] (lead = 3);
              bars_to_peak = t_peak - t exactly on [t_peak - lead, t_peak + late_window] (late_window = 5; signed,
              negative after the peak; where windows overlap the nearest upcoming peak wins). meta["events"] holds the
              table {asset_index, t_onset, t_peak, direction, run_len, source_id}.
price_feat    Alpha158-lite (23) + 3 limit features, see _qlib_prep.compute_features: ret{1,5,10,20}/(sigma sqrt h),
              volume z (20, 60), amount z (20) [the bundle has no shares outstanding, so no true turnover], log(H/L)/sigma,
              close position in range, open-close, gap, upper/lower shadow, distance to 20/60-bar max/min, close / MA
              5/20/60, sigma20/sigma60, corr(ret, dlog vol, 20), SUMP20; all on bars <= t, then rank-normalised to
              [-1, 1] per day against that day's CSI 300 members only (same-day data). Plus limit_up_t, limit_down_t
              (close of t at the limit; known at the close of t) and net_limit_hits_20 = (#up - #down)/20 over (t-19..t),
              left unranked.
liq           [Corwin-Schultz (2012) half-spread from daily H/L (2-day, overnight-adjusted, trailing 20-bar mean, /2,
              clipped [1e-5, 0.05]); amount / trailing 60-bar median; volume / trailing 20-bar ADV]; NaN -> ffill -> asset
              median.
Tradability   tradable = False when volume == 0 / missing (suspension) or mid is NaN. meta["can_buy"] False on limit-up
              closes, meta["can_sell"] False on limit-down closes (both also False when not tradable). Not shortable
              (no securities lending assumed): shortable all False, borrow_bps_per_bar = 0.
Smoke         cfg keys max_assets (first k codes in sorted order -- not outcome based), date_start, date_end.
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
from ...utils.registry import DATASETS
from ._common import require_files
from . import _qlib_prep as qp

SOURCE_URLS = ["https://github.com/chenditc/investment_data/releases/download/2026-10-05/qlib_bin.tar.gz",
               "https://github.com/chenditc/investment_data"]


def _limit_rate(code: str, dates: pd.DatetimeIndex) -> np.ndarray:
    c = code.upper()
    if c.startswith("SH688") or c.startswith("SH689"):
        return np.full(len(dates), 0.20)
    if c.startswith("SZ300") or c.startswith("SZ301"):
        return np.where(dates >= pd.Timestamp("2020-08-24"), 0.20, 0.10)
    if c.startswith("BJ"):
        return np.full(len(dates), 0.30)
    return np.full(len(dates), 0.10)


def _round2(x):
    return np.floor(x * 100.0 + 0.5 + 1e-7) / 100.0


@DATASETS.register("csi300_limit")
class CSI300LimitBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "csi300_limit", "daily", 243, (1, 2, 3, 5, 10, 20)

    def _key(self) -> str:
        c = self.cfg
        keys = ["max_assets", "date_start", "date_end", "lead", "late_window", "stale_bars", "tail", "guard_lookback",
                "horizons", "version"]
        sub = {k: c.get(k) for k in keys}
        default = all(c.get(k) is None for k in ("max_assets", "date_start", "date_end"))
        h = hashlib.md5(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:10]
        return ("panel" if default else "panel_sub") + "_" + h

    def build_panel(self) -> Panel:
        c = self.cfg
        bin_root = os.path.join(self.root, "qlib_cn", "qlib_bin")
        require_files(self.root, ["qlib_cn/qlib_bin/calendars/day.txt", "qlib_cn/qlib_bin/instruments/csi300.txt"],
                      "ssh c225; cd /date/zyy/warn2trade; /date/zyy/venvs/w2t/bin/python scripts/prep_qlib.py --cn")
        cache = os.path.join(self.root, self.name, "cache", self._key() + ".npz")
        if os.path.exists(cache) and not c.get("rebuild", False):
            arrays, meta = qp.load_cache(cache)
            p = Panel(assets=list(meta.pop("assets")), meta=meta, **arrays)
            p.validate()
            return p
        t0 = time.time()
        horizons = tuple(c.get("horizons", self.horizons))
        lead, late = int(c.get("lead", 3)), int(c.get("late_window", 5))
        cal = qp.read_calendar(bin_root)
        d0 = pd.Timestamp(c.get("date_start") or "2010-01-01")
        d1 = pd.Timestamp(c.get("date_end") or cal[-1])
        i_start = int(np.searchsorted(cal, d0)); i1 = int(np.searchsorted(cal, d1, side="right"))
        warm = int(c.get("warmup_bars", 130))
        i0 = max(0, i_start - warm)
        dates_all = cal[i0:i1]
        spells = qp.parse_membership(os.path.join(bin_root, "instruments", "csi300.txt"))
        mem_all = qp.membership_mask(spells, sorted(spells), dates_all)
        codes = [cd for j, cd in enumerate(sorted(spells)) if mem_all[i_start - i0:, j].any()]
        missing = [cd for cd in codes if not qp.has_features(bin_root, cd)]
        codes = [cd for cd in codes if cd not in missing]
        if c.get("max_assets"):
            codes = codes[: int(c["max_assets"])]
        N = len(codes)
        f = qp.load_fields(bin_root, codes, ["open", "high", "low", "close", "volume", "amount", "factor"], i0, i1, len(cal))
        o, h, l, cl, v, amt, fac = (f[k] for k in ["open", "high", "low", "close", "volume", "amount", "factor"])
        member = qp.membership_mask(spells, codes, dates_all)
        traded = np.isfinite(cl) & np.isfinite(v) & (v > 0) & np.isfinite(h) & np.isfinite(l) & np.isfinite(o)

        # ---- raw-price limit hits
        cl_ff = qp.ffill(np.where(traded, cl, np.nan))
        fac_ff = qp.ffill(np.where(np.isfinite(fac) & (fac > 0), fac, np.nan))
        prev_adj = np.full_like(cl_ff, np.nan); prev_adj[1:] = cl_ff[:-1]
        ref = prev_adj / fac_ff                               # ex-rights previous close in today's raw units
        raw_close = _round2(cl / fac_ff)
        lim = np.stack([_limit_rate(cd, dates_all) for cd in codes], 1)
        up_px, dn_px = _round2(ref * (1.0 + lim)), _round2(ref * (1.0 - lim))
        first = np.argmax(np.isfinite(cl), 0)
        age = np.arange(len(dates_all))[:, None] - first[None, :]
        lim_ok = traded & np.isfinite(ref) & (age >= 5)
        hit_up = lim_ok & (raw_close >= up_px - 1e-3)
        hit_dn = lim_ok & (raw_close <= dn_px + 1e-3)
        st5 = lim_ok & ~hit_up & ~hit_dn & ((np.abs(raw_close - _round2(ref * 1.05)) < 1e-3) | (np.abs(raw_close - _round2(ref * 0.95)) < 1e-3))

        # ---- features (on the warm-up-extended range), then trim
        dv = amt * 1000.0
        feats, names = qp.compute_features(o, h, l, cl, v, dv, traded, extended=False)
        s = i_start - i0
        sl = slice(s, None)
        trim = lambda x: x[sl]
        member, traded, hit_up, hit_dn, st5 = map(trim, (member, traded, hit_up, hit_dn, st5))
        dates = dates_all[s:]
        T = len(dates)
        stale = np.zeros((T, N), dtype=np.int64)
        tr_full = qp.ffill(np.where(traded, np.arange(T)[:, None].astype(float), np.nan))
        stale = np.arange(T)[:, None] - np.nan_to_num(tr_full, nan=-1e9)
        active = member & (stale <= int(c.get("stale_bars", 20)))
        pf = qp.rank_normalise([x[sl] for x in feats], member)
        nl = np.stack([hit_up.astype(np.float32), hit_dn.astype(np.float32),
                       ((qp._roll_count(hit_up, 20) - qp._roll_count(hit_dn, 20)) / 20.0).astype(np.float32)], -1)
        pf = np.concatenate([pf, nl], -1)
        names = names + ["limit_up_t", "limit_down_t", "net_limit_hits_20"]
        liq = qp.build_liq(h, l, cl, v, dv, traded)[sl]
        close_ff = trim(cl_ff)
        mid, tail_m, A, n_bridged = qp.assemble_universe(active, close_ff, horizons, tail=int(c.get("tail", 30)),
                                                         lookback=int(c.get("guard_lookback", 64)))
        pf[tail_m] = np.nan
        tradable = traded & np.isfinite(mid)
        can_buy = tradable & ~hit_up
        can_sell = tradable & ~hit_dn

        # ---- event table -> labels / bars_to_peak
        labels = np.zeros((T, N), dtype=np.int8)
        btp = np.full((T, N), np.nan)
        events = []
        for i in range(N):
            for dirn, hit in ((1, hit_up[:, i]), (-1, hit_dn[:, i])):
                starts = np.where(hit & ~np.concatenate([[False], hit[:-1]]))[0]
                for tp in starts:
                    if not A[tp, i]:
                        continue
                    run = 1
                    while tp + run < T and hit[tp + run]:
                        run += 1
                    events.append({"asset_index": int(i), "t_onset": int(tp), "t_peak": int(tp), "direction": int(dirn),
                                   "run_len": int(run), "source_id": f"{codes[i]}:{dates[tp].date()}:{'up' if dirn > 0 else 'down'}"})
        events.sort(key=lambda e: (e["asset_index"], e["t_peak"]))
        for e in events:                                      # pass 1: post-peak (late) segments
            i, tp = e["asset_index"], e["t_peak"]
            seg = np.arange(tp + 1, min(T, tp + late + 1))
            btp[seg, i] = tp - seg
        for e in sorted(events, key=lambda e: -e["t_peak"]):  # pass 2: pre-peak, earlier peaks win
            i, tp = e["asset_index"], e["t_peak"]
            seg = np.arange(max(0, tp - lead), tp + 1)
            btp[seg, i] = tp - seg
            labels[seg, i] = 1
        labels[~np.isfinite(mid)] = 0

        meta = {
            "name": self.name, "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(horizons),
            "label_type": "large_move", "label_source": "limit-hit event table (first limit close of a run)",
            "universe_rule": "CSI 300 constituents as of each date (instruments/csi300.txt, point in time) that traded "
                             "within the last 20 bars; 30-bar exit tail; short re-entry gaps bridged",
            "source_urls": SOURCE_URLS, "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "feat_names": names, "lead": lead, "late_window": late, "events": events, "n_events": len(events),
            "n_events_up": int(sum(e["direction"] > 0 for e in events)),
            "n_events_down": int(sum(e["direction"] < 0 for e in events)),
            "can_buy": can_buy, "can_sell": can_sell, "shortable": np.zeros(N, dtype=bool), "borrow_bps_per_bar": 0.0,
            "universe": A, "index_member": member, "volume": np.where(traded, v[sl], 0.0).astype(np.float32),
            "st_flagged": False, "n_possible_st_limit_closes": int((st5 & A).sum()),
            "n_bridged_bars": n_bridged, "n_missing_feature_dirs": len(missing), "missing_codes": missing,
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

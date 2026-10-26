"""A2 planted-leak testbed: a synthetic regime-switching panel with injected events and a minimal anomaly-to-trade
reference pipeline whose leakage channels can be switched on one at a time (ground truth for E4 / FPIT power).

Panel (registered as dataset "cg_planted"; raw, *un-normalised* features so that L1 is switchable downstream):
    returns      r_t,i = sigma_i * v_{z_t} * eps, two-state Markov variance regime z_t in {low, high} (Prop. 3 setting)
    events       onset t0, direction d, magnitude m, build-up over h* bars then partial reversion; pre-onset footprint of
                 `lead` bars in the volume feature so that a causal detector has something to learn
    labels       1 on [t0 - lead, t0 + h*]; bars_to_peak = (t0 + h*) - t; avail_label = event end bar (t0 + 2h*) for
                 event rows and t + max(horizon) elsewhere (a "no event" label is only knowable after the horizon)
    documents    (i) background chatter, published at its bar; (ii) one "insider note" per event whose *content date*
                 is t0 - lead but whose *publication* (availability) is t0 + news_lag: joining it on the content date is
                 the L5 label-time join
    price_feat   [log return, log volume, high-low range, log spread]  (raw; meta['feat_names'])

Leak switches of the reference pipeline (each maps to one channel of the taxonomy):
    global_norm              L1  z-score with full-sample mean / std            (causal: expanding Welford, warm-up abstains)
    test_quantile_threshold  L2  alarm threshold = (1 - alpha) quantile of the scores over the whole test block
                                                                                 (causal: rolling quantile of past scores)
    same_bar_fill            L7  decision at t filled at the close of t          (causal: fill at the close of t + latency)
    label_time_join          L5  documents joined on their content date          (causal: joined on their availability bar)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from warn2trade.data.base import BaseDatasetBuilder
from warn2trade.data.schema import Panel
from warn2trade.utils.registry import DATASETS
from .pit import attach_availability, default_availability, AsOfStamped, FiltrationGuard


def _register(name: str):
    def deco(cls):
        try:
            DATASETS.register(name)(cls)
        except KeyError:
            pass
        return cls
    return deco


@_register("cg_planted")
class PlantedLeakBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "cg_planted", "daily", 252, (1, 2, 3, 5, 10)

    def build_panel(self) -> Panel:
        c = self.cfg
        rng = np.random.default_rng(int(c.get("seed", 0)))
        T, N = int(c.get("T", 2500)), int(c.get("N", 8))
        M, Dt = int(c.get("text_slots", 2)), int(c.get("text_dim", 8))
        rate, lead = float(c.get("event_rate", 0.006)), int(c.get("lead", 3))
        m_lo, m_hi, rho = float(c.get("m_lo", 4.0)), float(c.get("m_hi", 7.0)), float(c.get("reversion", 0.5))
        news_lag, text_snr = int(c.get("news_lag", 2)), float(c.get("text_snr", 2.0))
        vol_mult = tuple(c.get("regime_vol_mult", (1.0, 3.0)))
        p_stay = tuple(c.get("regime_stay", (0.985, 0.95)))
        hmax = max(self.horizons)

        # ---- two-state Markov variance regime (shared across assets: a market-wide volatility regime)
        regime = np.zeros(T, dtype=np.int8)
        for t in range(1, T):
            z = regime[t - 1]
            regime[t] = z if rng.random() < p_stay[z] else 1 - z
        sigma = rng.uniform(0.008, 0.02, size=N)
        mult = np.asarray(vol_mult)[regime]                                      # (T,)
        ret = rng.standard_normal((T, N)) * sigma[None, :] * mult[:, None]
        log_vol = 10.0 + 0.6 * rng.standard_normal((T, N)) + 0.3 * (mult[:, None] - 1.0)
        spread = np.tile(rng.uniform(5e-4, 2e-3, size=N), (T, 1)) * (1.0 + 0.5 * (mult[:, None] - 1.0))
        tradable = np.ones((T, N), dtype=bool)
        labels = np.zeros((T, N), dtype=np.int8)
        btp = np.full((T, N), np.nan)
        avail_label = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None] + hmax, (T, N)).copy()
        text = np.zeros((T, N, M, Dt), dtype=np.float32)
        text_mask = np.zeros((T, N, M), dtype=bool)
        text_age = np.zeros((T, N, M), dtype=np.float32)
        avail_text = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None, None], (T, N, M)).copy()
        signal_vec = rng.standard_normal(Dt)
        signal_vec /= np.linalg.norm(signal_vec)

        # background chatter: published at its bar, orthogonal-ish noise
        chatter = rng.random((T, N, M)) < 0.10
        text[chatter] = rng.standard_normal((int(chatter.sum()), Dt)) / np.sqrt(Dt)
        text_mask |= chatter

        events: List[Dict[str, Any]] = []
        for i in range(N):
            t = 60
            while t < T - 4 * hmax - 10:
                if rng.random() < rate:
                    h_star = int(rng.choice([2, 3, 5]))
                    d = float(rng.choice([-1.0, 1.0]))
                    m = float(rng.uniform(m_lo, m_hi) * sigma[i] * np.sqrt(h_star))
                    events.append({"onset_t": t, "asset": i, "direction": d, "magnitude": m, "h_star": h_star,
                                   "peak_t": t + h_star, "end_t": t + 2 * h_star})
                    t += 3 * h_star + lead + 6
                else:
                    t += 1
        for e in events:
            t0, i, d, m, h = e["onset_t"], e["asset"], e["direction"], e["magnitude"], e["h_star"]
            peak, end = t0 + h, t0 + 2 * h
            ret[t0 + 1 : peak + 1, i] += d * m / h
            ret[peak + 1 : end + 1, i] -= d * rho * m / h
            lo = max(0, t0 - lead)
            labels[lo : peak + 1, i] = 1
            btp[lo : end + 1, i] = peak - np.arange(lo, end + 1)
            avail_label[lo : end + 1, i] = end                       # the event is only knowable once it has played out
            log_vol[lo : peak + 1, i] += np.linspace(0.3, 1.5, peak + 1 - lo)
            spread[peak - 1 : peak + 2, i] *= 2.0
            # insider note: content date t0 - lead, published t0 + news_lag
            row, slot = lo, 0
            text[row, i, slot] = d * text_snr * signal_vec + rng.standard_normal(Dt) / np.sqrt(Dt)
            text_mask[row, i, slot] = True
            avail_text[row, i, slot] = min(T - 1, t0 + news_lag)
            e["doc_row"], e["doc_avail"] = int(row), int(min(T - 1, t0 + news_lag))

        mid = 100.0 * np.exp(np.cumsum(ret, axis=0))
        hl_range = np.abs(rng.standard_normal((T, N))) * sigma[None, :] * mult[:, None] + np.abs(ret)
        price_feat = np.stack([ret, log_vol, hl_range, np.log(spread * 1e3)], -1).astype(np.float32)
        liq = np.stack([spread, np.ones((T, N)), np.full((T, N), 0.02)], -1).astype(np.float32)
        panel = Panel(times=np.arange(T, dtype=np.int64), assets=[f"PL{i:02d}" for i in range(N)], mid=mid,
                      price_feat=price_feat, liq=liq, tradable=tradable, labels=labels, bars_to_peak=btp,
                      text=text, text_mask=text_mask, text_age=text_age,
                      meta={"n_events": len(events), "freq": self.freq, "bars_per_year": self.bars_per_year,
                            "horizons": list(self.horizons), "feat_names": ["log_ret", "log_vol", "hl_range", "log_spread"],
                            "raw_features": True, "ret_col": 0, "regime": regime, "sigma": sigma, "signal_vec": signal_vec,
                            "events": events, "lead": lead, "news_lag": news_lag})
        panel.validate()
        attach_availability(panel, avail_label=avail_label, avail_text=avail_text, universe=np.ones((T, N), dtype=bool))
        default_availability(panel)
        return panel


# ======================================================================================= reference pipeline
@dataclass
class LeakSwitches:
    global_norm: bool = False               # L1
    test_quantile_threshold: bool = False   # L2
    same_bar_fill: bool = False             # L7
    label_time_join: bool = False           # L5

    def any(self) -> bool:
        return any(asdict(self).values())

    def names(self) -> List[str]:
        return [k for k, v in asdict(self).items() if v]


def expanding_zscore(x: np.ndarray, warmup: int) -> np.ndarray:
    """Causal expanding mean / std along axis 0 (NaN-aware); the first `warmup` bars abstain (NaN)."""
    x = x.astype(np.float64)
    ok = ~np.isnan(x)
    xf = np.where(ok, x, 0.0)
    n = np.cumsum(ok, axis=0)
    s1 = np.cumsum(xf, axis=0)
    s2 = np.cumsum(xf * xf, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / np.maximum(n, 1)
        var = s2 / np.maximum(n, 1) - mean ** 2
        z = (x - mean) / np.sqrt(np.clip(var, 1e-12, None))
    z[n < warmup] = np.nan
    return z


def rolling_quantile_past(score: np.ndarray, q: float, window: int, warmup: int, every: int = 5) -> np.ndarray:
    """thr[t] = q-quantile of the pooled (all assets) scores over bars [t - window, t - 1], recalibrated every `every`
    bars and held constant in between (still F_t-measurable); NaN before warm-up."""
    T = score.shape[0]
    thr = np.full(T, np.nan)
    last = np.nan
    for t in range(T):
        if t % every == 0:
            lo = max(0, t - window)
            seg = score[lo:t]
            seg = seg[np.isfinite(seg)]
            last = np.quantile(seg, q) if seg.size >= warmup else np.nan
        thr[t] = last
    return thr


class ReferencePipeline:
    """Minimal anomaly-to-trade pipeline: normaliser -> scorer -> threshold -> policy -> fill.

    Batch interface for the audit: `run(panel)` returns (T, N) arrays for every operator in `operators` plus "score",
    "thr", "alarm", "weight" and "pnl" (gross log-return PnL of the decision taken at t). The same code path is what a
    bar-by-bar replay would execute, so the FPIT verdict on it is exact.
    """
    operators = ("normaliser", "text_join", "scorer", "threshold", "policy")

    def __init__(self, leaks: Optional[LeakSwitches] = None, latency: int = 1, hold: int = 5, alpha: float = 0.05,
                 warmup: int = 60, window: int = 250, test_start: Optional[int] = None, doc_window: int = 5):
        self.leaks = leaks or LeakSwitches()
        self.latency, self.hold, self.alpha = int(latency), int(hold), float(alpha)
        self.warmup, self.window, self.test_start, self.doc_window = int(warmup), int(window), test_start, int(doc_window)
        self.guard = FiltrationGuard(strict=False)

    # ---- operators
    def normalise(self, panel: Panel) -> np.ndarray:
        x = panel.price_feat.astype(np.float64)
        if self.leaks.global_norm:
            mean = np.nanmean(x, axis=0, keepdims=True)
            std = np.nanstd(x, axis=0, keepdims=True)
            return (x - mean) / np.clip(std, 1e-12, None)
        return expanding_zscore(x, self.warmup)

    def text_signal(self, panel: Panel) -> np.ndarray:
        """(T, N) projection of the documents on the planted signal direction, aggregated over the last `doc_window`
        bars. Causal: a document contributes from its availability bar; leaky: from its content (row) date."""
        T, N = panel.mid.shape
        sv = np.asarray(panel.meta.get("signal_vec"))
        proj = np.einsum("tnmd,d->tnm", panel.text, sv) * panel.text_mask                        # (T, N, M)
        rows = np.broadcast_to(np.arange(T)[:, None, None], proj.shape)
        eff = rows if self.leaks.label_time_join else panel.avail_text                            # when it becomes usable
        tt, ii, mm = np.nonzero(panel.text_mask)
        e, v = eff[tt, ii, mm], proj[tt, ii, mm]
        acc = np.zeros((T + self.doc_window + 1, N))
        for k in range(self.doc_window):                       # a document contributes for doc_window bars from `eff`
            np.add.at(acc, (np.minimum(e + k, T + self.doc_window), ii), v)
        return acc[:T]

    def score(self, z: np.ndarray, text_sig: np.ndarray) -> np.ndarray:
        s = np.abs(z[..., 0]) + np.clip(z[..., 1], 0, None) + np.abs(text_sig)
        return s

    def threshold(self, score: np.ndarray) -> np.ndarray:
        T = score.shape[0]
        if self.leaks.test_quantile_threshold:
            t0 = self.test_start if self.test_start is not None else 0
            seg = score[t0:]
            thr_val = np.nanquantile(seg[np.isfinite(seg)], 1 - self.alpha)
            thr = np.full(T, np.nan)
            thr[t0:] = thr_val
            return thr
        return rolling_quantile_past(score, 1 - self.alpha, self.window, self.warmup)

    def policy(self, alarm: np.ndarray, panel: Panel) -> np.ndarray:
        lr = panel.price_feat[..., int(panel.meta.get("ret_col", 0))].astype(np.float64)
        mom = np.zeros_like(lr)
        for k in range(3):
            mom[k:] += lr[: lr.shape[0] - k] if k else lr
        return np.where(alarm, np.sign(mom), 0.0)

    def pnl(self, weight: np.ndarray, panel: Panel) -> np.ndarray:
        lp = np.log(panel.mid)
        T = lp.shape[0]
        lam = 0 if self.leaks.same_bar_fill else self.latency
        out = np.full_like(weight, np.nan)
        hi = T - lam - self.hold
        out[:hi] = weight[:hi] * (lp[lam + self.hold : lam + self.hold + hi] - lp[lam : lam + hi])
        return out

    # ---- batch run
    def run(self, panel: Panel) -> Dict[str, np.ndarray]:
        default_availability(panel)
        z = self.normalise(panel)
        ts = self.text_signal(panel)
        s = self.score(z, ts)
        thr = self.threshold(s)
        alarm = np.isfinite(s) & np.isfinite(thr)[:, None] & (s > thr[:, None])
        w = self.policy(alarm, panel)
        return {"normaliser": z[..., 0], "text_join": ts, "scorer": s, "threshold": np.broadcast_to(thr[:, None], s.shape).copy(),
                "policy": w, "score": s, "thr": np.broadcast_to(thr[:, None], s.shape).copy(), "alarm": alarm.astype(np.float64),
                "weight": w, "pnl": self.pnl(w, panel)}

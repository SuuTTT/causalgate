"""Synthetic multimodal panel with planted anomalies. For smoke tests, unit tests and controlled studies
(lead-time sweeps, slippage traps). NOT a benchmark result.

Event mechanics (per planted event at onset t0 on asset i):
    * direction d in {-1,+1}; magnitude m ~ U(m_lo, m_hi) * sigma * sqrt(h*); build-up over h* bars; partial reversion
      (fraction rho) over the next h* bars  -> impact peaks at t0 + h*.
    * pre-onset footprint of `lead` bars: volume z rises, text embeddings carry a direction-aligned signal vector,
      neighbours' features drift weakly (contagion) -> lead time is learnable.
    * slippage trap: relative spread widens by `spread_mult` around the peak; with prob p_halt the peak bar +-1 is
      untradable (limit-hit / halt emulation).
Labels are 1 on [t0 - lead, t0 + h*]; bars_to_peak = (t0 + h*) - t (negative after the peak).
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .schema import Panel
from .base import BaseDatasetBuilder
from ..utils.registry import DATASETS


@DATASETS.register("synthetic")
class SyntheticBuilder(BaseDatasetBuilder):
    name = "synthetic"
    freq = "daily"
    bars_per_year = 252
    horizons = (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        c = self.cfg
        rng = np.random.default_rng(int(c.get("seed", 0)))
        T, N = int(c.get("T", 3000)), int(c.get("N", 16))
        M, Dt, K, Dg = int(c.get("text_slots", 4)), int(c.get("text_dim", 32)), min(int(c.get("graph_slots", 4)), N - 1), int(c.get("graph_dim", 8))
        rate, lead = float(c.get("event_rate", 0.004)), int(c.get("lead", 3))
        m_lo, m_hi, rho = float(c.get("m_lo", 3.0)), float(c.get("m_hi", 6.0)), float(c.get("reversion", 0.5))
        spread_mult, p_halt = float(c.get("spread_mult", 4.0)), float(c.get("p_halt", 0.3))
        text_snr = float(c.get("text_snr", 1.5))

        sigma = rng.uniform(0.008, 0.02, size=N)
        ret = rng.standard_normal((T, N)) * sigma
        vol_z = rng.standard_normal((T, N)) * 0.7
        spread = np.tile(rng.uniform(5e-4, 2e-3, size=N), (T, 1))
        tradable = np.ones((T, N), dtype=bool)
        labels = np.zeros((T, N), dtype=np.int8)
        btp = np.full((T, N), np.nan)
        text = np.zeros((T, N, M, Dt), dtype=np.float32)
        text_mask = np.zeros((T, N, M), dtype=bool)
        text_age = np.zeros((T, N, M), dtype=np.float32)
        signal_vec = rng.standard_normal(Dt) / np.sqrt(Dt)
        nb = np.stack([rng.choice(np.delete(np.arange(N), i), K, replace=False) for i in range(N)])

        # background chatter
        chatter = rng.random((T, N, M)) < 0.15
        text[chatter] = rng.standard_normal((int(chatter.sum()), Dt)) / np.sqrt(Dt)
        text_mask |= chatter
        text_age[chatter] = rng.integers(0, 5, size=int(chatter.sum()))

        events = []
        for i in range(N):
            t = 100
            while t < T - 80:
                if rng.random() < rate:
                    h_star = int(rng.choice([2, 3, 5, 10]))
                    d = rng.choice([-1.0, 1.0])
                    m = rng.uniform(m_lo, m_hi) * sigma[i] * np.sqrt(h_star)
                    events.append((t, i, d, m, h_star))
                    t += 3 * h_star + lead + 5
                else:
                    t += 1
        for (t0, i, d, m, h_star) in events:
            peak = t0 + h_star
            ret[t0 + 1 : peak + 1, i] += d * m / h_star
            ret[peak + 1 : peak + 1 + h_star, i] -= d * rho * m / h_star
            lo = max(0, t0 - lead)
            labels[lo : peak + 1, i] = 1
            btp[lo : peak + 1 + h_star, i] = peak - np.arange(lo, peak + 1 + h_star)
            vol_z[lo : peak + 1, i] += np.linspace(0.5, 3.0, peak + 1 - lo)
            for tt in range(lo, t0 + 1):
                slot = rng.integers(0, M)
                text[tt, i, slot] = d * text_snr * signal_vec + rng.standard_normal(Dt) / np.sqrt(Dt)
                text_mask[tt, i, slot] = True
                text_age[tt, i, slot] = 0
            spread[peak - 1 : peak + 2, i] *= spread_mult
            if rng.random() < p_halt:
                tradable[peak - 1 : peak + 1, i] = False
            for j in nb[i]:
                vol_z[lo : t0 + 1, j] += 0.3

        mid = 100.0 * np.exp(np.cumsum(ret, axis=0))
        volume = np.exp(vol_z + 10.0)
        hl_range = np.abs(rng.standard_normal((T, N))) * sigma + np.abs(ret)
        price_feat = np.stack([ret / sigma, vol_z, hl_range / sigma, np.log(spread * 1e3)], -1).astype(np.float32)
        liq = np.stack([spread, np.full((T, N), 1.0), np.full((T, N), 0.02)], -1).astype(np.float32)
        graph = np.zeros((T, N, K, Dg), dtype=np.float32)
        for i in range(N):
            graph[:, i, :, :4] = price_feat[:, nb[i], :4]
        graph_mask = np.ones((T, N, K), dtype=bool)
        panel = Panel(times=np.arange(T, dtype=np.int64), assets=[f"SYN{i:02d}" for i in range(N)], mid=mid,
                      price_feat=price_feat, liq=liq, tradable=tradable, labels=labels, bars_to_peak=btp,
                      text=text, text_mask=text_mask, text_age=text_age, graph=graph, graph_mask=graph_mask,
                      meta={"n_events": len(events), "freq": self.freq, "bars_per_year": self.bars_per_year,
                            "horizons": list(self.horizons), "volume": volume,
                            "feat_names": ["ret_z", "vol_z", "hl_range_z", "log_spread"],
                            # planted-event table -> anchors.synthetic_anchor_table builds regulatory anchors from it
                            "events": [{"onset_t": int(t0), "asset": int(i), "direction": float(d), "magnitude": float(m),
                                        "peak_t": int(t0 + h_star), "end_t": int(t0 + 2 * h_star)}
                                       for (t0, i, d, m, h_star) in events]})
        panel.validate()
        return panel

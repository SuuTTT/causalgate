"""Panel = the single in-memory representation every dataset builder must produce.

Shapes (T = time bars, N = assets, M = max text docs per (t, i), K = max graph neighbours):
    times        (T,)            int64 (ns since epoch) or bar index
    mid          (T, N)          mid / close price (float64). NaN where the asset does not exist.
    price_feat   (T, N, Fp)      per-bar price/volume features (already stationary: returns, z-scores, ...)
    liq          (T, N, 3)       [relative half-spread, depth or quote size (normalised), ADV fraction]
    tradable     (T, N)          bool. False = halted / limit-hit / liquidity freeze -> cannot enter or exit
    labels       (T, N)          weak anomaly label in {0, 1} (see labels.py). Used ONLY as a training target.
    bars_to_peak (T, N)          signed bars until the event's peak market impact (NaN if labels == 0).
                                 Positive = warning is early, negative = warning is late. Drives WLT.
    lob          (T, N, Fl)      optional LOB snapshot features (levels x [bid_px, bid_sz, ask_px, ask_sz] ...)
    text         (T, N, M, Dt)   optional pre-computed document embeddings (zeros where padded)
    text_mask    (T, N, M)       optional bool, True for a real document
    text_age     (T, N, M)       optional bars elapsed since the document was published (>= 0)
    graph        (T, N, K, Dg)   optional neighbour embeddings (entity / sector / co-movement graph)
    graph_mask   (T, N, K)       optional bool
    meta         dict            free-form (frequency, bars_per_year, horizons, asset names, ...)

Storage note: for the large benchmarks (LOBSTER, FNSPID, EDGAR) keep `text` / `lob` as np.memmap on disk
and let EventDataset slice lazily; the Panel contract is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

LIQ_FIELDS = ("rel_half_spread", "depth", "adv_frac")


@dataclass
class Panel:
    times: np.ndarray
    assets: List[str]
    mid: np.ndarray
    price_feat: np.ndarray
    liq: np.ndarray
    tradable: np.ndarray
    labels: np.ndarray
    bars_to_peak: np.ndarray
    lob: Optional[np.ndarray] = None
    text: Optional[np.ndarray] = None
    text_mask: Optional[np.ndarray] = None
    text_age: Optional[np.ndarray] = None
    graph: Optional[np.ndarray] = None
    graph_mask: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.mid.shape[0])

    @property
    def N(self) -> int:
        return int(self.mid.shape[1])

    def dims(self) -> Dict[str, int]:
        return {
            "price": int(self.price_feat.shape[-1]),
            "lob": int(self.lob.shape[-1]) if self.lob is not None else 0,
            "text": int(self.text.shape[-1]) if self.text is not None else 0,
            "text_slots": int(self.text.shape[2]) if self.text is not None else 0,
            "graph": int(self.graph.shape[-1]) if self.graph is not None else 0,
            "graph_slots": int(self.graph.shape[2]) if self.graph is not None else 0,
            "liq": int(self.liq.shape[-1]),
        }

    def validate(self) -> None:
        T, N = self.mid.shape
        assert self.times.shape == (T,)
        assert len(self.assets) == N
        assert self.price_feat.shape[:2] == (T, N)
        assert self.liq.shape == (T, N, 3), self.liq.shape
        assert self.tradable.shape == (T, N) and self.tradable.dtype == bool
        assert self.labels.shape == (T, N)
        assert self.bars_to_peak.shape == (T, N)
        for name in ("lob", "text", "graph"):
            arr = getattr(self, name)
            if arr is not None:
                assert arr.shape[:2] == (T, N), f"{name}: {arr.shape}"
        if self.text is not None:
            assert self.text_mask is not None and self.text_age is not None
        if self.graph is not None:
            assert self.graph_mask is not None


# ----------------------------------------------------------------------------------------------------------------
# CausalGate track (append-only, 2026-10-03): optional point-in-time availability fields. They are attached to a Panel
# *instance* by causalgate.data.pit.attach_availability (never required by validate(), never read by other tracks):
#   avail_feat (T,N) int64, avail_text (T,N,M) int64, avail_label (T,N) int64, avail_graph (T,N,K) int64,
#   universe (T,N) bool.  Names are listed in panel.meta["pit_fields"]. See causalgate/data/pit.py.
PIT_OPTIONAL_FIELDS = ("avail_feat", "avail_text", "avail_label", "avail_graph", "universe")

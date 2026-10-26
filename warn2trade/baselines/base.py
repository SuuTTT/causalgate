from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.base import build_loader
from ..backtest.costs import CostModel
from ..backtest.engine import EventBacktester
from ..utils.registry import Registry

SEMI_BASELINES = Registry("semi_baselines")   # score-only baselines of the label-scarcity track; separate from BASELINES


class ScoreBaseline:
    """fit(SemiEventDataset) -> self ; score(dataset) -> (len(ds),) higher = more anomalous."""
    name = "base"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}

    def fit(self, ds):  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, ds) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


def flatten_windows(ds, batch_size: int = 1024, max_items: Optional[int] = None, key: str = "price") -> np.ndarray:
    """(n, L*F) flattened model-visible windows in dataset order."""
    xs, n = [], 0
    for b in build_loader(ds, batch_size, False):
        x = b[key].reshape(b[key].shape[0], -1).numpy()
        xs.append(x)
        n += len(x)
        if max_items is not None and n >= max_items:
            break
    return np.concatenate(xs)[: max_items] if xs else np.zeros((0, 1))


class ScoreToTradeAdapter:
    """The common trade rule for detection-only baselines (stated in the paper's protocol section).

    fire      iff score >= thr (thr = train-score quantile)
    direction 'fade' = against the trailing return (manipulation reverts), 'momentum' = with it, 'long' / 'short' fixed
    size w, holding horizon hold_h bars (nearest grid horizon), immediate execution (kappa = 1).
    a_prob = rank of the score among the training scores (a monotone [0, 1] map so AUC metrics are unchanged).
    """

    def __init__(self, horizons: Sequence[int], latency: int, cost_model: CostModel, thr: float, w: float = 0.5, hold_h: int = 5,
                 direction: str = "fade", train_scores: Optional[np.ndarray] = None, ret_feat: int = 0, trail: int = 5):
        self.horizons, self.latency, self.thr, self.w, self.direction = list(horizons), latency, thr, w, direction
        self.h_idx = int(np.argmin(np.abs(np.asarray(self.horizons) - hold_h)))
        self.bt = EventBacktester(self.horizons, latency, cost_model)
        self.ref = np.sort(train_scores[np.isfinite(train_scores)]) if train_scores is not None else None
        self.ret_feat, self.trail = ret_feat, trail

    def _prob(self, s: np.ndarray) -> np.ndarray:
        if self.ref is None or len(self.ref) == 0:
            return 1.0 / (1.0 + np.exp(-s))
        return np.searchsorted(self.ref, s, side="right") / (len(self.ref) + 1.0)

    @torch.no_grad()
    def collect(self, ds, scores: np.ndarray, batch_size: int = 256) -> pd.DataFrame:
        frames, pos = [], 0
        H = len(self.horizons)
        for b in build_loader(ds, batch_size, False):
            B = b["price"].shape[0]
            s = scores[pos : pos + B]
            pos += B
            fire = torch.as_tensor(s >= self.thr)
            trail = b["price"][:, -self.trail :, self.ret_feat].sum(-1)
            if self.direction == "fade":
                d = -torch.sign(trail)
            elif self.direction == "momentum":
                d = torch.sign(trail)
            elif self.direction == "long":
                d = torch.ones(B)
            else:
                d = -torch.ones(B)
            d = torch.where(d == 0, torch.ones_like(d), d)
            w = torch.where(fire, self.w * d, torch.zeros(B)).float()
            out = {"w": w, "abstain": (~fire).float(), "hold_p": F.one_hot(torch.full((B,), self.h_idx), H).float(),
                   "kappa_p": F.one_hot(torch.zeros(B, dtype=torch.long), 4).float(),
                   "a_prob": torch.as_tensor(self._prob(s), dtype=torch.float32)}
            frames.append(self.bt.collect(out, b))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

"""Rule baselines: statistical anomaly score + fixed direction rule. These are the 'open-loop' reference points."""
from __future__ import annotations
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader

from .event_base import BaseBaseline, register_event_baseline
from ..utils.registry import BASELINES


class _ZScoreBase(BaseBaseline):
    direction_sign = 1.0   # +1 momentum, -1 reversal

    def __init__(self, cfg, dims, horizons, device):
        super().__init__(cfg, dims, horizons, device)
        self.thr = float(cfg.get("z_thr", 2.5))
        self.hold_index = int(cfg.get("hold_index", 2))
        self.ret_col, self.vol_col = int(cfg.get("ret_col", 0)), int(cfg.get("vol_col", 1))

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        return None

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["price"]
        last_ret = x[:, -3:, self.ret_col].sum(-1)
        vol_z = x[:, -1, self.vol_col]
        z = torch.maximum(last_ret.abs(), vol_z)
        a = torch.sigmoid(z - self.thr)
        trade = z > self.thr
        w = torch.where(trade, self.direction_sign * torch.sign(last_ret) * self.w_max, torch.zeros_like(last_ret))
        B = x.shape[0]
        return {"a_prob": a, "w": w, "abstain": (~trade).float(), "hold_p": self._hold_fixed(B, self.hold_index), "kappa_p": self._immediate(B)}


@register_event_baseline("zscore_momentum")
class ZScoreMomentum(_ZScoreBase):
    name, direction_sign = "zscore_momentum", 1.0
    paper = "Rule baseline: volume/return z-score anomaly -> trade with the move (continuation)."


@register_event_baseline("zscore_reversal")
class ZScoreReversal(_ZScoreBase):
    name, direction_sign = "zscore_reversal", -1.0
    paper = "Rule baseline: volume/return z-score anomaly -> trade against the move (mean reversion)."

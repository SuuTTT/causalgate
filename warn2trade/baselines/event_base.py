"""Event-level baseline contract for the Warn2Trade track: every baseline turns a batch into the same `out` dict as
Warn2Trade so the event backtester and the dual metric matrix are shared verbatim.

    out = {"a_prob": (B,), "w": (B,), "hold_p": (B, H), "kappa_p": (B, n_exec)}   (+ optional "abstain")

(Kept in its own module so that other experiment tracks sharing this checkout can define their own base classes
in baselines/base.py without breaking these runners.)
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import torch
from torch.utils.data import DataLoader

from ..utils.registry import BASELINES


class BaseBaseline:
    name: str = "base"
    paper: str = ""
    official_repo: str = ""
    status: str = "implemented"   # implemented | wrapper-needed | placeholder-unverified

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        self.cfg, self.dims, self.horizons, self.device = cfg, dims, list(horizons), device
        self.n_exec = int(cfg.get("n_exec", 4))
        self.w_max = float(cfg.get("w_max", 1.0))

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def _immediate(self, B: int) -> torch.Tensor:
        k = torch.zeros(B, self.n_exec, device=self.device)
        k[:, 0] = 1.0
        return k

    def _hold_fixed(self, B: int, h_index: int) -> torch.Tensor:
        p = torch.zeros(B, len(self.horizons), device=self.device)
        p[:, h_index] = 1.0
        return p

    def to(self, device):
        self.device = device
        return self


def register_event_baseline(name: str):
    """Register unless another track already took the name (first registration wins, no crash)."""
    def deco(cls):
        try:
            BASELINES.register(name)(cls)
        except KeyError:
            pass
        return cls
    return deco


EVENT_BASELINE_MODULES = ("rules", "dlinear", "detect_then_rule", "stubs")

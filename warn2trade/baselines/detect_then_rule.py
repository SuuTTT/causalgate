"""The status-quo 'open loop': train the SAME multimodal detector with plain BCE (no utility weighting), then hand
warnings to a fixed trading rule (momentum or reversal, fixed holding period). Isolates the value of closing the loop."""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .event_base import BaseBaseline, register_event_baseline
from ..models.anomaly_encoder import MultimodalAnomalyEncoder
from ..utils.registry import BASELINES


@register_event_baseline("detect_then_rule")
class DetectThenRule(BaseBaseline):
    name = "detect_then_rule"
    paper = "Open-loop reference: Warn2Trade's detector trained with F1-style BCE, trades by a fixed rule."

    def __init__(self, cfg, dims, horizons, device):
        super().__init__(cfg, dims, horizons, device)
        self.det = MultimodalAnomalyEncoder(dims, cfg.get("encoder", {})).to(device)
        self.epochs, self.lr = int(cfg.get("epochs", 5)), float(cfg.get("lr", 3e-4))
        self.thr = float(cfg.get("thr", 0.5))
        self.rule = cfg.get("rule", "momentum")
        self.hold_index = int(cfg.get("hold_index", 2))
        self.pos_weight = cfg.get("pos_weight", None)

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        opt = torch.optim.AdamW(self.det.parameters(), lr=self.lr, weight_decay=1e-4)
        pw = None if self.pos_weight is None else torch.tensor(float(self.pos_weight), device=self.device)
        self.det.train()
        for _ in range(self.epochs):
            for b in train:
                b = {k: v.to(self.device) for k, v in b.items()}
                loss = F.binary_cross_entropy_with_logits(self.det(b)["logit"], b["y"], pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
        self.det.eval()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a = torch.sigmoid(self.det(batch)["logit"])
        last_ret = batch["price"][:, -3:, 0].sum(-1)
        sgn = torch.sign(last_ret) * (1.0 if self.rule == "momentum" else -1.0)
        trade = a > self.thr
        w = torch.where(trade, sgn * self.w_max, torch.zeros_like(a))
        B = a.shape[0]
        return {"a_prob": a, "w": w, "abstain": (~trade).float(), "hold_p": self._hold_fixed(B, self.hold_index), "kappa_p": self._immediate(B)}

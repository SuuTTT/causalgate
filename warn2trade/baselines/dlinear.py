"""DLinear (Zeng et al., AAAI 2023) adapted to multi-horizon impact forecasting -> sign/size rule."""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .event_base import BaseBaseline, register_event_baseline
from ..utils.registry import BASELINES


class DLinear(nn.Module):
    def __init__(self, L: int, F_in: int, H: int, kernel: int = 7):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(L * F_in, H)
        self.season = nn.Linear(L * F_in, H)

    def forward(self, x):                       # (B, L, F)
        pad = self.kernel // 2
        xt = x.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"), self.kernel, stride=1).transpose(1, 2)
        season = x - trend
        return self.trend(trend.flatten(1)) + self.season(season.flatten(1))


@register_event_baseline("dlinear")
class DLinearBaseline(BaseBaseline):
    name = "dlinear"
    paper = "Zeng, Chen, Zhang, Xu. Are Transformers Effective for Time Series Forecasting? AAAI 2023"
    official_repo = "https://github.com/cure-lab/LTSF-Linear"

    def __init__(self, cfg, dims, horizons, device):
        super().__init__(cfg, dims, horizons, device)
        self.L = int(cfg["lookback"])
        self.net = DLinear(self.L, dims["price"], len(horizons)).to(device)
        self.epochs, self.lr = int(cfg.get("epochs", 5)), float(cfg.get("lr", 1e-3))
        self.k_trade = float(cfg.get("k_trade", 1.0))     # trade if |pred| > k * sigma * sqrt(h)

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for b in train:
                x, y = b["price"].to(self.device), b["impact"].to(self.device)
                loss = F.mse_loss(self.net(x), y / (b["sigma"].to(self.device).unsqueeze(-1) + 1e-8))
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sigma = batch["sigma"]
        pred = self.net(batch["price"]) * sigma.unsqueeze(-1)                      # (B, H)
        hz = torch.tensor(self.horizons, device=pred.device, dtype=torch.float32)
        snr = pred.abs() / (sigma.unsqueeze(-1) * hz.sqrt() + 1e-8)
        h_star = snr.argmax(-1)
        bi = torch.arange(pred.shape[0], device=pred.device)
        best = snr[bi, h_star]
        trade = best > self.k_trade
        w = torch.where(trade, torch.sign(pred[bi, h_star]) * torch.clamp(best / 3.0, max=1.0) * self.w_max, torch.zeros_like(best))
        return {"a_prob": torch.sigmoid(best - self.k_trade), "w": w, "abstain": (~trade).float(),
                "hold_p": F.one_hot(h_star, len(self.horizons)).float(), "kappa_p": self._immediate(pred.shape[0])}

"""Dataset builder contract + the windowed EventDataset consumed by every model and baseline."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .schema import Panel
from .labels import impact_curves, realized_vol
from ..utils.seed import seed_worker


class BaseDatasetBuilder:
    """Subclass per benchmark. `build_panel` must return a validated Panel; everything downstream is shared.

    Required class attributes
        name            registry key
        freq            'daily' | '1min' | 'lob'
        bars_per_year   annualisation constant used by the trading metrics
        horizons        default impact-horizon grid (in bars)
    """
    name: str = "base"
    freq: str = "daily"
    bars_per_year: int = 252
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 20)

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.root = cfg.get("root", "data_raw")

    def build_panel(self) -> Panel:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(self.horizons)}


class EventDataset(Dataset):
    """Lazy windowed view over a Panel restricted to a time index set (one walk-forward fold slice).

    Each item is one decision point (t, i). Future quantities (`impact`, `fut_ret`, `fut_liq`, `fut_tradable`) are
    targets for training and inputs to the *backtest only*; models never see them in forward().
    """

    def __init__(self, panel: Panel, time_idx: np.ndarray, lookback: int, horizons: Sequence[int],
                 latency: int = 1, exec_max: int = 4, vol_window: int = 60, drop_nan: bool = True,
                 neg_subsample: Optional[float] = None, rng: Optional[np.random.Generator] = None):
        self.p = panel
        self.L = lookback
        self.horizons = list(horizons)
        self.hmax = max(self.horizons) + latency + exec_max
        self.latency = latency
        self.vol_window = vol_window
        T, N = panel.mid.shape
        self._impact = impact_curves(panel.mid, self.horizons)
        self._sigma = realized_vol(panel.mid, vol_window)
        self._logret = np.zeros_like(panel.mid)
        self._logret[1:] = np.log(panel.mid[1:]) - np.log(panel.mid[:-1])
        lo, hi = max(int(time_idx.min()), lookback), min(int(time_idx.max()) + 1, T - self.hmax)
        ts = np.arange(lo, hi)
        tt, ii = np.meshgrid(ts, np.arange(N), indexing="ij")
        tt, ii = tt.ravel(), ii.ravel()
        if drop_nan:
            ok = ~np.isnan(panel.mid[tt, ii]) & ~np.isnan(self._sigma[tt, ii]) & ~np.isnan(self._impact[tt, ii]).any(-1)
            ok &= ~np.isnan(panel.price_feat[tt, ii]).any(-1)
            tt, ii = tt[ok], ii[ok]
        if neg_subsample is not None and neg_subsample < 1.0:
            rng = rng or np.random.default_rng(0)
            y = panel.labels[tt, ii]
            keep = (y == 1) | (rng.random(len(tt)) < neg_subsample)
            tt, ii = tt[keep], ii[keep]
        self.index = np.stack([tt, ii], 1)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def pos_rate(self) -> float:
        return float(self.p.labels[self.index[:, 0], self.index[:, 1]].mean()) if len(self) else 0.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t, i = self.index[idx]
        p = self.p
        L, H = self.L, self.hmax
        item: Dict[str, Any] = {
            "price": torch.as_tensor(p.price_feat[t - L + 1 : t + 1, i], dtype=torch.float32),
            "liq": torch.as_tensor(p.liq[t, i], dtype=torch.float32),
            "sigma": torch.tensor(float(self._sigma[t, i]), dtype=torch.float32),
            "y": torch.tensor(float(p.labels[t, i]), dtype=torch.float32),
            "bars_to_peak": torch.tensor(float(p.bars_to_peak[t, i]), dtype=torch.float32),
            "impact": torch.as_tensor(self._impact[t, i], dtype=torch.float32),
            "fut_ret": torch.as_tensor(self._logret[t + 1 : t + 1 + H, i], dtype=torch.float32),
            "fut_liq": torch.as_tensor(p.liq[t + 1 : t + 1 + H, i], dtype=torch.float32),
            "fut_tradable": torch.as_tensor(p.tradable[t + 1 : t + 1 + H, i]),
            "asset": torch.tensor(int(i)),
            "t": torch.tensor(int(t)),
        }
        if p.lob is not None:
            item["lob"] = torch.as_tensor(p.lob[t - L + 1 : t + 1, i], dtype=torch.float32)
        if p.text is not None:
            item["text"] = torch.as_tensor(p.text[t, i], dtype=torch.float32)
            item["text_mask"] = torch.as_tensor(p.text_mask[t, i])
            item["text_age"] = torch.as_tensor(p.text_age[t, i], dtype=torch.float32)
        if p.graph is not None:
            item["graph"] = torch.as_tensor(p.graph[t, i], dtype=torch.float32)
            item["graph_mask"] = torch.as_tensor(p.graph_mask[t, i])
        return item


def build_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 42) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      worker_init_fn=seed_worker, generator=g, drop_last=False)

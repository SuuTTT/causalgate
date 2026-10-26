"""Causal anomaly scorers. Each consumes windows (B, L, F) of *causally normalised* features ending at the decision
bar and returns one score per window; no scorer reads anything after the window's last bar.

    (a) CausalTCNScorer        dilated causal 1-D convolutions predicting x_t from x_{<t}; score = standardised
                               one-step prediction error at the last bar (residual scale from the training fold).
    (b) PrequentialAEScorer    autoencoder over the window; score = reconstruction error of the last k bars; in
                               refit_policy='streaming' the window is scored *then* used for one SGD step
                               (prequential / predict-then-update, Dawid 1984).
    (c) LeftMatrixProfileScorer DAMP-style left matrix profile (Lu et al., KDD 2022): z-normalised Euclidean distance
                               from the last m-subsequence to its nearest *earlier* neighbour (exclusion zone m).
                               Non-parametric reference; `score_series` gives the full left profile for a series.

refit_policy in {static, periodic, streaming} is an explicit experimental axis (E6): causality never
requires streaming updates (Parrino et al., KDD 2026); a static model applied causally is leakage-free.
    static     fitted once on the purged train fold (the ProtocolRunner builds a fresh model per walk-forward fold, so
               this is already a per-fold refit; a separate 'per_fold' value would be identical and was removed);
    periodic   every `refit_every` decision bars the owner (CausalGate) calls `refit(X_past, t_end)` with the trailing
               purged window of past windows (all ending <= t_end < the decision bar); the scorer is fine-tuned for
               `refit_epochs` epochs and re-stamped as of t_end;
    streaming  after each predict batch the scored windows are used for one SGD step (prequential).
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.pit import AsOfStamped

REFIT_POLICIES = ("static", "periodic", "streaming")


def _nan_to_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class CausalScorer(AsOfStamped):
    name = "base"

    def __init__(self, in_dim: int, lookback: int, cfg: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None):
        self.in_dim, self.L, self.cfg = int(in_dim), int(lookback), dict(cfg or {})
        self.device = device or torch.device("cpu")
        self.refit_policy = self.cfg.get("refit_policy", "static")
        assert self.refit_policy in REFIT_POLICIES, self.refit_policy
        self.refit_every = int(self.cfg.get("refit_every", 250))
        self.refit_window = int(self.cfg.get("refit_window", 5000))     # max past windows used by a periodic refit
        self.refit_epochs = int(self.cfg.get("refit_epochs", 1))

    def fit(self, X: np.ndarray, t_end: int) -> "CausalScorer":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, X: np.ndarray, t: int) -> None:
        """Absorb windows already scored (refit_policy='streaming'). Default: no-op."""
        return None

    def refit(self, X: np.ndarray, t_end: int) -> "CausalScorer":
        """Periodic refit on past windows (all ending <= t_end): fine-tune for `refit_epochs` epochs, re-stamp."""
        epochs = getattr(self, "epochs", None)
        if epochs is not None:
            self.epochs = self.refit_epochs
        try:
            with torch.random.fork_rng(devices=[]), torch.enable_grad():   # dropout RNG seeded by t_end: replays coincide
                torch.manual_seed(int(self.cfg.get("seed", 0)) * 1_000_003 + int(t_end))
                self.fit(X, t_end)
        finally:
            if epochs is not None:
                self.epochs = epochs
        return self

    def parameters(self):
        return []


# --------------------------------------------------------------------------------------- (a) causal TCN
class _CausalConvBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(F.pad(x, (self.pad, 0))))
        h = self.drop(h)
        h = self.conv2(F.pad(h, (self.pad, 0)))
        return F.gelu(x + h)


class CausalTCN(nn.Module):
    """x (B, L, F) -> one-step-ahead prediction (B, L, F): output at position t uses inputs <= t only."""

    def __init__(self, in_dim: int, hidden: int = 32, n_layers: int = 4, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList([_CausalConvBlock(hidden, k, 2 ** i, dropout) for i in range(n_layers)])
        self.out = nn.Conv1d(hidden, in_dim, 1)
        self.receptive = 1 + sum(2 * (k - 1) * 2 ** i for i in range(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out(h).transpose(1, 2)


class CausalTCNScorer(CausalScorer):
    name = "tcn"

    def __init__(self, in_dim, lookback, cfg=None, device=None):
        super().__init__(in_dim, lookback, cfg, device)
        c = self.cfg
        torch.manual_seed(int(c.get("seed", 0)))
        self.net = CausalTCN(self.in_dim, int(c.get("hidden", 32)), int(c.get("n_layers", 4)), int(c.get("kernel", 3)), float(c.get("dropout", 0.1))).to(self.device)
        self.epochs, self.lr, self.bs = int(c.get("epochs", 5)), float(c.get("lr", 1e-3)), int(c.get("batch_size", 256))
        self.resid_scale = np.ones(self.in_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.net(x)[:, :-1]
        tgt = x[:, 1:]
        return F.mse_loss(pred, tgt)

    def fit(self, X, t_end):
        x = _nan_to_zero(torch.as_tensor(np.asarray(X), dtype=torch.float32)).to(self.device)
        n = x.shape[0]
        self.net.train()
        g = torch.Generator().manual_seed(int(self.cfg.get("seed", 0)))
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.bs):
                xb = x[perm[i : i + self.bs]]
                loss = self._loss(xb)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.net.eval()
        with torch.no_grad():
            res = []
            for i in range(0, n, 1024):
                xb = x[i : i + 1024]
                res.append((xb[:, -1] - self.net(xb)[:, -2]).cpu().numpy())
            res = np.concatenate(res) if res else np.zeros((1, self.in_dim))
        self.resid_scale = np.maximum(np.nanstd(res, axis=0), 1e-6)
        self.stamp(int(t_end))
        return self

    @torch.no_grad()
    def score(self, X):
        Xn = np.asarray(X, dtype=np.float32)
        x = _nan_to_zero(torch.as_tensor(Xn)).to(self.device)
        self.net.eval()
        pred = self.net(x)[:, -2]
        res = (x[:, -1] - pred).cpu().numpy() / self.resid_scale
        s = np.log1p(np.mean(res ** 2, axis=-1))
        s[np.isnan(Xn[:, -1]).any(-1)] = np.nan                     # warm-up abstention propagates
        return s

    def update(self, X, t):
        if self.refit_policy == "streaming":
            x = _nan_to_zero(torch.as_tensor(np.asarray(X), dtype=torch.float32)).to(self.device)
            self.net.train()
            with torch.random.fork_rng(devices=[]), torch.enable_grad():   # callers (predict) run under no_grad
                torch.manual_seed(int(self.cfg.get("seed", 0)) * 1_000_003 + int(t))
                loss = self._loss(x)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.net.eval()
            self.stamp(int(t))

    def parameters(self):
        return list(self.net.parameters())


# --------------------------------------------------------------------------------------- (b) prequential AE
class _AE(nn.Module):
    def __init__(self, d_in: int, hidden: int, latent: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class PrequentialAEScorer(CausalScorer):
    name = "ae"

    def __init__(self, in_dim, lookback, cfg=None, device=None):
        super().__init__(in_dim, lookback, cfg, device)
        c = self.cfg
        torch.manual_seed(int(c.get("seed", 0)))
        self.k_last = int(c.get("k_last", 1))
        self.net = _AE(self.L * self.in_dim, int(c.get("hidden", 64)), int(c.get("latent", 8))).to(self.device)
        self.epochs, self.lr, self.bs = int(c.get("epochs", 5)), float(c.get("lr", 1e-3)), int(c.get("batch_size", 256))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.err_scale = 1.0

    def _flat(self, X) -> torch.Tensor:
        x = _nan_to_zero(torch.as_tensor(np.asarray(X), dtype=torch.float32)).to(self.device)
        return x.reshape(x.shape[0], -1)

    def _err(self, xf: torch.Tensor) -> torch.Tensor:
        rec = self.net(xf).reshape(xf.shape[0], self.L, self.in_dim)
        x = xf.reshape(xf.shape[0], self.L, self.in_dim)
        return ((rec[:, -self.k_last :] - x[:, -self.k_last :]) ** 2).mean(dim=(1, 2))

    def fit(self, X, t_end):
        xf = self._flat(X)
        n = xf.shape[0]
        g = torch.Generator().manual_seed(int(self.cfg.get("seed", 0)))
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.bs):
                loss = F.mse_loss(self.net(xf[perm[i : i + self.bs]]), xf[perm[i : i + self.bs]])
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.net.eval()
        with torch.no_grad():
            e = torch.cat([self._err(xf[i : i + 1024]) for i in range(0, n, 1024)]).cpu().numpy() if n else np.ones(1)
        self.err_scale = float(max(np.nanmean(e), 1e-8))
        self.stamp(int(t_end))
        return self

    @torch.no_grad()
    def score(self, X):
        Xn = np.asarray(X, dtype=np.float32)
        self.net.eval()
        s = np.log1p(self._err(self._flat(Xn)).cpu().numpy() / self.err_scale)
        s[np.isnan(Xn[:, -1]).any(-1)] = np.nan
        return s

    def update(self, X, t):
        if self.refit_policy == "streaming":
            xf = self._flat(X)
            self.net.train()
            with torch.random.fork_rng(devices=[]), torch.enable_grad():
                torch.manual_seed(int(self.cfg.get("seed", 0)) * 1_000_003 + int(t))
                loss = F.mse_loss(self.net(xf), xf)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.net.eval()
            self.stamp(int(t))

    def parameters(self):
        return list(self.net.parameters())


# --------------------------------------------------------------------------------------- (c) left matrix profile
def _znorm(a: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    m = a.mean(axis=axis, keepdims=True)
    s = a.std(axis=axis, keepdims=True)
    return (a - m) / np.maximum(s, eps)


class LeftMatrixProfileScorer(CausalScorer):
    """Window-local left matrix profile: the last m-subsequence of the window vs all earlier subsequences of the
    same window that end at least m bars before it. Per-feature z-normalised Euclidean distance, averaged."""
    name = "lmp"

    def __init__(self, in_dim, lookback, cfg=None, device=None):
        super().__init__(in_dim, lookback, cfg, device)
        self.m = int(self.cfg.get("m", 8))
        assert 2 * self.m <= self.L, "lookback must be >= 2 m"
        self.scale = 1.0

    def fit(self, X, t_end):
        s = self.score(np.asarray(X)[: 2048])
        self.scale = float(max(np.nanmedian(s), 1e-8))
        self.stamp(int(t_end))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        B, L, Fd = X.shape
        m = self.m
        q = _znorm(X[:, L - m :, :], axis=1)                                            # (B, m, F)
        n_cand = L - 2 * m + 1
        cand = np.stack([X[:, j : j + m, :] for j in range(n_cand)], 1)                  # (B, C, m, F)
        cand = _znorm(cand, axis=2)
        d = np.sqrt(((cand - q[:, None]) ** 2).sum(2)).mean(-1)                          # (B, C)
        d[np.isnan(d)] = np.inf
        s = d.min(1) / max(self.scale, 1e-8)
        s[np.isnan(X[:, -1]).any(-1) | ~np.isfinite(s)] = np.nan
        return s

    @staticmethod
    def score_series(x: np.ndarray, m: int, lookback: int = 1000, t_lo: int = 0) -> np.ndarray:
        """Full left matrix profile of a univariate series x (T,): lmp[t] = distance from x[t-m+1..t] to its nearest
        neighbour among subsequences ending in [t - lookback, t - m]. NaN where no candidate exists."""
        x = np.asarray(x, dtype=np.float64)
        T = x.shape[0]
        out = np.full(T, np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        if T < 2 * m:
            return out
        subs = _znorm(sliding_window_view(x, m), axis=1)                                 # (T-m+1, m) ending at j+m-1
        for t in range(max(t_lo, 2 * m - 1), T):
            e = t - m + 1                                                                # index of the query subsequence
            lo = max(0, e - lookback)
            hi = e - m + 1                                                               # candidates ending <= t - m
            if hi <= lo:
                continue
            d = np.sqrt(((subs[lo:hi] - subs[e]) ** 2).sum(1))
            out[t] = d.min()
        return out


SCORERS = {"tcn": CausalTCNScorer, "ae": PrequentialAEScorer, "lmp": LeftMatrixProfileScorer}


def make_scorer(cfg: Dict[str, Any], in_dim: int, lookback: int, device: Optional[torch.device] = None) -> CausalScorer:
    return SCORERS[cfg.get("type", "tcn")](in_dim, lookback, cfg, device)

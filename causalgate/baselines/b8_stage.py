"""B8 STAGE: Shi, Rasli, Wang, "STAGE: ..." PLOS ONE 2025, doi:10.1371/journal.pone.0318939. status = reimplemented.

Pipeline as described in the paper's Methodology: an undirected stock graph G = (V, E) with adjacency A (construction
not specified), a Graph Attention Network layer (Eq. 5: h'_i = sigma(sum_j alpha_ij W x_j); Eq. 6: alpha_ij =
softmax_j(LeakyReLU(a^T [W x_i || W x_j]))), a Variational Autoencoder on the node features (Eqs. 7-9, loss Eq. 10 =
reconstruction + KL), an anomaly score per node (Eq. 19) and the anomaly threshold of Eq. 20,
tau = mu_S + delta * sigma_S, "the mean and standard deviation of the anomaly scores S". The paper does not say which
sample mu_S and sigma_S are computed over, nor how the data are split or normalised: this wrapper is the canonical
*protocol-ambiguous* baseline and implements the three readings of Eq. 20 as cfg['stage_reading'] in
{'full_sample', 'test_fold', 'train_fold'} so that the gap between them can be reported.

Compact re-implementation: graph = kNN on the rolling correlation of past returns (window and k are ours), one GAT
layer, a VAE reconstructing the node's window features, score = reconstruction error + kl_weight * KL, trend = moving-
average crossover direction, trading rule = the scaffold's alarm -> trade in the trend direction (the paper reports
Sharpe 1.78 / MDD -16.5 % but states no trading rule).
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.pit import AsOfStamped
from .common import ScoreToTradeBaseline, register
from .b1_cpd_dmn import RowAwareMixin

READINGS = ("full_sample", "test_fold", "train_fold")


class _STAGEModel(AsOfStamped):
    """Fitted GAT + VAE; stamped at the end of the training fold."""


class _STAGEThreshold(AsOfStamped):
    """Eq. 20 threshold; stamped at the last bar of the sample its mean / std were computed on (the reading)."""


class _GATLayer(nn.Module):
    """Eqs. 5-6 (single head; the number of heads is not stated)."""

    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.W = nn.Linear(in_dim, d, bias=False)
        self.a_src = nn.Linear(d, 1, bias=False)
        self.a_dst = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.W(x)                                                                 # (B, N, d)
        e = F.leaky_relu(self.a_src(h) + self.a_dst(h).transpose(1, 2), 0.2)         # a^T [W x_i || W x_j]
        e = e.masked_fill(adj <= 0, float("-inf"))
        att = torch.softmax(e, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        return F.elu(att @ h), att


class _VAE(nn.Module):
    """Eqs. 7-9 with a Gaussian posterior; the decoder reconstructs the node's own input features."""

    def __init__(self, d: int, hidden: int, latent: int, out_dim: int):
        super().__init__()
        self.enc = nn.Linear(d, hidden)
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.dec1, self.dec2 = nn.Linear(latent, hidden), nn.Linear(hidden, out_dim)

    def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = F.relu(self.enc(h))
        return self.mu(e), self.logvar(e).clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec2(F.relu(self.dec1(z)))


class _STAGENet(nn.Module):
    def __init__(self, in_dim: int, d: int, hidden: int, latent: int):
        super().__init__()
        self.gat = _GATLayer(in_dim, d)
        self.vae = _VAE(d, hidden, latent, in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, sample: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns per-node (reconstruction MSE, KL): (B, N) each."""
        h, _ = self.gat(x, adj)
        mu, logvar = self.vae.encode(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        xhat = self.vae.decode(z)
        rec = ((xhat - x) ** 2).mean(-1)
        kl = -0.5 * (1.0 + logvar - mu ** 2 - logvar.exp()).sum(-1)
        return rec, kl


@register("cg_stage")
class STAGEBaseline(RowAwareMixin, ScoreToTradeBaseline):
    name = "cg_stage"
    paper = "Shi, Rasli, Wang, STAGE (GAT + VAE anomaly-aware stock trend prediction), PLOS ONE 2025, doi:10.1371/journal.pone.0318939"
    official_repo = ""
    status = "reimplemented"
    year = 2025
    cutoff_date = None
    conventions = {
        "L1": "Not stated (no normalisation / standardisation / scaling sentence in the paper). published mode: scaffold OfflineScaler "
              "(`published_norm`, default global); causal mode: streaming z-score. Both are readings, neither is the paper's.",
        "L2": "Eq. 20: tau = mu_S + delta * sigma_S with mu_S, sigma_S 'the mean and standard deviation of the anomaly scores S'; the sample "
              "is not stated. cfg['stage_reading'] in {full_sample, test_fold, train_fold}; delta ('adjustment parameter') not stated -> 2.",
        "L3": "Split not stated: results are reported over 'Window 1, Window 2, Window 3' (Fig. 5 / Fig. 8) without dates or proportions; "
              "20 epochs stated; other hyper-parameters in 'Table 2' (not reproduced in the text). No validation procedure described.",
        "L4": "No point-adjust mentioned; evaluation is prediction accuracy (85 % -> 95 % with anomaly detection) plus Sharpe / MDD (Table 5).",
        "L5": "Combined_News_DJIA.csv (25 daily headlines) is described but the headlines are not used in the reported experiments; price features only.",
        "L6": "DJIA constituents (which vintage is not stated).",
        "L7": "Execution timing not stated; no transaction costs. The protocol runner prices decisions at t (published) or t + latency (causal).",
        "L8": "No pre-trained component.",
    }
    unspecified = [
        "Graph construction (Methodology only defines G = (V, E) and A): we use a kNN graph on the rolling correlation of the past "
        "`graph.window` bars of returns (default 60) with k = `graph.k` (default 2) plus self-loops, rebuilt causally at every bar in both modes.",
        "Node features: the asset's last `node_bars` bars (capped at the lookback; pre-sample bars zero-filled) of scaffold-normalised features, flattened; the paper's X in R^{N x F} is not itemised.",
        "GAT: one layer, one head (heads not stated), ELU; the extra regularisation terms printed in Eq. 5 / Eq. 10 / Eq. 19 are dropped.",
        "VAE: hidden `vae.hidden`, latent `vae.latent`, KL weight `vae.kl_weight` (Table 2 values are not in the text); Adam, `epochs` epochs.",
        "Anomaly score (Eq. 19, 'attention-weighted sum of node features'): read as the node's reconstruction error + kl_weight * KL "
        "evaluated at the posterior mean.",
        "Eq. 20 sample (the ambiguity this baseline exists for): three readings, default `test_fold` (the literal 'scores being thresholded'); "
        "delta = 2 (not stated).",
        "Trend module and trading rule are not described: trend = sign(MA_short - MA_long) of the window's cumulative return (5 / 20 bars), "
        "trade = scaffold alarm -> position in the trend direction, hold h bars.",
        "Split / normalisation: not stated; we use the fold's train/val/test and the scaffold's L1 switch.",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "train_mean_std")
        super().__init__(cfg, dims, horizons, device)
        self._init_rows()
        g = dict(cfg.get("graph", {}))
        self.graph_window, self.graph_k = int(g.get("window", 60)), int(g.get("k", 2))
        self.d = int(dict(cfg.get("gat", {})).get("d", 16))
        v = dict(cfg.get("vae", {}))
        self.hidden, self.latent, self.kl_weight = int(v.get("hidden", 16)), int(v.get("latent", 4)), float(v.get("kl_weight", 0.1))
        self.epochs, self.lr, self.batch_size = int(cfg.get("epochs", 2)), float(cfg.get("lr", 1e-3)), int(cfg.get("batch_size", 64))
        nb_cfg = int(cfg.get("node_bars", self.L))
        self.node_bars = max(1, min(nb_cfg, self.L))                    # never more bars than the decision window (no wrap to future rows)
        self.diag["node_bars_effective"] = self.node_bars
        self.stage_reading = str(cfg.get("stage_reading", "test_fold"))
        assert self.stage_reading in READINGS, self.stage_reading
        self.stage_delta = float(cfg.get("stage_delta", 2.0))
        tr = dict(cfg.get("trend", {}))
        self.ma_short, self.ma_long = int(tr.get("short", 5)), int(tr.get("long", 20))
        self.model_op = self._register_op(_STAGEModel())
        self.thr_op = self._register_op(_STAGEThreshold())
        self.net: Optional[_STAGENet] = None
        self._adj_cache: Dict[int, np.ndarray] = {}
        self._s_test: Optional[np.ndarray] = None
        self._t_train_end = -1
        self._t_test_end = -1

    # ------------------------------------------------------------------ graph + cross-sections
    def _adjacency(self, t: int) -> np.ndarray:
        if t in self._adj_cache:
            return self._adj_cache[t]
        Z = self._panel_Z()
        N = Z.shape[1]
        A = np.eye(N, dtype=np.float32)
        lo = max(0, t - self.graph_window + 1)
        R = Z[lo : t + 1, :, self.ret_col].astype(np.float64)
        R = R[np.isfinite(R).all(1)]
        if R.shape[0] >= 10 and N > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.nan_to_num(np.corrcoef(R.T))
            np.fill_diagonal(C, -np.inf)
            k = min(self.graph_k, N - 1)
            nn_idx = np.argsort(-np.abs(C), axis=1)[:, :k]
            for i in range(N):
                A[i, nn_idx[i]] = 1.0
            A = np.maximum(A, A.T)
        self._adj_cache[t] = A
        return A

    def _cross_sections(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(U, N, D) node inputs and (U, N, N) adjacencies for the bars `ts` (NaN -> 0)."""
        Z = self._panel_Z()
        nb = self.node_bars
        idx = ts[:, None] - nb + 1 + np.arange(nb)[None, :]
        neg = idx < 0                                                   # pre-sample bars: zero-filled, never wrapped to the panel end
        Xcs = Z[np.maximum(idx, 0)].astype(np.float32, copy=True)       # (U, nb, N, F)
        if neg.any():
            Xcs[neg] = np.nan
        Xcs = np.nan_to_num(Xcs.transpose(0, 2, 1, 3).reshape(len(ts), Z.shape[1], -1)).astype(np.float32)
        A = np.stack([self._adjacency(int(t)) for t in ts])
        return Xcs, A

    def _single_node(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xn = np.nan_to_num(X[:, -self.node_bars :, :]).reshape(len(X), 1, -1).astype(np.float32)
        return Xn, np.ones((len(X), 1, 1), dtype=np.float32)

    def _node_scores(self, Xcs: np.ndarray, A: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for a in range(0, len(Xcs), 256):
                rec, kl = self.net(torch.as_tensor(Xcs[a : a + 256], device=self.device), torch.as_tensor(A[a : a + 256], device=self.device), sample=False)
                out.append((rec + self.kl_weight * kl).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, Xcs.shape[1]))

    # ------------------------------------------------------------------ hooks
    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        rows = self._rows_for(X)
        if rows is not None and self.panel is not None:
            ts = np.unique(rows["t"])
            Xcs, A = self._cross_sections(ts)
        else:
            Xcs, A = self._single_node(X)
        torch.manual_seed(int(self.cfg.get("seed", 0)))
        self.net = _STAGENet(Xcs.shape[-1], self.d, self.hidden, self.latent).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n = len(Xcs)
        if n >= 2:
            xt, at = torch.as_tensor(Xcs, device=self.device), torch.as_tensor(A, device=self.device)
            g = torch.Generator().manual_seed(int(self.cfg.get("seed", 0)))
            self.net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                for a in range(0, n, self.batch_size):
                    idx = perm[a : a + self.batch_size]
                    rec, kl = self.net(xt[idx], at[idx], sample=True)
                    loss = rec.mean() + self.kl_weight * kl.mean()                       # Eq. 10 (first two terms)
                    opt.zero_grad(); loss.backward(); opt.step()
        self._t_train_end = int(t_end)
        self._stamp_ops(t_end)
        self.diag.update({"n_cross_sections": int(n), "node_dim": int(Xcs.shape[-1]), "stage_reading": self.stage_reading, "stage_delta": self.stage_delta})

    def _score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        rows = self._rows_for(X)
        bad = ~np.isfinite(X[:, -1, :]).all(-1)                          # normaliser warm-up -> abstain
        if rows is not None and self.panel is not None:
            ts, inv = np.unique(rows["t"], return_inverse=True)
            Xcs, A = self._cross_sections(ts)
            S = self._node_scores(Xcs, A)                                 # (U, N)
            s = S[inv, rows["asset"]]
        else:
            Xn, A = self._single_node(X)
            s = self._node_scores(Xn, A)[:, 0]
        s = s.astype(np.float64)
        s[bad] = np.nan
        return s

    def _direction(self, X: np.ndarray, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        R = np.nan_to_num(self._raw_returns(X))
        p = np.cumsum(R, 1)
        d = np.sign(p[:, -self.ma_short :].mean(1) - p[:, -self.ma_long :].mean(1))
        return d * self.direction_sign

    # ------------------------------------------------------------------ Eq. 20 readings
    def threshold_for_reading(self, reading: str, s_train=None, s_val=None, s_test=None) -> float:
        assert reading in READINGS, reading
        fin = lambda a: np.asarray(a, dtype=np.float64)[np.isfinite(np.asarray(a, dtype=np.float64))] if a is not None else np.zeros(0)
        s_train = self._s_train if s_train is None else s_train
        s_val = self._s_val if s_val is None else s_val
        s_test = self._s_test if s_test is None else s_test
        pool = {"full_sample": np.concatenate([fin(s_train), fin(s_val), fin(s_test)]), "test_fold": fin(s_test), "train_fold": fin(s_train)}[reading]
        if pool.size < 2:
            return float("inf")
        return float(pool.mean() + self.stage_delta * pool.std())

    def thresholds_by_reading(self) -> Dict[str, float]:
        return {r: self.threshold_for_reading(r) for r in READINGS}

    def _published_threshold(self, s_train, s_val, s_test, y_test) -> float:
        self._s_test = np.asarray(s_test, dtype=np.float64)
        rows = self._rows
        self._t_test_end = int(rows["t"].max()) if rows is not None and len(rows["t"]) else self._t_train_end
        thr = self.threshold_for_reading(self.stage_reading, s_train, s_val, s_test)
        self.thr_op.stamp(self._t_train_end if self.stage_reading == "train_fold" else self._t_test_end)
        self.diag["stage_thresholds"] = {r: self.threshold_for_reading(r, s_train, s_val, s_test) for r in READINGS}
        return thr

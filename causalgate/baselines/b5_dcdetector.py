"""B5 DCdetector (Yang, Zhang, Zhou, Wen, Sun, KDD 2023) as a score -> alarm -> trade baseline.

Official code: https://github.com/DAMO-DI-ML/KDD2023-DCdetector  (model/DCdetector.py, model/attn.py, solver.py,
data_factory/data_loader.py, metrics/metrics.py, main.py). It is a fork of the Anomaly-Transformer code base: same
data loader (StandardScaler on the train file, val = test), same solver skeleton (threshold = percentile of the
combined train + test energy, point-adjust) with the model replaced by a dual-attention contrastive structure and the
reconstruction term dropped. The repository needs einops + tkinter and cannot be vendored here, so the model is
re-implemented compactly (status 'reimplemented-fallback'); when third_party/KDD2023-DCdetector/ is importable the
official `DCdetector` module is used (status 'official').

Model (model/DCdetector.py, model/attn.py; line numbers ~ of the 2026-10-03 fetch):
    instance normalisation of each window (RevIN inside forward, DCdetector.py ~L62-65)
    for every patch size p (win_size must be divisible by p):
        patch-wise view   'b m (n p) -> (b m) n p'  (N = L / p tokens of p consecutive values, ~L74)
        in-patch view     'b m (p n) -> (b m) p n'  (p tokens of N consecutive values, ~L76 - note: a block split, not a transpose)
        each view -> DataEmbedding -> softmax self-attention (attn.py ~L24-36) -> upsampled to L x L by repetition
        (~L38-40) and averaged over channels (~L41-42); e_layers independent attention layers read the SAME embeddings
Training loss (solver.py ~L168): loss = prior_loss - series_loss with the stop-gradient KL pairs of the Anomaly
Transformer (series = patch-wise, prior = in-patch). Score (solver.py ~L223, ~L255, ~L290):
    softmax_t( -(sum_u KL(series_u, prior_u.detach()) + KL(prior_u, series_u.detach())) * temperature )  (no reconstruction)
Both the sign of the loss and the sign inside the softmax are kept exactly as in the official code
(cfg['model']['variant'] = 'paper' switches to the paper's additive loss and the raw discrepancy as score).
"""
from __future__ import annotations
import importlib
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .common import ScoreToTradeBaseline, register
from ..data.pit import AsOfStamped

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party")


def _import_official(repo: str, module: str, attr: str) -> Tuple[Optional[Any], str]:
    root = os.path.join(THIRD_PARTY, repo)
    if not os.path.isdir(root):
        return None, f"third_party/{repo} not found"
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr), "ok"
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


class _ModelStamp(AsOfStamped):
    pass


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """solver.py ~L15 my_kl_loss: sum over keys, mean over heads -> (B, L)."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def _norm(p: torch.Tensor) -> torch.Tensor:
    return p / torch.sum(p, dim=-1, keepdim=True)


def _sinusoid(max_len: int, d: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
    return pe


# --------------------------------------------------------------------------------------- compact model
class _DualAttention(nn.Module):
    """attn.py AttentionLayer + DAC_structure: shared q/k projections for both views (patch_query_projection /
    patch_key_projection are reused, attn.py AttentionLayer.forward), softmax attention per view, upsampling to L x L
    (repeat for the patch-wise view, tiling for the in-patch view) and mean over channels."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.H, self.E = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, self.H * self.E)
        self.k = nn.Linear(d_model, self.H * self.E)
        self.drop = nn.Dropout(dropout)

    def _attn(self, e: torch.Tensor) -> torch.Tensor:
        B, L, _ = e.shape
        q = self.q(e).view(B, L, self.H, self.E)
        k = self.k(e).view(B, L, self.H, self.E)
        scores = torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(self.E)
        return self.drop(torch.softmax(scores, dim=-1))                       # (B, H, L, L)

    def forward(self, e_ps: torch.Tensor, e_pn: torch.Tensor, B: int, M: int, p: int, N: int):
        a_ps = self._attn(e_ps)                                               # (B*M, H, N, N)
        a_pn = self._attn(e_pn)                                               # (B*M, H, p, p)
        a_ps = a_ps.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)   # attn.py ~L39 repeat -> (B*M, H, L, L)
        a_pn = a_pn.repeat(1, 1, N, N)                                        # attn.py ~L40 tiling  -> (B*M, H, L, L)
        L = p * N
        series = a_ps.view(B, M, self.H, L, L).mean(1)                        # attn.py ~L41 mean over channels
        prior = a_pn.view(B, M, self.H, L, L).mean(1)                         # attn.py ~L42
        return series, prior


class DCdetectorNet(nn.Module):
    def __init__(self, win_size: int, enc_in: int, patch_sizes: List[int], d_model: int = 32, n_heads: int = 1, e_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.L, self.M, self.patch_sizes = int(win_size), int(enc_in), [int(p) for p in patch_sizes]
        for p in self.patch_sizes:
            assert self.L % p == 0, f"win_size {self.L} must be divisible by patch size {p}"
        self.emb_ps = nn.ModuleList([nn.Linear(p, d_model) for p in self.patch_sizes])
        self.emb_pn = nn.ModuleList([nn.Linear(self.L // p, d_model) for p in self.patch_sizes])
        self.register_buffer("pos", _sinusoid(self.L, d_model))
        self.layers = nn.ModuleList([_DualAttention(d_model, n_heads, dropout) for _ in range(e_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, L, M = x.shape
        mu = x.mean(1, keepdim=True); sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - mu) / sd                                                     # RevIN 'norm' (affine params at identity)
        xt = x.permute(0, 2, 1).reshape(B * M, L)
        series_l: List[torch.Tensor] = []
        prior_l: List[torch.Tensor] = []
        for pi, p in enumerate(self.patch_sizes):
            N = L // p
            e_ps = self.drop(self.emb_ps[pi](xt.reshape(B * M, N, p)) + self.pos[:N])     # 'b m (n p) -> (b m) n p'
            e_pn = self.drop(self.emb_pn[pi](xt.reshape(B * M, p, N)) + self.pos[:p])     # 'b m (p n) -> (b m) p n'
            for layer in self.layers:
                s, pr = layer(e_ps, e_pn, B, M, p, N)
                series_l.append(s); prior_l.append(pr)
        return series_l, prior_l


# --------------------------------------------------------------------------------------- baseline
@register("cg_dcdetector")
class DCdetectorBaseline(ScoreToTradeBaseline):
    name = "cg_dcdetector"
    paper = "Yang, Zhang, Zhou, Wen, Sun. DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection. KDD 2023"
    official_repo = "https://github.com/DAMO-DI-ML/KDD2023-DCdetector"
    status = "reimplemented-fallback"
    year = 2023
    cutoff_date = None
    conventions = {
        "L1": "official: StandardScaler fitted on the TRAIN file, applied to train and test; val = test (data_factory/data_loader.py "
              "PSMSegLoader ~L21-34, same code as Anomaly-Transformer) + per-window instance normalisation (RevIN) inside the model "
              "(model/DCdetector.py ~L62-65). causal: StreamingNormaliser + the same per-window RevIN (window-internal, F_t-measurable).",
        "L2": "official: thresh = np.percentile(combined train+test energy, 100 - anormly_ratio) (solver.py test() ~L261-262); "
              "anormly_ratio per dataset (main.py default 4.0 %; scripts 0.5-1 %). causal: delayed-ACI gate.",
        "L3": "official: early stopping on the TEST loader (solver.py train() ~L180 `self.vali(self.test_loader)`); anomaly ratio = known "
              "test ratio; batch size chosen from the data length (main.py find_nearest). causal: fixed epochs.",
        "L4": "official: point-adjust before P/R/F1 (solver.py ~L309-329) and, in addition, affiliation / VUS / PA-F1 through "
              "metrics.combine_all_evaluation_scores (~L304; called with (pred, gt, energy) - note the argument order vs its signature "
              "(y_test, pred_labels, scores)). causal: none.",
        "L5": "not applicable.",
        "L6": "not applicable.",
        "L7": "official: detection only; every point of non-overlapping test windows is scored. Here: sliding window ending at t, score = "
              "softmax energy at the last position; fill at t + latency.",
        "L8": "none.",
    }
    unspecified = [
        "DataEmbedding: Linear + sinusoidal positions instead of the official circular Conv1d token embedding (model/embed.py).",
        "Patch sizes: official scripts use [3,5,7] with win_size 105 (SMAP/SWaT; other datasets other pairs, e.g. MSL [3,5] @ 90), main.py default win_size 100 with [5]; here cfg['model']['patch_sizes'] and the window must be "
        "divisible by each (non-divisors are dropped with a diag note).",
        "e_layers attention layers all read the same patch embeddings (as the official Encoder loops over attn_layers without stacking).",
        "Loss sign: official code minimises prior_loss - series_loss (solver.py ~L168) although the paper's Eq. adds the two stop-gradient "
        "KL terms; the official sign is the default (variant='official_code'); variant='paper' uses the sum and scores + discrepancy.",
        "d_model/n_heads/e_layers from cfg['model'] (official 256/1/3, batch 128, lr 1e-4, 10 epochs); no early stopping; lr halving per epoch kept.",
        "Score position: last position of the window (official scores all points of non-overlapping windows).",
        "Score = softmax(-discrepancy * 50)[last] ^ (1/50) computed in log space: a positive monotone transform of the official metric "
        "(same ranking / quantiles) that avoids float32 underflow of the temperature-50 softmax at the window edge.",
        "Capacity matters for the official-code sign: with d_model 16 / 1 layer / 1 epoch the official-sign score ranked the planted "
        "anomalies below normal bars (AUC_ROC 0.33) while d_model 32 / 2 layers / 3 epochs gave 0.63; the smoke config uses the latter sizes.",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons, device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "combined_quantile")
        cfg.setdefault("published_norm", "train")
        super().__init__(cfg, dims, horizons, device)
        m = dict(cfg.get("model", {}))
        self.d_model = int(m.get("d_model", 32)); self.n_heads = int(m.get("n_heads", 1)); self.e_layers = int(m.get("e_layers", 2))
        self.dropout = float(m.get("dropout", 0.0)); self.temperature = float(m.get("temperature", 50.0))
        self.variant = str(m.get("variant", "official_code")); assert self.variant in ("official_code", "paper"), self.variant
        self.epochs = int(m.get("epochs", 3)); self.lr = float(m.get("lr", 1e-4)); self.bs = int(m.get("batch_size", 128))
        self.seed = int(m.get("seed", 0)); self.lr_decay = bool(m.get("lr_decay", True))
        ps = [int(p) for p in m.get("patch_sizes", [4, 8])]
        self.patch_sizes = [p for p in ps if self.L % p == 0] or [max(1, self.L // 8)]
        if self.patch_sizes != ps:
            self.diag["patch_sizes_dropped"] = [p for p in ps if p not in self.patch_sizes]
        self.net: Optional[nn.Module] = None
        self.official_cls, reason = (_import_official("KDD2023-DCdetector", "model.DCdetector", "DCdetector")
                                     if m.get("use_official", True) else (None, "disabled by cfg"))
        self.status = "reimplemented-fallback"
        self.diag["official_import"] = reason
        self._stamp = _ModelStamp()
        self.guard.register(self._stamp)

    def _build(self, n_feat: int) -> nn.Module:
        if self.official_cls is not None:
            try:
                net = self.official_cls(win_size=self.L, enc_in=n_feat, c_out=n_feat, n_heads=self.n_heads, d_model=self.d_model,
                                        e_layers=self.e_layers, patch_size=list(self.patch_sizes), channel=n_feat, dropout=self.dropout)
                self.status = "official"
                return net.to(self.device)
            except Exception as e:                           # noqa: BLE001
                self.diag["official_import"] = f"official DCdetector failed to build: {type(e).__name__}: {e}"
        self.status = "reimplemented-fallback"
        return DCdetectorNet(self.L, n_feat, self.patch_sizes, self.d_model, self.n_heads, self.e_layers, self.dropout).to(self.device)

    def _tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.nan_to_num(torch.as_tensor(np.asarray(X, dtype=np.float32)), nan=0.0, posinf=0.0, neginf=0.0).to(self.device)

    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        torch.manual_seed(self.seed)
        x = self._tensor(X)
        n, L, Fd = x.shape
        self.net = self._build(Fd)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        last = float("nan")
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.bs):
                xb = x[perm[i : i + self.bs]]
                series, prior = self.net(xb)
                series_loss = 0.0; prior_loss = 0.0
                for u in range(len(prior)):
                    pn = _norm(prior[u])
                    series_loss = series_loss + torch.mean(_kl(series[u], pn.detach())) + torch.mean(_kl(pn.detach(), series[u]))
                    prior_loss = prior_loss + torch.mean(_kl(pn, series[u].detach())) + torch.mean(_kl(series[u].detach(), pn))
                series_loss = series_loss / len(prior); prior_loss = prior_loss / len(prior)
                loss = (prior_loss - series_loss) if self.variant == "official_code" else (prior_loss + series_loss)   # solver.py ~L168
                opt.zero_grad(); loss.backward(); opt.step()
                last = float(loss.item())
            if self.lr_decay:
                for pg in opt.param_groups:
                    pg["lr"] = self.lr * (0.5 ** (ep + 1))
        self.net.eval()
        self._stamp.stamp(int(t_end))
        self.diag.update({"status": self.status, "train_loss": last, "epochs": self.epochs, "patch_sizes": list(self.patch_sizes)})

    @torch.no_grad()
    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        series, prior = self.net(x)
        sl = 0.0; pl = 0.0
        for u in range(len(prior)):
            pn = _norm(prior[u])
            sl = sl + _kl(series[u], pn.detach()) * self.temperature
            pl = pl + _kl(pn, series[u].detach()) * self.temperature
        if self.variant == "paper":
            return (sl + pl) / self.temperature
        return torch.exp(torch.log_softmax(-sl - pl, dim=-1) / self.temperature)   # solver.py ~L290 metric ^ (1/T), log-space (no underflow)

    def _score(self, X: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float32)
        if len(Xn) == 0:
            return np.zeros(0)
        self.net.eval()
        x = self._tensor(Xn)
        out = [self._energy(x[i : i + 512])[:, -1].cpu().numpy() for i in range(0, len(x), 512)]
        s = np.concatenate(out).astype(np.float64)
        s[np.isnan(Xn[:, -1]).any(-1)] = np.nan
        return s

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._stamp.check(int(batch["t"].min()))
        return super().predict(batch)

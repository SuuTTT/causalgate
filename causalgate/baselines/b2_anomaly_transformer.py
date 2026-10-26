"""B2 Anomaly Transformer (Xu, Wu, Wang & Long, ICLR 2022) as a score -> alarm -> trade baseline.

Official code: https://github.com/thuml/Anomaly-Transformer  (model/AnomalyTransformer.py, model/attn.py, solver.py,
data_factory/data_loader.py, main.py). The repository could not be cloned in the build session (its files were read through the GitHub raw endpoints only), so this
file carries a compact re-implementation of the *model* (prior-association = Gaussian kernel with a learned sigma per
position and head, series-association = softmax self-attention, minimax association-discrepancy training) and
*replicates solver.py* for the training loop and the association-based criterion. When
third_party/Anomaly-Transformer/ is present the official `AnomalyTransformer` module is used instead (status
'official'); note that the official attn.py calls `.cuda()` unconditionally (attn.py:29, :51), so the official module
is only usable on a CUDA machine - on CPU the fallback is used and reported.

What the scaffold (common.ScoreToTradeBaseline) owns: normalisation (L1), threshold (L2/L3), gate, out dict, trading
rule. What this file owns: the scoring function s(window) and the convention sheet (`conventions`, `unspecified`).

Scoring (solver.py test(), lines ~212-331 of the 2026-10-03 fetch; see conventions_b2_b7.md):
    energy_t = softmax_t( -(sum_u KL(series_u, prior_u.detach()) + KL(prior_u, series_u.detach())) * temperature )
               * mean_f (x_t - x_hat_t)^2
    The official code scores every point of non-overlapping test windows; here every window ends at the decision bar
    t and the score is energy at the last position, which keeps the operator F_t-measurable (the softmax only sees
    positions <= t).
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
import torch.nn.functional as F

from .common import ScoreToTradeBaseline, register
from ..data.pit import AsOfStamped

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party")


def _import_official(repo: str, module: str, attr: str) -> Tuple[Optional[Any], str]:
    """Import `attr` from `module` inside third_party/<repo>/ (see THIRD_PARTY.md). Returns (obj or None, reason)."""
    root = os.path.join(THIRD_PARTY, repo)
    if not os.path.isdir(root):
        return None, f"third_party/{repo} not found"
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr), "ok"
    except Exception as e:                                   # noqa: BLE001  (any import-time failure -> fallback)
        return None, f"{type(e).__name__}: {e}"


class _ModelStamp(AsOfStamped):
    """As-of stamp of the fitted scoring model, registered with the baseline's Filtration Guard."""


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """solver.py:12 my_kl_loss: sum over keys, mean over heads -> (B, L)."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def _norm_prior(prior: torch.Tensor) -> torch.Tensor:
    return prior / torch.sum(prior, dim=-1, keepdim=True)


def _sinusoid(max_len: int, d: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
    return pe


# --------------------------------------------------------------------------------------- compact model
class _AnomalyAttention(nn.Module):
    """attn.py AnomalyAttention + AttentionLayer: series = softmax(QK^T / sqrt(E)); prior = Gaussian(|i - j|; sigma_i)
    with sigma = 3^(sigmoid(5 * w x_i) + 1e-5) - 1 (attn.py:48-52). mask_flag=False as in the official model
    (AnomalyTransformer.py:68 passes False): full attention inside the window."""

    def __init__(self, L: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.H, self.E = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, self.H * self.E)
        self.k = nn.Linear(d_model, self.H * self.E)
        self.v = nn.Linear(d_model, self.H * self.E)
        self.sigma = nn.Linear(d_model, self.H)
        self.out = nn.Linear(self.H * self.E, d_model)
        self.drop = nn.Dropout(dropout)
        idx = torch.arange(L)
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs().float())

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        H, E = self.H, self.E
        q = self.q(x).view(B, L, H, E)
        k = self.k(x).view(B, L, H, E)
        v = self.v(x).view(B, L, H, E)
        scores = torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(E)
        series = self.drop(torch.softmax(scores, dim=-1))                                  # (B, H, L, L)
        sig = self.sigma(x).transpose(1, 2)                                                # (B, H, L)
        sig = torch.sigmoid(sig * 5) + 1e-5
        sig = torch.pow(3, sig) - 1
        sig = sig.unsqueeze(-1)                                                            # (B, H, L, 1)
        d = self.dist[:L, :L]
        prior = 1.0 / (math.sqrt(2 * math.pi) * sig) * torch.exp(-d ** 2 / 2 / (sig ** 2))   # (B, H, L, L)
        out = torch.einsum("bhls,bshd->blhd", series, v).reshape(B, L, H * E)
        return self.out(out), series, prior


class _EncoderLayer(nn.Module):
    def __init__(self, L: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = _AnomalyAttention(L, d_model, n_heads, dropout)
        self.ff1 = nn.Linear(d_model, d_ff)                 # official: Conv1d(kernel_size=1) == position-wise Linear
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        new_x, series, prior = self.attn(x)
        x = self.norm1(x + self.drop(new_x))
        y = self.drop(self.ff2(self.drop(F.gelu(self.ff1(x)))))
        return self.norm2(x + y), series, prior


class AnomalyTransformerNet(nn.Module):
    """Compact AnomalyTransformer: Linear token embedding + sinusoidal positions (official: circular Conv1d k=3 token
    embedding + sinusoidal positions), e_layers encoder layers, LayerNorm, Linear projection back to the inputs."""

    def __init__(self, win_size: int, enc_in: int, d_model: int = 32, n_heads: int = 4, e_layers: int = 2, d_ff: int = 64, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(enc_in, d_model)
        self.register_buffer("pos", _sinusoid(win_size, d_model))
        self.layers = nn.ModuleList([_EncoderLayer(win_size, d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, enc_in)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        L = x.shape[1]
        h = self.drop(self.embed(x) + self.pos[:L])
        series_l: List[torch.Tensor] = []
        prior_l: List[torch.Tensor] = []
        for layer in self.layers:
            h, s, p = layer(h)
            series_l.append(s)
            prior_l.append(p)
        return self.proj(self.norm(h)), series_l, prior_l


# --------------------------------------------------------------------------------------- baseline
@register("cg_anomaly_transformer")
class AnomalyTransformerBaseline(ScoreToTradeBaseline):
    name = "cg_anomaly_transformer"
    paper = "Xu, Wu, Wang, Long. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR 2022"
    official_repo = "https://github.com/thuml/Anomaly-Transformer"
    status = "reimplemented-fallback"
    year = 2022
    cutoff_date = None
    conventions = {
        "L1": "official: sklearn StandardScaler fitted on the TRAIN file and applied to train and test (data_factory/data_loader.py PSMSegLoader "
              "~L21-34: `self.scaler.fit(data)` on train.csv, `self.test = self.scaler.transform(test_data)`); the 'val' split IS the test "
              "set (~L37 `self.val = self.test`). causal: StreamingNormaliser (scaffold).",
        "L2": "official: threshold = np.percentile(combined train+test energy, 100 - anormly_ratio) (solver.py test() ~L287-288), i.e. a "
              "quantile that reads the test scores; anormly_ratio is a per-dataset hyper-parameter (main.py default 4.0 %; scripts 0.5-1 %). "
              "causal: delayed-ACI gate calibrated on the validation fold (scaffold).",
        "L3": "official: early stopping / checkpoint selection on the TEST loader (solver.py train() ~L196 `self.vali(self.test_loader)`) and "
              "anormly_ratio set to the known test anomaly ratio. causal: fixed epochs, no test access.",
        "L4": "official: point-adjust before precision/recall/F1 (solver.py test() ~L338-358, comment cites issue #14). causal: none "
              "(event-wise metrics by causalgate.eval).",
        "L5": "not applicable (no text / late labels in the official benchmarks); the protocol runner decides the join.",
        "L6": "not applicable (single multivariate series per dataset).",
        "L7": "official: detection-only, no execution; scores every point of non-overlapping test windows (thre loader). Here: window ends at t, "
              "score = energy at the last position, fill at t + latency by the protocol runner.",
        "L8": "none (trained from scratch per dataset).",
    }
    unspecified = [
        "Token embedding: Linear instead of the official circular Conv1d(k=3) TokenEmbedding (model/embed.py); positions sinusoidal as official.",
        "Sizes: d_model/n_heads/e_layers/d_ff from cfg['model'] (official 512/8/3/512, win_size 100, batch 1024, lr 1e-4, 10 epochs, k=3); "
        "the smoke config uses 16/2/1/32 and 1 epoch.",
        "Minimax step: official calls loss1.backward(retain_graph=True); loss2.backward(); step  (solver.py ~L188-190); here (loss1 + loss2).backward() "
        "which yields identical gradients.",
        "Score position: official scores all L points of each non-overlapping window; here the energy of the last position of a sliding window "
        "ending at the decision bar (the only F_t-measurable choice).",
        "No early stopping (official: patience 3 on the test loader); learning-rate halving per epoch (adjust_learning_rate) kept.",
        "Reconstruction criterion per point = mean over features of squared error (nn.MSELoss(reduce=False) then mean(-1), solver.py ~L221).",
        "Score = (official energy)^(1/temperature) computed in log space, exp((log softmax(-dis * 50)[last] + log rec_err[last]) / 50): "
        "a positive monotone transform (same ranking / quantiles) that avoids float32 underflow at the window edge, where the one-sided "
        "prior makes the discrepancy systematically larger (measured: mean last-position weight ~3e-14, ~59 % of windows exactly 0).",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons, device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "combined_quantile")
        cfg.setdefault("published_norm", "train")
        super().__init__(cfg, dims, horizons, device)
        m = dict(cfg.get("model", {}))
        self.d_model = int(m.get("d_model", 32)); self.n_heads = int(m.get("n_heads", 4)); self.e_layers = int(m.get("e_layers", 2))
        self.d_ff = int(m.get("d_ff", 64)); self.dropout = float(m.get("dropout", 0.0))
        self.k = float(m.get("k", 3.0)); self.temperature = float(m.get("temperature", 50.0))
        self.epochs = int(m.get("epochs", 3)); self.lr = float(m.get("lr", 1e-4)); self.bs = int(m.get("batch_size", 256))
        self.seed = int(m.get("seed", 0)); self.lr_decay = bool(m.get("lr_decay", True))
        self.net: Optional[nn.Module] = None
        self.official_cls, reason = (_import_official("Anomaly-Transformer", "model.AnomalyTransformer", "AnomalyTransformer")
                                     if m.get("use_official", True) else (None, "disabled by cfg"))
        self.status = "reimplemented-fallback"
        self.diag["official_import"] = reason
        self._stamp = _ModelStamp()
        self.guard.register(self._stamp)

    # ---- model
    def _build(self, n_feat: int) -> nn.Module:
        if self.official_cls is not None and torch.cuda.is_available():
            net = self.official_cls(win_size=self.L, enc_in=n_feat, c_out=n_feat, d_model=self.d_model, n_heads=self.n_heads,
                                    e_layers=self.e_layers, d_ff=self.d_ff, dropout=self.dropout, output_attention=True)
            self.status = "official"
        else:
            if self.official_cls is not None:
                self.diag["official_import"] = "found but attn.py requires CUDA (.cuda() calls); CPU fallback used"
            net = AnomalyTransformerNet(self.L, n_feat, self.d_model, self.n_heads, self.e_layers, self.d_ff, self.dropout)
            self.status = "reimplemented-fallback"
        return net.to(self.device)

    def _forward(self, x: torch.Tensor):
        out = self.net(x)
        rec, series, prior = out[0], out[1], out[2]                        # official returns (enc_out, series, prior, sigmas)
        return rec, series, prior

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
                rec, series, prior = self._forward(xb)
                series_loss = 0.0; prior_loss = 0.0
                for u in range(len(prior)):
                    pn = _norm_prior(prior[u])
                    series_loss = series_loss + torch.mean(_kl(series[u], pn.detach())) + torch.mean(_kl(pn.detach(), series[u]))
                    prior_loss = prior_loss + torch.mean(_kl(pn, series[u].detach())) + torch.mean(_kl(series[u].detach(), pn))
                series_loss = series_loss / len(prior); prior_loss = prior_loss / len(prior)
                rec_loss = F.mse_loss(rec, xb)
                loss1 = rec_loss - self.k * series_loss                     # solver.py ~L178
                loss2 = rec_loss + self.k * prior_loss                      # solver.py ~L179
                opt.zero_grad()
                (loss1 + loss2).backward()                                   # == loss1.backward(retain_graph) + loss2.backward()
                opt.step()
                last = float(loss1.item())
            if self.lr_decay:                                                # solver.py adjust_learning_rate: lr * 0.5 ** epoch
                for pg in opt.param_groups:
                    pg["lr"] = self.lr * (0.5 ** (ep + 1))
        self.net.eval()
        self._stamp.stamp(int(t_end))
        self.diag.update({"status": self.status, "train_loss1": last, "epochs": self.epochs})

    @torch.no_grad()
    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        """Official energy  softmax_t(-(series_loss + prior_loss) * temperature) * rec_err_t  (solver.py ~L244-245), reported as
        its temperature-th root computed in log space: exp((log_softmax + log rec_err) / temperature). A positive, monotone
        transform (identical ranking and quantile thresholds) that avoids the float32 underflow of exp(-50 * discrepancy): the
        window edges carry a systematically larger discrepancy (one-sided Gaussian prior), so the last-position softmax weight
        is ~1e-14 on average and underflows to exactly 0 for about half of the windows."""
        rec, series, prior = self._forward(x)
        loss = torch.mean((x - rec) ** 2, dim=-1)                            # (B, L) per-point reconstruction error
        sl = 0.0; pl = 0.0
        for u in range(len(prior)):
            pn = _norm_prior(prior[u])
            sl = sl + _kl(series[u], pn.detach()) * self.temperature
            pl = pl + _kl(pn, series[u].detach()) * self.temperature
        log_metric = torch.log_softmax(-sl - pl, dim=-1)                     # solver.py ~L244 metric, in log space
        return torch.exp((log_metric + torch.log(loss + 1e-12)) / self.temperature)

    def _score(self, X: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float32)
        if len(Xn) == 0:
            return np.zeros(0)
        self.net.eval()
        x = self._tensor(Xn)
        out = [self._energy(x[i : i + 1024])[:, -1].cpu().numpy() for i in range(0, len(x), 1024)]
        s = np.concatenate(out).astype(np.float64)
        s[np.isnan(Xn[:, -1]).any(-1)] = np.nan                               # warm-up abstention propagates
        return s

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._stamp.check(int(batch["t"].min()))
        return super().predict(batch)

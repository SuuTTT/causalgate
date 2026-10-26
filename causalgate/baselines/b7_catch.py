"""B7 CATCH - Channel-Aware multivariate Time series anomaly detection via frequency patCHing (Wu, Qiu, Li, Wang, Hu,
Guo, Xiong, Yang, ICLR 2025) as a score -> alarm -> trade baseline.

Official code: https://github.com/decisionintelligence/CATCH (branch master; ts_benchmark/baselines/catch/CATCH.py,
models/CATCH_model.py, layers/channel_mask.py, layers/cross_channel_Transformer.py, layers/RevIN.py; evaluation in
ts_benchmark/evaluation/strategy/anomaly_detect.py and evaluation/metrics/*). The benchmark package (TFB) cannot be
vendored here, so this file carries a compact re-implementation (status 'reimplemented-fallback'); with
third_party/CATCH/ on the path the official `CATCHModel` is used (status 'official').

Official model (CATCH_model.py forward): RevIN 'norm' -> torch.fft.fft along time per channel -> real and imaginary
parts patched with unfold(patch_size, patch_stride) along the frequency axis -> per (window, patch) the channels are the
tokens: z_cat = [real patch, imag patch] (2 * patch_size) -> channel_mask_generator (channel_mask.py: zero-initialised
Linear(2p, n_vars) + sigmoid, Bernoulli via hard Gumbel-softmax, diagonal forced to 1) -> Trans_C frequency transformer
with the mask (bi-level / channel-fusion attention, returns a 'dcloss' regulariser) -> two heads (real / imag, flattened
patches -> seq_len) -> torch.complex -> torch.fft.ifft -> Linear(2 seq_len -> seq_len) on [real, imag] -> RevIN 'denorm'.
Training loss (CATCH.py detect_fit): MSE(rec, x) + dc_lambda * dcloss + auxi_lambda * frequency_loss(complex_z, RevIN(x)).
Score (CATCH.py detect_label / detect_score): mean over channels of the per-point MSE + score_lambda * frequency_criterion.

Compact fallback (differences listed in `unspecified`): the same FFT -> real/imag patching -> channel tokens -> mask
generator -> masked multi-head channel attention (mask applied multiplicatively to the attention weights, re-normalised)
-> real/imag heads -> iFFT -> Linear -> denorm. dcloss and the frequency_criterion score term are omitted; the frequency
auxiliary loss is an MAE on real/imag of the spectrum; the mask is deterministic (p > 0.5) at scoring time.
"""
from __future__ import annotations
import importlib
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# --------------------------------------------------------------------------------------- compact model
class _ChannelMaskGenerator(nn.Module):
    """channel_mask.py channel_mask_generator: sigmoid(Linear(2p -> n_vars, no bias, zero init)) -> Bernoulli sample
    (hard Gumbel-softmax while training; p > 0.5 at eval) with the diagonal forced to 1."""

    def __init__(self, input_size: int, n_vars: int):
        super().__init__()
        self.gen = nn.Linear(input_size * 2, n_vars, bias=False)
        with torch.no_grad():
            self.gen.weight.zero_()
        self.n_vars = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:                      # x: (Bp, C, 2p)
        prob = torch.sigmoid(self.gen(x)).clamp(1e-4, 1 - 1e-4)              # (Bp, C, C)
        if self.training:
            logits = torch.stack([torch.log(prob / (1 - prob)), torch.log((1 - prob) / prob)], -1)
            sample = F.gumbel_softmax(logits, hard=True)[..., 0]
        else:
            sample = (prob > 0.5).float()
        eye = torch.eye(self.n_vars, device=x.device)
        return sample * (1 - eye) + eye


class _MaskedChannelAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.H, self.E = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * self.H * self.E)
        self.out = nn.Linear(self.H * self.E, d)
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x (Bp, C, d), mask (Bp, C, C)
        Bp, C, _ = x.shape
        q, k, v = self.qkv(x).view(Bp, C, 3, self.H, self.E).unbind(2)
        scores = torch.einsum("bche,bshe->bhcs", q, k) / math.sqrt(self.E)
        attn = torch.softmax(scores, -1) * mask[:, None]
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        h = torch.einsum("bhcs,bshe->bche", self.drop(attn), v).reshape(Bp, C, self.H * self.E)
        x = self.norm1(x + self.drop(self.out(h)))
        return self.norm2(x + self.drop(self.ff(x)))


class CatchNet(nn.Module):
    def __init__(self, c_in: int, seq_len: int, patch_size: int = 8, patch_stride: int = 4, d_model: int = 32, n_heads: int = 2,
                 e_layers: int = 2, d_ff: int = 64, dropout: float = 0.0, head_dropout: float = 0.0):
        super().__init__()
        self.C, self.L, self.p, self.s = int(c_in), int(seq_len), int(patch_size), int(patch_stride)
        assert self.p <= self.L, "patch_size must be <= seq_len"
        self.patch_num = (self.L - self.p) // self.s + 1
        self.tok = nn.Linear(2 * self.p, d_model)
        self.mask_gen = _ChannelMaskGenerator(self.p, self.C)
        self.layers = nn.ModuleList([_MaskedChannelAttention(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.get_r = nn.Linear(d_model, d_model); self.get_i = nn.Linear(d_model, d_model)
        nf = d_model * self.patch_num
        self.head_r = nn.Sequential(nn.Flatten(-2), nn.Linear(nf, self.L), nn.Dropout(head_dropout))
        self.head_i = nn.Sequential(nn.Flatten(-2), nn.Linear(nf, self.L), nn.Dropout(head_dropout))
        self.ircom = nn.Linear(2 * self.L, self.L)

    def forward(self, z: torch.Tensor):                                       # z: (B, L, C)
        B, L, C = z.shape
        mu = z.mean(1, keepdim=True); sd = torch.sqrt(z.var(1, keepdim=True, unbiased=False) + 1e-5)
        zn = ((z - mu) / sd).permute(0, 2, 1)                                 # RevIN 'norm' -> (B, C, L)
        spec = torch.fft.fft(zn)                                              # CATCH_model.py: torch.fft.fft(z)
        z1 = spec.real.unfold(-1, self.p, self.s)                             # (B, C, P, p)
        z2 = spec.imag.unfold(-1, self.p, self.s)
        P = z1.shape[2]
        z_cat = torch.cat((z1, z2), -1).permute(0, 2, 1, 3).reshape(B * P, C, 2 * self.p)
        mask = self.mask_gen(z_cat)                                           # (B*P, C, C)
        h = self.tok(z_cat)
        for layer in self.layers:
            h = layer(h, mask)
        hr = self.get_r(h).view(B, P, C, -1).permute(0, 2, 1, 3)              # (B, C, P, d)
        hi = self.get_i(h).view(B, P, C, -1).permute(0, 2, 1, 3)
        zr = self.head_r(hr); zi = self.head_i(hi)                            # (B, C, L)
        complex_z = torch.complex(zr, zi)
        zt = torch.fft.ifft(complex_z)
        rec = self.ircom(torch.cat((zt.real, zt.imag), -1))                   # (B, C, L)
        rec = rec.permute(0, 2, 1) * sd + mu                                  # RevIN 'denorm' -> (B, L, C)
        return rec, complex_z, spec


# --------------------------------------------------------------------------------------- baseline
@register("cg_catch")
class CatchBaseline(ScoreToTradeBaseline):
    name = "cg_catch"
    paper = "Wu, Qiu, Li, Wang, Hu, Guo, Xiong, Yang. CATCH: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching. ICLR 2025"
    official_repo = "https://github.com/decisionintelligence/CATCH"
    status = "reimplemented-fallback"
    year = 2025
    cutoff_date = None
    conventions = {
        "L1": "official: sklearn StandardScaler fitted on the first 80 % of the TRAIN split (CATCH.py detect_fit: `train_val_split(train_data, 0.8)`, "
              "`self.scaler.fit(train_data_value.values)`) and applied to the test split (detect_label / detect_score: "
              "`self.scaler.transform(test.values)`), plus per-window RevIN inside the model (CATCH_model.py forward, revin=1). "
              "causal: StreamingNormaliser + the same window-internal RevIN.",
        "L2": "official (detect_label): threshold = np.percentile(combined train+test energy, 100 - ratio) for EVERY ratio in anomaly_ratio "
              "(default list [0.1, 0.5, 1, 2, 3, 5, 10, 15, 20, 25]; the MSL label script passes 5.0), one prediction per ratio returned; "
              "detect_score returns raw energies for threshold-free metrics. causal: delayed-ACI gate.",
        "L3": "official: validation = last 20 % of train (early stopping, patience 3); the label protocol evaluates each anomaly ratio on the "
              "test labels (one result row per ratio, anomaly_detect.py execute()); which ratio is reported is decided downstream "
              "(report_config aggregate 'mean' - not verified). causal: fixed epochs, single ratio.",
        "L4": "official: metrics 'all' of TFB: label metrics include adjust_precision / adjust_recall / adjust_f_score (point-adjust, "
              "classification_metrics_label.py adjust_predicts) and affiliation_f (reported in unfixed_detect_label_multi_config.json); "
              "score metrics best_f_score / auc_roc / auc_pr / R_AUC / VUS (classification_metrics_score.py, best_* search the test labels). "
              "causal: none.",
        "L5": "not applicable.",
        "L6": "not applicable.",
        "L7": "official: detection only, step-1 sliding windows (anomaly_detection_data_provider step=1), every point scored. Here: window "
              "ending at t, score = per-point energy at the last bar, fill at t + latency.",
        "L8": "none (trained from scratch per dataset).",
    }
    unspecified = [
        "dcloss (the channel-correlation regulariser of Trans_C, cross_channel_Transformer.py) omitted; auxi loss = MAE on real/imag of the "
        "spectrum vs the FFT of the normalised input (official frequency_loss, auxi_type complex, auxi_mode fft, auxi_loss MAE).",
        "Score = per-point temporal MSE only (official adds score_lambda * frequency_criterion, utils/fre_rec_loss.py).",
        "Mask: hard Gumbel-softmax Bernoulli while training as official; deterministic p > 0.5 at scoring (official samples at test time).",
        "Masked attention: mask multiplies the softmax weights (re-normalised); the official Trans_C masking rule was not read.",
        "Heads: one Linear(d * patch_num -> seq_len) per real/imag (official Flatten_Head: three residual ReLU Linear layers then Linear).",
        "One Adam optimiser with two groups (lr for the model, Mlr for the mask generator; official steps the mask optimiser every "
        "min(len(loader)/10, 100) iterations); OneCycleLR schedule not replicated.",
        "Sizes from cfg['model'] (official seq_len 192, patch 16 / stride 8, d_model 128-256, e_layers 3, 3-5 epochs, batch 128).",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons, device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "combined_quantile")
        cfg.setdefault("published_norm", "train")
        super().__init__(cfg, dims, horizons, device)
        m = dict(cfg.get("model", {}))
        self.patch_size = int(m.get("patch_size", 8)); self.patch_stride = int(m.get("patch_stride", 4))
        self.d_model = int(m.get("d_model", 32)); self.n_heads = int(m.get("n_heads", 2)); self.e_layers = int(m.get("e_layers", 2))
        self.d_ff = int(m.get("d_ff", 64)); self.dropout = float(m.get("dropout", 0.0)); self.head_dropout = float(m.get("head_dropout", 0.0))
        self.auxi_lambda = float(m.get("auxi_lambda", 0.005)); self.dc_lambda = float(m.get("dc_lambda", 0.005))
        self.epochs = int(m.get("epochs", 3)); self.lr = float(m.get("lr", 1e-4)); self.mlr = float(m.get("Mlr", 1e-5))
        self.bs = int(m.get("batch_size", 128)); self.seed = int(m.get("seed", 0))
        self.net: Optional[nn.Module] = None
        self.official_cls, reason = (_import_official("CATCH", "ts_benchmark.baselines.catch.models.CATCH_model", "CATCHModel")
                                     if m.get("use_official", True) else (None, "disabled by cfg"))
        self.status = "reimplemented-fallback"
        self.diag["official_import"] = reason
        self._stamp = _ModelStamp()
        self.guard.register(self._stamp)

    def _build(self, n_feat: int) -> nn.Module:
        if self.official_cls is not None:
            try:
                from types import SimpleNamespace
                cfg = SimpleNamespace(c_in=n_feat, affine=0, subtract_last=0, patch_size=self.patch_size, patch_stride=self.patch_stride,
                                      seq_len=self.L, cf_dim=self.d_model, e_layers=self.e_layers, n_heads=self.n_heads, d_ff=self.d_ff,
                                      head_dim=max(self.d_model // self.n_heads, 8), dropout=self.dropout, d_model=self.d_model,
                                      regular_lambda=0.5, temperature=0.07, individual=0, head_dropout=self.head_dropout)
                net = self.official_cls(cfg)
                self.status = "official"
                return net.to(self.device)
            except Exception as e:                           # noqa: BLE001
                self.diag["official_import"] = f"official CATCHModel failed to build: {type(e).__name__}: {e}"
        self.status = "reimplemented-fallback"
        return CatchNet(n_feat, self.L, self.patch_size, self.patch_stride, self.d_model, self.n_heads, self.e_layers, self.d_ff,
                        self.dropout, self.head_dropout).to(self.device)

    def _forward(self, x: torch.Tensor):
        out = self.net(x)
        if self.status == "official":                        # (rec, complex_z (B, L, C), dcloss)
            rec, cz, dcloss = out
            mu = x.mean(1, keepdim=True); sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            spec = torch.fft.fft(((x - mu) / sd).permute(0, 2, 1))
            return rec, cz.permute(0, 2, 1), spec, dcloss
        rec, cz, spec = out
        return rec, cz, spec, torch.zeros((), device=x.device)

    def _tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.nan_to_num(torch.as_tensor(np.asarray(X, dtype=np.float32)), nan=0.0, posinf=0.0, neginf=0.0).to(self.device)

    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        torch.manual_seed(self.seed)
        x = self._tensor(X)
        n, L, Fd = x.shape
        self.net = self._build(Fd)
        mask_params = [p for nme, p in self.net.named_parameters() if "mask_gen" in nme or "mask_generator" in nme]
        main_params = [p for nme, p in self.net.named_parameters() if not ("mask_gen" in nme or "mask_generator" in nme)]
        opt = torch.optim.Adam([{"params": main_params, "lr": self.lr}, {"params": mask_params, "lr": self.mlr}])
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        last = float("nan")
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.bs):
                xb = x[perm[i : i + self.bs]]
                rec, cz, spec, dcloss = self._forward(xb)
                rec_loss = F.mse_loss(rec, xb)
                auxi = (cz.real - spec.real).abs().mean() + (cz.imag - spec.imag).abs().mean()
                loss = rec_loss + self.dc_lambda * dcloss + self.auxi_lambda * auxi
                opt.zero_grad(); loss.backward(); opt.step()
                last = float(loss.item())
        self.net.eval()
        self._stamp.stamp(int(t_end))
        self.diag.update({"status": self.status, "train_loss": last, "epochs": self.epochs, "patch_num": (L - self.patch_size) // self.patch_stride + 1})

    @torch.no_grad()
    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        rec, _, _, _ = self._forward(x)
        return torch.mean((x - rec) ** 2, dim=-1)                              # (B, L) per-point temporal score

    def _score(self, X: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float32)
        if len(Xn) == 0:
            return np.zeros(0)
        self.net.eval()
        x = self._tensor(Xn)
        out = [self._energy(x[i : i + 1024])[:, -1].cpu().numpy() for i in range(0, len(x), 1024)]
        s = np.concatenate(out).astype(np.float64)
        s[np.isnan(Xn[:, -1]).any(-1)] = np.nan
        return s

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._stamp.check(int(batch["t"].min()))
        return super().predict(batch)

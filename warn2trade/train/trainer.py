"""Stage-wise pre-training followed by end-to-end utility fine-tuning through the differentiable event backtest.

Default schedule (cfg.train.stages):
    detect   : detector only            L = L_det(utility-weighted BCE) + b_dir L_dir(sign of peak impact) + b_svdd L_svdd
    extrap   : extrapolator only        L = L_imp + b_liq L_liq + b_peak L_peak      (upstream detached)
    policy   : policy only              L = -U(pnl) + b_to Omega                       (upstream detached)
    e2e      : all modules              L = sum of the above with the policy utility back-propagating into the detector
Ablation `two_stage` skips `e2e`; `no_anomaly_guidance` zeroes the detection loss.
"""
from __future__ import annotations
import copy
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models import losses as Lz
from ..backtest import CostModel, event_pnl, EventBacktester
from ..backtest import metrics as M
from ..utils.logger import get_logger
from ..models.encoders import liq_transform

log = get_logger("trainer")


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")   # MPS is excluded on purpose: torch.use_deterministic_algorithms is not honoured there


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Trainer:
    def __init__(self, cfg: Dict[str, Any], model: torch.nn.Module, loaders: Dict[str, DataLoader], cost_model: CostModel,
                 device: torch.device, bars_per_year: int):
        self.cfg, self.model, self.loaders, self.cost, self.device = cfg, model.to(device), loaders, cost_model, device
        self.horizons = list(cfg["model"]["horizons"])
        self.latency = int(cfg["data"]["latency"])
        self.bars_per_year = bars_per_year
        self.tcfg = cfg["train"]
        self.taus = self.model.extrapolator.taus
        self.ab = getattr(model, "ab", {})

    # ------------------------------------------------------------------ losses
    def _exec_alpha(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        c_rt = self.cost.round_trip_estimate(batch["liq"][:, 0], batch["sigma"], batch["liq"][:, 2])
        return torch.clamp(batch["impact"].abs() - c_rt.unsqueeze(-1), min=0.0).max(-1).values

    def compute_losses(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        L: Dict[str, torch.Tensor] = {}
        lc = self.tcfg.get("loss", {})
        if weights.get("det", 0) > 0 and not self.ab.get("no_anomaly_guidance", False):
            L["det"] = Lz.utility_weighted_bce(out["logit"], batch["y"], self._exec_alpha(batch), rho=float(lc.get("rho", 2.0)), pos_weight=lc.get("pos_weight"))
        if weights.get("dir", 0) > 0 and "dir_logit" in out:
            L["dir"] = Lz.direction_loss(out["dir_logit"], batch["impact"], batch["y"])
        if weights.get("svdd", 0) > 0:
            L["svdd"] = Lz.svdd_regularizer(out["z"], self.model.detector.center, batch["y"])
        if weights.get("imp", 0) > 0:
            wgt = out["a_prob"].detach() + float(lc.get("imp_floor", 0.1))
            hz = torch.tensor(self.horizons, device=out["q"].device, dtype=torch.float32)
            target_n = batch["impact"] / (batch["sigma"].clamp_min(1e-6).unsqueeze(-1) * hz.sqrt())
            L["imp"] = Lz.pinball_loss(out["q_norm"], target_n, self.taus, weight=wgt)
        if weights.get("liq", 0) > 0:
            fut = batch["fut_liq"][:, [self.latency + h - 1 for h in self.horizons]]
            L["liq"] = Lz.liquidity_loss(out["liq_hat"], liq_transform(fut), weight=out["a_prob"].detach() + 0.1)
        if weights.get("peak", 0) > 0:
            L["peak"] = Lz.peak_time_ce(out["tpeak_logit"], batch["impact"], batch["y"])
        if weights.get("utility", 0) > 0:
            r = event_pnl(out, batch, self.horizons, self.latency, self.cost, hard=False)
            scale = float(lc.get("pnl_scale", 100.0))     # percent units: keeps U at O(0.1-1) regardless of bar frequency
            L["utility"] = -Lz.utility(r["pnl"] * scale, float(lc.get("lam_down", 1.0)), float(lc.get("lam_cvar", 0.5)), float(lc.get("alpha", 0.05)))
            if weights.get("turnover", 0) > 0:
                L["turnover"] = Lz.turnover_penalty(out["w"], batch.get("prev_w", torch.zeros_like(out["w"])))
        total = sum(weights[k] * v for k, v in L.items())
        L["total"] = total
        return L

    # ------------------------------------------------------------------ training
    def _optimizer(self, modules: List[str], lr: float) -> torch.optim.Optimizer:
        groups = self.model.parameter_groups()
        params = [p for m in modules for p in groups[m]]
        return torch.optim.AdamW(params, lr=lr, weight_decay=float(self.tcfg.get("weight_decay", 1e-4)))

    def _set_train_modules(self, modules: List[str]) -> None:
        for name, params in self.model.parameter_groups().items():
            for p in params:
                p.requires_grad_(name in modules)

    @torch.no_grad()
    def _init_center(self) -> None:
        self.model.eval()
        zs = []
        for i, b in enumerate(self.loaders["train"]):
            b = to_device(b, self.device)
            out = self.model.detector(b)
            zs.append(out["z"][b["y"] < 0.5])
            if i >= 20:
                break
        self.model.detector.set_center(torch.cat(zs))

    def fit(self) -> Dict[str, Any]:
        history = []
        stages = self.tcfg["stages"]
        for st in stages:
            if st["name"] == "e2e" and self.ab.get("two_stage", False):
                log.info("ablation two_stage: skipping e2e stage")
                continue
            if st["name"] == "policy" and self.ab.get("plugin_policy", False):
                log.info("ablation plugin_policy: skipping policy stage")
                continue
            if st["name"] == "detect" and st.get("init_center", True):
                self._init_center()
            self._set_train_modules(st["modules"])
            opt = self._optimizer(st["modules"], float(st["lr"]))
            detach = bool(st.get("detach_upstream", False))
            best, best_state, bad = -float("inf"), None, 0
            for ep in range(int(st["epochs"])):
                self.model.train()
                t0, agg, n = time.time(), {}, 0
                for b in self.loaders["train"]:
                    b = to_device(b, self.device)
                    out = self.model(b, detach_upstream=detach)
                    L = self.compute_losses(out, b, st["losses"])
                    opt.zero_grad(set_to_none=True)
                    L["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg.get("grad_clip", 1.0)))
                    opt.step()
                    for k, v in L.items():
                        agg[k] = agg.get(k, 0.0) + float(v.detach())
                    n += 1
                val_score = self.validate(st["losses"])
                rec = {"stage": st["name"], "epoch": ep, "val_score": val_score, "sec": time.time() - t0, **{k: v / max(n, 1) for k, v in agg.items()}}
                history.append(rec)
                log.info(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
                if val_score > best:
                    best, best_state, bad = val_score, copy.deepcopy(self.model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= int(self.tcfg.get("patience", 3)):
                        break
            if best_state is not None:
                self.model.load_state_dict(best_state)
        return {"history": history}

    @torch.no_grad()
    def validate(self, weights: Dict[str, float]) -> float:
        """Model selection score on the validation fold: utility if the policy is being trained, else -loss."""
        self.model.eval()
        tot, n, pnls = 0.0, 0, []
        for b in self.loaders["val"]:
            b = to_device(b, self.device)
            out = self.model(b)
            if weights.get("utility", 0) > 0:
                pnls.append(event_pnl(out, b, self.horizons, self.latency, self.cost, hard=True)["pnl"])
            else:
                tot += float(self.compute_losses(out, b, weights)["total"]); n += 1
        if pnls:
            p = torch.cat(pnls) * float(self.tcfg.get("loss", {}).get("pnl_scale", 100.0))
            lc = self.tcfg.get("loss", {})
            return float(Lz.utility(p, float(lc.get("lam_down", 1.0)), float(lc.get("lam_cvar", 0.5)), float(lc.get("alpha", 0.05))))
        return -tot / max(n, 1)

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def predict_trades(self, loader: DataLoader, cost_model: Optional[CostModel] = None) -> pd.DataFrame:
        self.model.eval()
        bt = EventBacktester(self.horizons, self.latency, cost_model or self.cost)
        frames = [bt.collect(self.model(b), b) for b in (to_device(bb, self.device) for bb in loader)]
        return pd.concat(frames, ignore_index=True)


def evaluate_outputs(trades: pd.DataFrame, bars_per_year: int, n_bars: int, t0: int, bench: Optional[np.ndarray] = None,
                     n_assets: Optional[int] = None) -> Dict[str, float]:
    """Event trades -> strategy return series. Each asset owns a fixed 1/N capital slot (w in [-w_max, w_max] of that slot),
    so concurrent trades never lever the book beyond N * w_max; a trade's net PnL is booked on its exit bar."""
    N = int(n_assets or max(int(trades["asset"].max()) + 1, 1))
    r = EventBacktester.to_bar_returns(trades, n_bars, t0) / N
    to = np.zeros(n_bars)
    tr = trades[trades["w"] != 0]
    np.add.at(to, np.clip(tr["t"].to_numpy() - t0, 0, n_bars - 1), 2 * np.abs(tr["w"].to_numpy()) / N)
    out = {}
    out.update(M.summarize_detection(trades))
    out.update(M.summarize_trading(r, bars_per_year, to, bench))
    out["n_trades"] = int((trades["w"] != 0).sum())
    return out

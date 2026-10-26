"""SemiSupervisedTrainer: anchored weak-to-semi-supervised training with a frozen BH-conformal gate.

Stage A  ssl    epochs_ssl    L_ssl on every row (labelled, unlabelled D_u and calibration C0 alike)  [if use_ssl]
                                -> Deep SAD centre c from the labelled normals
Stage W  warm   epochs_warm   L_sup(D_l^K) + lam_ssl L_ssl(D_u + C0) + lam_imp L_imp(D_l)
NOTE (2026-10-05 finding): at manipulation-level prevalence (~1%) the valid gate selects nothing (calibration contamination
and k_min = m / (q (n + 1))), and power-restoring trimming breaks validity (measured FDR 0.71 at q = 0.1). The gate is
therefore OFF by default (semi.use_conformal = false) and reported as an ablation, not as a contribution.
                                The score is now a function of D_l and of the unordered set D_u + C0 (symmetric), which is
                                the condition of Proposition 2. Then, ONCE:
                                  label-model posterior on all rows, outcome weights on all rows,
                                  conformal p-values of D_u vs C0 and BH(q) -> gate set G (frozen)
Stage B  semi   epochs_semi   L_sup + lam_cons L_cons(D_u) [+ lam_ssl L_ssl(D_u)] + lam_imp L_imp
                                teacher positive allowed only if its row is in G; weight omega on teacher positives
Stage C  joint  epochs_joint  Stage B + lam_T(ep) L_trade (differentiable event backtest) on D_l and D_u
Teacher (semi.teacher): 'mixture' p_t = g(ep) p_LM + (1 - g(ep)) p_model(weak view); 'labelmodel'; 'model' (FixMatch).
semi.gate_refresh = true recomputes G every epoch from the current model (heuristic ablation, no guarantee).
semi.anchors_for = 'sign_only' feeds the anchors to the label model only (L_sup sees confident normals only).
The trainer never reads batch['y'] (evaluation labels); they are used only for PLCA / FDR reporting.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.semi_dataset import SemiEventDataset, TwoStreamLoader, make_two_stream_loaders
from ..data.base import build_loader
from ..models.ssl_heads import SSLHeads
from ..models.losses import pinball_loss
from ..backtest.costs import CostModel
from ..backtest.engine import EventBacktester
from .augment import augment_batch
from .losses import (supervised_anchor_loss, DynamicThreshold, consistency_loss, nt_xent, masked_reconstruction_loss,
                     closed_loop_trade_loss)
from .pseudo_labels import PseudoLabeler


class SemiSupervisedTrainer:
    def __init__(self, model, cfg: Dict[str, Any], cost_model: CostModel, device: str = "cpu",
                 logger: Optional[logging.Logger] = None, report_oracle: bool = True):
        self.m = model.to(device)
        self.cfg, self.device = cfg, torch.device(device)
        self.semi, self.lw, self.tr = dict(cfg.get("semi", {})), dict(cfg.get("loss", {})), dict(cfg.get("train", {}))
        self.aug_cfg = dict(cfg.get("augment", {}))
        self.cost_model = cost_model
        self.horizons = list(model.horizons)
        self.latency = int(cfg.get("data", {}).get("latency", 1))
        self.bs = int(cfg.get("data", {}).get("batch_size", 256))
        self.log = logger or logging.getLogger("warn2trade.trainer")
        self.report_oracle = report_oracle
        self.gen = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        self.dyn = DynamicThreshold(self.semi.get("tau", 0.9), self.semi.get("threshold_momentum", 0.99)) if self.semi.get("dynamic_threshold", True) else None
        self.teacher = str(self.semi.get("teacher", "mixture"))
        self.score_mode = str(cfg.get("eval", {}).get("score", "logit"))
        self.gate_score_mode = str(self.semi.get("gate_score", self.score_mode))
        self.heads: Optional[SSLHeads] = None
        self.history: List[Dict[str, float]] = []
        self.pseudo_report: Dict[str, Any] = {}
        self._pl: Optional[PseudoLabeler] = None

    # ------------------------------------------------------------------ helpers
    def _to(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _flag(self, key: str, default: bool = True) -> bool:
        return bool(self.semi.get(key, default))

    def _build_heads(self, ds: SemiEventDataset) -> None:
        L, Fp = ds[0]["price"].shape
        self.heads = SSLHeads(self.m.detector.d_model, L, Fp, int(self.semi.get("proj_dim", 64))).to(self.device)

    def _params(self):
        return list(self.m.parameters()) + list(self.heads.parameters())

    def _trade_weight(self, ep: int, e_semi: int) -> float:
        ramp = max(1, int(self.tr.get("trade_ramp_epochs", 2)))
        return float(self.lw.get("lam_trade", 1.0)) * min(1.0, (ep - e_semi + 1) / ramp)

    def _gamma(self, ep: int, e_total: int) -> float:
        if self.teacher == "labelmodel":
            return 1.0
        if self.teacher == "model":
            return 0.0
        final = float(self.semi.get("teacher_mix_final", 0.3))
        start = int(self.tr.get("epochs_semi", 1)) / 2.0
        return 1.0 if ep < start else max(final, 1.0 - (ep - start) / max(1.0, e_total - start))

    # ------------------------------------------------------------------ public API
    def fit(self, ds: SemiEventDataset) -> List[Dict[str, float]]:
        if self.heads is None:
            self._build_heads(ds)
        seed = int(self.cfg.get("seed", 42))
        if not self._flag("use_anchors", True):
            ds.y_l = np.where(ds.y_l == 1, -1, ds.y_l).astype(np.int8)       # pure-weak ablation: forget the anchors
        L = make_two_stream_loaders(ds, self.bs, seed, int(self.cfg.get("data", {}).get("num_workers", 0)), self.semi.get("labeled_batch_size"))
        opt = torch.optim.AdamW(self._params(), lr=float(self.tr.get("lr", 1e-3)), weight_decay=float(self.tr.get("weight_decay", 1e-4)))
        e_ssl, e_warm = int(self.tr.get("epochs_ssl", 1)), int(self.tr.get("epochs_warm", 1))
        e_semi, e_joint = int(self.tr.get("epochs_semi", 1)), int(self.tr.get("epochs_joint", 1))
        self.log.info("supervision: %s", ds.supervision_summary())
        if self._flag("use_ssl"):
            all_shuffled = build_loader(ds, self.bs, True, 0, seed + 5)
            for ep in range(e_ssl):
                self._run_epoch(TwoStreamLoader(None, all_shuffled), opt, "ssl", ep, e_semi, e_semi + e_joint)
        self._set_center(L["labeled"] or L["unlabeled"])
        for ep in range(e_warm):
            self._run_epoch(TwoStreamLoader(L["labeled"], L["unlabeled_cal"]), opt, "warm", ep, e_semi, e_semi + e_joint)
        self._pl = PseudoLabeler(self.semi).fit(ds.flat(ds.votes), ds.flat(ds.y_l))
        self._refresh_pseudo(ds, L["all"], first=True)
        for ep in range(e_semi + e_joint):
            stage = "joint" if ep >= e_semi else "semi"
            if ep > 0 and self._flag("gate_refresh", False):
                self._refresh_pseudo(ds, L["all"], first=False)
            self._run_epoch(TwoStreamLoader(L["labeled"], L["unlabeled"]), opt, stage, ep, e_semi, e_semi + e_joint)
        return self.history

    @torch.no_grad()
    def predict_scores(self, ds, loader: Optional[DataLoader] = None, mode: Optional[str] = None) -> np.ndarray:
        """Anomaly scores in dataset order (clean view). mode 'logit' = score head, 'svdd' = ||z - c||^2."""
        mode = mode or self.score_mode
        self.m.eval()
        loader = loader or build_loader(ds, self.bs, False)
        out = []
        for b in loader:
            det = self.m.detector(self._to(b))
            out.append((det["logit"] if mode == "logit" else ((det["z"] - self.m.detector.center) ** 2).sum(-1)).cpu())
        self.m.train()
        return torch.cat(out).numpy() if out else np.zeros(0)

    @torch.no_grad()
    def collect_trades(self, ds, backtester: EventBacktester, batch_size: Optional[int] = None) -> pd.DataFrame:
        self.m.eval()
        frames = []
        for b in build_loader(ds, batch_size or self.bs, False):
            b = self._to(b)
            frames.append(backtester.collect(self.m(b), b))
        self.m.train()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ stages
    @torch.no_grad()
    def _set_center(self, loader: Optional[DataLoader], max_items: int = 4096) -> None:
        if loader is None:
            return
        self.m.eval()
        zs, n = [], 0
        for b in loader:
            b = self._to(b)
            z = self.m.detector(b)["z"]
            normal = b["y_l"] == 0
            zs.append(z[normal] if normal.any() else z)
            n += zs[-1].shape[0]
            if n >= max_items:
                break
        zs = [z for z in zs if z.shape[0]]
        if zs:
            self.m.detector.set_center(torch.cat(zs))
        self.m.train()

    def _refresh_pseudo(self, ds: SemiEventDataset, loader_all: DataLoader, first: bool) -> None:
        idx_u, idx_c = ds.unlabeled_indices(), ds.cal_indices()
        n = len(ds)
        gate_ok = np.ones(n, dtype=bool)
        prob = self._pl.posterior(ds.flat(ds.votes)) if first else ds.flat(ds.pseudo)
        g = {}
        if self._flag("use_conformal"):
            if self.gate_score_mode == "labelmodel":                       # LM logit, latent-distance rank as tie-breaker
                from scipy.stats import rankdata
                lm = np.log(np.clip(prob, 1e-9, 1.0)) - np.log(np.clip(1.0 - prob, 1e-9, 1.0))
                sv = self.predict_scores(ds, loader_all, "svdd")
                scores = lm + 1e-3 * rankdata(sv) / max(len(sv), 1)
            else:
                scores = self.predict_scores(ds, loader_all, self.gate_score_mode)
            trim = self.semi.get("cal_trim_tau", None)
            g = self._pl.gate(scores[idx_u], scores[idx_c], prob[idx_c], None if trim is None else float(trim))
            m, n = len(idx_u), int(g.get("n_cal_used", len(idx_c)))
            self.gate_k_min = int(np.ceil(m / (float(self._pl.q) * (n + 1)))) if n else -1
            self.log.info("BH gate feasibility: m=%d unlabelled, n=%d calibration, q=%.2f -> selects only if >= %d rows beat every calibration score", m, n, self._pl.q, self.gate_k_min)
            gate_ok[idx_u] = g["selected"]
            if not g["active"]:
                self.log.warning("conformal gate inactive: %d calibration rows < min_calibration", len(idx_c))
        if first:
            omega = self._pl.outcome(ds.flat_impact(), ds.flat_sigma(), self.horizons)
            ds.set_pseudo(prob, gate_ok, omega)
        else:
            ds.set_pseudo(None, gate_ok, None)
        y_eval = ds.flat_eval("y") if self.report_oracle else None
        self.pseudo_report = self._pl.report(ds.flat(ds.pseudo), gate_ok[idx_u], idx_u, y_eval)
        self.pseudo_report["gate_k_min"] = getattr(self, "gate_k_min", -1)
        self.pseudo_report.update({k: g[k] for k in ("n_cal_used", "n_cal_trimmed", "t0_hat", "validity_slack") if k in g})
        self.log.info("gate/pseudo-labels: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.pseudo_report.items()})

    def _run_epoch(self, stream: TwoStreamLoader, opt, stage: str, ep: int, e_semi: int, e_total: int) -> None:
        self.m.train()
        logs: Dict[str, List[float]] = {}
        clip = float(self.tr.get("grad_clip", 1.0))
        for bl, bu in stream:
            if bu["price"].shape[0] < 2:                                  # policy input BatchNorm needs >= 2 rows
                continue
            if bl is not None and bl["price"].shape[0] < 2:
                bl = None
            loss, step_logs = self._step(bl, bu, stage, ep, e_semi, e_total)
            if not loss.requires_grad:
                continue
            if not torch.isfinite(loss):
                self.log.warning("non-finite loss at stage %s epoch %d; step skipped", stage, ep)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # CostModel's sqrt market impact has an infinite derivative at a zero position (inf * 0 = NaN): never step on it
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self._params()):
                self.n_bad_grad = getattr(self, "n_bad_grad", 0) + 1
                opt.zero_grad(set_to_none=True)
                continue
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self._params(), clip)
            opt.step()
            for k, v in step_logs.items():
                logs.setdefault(k, []).append(float(v))
        row = {"stage": stage, "epoch": ep, "bad_grad_steps": getattr(self, "n_bad_grad", 0), **{k: float(np.mean(v)) for k, v in logs.items()}}
        self.history.append(row)
        self.log.info("epoch %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    def _pinball(self, out, batch, weight=None) -> torch.Tensor:
        """Pinball on the sigma*sqrt(h)-normalised quantiles (q_norm), matching the extrapolator's parameterisation."""
        hz = torch.tensor(self.horizons, device=batch["impact"].device, dtype=torch.float32)
        target_n = batch["impact"] / (batch["sigma"].clamp_min(1e-6).unsqueeze(-1) * hz.sqrt())
        return pinball_loss(out["q_norm"], target_n, self.m.extrapolator.taus, weight)

    def _ssl_terms(self, out_s, out_w, mask, x_clean) -> torch.Tensor:
        con = nt_xent(self.heads.project(out_s["z"]), self.heads.project(out_w["z"]).detach(), float(self.lw.get("temperature", 0.2)))
        rec = masked_reconstruction_loss(self.heads.reconstruct(out_s["z"]), x_clean, mask)
        return con + float(self.lw.get("lam_rec", 0.5)) * rec

    def _trade(self, out, batch) -> Dict[str, torch.Tensor]:
        return closed_loop_trade_loss(out, batch, self.horizons, self.latency, self.cost_model, self.lw.get("lam_down", 1.0),
                                      self.lw.get("lam_cvar", 0.5), self.lw.get("cvar_alpha", 0.05), self.lw.get("lam_turnover", 0.01))

    def _step(self, bl, bu, stage: str, ep: int, e_semi: int, e_total: int):
        logs: Dict[str, float] = {}
        lam_imp = float(self.lw.get("lam_imp", 0.5))
        bu = self._to(bu)
        x_clean = bu["price"]
        bw, _ = augment_batch(bu, "weak", self.aug_cfg, self.gen)
        bs_, mask = augment_batch(bu, "strong", self.aug_cfg, self.gen)
        out_s = self.m(bs_)
        with torch.no_grad():
            out_w = self.m(bw)
        if stage == "ssl":
            loss = self._ssl_terms(out_s, out_w, mask, x_clean)
            logs["L_ssl"] = loss.item()
            return loss, logs

        total = out_s["logit"].sum() * 0.0
        use_ssl_here = self._flag("use_ssl") and (stage == "warm" or self._flag("ssl_in_semi", True))
        if use_ssl_here:
            l_ssl = self._ssl_terms(out_s, out_w, mask, x_clean)
            total = total + float(self.lw.get("lam_ssl", 0.5)) * l_ssl
            logs["L_ssl"] = l_ssl.item()
        p_t = None
        # ---- L_cons (semi / joint only): every teacher positive must be in the frozen gate set
        if stage in ("semi", "joint") and self._flag("use_consistency"):
            p_model = torch.sigmoid(out_w["logit"])
            p_lm = bu["pseudo"]
            g = self._gamma(ep, e_total)
            p_t = torch.where(torch.isfinite(p_lm), g * torch.nan_to_num(p_lm) + (1.0 - g) * p_model, p_model)
            hard = (p_t >= 0.5).float()
            thr = self.dyn.per_sample(hard) if self.dyn is not None else float(self.semi.get("tau", 0.9))
            gate = bu["pseudo_ok"] if self._flag("use_conformal") else None
            omega = torch.where(hard > 0.5, bu["omega"], torch.ones_like(bu["omega"])) if self._flag("use_outcome_weights") else None
            l_cons, mask_rate, _ = consistency_loss(out_s["logit"], p_t, thr, omega, gate)
            if self.dyn is not None:
                self.dyn.update(torch.maximum(p_t, 1 - p_t), hard)
            total = total + float(self.lw.get("lam_cons", 1.0)) * l_cons
            pos_used = (hard > 0.5) & (gate if gate is not None else torch.ones_like(hard, dtype=torch.bool))
            logs.update(L_cons=l_cons.item(), mask_rate=mask_rate.item(), teacher_pos_rate=float(hard.mean()), gated_pos_rate=float(pos_used.float().mean()))
        # ---- extrapolator (+ trade) on the unlabelled stream
        joint = stage == "joint" and self._flag("use_trade_loss")
        if joint:
            out_u = self.m(bu)
            total = total + lam_imp * self._pinball(out_u, bu, out_u["a_prob"].detach())
            if bu["price"].shape[0] >= 8:
                tl = self._trade(out_u, bu)
                lam_T = self._trade_weight(ep, e_semi)
                total = total + lam_T * tl["loss"]
                logs.update(U_trade=tl["utility"].item(), pnl_u=tl["pnl"].mean().item(), frozen=tl["frozen_rate"].item(), lam_T=lam_T)
        elif stage in ("semi", "joint"):
            total = total + lam_imp * self._pinball(out_s, bu, p_t.detach() if p_t is not None else None)
        # ---- L_sup on the labelled stream (anchors + confident normals)
        if bl is not None:
            bl = self._to(bl)
            if self.semi.get("anchors_for", "all") == "sign_only":
                keep = bl["y_l"] == 0
                bl = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in bl.items()}
            if bl["price"].shape[0] >= 2:
                out_l = self.m(bl)
                sup = supervised_anchor_loss(out_l["z"], self.m.detector.center, out_l["logit"], bl["y_l"], float(self.lw.get("eta", 1.0)),
                                             float(self.lw.get("lam_dev", 0.5)), float(self.lw.get("dev_margin", 5.0)), self.gen,
                                             float(self.lw.get("lam_sad", 1.0)))
                total = total + sup["loss"] + lam_imp * self._pinball(out_l, bl)
                logs.update(L_sup=sup["loss"].item(), L_sad=sup["sad"].item(), L_dev=sup["dev"].item())
                if joint and bl["price"].shape[0] >= 8:
                    tl = self._trade(out_l, bl)
                    total = total + self._trade_weight(ep, e_semi) * tl["loss"]
                    logs["U_trade_l"] = tl["utility"].item()
        logs["loss"] = total.item()
        return total, logs

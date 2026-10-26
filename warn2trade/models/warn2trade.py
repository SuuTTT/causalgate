"""Warn2Trade: MAE -> TDE -> REP with ablation switches.

Ablations (cfg.ablation.*):
    no_anomaly_guidance   detector output replaced by a constant 0.5 and L_det disabled (policy sees z only via TDE)
    no_extrapolator       policy consumes [a, z, liq] directly (no impact quantiles / liquidity forecast / h*)
    no_abstain            abstain gate removed (always in the market when a trade is proposed)
    two_stage             detector and extrapolator are frozen before the policy stage (no end-to-end gradient)
    plugin_policy         replaces the learned policy with the closed-form Bayes rule
    no_text / no_graph / no_lob   drop a modality
"""
from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn

from .anomaly_encoder import MultimodalAnomalyEncoder
from .extrapolator import TemporalDynamicsExtrapolator
from .policy import RiskAwarePolicy, BayesPlugInPolicy, policy_features, policy_in_dim
from .encoders import liq_transform
from ..utils.registry import MODELS


@MODELS.register("warn2trade")
class Warn2Trade(nn.Module):
    def __init__(self, dims: Dict[str, int], cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        ab = cfg.get("ablation", {}) or {}
        self.ab = {k: bool(ab.get(k, False)) for k in ("no_anomaly_guidance", "no_extrapolator", "no_abstain", "two_stage", "plugin_policy", "no_text", "no_graph", "no_lob")}
        enc_cfg = dict(cfg.get("encoder", {}))
        enc_cfg.update({"use_text": not self.ab["no_text"], "use_graph": not self.ab["no_graph"], "use_lob": not self.ab["no_lob"]})
        d = int(enc_cfg.get("d_model", 128))
        self.horizons = list(cfg["horizons"])
        self.taus = list(cfg.get("taus", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
        H, Q, liq_dim = len(self.horizons), len(self.taus), dims["liq"]
        self.detector = MultimodalAnomalyEncoder(dims, enc_cfg)
        self.extrapolator = TemporalDynamicsExtrapolator(d, self.horizons, self.taus, liq_dim, cfg.get("tde_hidden", 256), enc_cfg.get("dropout", 0.1))
        pol = cfg.get("policy", {})
        in_dim = policy_in_dim(H, Q, liq_dim) if not self.ab["no_extrapolator"] else (1 + d + liq_dim + 2)
        self.policy = RiskAwarePolicy(in_dim, H, pol.get("n_exec", 4), pol.get("w_max", 1.0), pol.get("hidden", 256),
                                      use_abstain=not self.ab["no_abstain"], dropout=enc_cfg.get("dropout", 0.1))
        self.plugin = BayesPlugInPolicy(self.extrapolator.taus.clone(), pol.get("cost_rt", 0.002), pol.get("lam_cvar", 0.5),
                                        pol.get("alpha", 0.1), pol.get("kelly_frac", 0.25), pol.get("w_max", 1.0), pol.get("a_min", 0.5), pol.get("n_exec", 4))

    def forward(self, batch: Dict[str, torch.Tensor], detach_upstream: bool = False) -> Dict[str, torch.Tensor]:
        det = self.detector(batch)
        logit, z = det["logit"], det["z"]
        if self.ab["no_anomaly_guidance"]:
            a_prob = torch.full_like(logit, 0.5)
        else:
            a_prob = torch.sigmoid(logit)
        if detach_upstream or self.ab["two_stage"]:
            z_in, a_in = z.detach(), a_prob.detach()
        else:
            z_in, a_in = z, a_prob
        tde = self.extrapolator(z_in, a_in, batch["liq"], batch["sigma"])
        prev_w = batch.get("prev_w", torch.zeros_like(a_prob))
        if self.ab["plugin_policy"]:
            pol = self.plugin(a_prob, tde["q"])
        else:
            if self.ab["no_extrapolator"]:
                feats = torch.cat([a_in.unsqueeze(-1), z_in, liq_transform(batch["liq"]), prev_w.unsqueeze(-1), torch.log(batch["sigma"].unsqueeze(-1) * 1e2 + 1e-4)], -1)
            else:
                q_in = tde["q_norm"] if not (detach_upstream or self.ab["two_stage"]) else tde["q_norm"].detach()
                feats = policy_features(a_in, q_in, tde["liq_hat"], tde["tpeak_logit"], prev_w, batch["sigma"], batch["liq"])
            pol = self.policy(feats)
        out = {"logit": logit, "a_prob": a_prob, "z": z, "dir_logit": det["dir_logit"]}
        out.update(tde)
        out.update(pol)
        return out

    def parameter_groups(self):
        return {"detector": list(self.detector.parameters()), "extrapolator": list(self.extrapolator.parameters()), "policy": list(self.policy.parameters())}

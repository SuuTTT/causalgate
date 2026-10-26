"""Warn2Trade v2 model with the identity comparators and corrected ablations from the CCF-A readiness review.

New ablation flags (cfg.model.ablation.*), on top of Warn2Trade's:
    label_free     the anomaly label y is never used: a := 1, no det/dir/peak/svdd losses, unweighted quantile loss
                   => "direct quantile forecaster" (+ plug-in Kelly/CVaR rule, or + learned policy)
    raw_policy     encoders -> RiskAwarePolicy on [z, liq, sigma], trained only on the trading utility (decision-focused,
                   no extrapolator supervision, no anomaly label)  => "e2e_policy_raw"
two_stage now detaches only where the stage asks for it (policy stage); the extrap stage still trains MAE+TDE jointly.
"""
from __future__ import annotations
from typing import Dict
import torch

from .warn2trade import Warn2Trade
from .encoders import liq_transform
from .policy import policy_features
from ..utils.registry import MODELS


@MODELS.register("warn2trade_v2")
class Warn2TradeV2(Warn2Trade):
    def __init__(self, dims, cfg):
        ab = dict(cfg.get("ablation", {}) or {})
        self.label_free = bool(ab.get("label_free", False))
        self.raw_policy = bool(ab.get("raw_policy", False))
        if self.raw_policy:
            ab["no_extrapolator"] = True
            self.label_free = True
        cfg = dict(cfg); cfg["ablation"] = ab
        super().__init__(dims, cfg)
        self.ab["label_free"] = self.label_free
        self.ab["raw_policy"] = self.raw_policy

    def forward(self, batch: Dict[str, torch.Tensor], detach_upstream: bool = False) -> Dict[str, torch.Tensor]:
        det = self.detector(batch)
        logit, z = det["logit"], det["z"]
        if self.label_free or self.ab["no_anomaly_guidance"]:
            a_prob = torch.ones_like(logit) if self.label_free else torch.full_like(logit, 0.5)
        else:
            a_prob = torch.sigmoid(logit)
        z_in, a_in = (z.detach(), a_prob.detach()) if detach_upstream else (z, a_prob)
        tde = self.extrapolator(z_in, a_in, batch["liq"], batch["sigma"])
        prev_w = batch.get("prev_w", torch.zeros_like(a_prob))
        if self.ab["plugin_policy"]:
            pol = self.plugin(a_prob, tde["q"])
        elif self.ab["no_extrapolator"]:
            feats = torch.cat([a_in.unsqueeze(-1), z_in, liq_transform(batch["liq"]), prev_w.unsqueeze(-1),
                               torch.log(batch["sigma"].unsqueeze(-1) * 1e2 + 1e-4)], -1)
            pol = self.policy(feats)
        else:
            q_in = tde["q_norm"].detach() if detach_upstream else tde["q_norm"]
            pol = self.policy(policy_features(a_in, q_in, tde["liq_hat"], tde["tpeak_logit"], prev_w, batch["sigma"], batch["liq"]))
        out = {"logit": logit, "a_prob": a_prob, "z": z, "dir_logit": det["dir_logit"]}
        out.update(tde)
        out.update(pol)
        if self.label_free:
            # detection score for the metric tables = predicted |median impact| at the best horizon (sigma units)
            out["score"] = tde["q_norm"][..., tde["q_norm"].shape[-1] // 2].abs().max(-1).values
        return out

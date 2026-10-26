"""PseudoLabeler: label model posterior + BH-conformal gate + outcome weights.

fit(votes, y_l)            anchored EM label model (or majority vote) over the training rows' votes
gate(s_u, s_cal, q)        conformal p-values of the unlabelled rows against the held-out calibration rows, then BH(q);
                           the returned mask is the ONLY set from which any teacher positive may reach the loss
outcome(impact, sigma)     realised post-warning footprint as a delayed weak label, computed for every row, applied by
                           the trainer to teacher positives only (training windows only, never at test time)
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np

from .label_model import AnchoredLabelModel, majority_vote_proba
from .calibration import conformal_pvalues, bh_reject, plca, empirical_fdr


def outcome_weights_np(impact: np.ndarray, sigma: np.ndarray, horizons: Sequence[int], m_out: float, kappa: float, floor: float) -> np.ndarray:
    hz = np.sqrt(np.asarray(list(horizons), dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.nanmax(np.abs(impact) / (sigma[:, None] * hz[None, :] + 1e-8), axis=-1)
    z = np.nan_to_num(z, nan=0.0)
    return floor + (1.0 - floor) / (1.0 + np.exp(-kappa * (z - m_out)))


class PseudoLabeler:
    def __init__(self, cfg: Dict[str, Any]):
        self.use_label_model = bool(cfg.get("use_label_model", True))
        self.q = float(cfg.get("gate_q", cfg.get("conformal_alpha", 0.1)))
        self.tau = float(cfg.get("tau", 0.9))
        self.m_out, self.kappa, self.floor = float(cfg.get("outcome_m", 2.0)), float(cfg.get("outcome_kappa", 4.0)), float(cfg.get("outcome_floor", 0.1))
        self.prior_pos = float(cfg.get("prior_pos", 0.01))
        self.min_cal = int(cfg.get("min_calibration", 20))
        self.lm: Optional[AnchoredLabelModel] = None

    def fit(self, votes: np.ndarray, y_l: Optional[np.ndarray]) -> "PseudoLabeler":
        if self.use_label_model:
            self.lm = AnchoredLabelModel(votes.shape[-1], prior_pos=self.prior_pos).fit(votes, y_l)
        return self

    def posterior(self, votes: np.ndarray) -> np.ndarray:
        return self.lm.predict_proba(votes) if self.lm is not None else majority_vote_proba(votes)

    def gate(self, scores_u: np.ndarray, scores_cal: np.ndarray, cal_posterior: Optional[np.ndarray] = None,
             trim_tau: Optional[float] = None) -> Dict[str, Any]:
        """BH over conformal p-values. Calibration rows are drawn from the (contaminated) unlabelled pool; with trim_tau set,
        rows whose label-model posterior >= trim_tau are removed from C0 first (Corollary 2.1): power is restored at a
        validity cost bounded by t0 / (n0 + 1), t0 = trimmed normals, estimated here as sum(1 - posterior) over trimmed rows."""
        keep = np.ones(len(scores_cal), dtype=bool)
        t0_hat = 0.0
        if trim_tau is not None and cal_posterior is not None:
            trim = cal_posterior >= trim_tau
            keep = ~trim
            t0_hat = float((1.0 - cal_posterior[trim]).sum())
        sc = scores_cal[keep]
        info = {"n_cal_used": int(keep.sum()), "n_cal_trimmed": int((~keep).sum()), "t0_hat": t0_hat,
                "validity_slack": t0_hat / (keep.sum() + 1.0)}
        if len(sc) < self.min_cal:
            return {"pvalue": np.full(len(scores_u), np.nan), "selected": np.ones(len(scores_u), dtype=bool), "active": False, **info}
        pv = conformal_pvalues(sc, scores_u)
        return {"pvalue": pv, "selected": bh_reject(pv, self.q), "active": True, **info}

    def outcome(self, impact: np.ndarray, sigma: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
        return outcome_weights_np(impact, sigma, horizons, self.m_out, self.kappa, self.floor).astype(np.float32)

    def report(self, prob: np.ndarray, selected_u: np.ndarray, idx_u: np.ndarray, y_eval: Optional[np.ndarray]) -> Dict[str, Any]:
        """Coverage always; PLCA / empirical FDR only against evaluation labels (reporting only, never used in training)."""
        lm_pos = prob >= self.tau
        out = {"n_bh_selected": int(selected_u.sum()), "n_unlabeled": int(len(idx_u)), "cov_lm_pos": float(lm_pos.mean()),
               "n_lm_pos_and_gate": int((lm_pos[idx_u] & selected_u).sum())}
        if self.lm is not None:
            out.update(lm_pi=float(self.lm.pi), lm_alpha_mean=float(self.lm.alpha.mean()), lm_flipped=bool(getattr(self.lm, "flipped_", False)))
        if y_eval is not None:
            yu = y_eval[idx_u]
            out["gate_FDR_emp"] = empirical_fdr(selected_u, yu)
            out["PLCA_pos"] = plca(lm_pos[idx_u] & selected_u, yu, 1)
            out["PLCA_neg"] = plca(~lm_pos[idx_u] & (prob[idx_u] <= 1 - self.tau), yu, 0)
            out["gate_recall"] = float((selected_u & (yu == 1)).sum() / max((yu == 1).sum(), 1))
        return out

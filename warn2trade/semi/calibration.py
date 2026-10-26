"""Confidence calibration and finite-sample control of the pseudo-positive set.

Conformal p-values (Vovk; Bates, Candes, Lei, Romano, Sesia, Ann. Statist. 2023): with a calibration set C0 of n rows
and a score s (higher = more anomalous), p(x) = (1 + #{c in C0 : s(c) >= s(x)}) / (n + 1).
Pseudo-positives are selected by Benjamini-Hochberg over the unlabelled rows (bh_reject). Under the conditions of
Proposition 2 (docs/sparse_label_research_plan.md: C0 drawn uniformly from the unlabelled pool, score fitted
symmetrically in C0 and D_u, exchangeable normal rows, anomalies stochastically larger than normals), conformal
p-values are valid and PRDS and BH controls the FDR of the selected set at level q * pi0 <= q.
The guarantee is for ONE fixed score function; the trainer therefore freezes the gate after the warm stage
(semi.gate_refresh = false). Refreshing the gate every epoch is kept as a heuristic ablation.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np


class TemperatureScaler:
    """Single-parameter temperature scaling fitted on labelled rows (anchors + confident normals) by grid search."""

    def __init__(self, grid: Optional[np.ndarray] = None):
        self.grid = np.logspace(-1, 1, 61) if grid is None else grid
        self.T = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)
        best, best_T = np.inf, 1.0
        for T in self.grid:
            z = logits / T
            nll = np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y))
            if nll < best:
                best, best_T = nll, T
        self.T = float(best_T)
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits) / self.T


def conformal_pvalues(cal_scores_normal: np.ndarray, scores: np.ndarray) -> np.ndarray:
    cal = np.sort(np.asarray(cal_scores_normal, dtype=np.float64))
    n = len(cal)
    s = np.asarray(scores, dtype=np.float64)
    n_ge = n - np.searchsorted(cal, s, side="left")               # #cal >= s
    return (1.0 + n_ge) / (n + 1.0)


def bh_reject(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up at level q. Returns a boolean rejection mask (True = selected as pseudo-positive)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    k = int(np.where(below)[0].max())
    rej = np.zeros(m, dtype=bool)
    rej[order[: k + 1]] = True
    return rej


def empirical_fdr(selected: np.ndarray, y_true: np.ndarray) -> float:
    sel = np.asarray(selected, dtype=bool)
    return float((np.asarray(y_true)[sel] == 0).mean()) if sel.any() else 0.0


def conformal_positive_mask(scores: np.ndarray, cal_scores_normal: np.ndarray, alpha: float) -> np.ndarray:
    return conformal_pvalues(cal_scores_normal, scores) <= alpha


def plca(pseudo_mask: np.ndarray, y_true: np.ndarray, target: int = 1) -> float:
    """Pseudo-Label Confirmation Accuracy: precision of the pseudo-labels of class `target` against oracle labels."""
    m = np.asarray(pseudo_mask, dtype=bool)
    if m.sum() == 0:
        return float("nan")
    return float((np.asarray(y_true)[m] == target).mean())


def pseudo_label_report(hard: np.ndarray, y_true: Optional[np.ndarray]) -> Dict[str, float]:
    """hard in {-1 abstain, 0, 1}. Coverage always; PLCA only if oracle labels are supplied (evaluation only)."""
    hard = np.asarray(hard)
    out = {"cov_pos": float((hard == 1).mean()), "cov_neg": float((hard == 0).mean()),
           "n_pseudo_pos": int((hard == 1).sum()), "n_pseudo_neg": int((hard == 0).sum())}
    if y_true is not None:
        out["PLCA_pos"] = plca(hard == 1, y_true, 1)
        out["PLCA_neg"] = plca(hard == 0, y_true, 0)
        y = np.asarray(y_true)
        out["pseudo_recall_pos"] = float(((hard == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    return out


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    prob, y = np.asarray(prob), np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi)
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(ece)

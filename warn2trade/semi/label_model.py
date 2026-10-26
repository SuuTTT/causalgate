"""Anchor-grounded generative label model: semi-supervised naive Bayes over LF votes, fitted by EM.

Model (Ratner et al. 2016/2017 style, conditionally independent LFs given y):
    y ~ Bernoulli(pi)
    P(Lambda_j != 0)                      = beta_j     (propensity, class independent)
    P(Lambda_j = y | Lambda_j != 0, y)    = alpha_j    (accuracy)
Posterior log-odds for a row with votes Lambda in {-1, 0, 1}^J:
    logit P(y = 1 | Lambda) = log(pi / (1 - pi)) + sum_{j : Lambda_j != 0} Lambda_j * log(alpha_j / (1 - alpha_j)).

Proposition 1 (sign ambiguity; why anchors matter). The marginal likelihood of the votes is invariant under the
joint flip alpha_j -> 1 - alpha_j (all j), pi -> 1 - pi. With no labels the *sign* of every LF accuracy is
unidentifiable, and practitioners resolve it by assuming alpha_j > 1/2 for every LF. One anchor of each class breaks
the symmetry; K anchors per class pin alpha_j with a Beta posterior of width O((K + n_j)^{-1/2}), n_j = number of
unlabelled rows covered by LF j weighted by posterior confidence. Anchors enter the E-step as observed y (hard
constraints); a weak Beta(a0, b0) prior on alpha keeps the MAP finite under tiny K.
"""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class AnchoredLabelModel:
    def __init__(self, n_lf: int, prior_pos: float = 0.01, alpha_init: float = 0.7, a0: float = 2.0, b0: float = 2.0,
                 n_iter: int = 200, tol: float = 1e-6, fix_prior: bool = False, min_alpha: float = 1e-3):
        self.J = int(n_lf)
        self.pi = float(prior_pos)
        self.alpha = np.full(self.J, float(alpha_init))
        self.beta = np.full(self.J, 0.5)
        self.a0, self.b0, self.n_iter, self.tol, self.fix_prior, self.min_alpha = a0, b0, n_iter, tol, fix_prior, min_alpha
        self.n_iter_ = 0
        self.n_anchors_ = 0
        self.flipped_ = False

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _anchor_loglik(Vk: np.ndarray, yk: np.ndarray, alpha: np.ndarray) -> float:
        """LF-accuracy log-likelihood of the anchors (prior term excluded: anchors are not a random sample of y)."""
        sign = np.where(yk == 1, 1, -1)[:, None]
        cov = Vk != 0
        agree = cov & (Vk == sign)
        return float((agree * np.log(alpha)[None, :] + (cov & ~agree) * np.log(1.0 - alpha)[None, :]).sum())

    def _anchor_init(self, Vk: np.ndarray, yk: np.ndarray) -> None:
        """Start EM in the anchors' basin: alpha_j^(0) = MAP accuracy of LF j on the anchors it covers."""
        sign = np.where(yk == 1, 1, -1)[:, None]
        cov = Vk != 0
        agree = (cov & (Vk == sign)).sum(0)
        n_cov = cov.sum(0)
        a0 = (agree + self.a0 - 1.0) / (n_cov + self.a0 + self.b0 - 2.0)
        self.alpha = np.clip(np.where(n_cov > 0, a0, self.alpha), self.min_alpha, 1.0 - self.min_alpha)

    def _em(self, V: np.ndarray, y: np.ndarray, known: np.ndarray) -> None:
        n = V.shape[0]
        cov = V != 0
        q = _sigmoid(np.log(self.pi / (1 - self.pi)) + V @ np.log(self.alpha / (1 - self.alpha)))
        q[known] = y[known]
        for it in range(self.n_iter):
            alpha_old = self.alpha.copy()
            # M-step (MAP with Beta(a0, b0) prior on alpha)
            agree = (V == 1) * q[:, None] + (V == -1) * (1.0 - q)[:, None]
            self.alpha = (agree.sum(0) + self.a0 - 1.0) / (cov.sum(0) + self.a0 + self.b0 - 2.0)
            self.alpha = np.clip(self.alpha, self.min_alpha, 1.0 - self.min_alpha)
            self.beta = cov.mean(0)
            if not self.fix_prior:
                self.pi = float(np.clip((q.sum() + 1.0) / (n + 2.0), 1e-5, 1 - 1e-5))
            # E-step
            q = _sigmoid(np.log(self.pi / (1 - self.pi)) + V @ np.log(self.alpha / (1 - self.alpha)))
            q[known] = y[known]
            self.n_iter_ = it + 1
            if np.max(np.abs(self.alpha - alpha_old)) < self.tol:
                break

    def fit(self, votes: np.ndarray, y_anchor: Optional[np.ndarray] = None) -> "AnchoredLabelModel":
        """EM over all rows; anchors enter as observed y. Two anchor-specific devices realise Proposition 1 in practice,
        because EM is local and the unlabelled rows (which outnumber anchors by orders of magnitude) are self-consistent
        in *both* basins: (i) alpha is initialised from the anchors; (ii) after convergence the anchor log-likelihood of
        (alpha, pi) and of the mirror (1 - alpha, 1 - pi) is compared and the better basin is kept (`flipped_` records it).
        """
        V = votes.reshape(-1, self.J).astype(np.int16)
        n = V.shape[0]
        y = np.full(n, -1, dtype=np.int16) if y_anchor is None else y_anchor.reshape(-1).astype(np.int16)
        known = y >= 0
        self.n_anchors_ = int(known.sum())
        self.flipped_ = False
        if known.any():
            self._anchor_init(V[known], y[known])
        self._em(V, y, known)
        if known.any():
            ll = self._anchor_loglik(V[known], y[known], self.alpha)
            ll_mirror = self._anchor_loglik(V[known], y[known], 1.0 - self.alpha)
            if ll_mirror > ll:
                self.alpha, self.pi, self.flipped_ = 1.0 - self.alpha, 1.0 - self.pi, True
                self._em(V, y, known)
        return self

    # ------------------------------------------------------------------ inference
    def predict_logit(self, votes: np.ndarray) -> np.ndarray:
        V = votes.reshape(-1, self.J).astype(np.int16)
        return np.log(self.pi / (1 - self.pi)) + V @ np.log(self.alpha / (1 - self.alpha))

    def predict_proba(self, votes: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logit(votes))

    def accuracy_table(self, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        names = list(names) if names is not None else [f"lf{j}" for j in range(self.J)]
        return pd.DataFrame({"lf": names, "alpha": self.alpha, "beta": self.beta,
                             "log_odds_weight": np.log(self.alpha / (1 - self.alpha))})


def majority_vote_proba(votes: np.ndarray) -> np.ndarray:
    """Baseline label model: (mean non-abstaining vote + 1) / 2 ; 0.5 when no LF fires."""
    V = votes.reshape(votes.shape[0] if votes.ndim == 2 else -1, votes.shape[-1]).astype(np.float64)
    cov = (V != 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cov > 0, V.sum(1) / np.maximum(cov, 1), 0.0)
    return 0.5 * (m + 1.0)


# ----------------------------------------------------------------------------- Theorem 1 (anchor basin selection)
def basin_selection_bound(alpha: np.ndarray, beta: np.ndarray, rows_per_case: int, K: int) -> float:
    """Hoeffding bound on P(anchor log-likelihood test keeps the WRONG basin) with K independent cases.

    Per case k the test statistic contributes D_k = sum_{rows m in k} sum_{j covering m} s_mj w_j, s = +1 if LF j agrees
    with the case label, -1 otherwise, w_j = log(alpha_j / (1 - alpha_j)) > 0 (true-basin orientation). Rows inside a case
    may be arbitrarily dependent; cases are independent. With |D_k| <= B = n * sum_j w_j and
    E D_k >= mu = n * sum_j beta_j (2 alpha_j - 1) w_j,
        P(sum_k D_k <= 0) <= exp(-K mu^2 / (2 B^2)).
    """
    a, b = np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    w = np.log(a / (1 - a))
    n = float(rows_per_case)
    mu, B = n * float((b * (2 * a - 1) * w).sum()), n * float(w.sum())
    return float(np.exp(-K * mu ** 2 / (2 * B ** 2))) if B > 0 else 1.0


def cases_needed(alpha: np.ndarray, beta: np.ndarray, delta: float) -> int:
    """Smallest K with basin_selection_bound <= delta (independent of the rows per case)."""
    a, b = np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    w = np.log(a / (1 - a))
    rho = float((b * (2 * a - 1) * w).sum() / w.sum())
    return int(np.ceil(2 * np.log(1 / delta) / rho ** 2))

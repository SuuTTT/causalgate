"""Unsupervised tabular baseline: Isolation Forest on the flattened price window (Liu et al. 2008)."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest

from .base import ScoreBaseline, flatten_windows, SEMI_BASELINES


@SEMI_BASELINES.register("isolation_forest")
class IsolationForestBaseline(ScoreBaseline):
    name = "isolation_forest"

    def fit(self, ds):
        X = flatten_windows(ds, max_items=int(self.cfg.get("max_fit_items", 50000)))
        self.model = IsolationForest(n_estimators=int(self.cfg.get("n_estimators", 200)), contamination="auto",
                                     random_state=int(self.cfg.get("seed", 42)), n_jobs=-1).fit(X)
        return self

    def score(self, ds) -> np.ndarray:
        return -self.model.score_samples(flatten_windows(ds))

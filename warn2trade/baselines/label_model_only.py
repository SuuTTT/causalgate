"""Weak-supervision-only baselines: the anchored label model / majority vote *is* the detector (no representation
learning). Isolates how much of the gain comes from the LFs themselves versus the semi-supervised network.
Protocol: fit on all training votes (targets may look forward), but score with *causal* votes only
(SemiEventDataset.causal_votes_flat), otherwise a forward-looking LF would leak the future into a test-time detector."""
from __future__ import annotations
import numpy as np

from .base import ScoreBaseline, SEMI_BASELINES
from ..semi.label_model import AnchoredLabelModel, majority_vote_proba


@SEMI_BASELINES.register("label_model_only")
class LabelModelOnly(ScoreBaseline):
    name = "label_model_only"

    def fit(self, ds):
        votes, y_l = ds.flat(ds.votes), ds.flat(ds.y_l)
        self.lm = AnchoredLabelModel(votes.shape[-1], prior_pos=float(self.cfg.get("prior_pos", 0.01))).fit(votes, y_l if self.cfg.get("use_anchors", True) else None)
        return self

    def score(self, ds) -> np.ndarray:
        return self.lm.predict_logit(ds.causal_votes_flat())


@SEMI_BASELINES.register("majority_vote")
class MajorityVote(ScoreBaseline):
    name = "majority_vote"

    def fit(self, ds):
        return self

    def score(self, ds) -> np.ndarray:
        return majority_vote_proba(ds.causal_votes_flat())

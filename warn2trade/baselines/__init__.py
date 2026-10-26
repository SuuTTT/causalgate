"""Unified baseline interface. Two families:

(1) Score-only detectors (ScoreBaseline): emit an anomaly score per decision point; trades come from the shared
    ScoreToTradeAdapter rule so every baseline is priced with the same latency / cost / liquidity model.
(2) Trainer-flag baselines: Deep SAD (anchors only), FixMatch-Fin (consistency only), Supervised-K, Weak-only
    (label model, no anchors) are the *same* SparseWarn2Trade network with cfg['semi'] switches; see configs/baselines/.
Every detector, including ours, is traded through the same ScoreToTradeAdapter with a validation-fold threshold.
Foundation-model and RL baselines (Time-LLM, FPT, UniTS, MacroHFT/EarnHFT) are registered (in SEMI_BASELINES, never in the
Warn2Trade track's BASELINES registry) as placeholders with the upstream repositories to wrap; they are not re-implemented here.
"""
from .base import ScoreBaseline, ScoreToTradeAdapter, flatten_windows, SEMI_BASELINES
from . import isolation_forest, label_model_only, placeholders  # noqa: F401  (registration side effects)

"""Warn2Trade / SparseWarn2Trade: from ultra-sparse regulatory anchors to executable, risk-constrained trades.

Package layout
--------------
warn2trade.data        Panel schema, EventDataset, purged walk-forward splits, 10 benchmark builders,
                       regulatory anchors (release-date-aware availability), weak labeling functions,
                       label-budget SemiEventDataset
warn2trade.semi        semi_engine: anchored label model, conformal pseudo-label gate, augmentations,
                       L_sup / L_cons / L_ssl / L_trade, three-stage SemiSupervisedTrainer
warn2trade.models      Multimodal Anomaly Encoder -> Temporal Dynamics Extrapolator -> Risk-Aware Execution Policy (+ SSL heads)
warn2trade.backtest    cost model, differentiable event backtest, dual metric matrix (+ LES, PLCA, WAD, ACR, cost resilience),
                       RollingExperiment driver (folds x protocols x label budgets x seeds)
warn2trade.baselines   unified baseline interface, shared score-to-trade rule, trainer-flag baselines via configs/baselines
"""
__version__ = "0.2.0"

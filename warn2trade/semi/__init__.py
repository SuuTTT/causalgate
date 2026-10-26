"""semi_engine: anchor-grounded weak -> semi-supervised pseudo-labelling and the closed-loop trainer.

label_model     AnchoredLabelModel (EM over LF votes with anchors as hard constraints), majority vote
calibration     temperature scaling, split-conformal p-values, Benjamini-Hochberg gate, PLCA / ECE diagnostics
augment         weak / strong time-series views for consistency + masked reconstruction
losses          L_sup (Deep SAD + deviation), L_cons (FixMatch w/ dynamic threshold), L_ssl (NT-Xent + masked recon),
                outcome-consistency weights, L_trade wrapper (closed loop)
pseudo_labels   PseudoLabeler: label model posterior, BH-conformal gate on held-out calibration rows, outcome weights
trainer         SemiSupervisedTrainer: SSL -> warm (anchors) -> frozen gate -> consistency -> joint anomaly-to-trade
"""
from .label_model import AnchoredLabelModel, majority_vote_proba
from .calibration import TemperatureScaler, conformal_pvalues, conformal_positive_mask, bh_reject, empirical_fdr, plca, pseudo_label_report, expected_calibration_error
from .pseudo_labels import PseudoLabeler
from .trainer import SemiSupervisedTrainer
from . import losses, augment

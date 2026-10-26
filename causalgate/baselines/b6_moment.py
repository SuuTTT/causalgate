"""B6 MOMENT zero-shot anomaly detection (Goswami, Szafer, Choudhry, Cai, Li, Dubrawski, ICML 2024) as a
score -> alarm -> trade baseline.

Official code / weights: https://github.com/moment-timeseries-foundation-model/moment  (pip package `momentfm`,
Hugging Face AutonLab/MOMENT-1-{large,base,small}; tutorial tutorials/anomaly_detection.ipynb). The pre-trained weights
are a *downloaded artefact*: this wrapper only works when `momentfm` is installed and the checkpoint is available
(status 'official'); otherwise status = 'wrapper-needed' and `_fit_model` raises ImportError with the install line, so
runners and tests skip it gracefully.

Zero-shot protocol of the tutorial (anomaly_detection.ipynb, code cells in order):
    model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", model_kwargs={"task_name": "reconstruction"}); model.init()
    output = model(x_enc=batch_x, input_mask=batch_masks)                     # 512-step windows, C channels
    anomaly_scores = (trues - preds) ** 2                                      # per-point squared reconstruction error
    adjbestf1(y_true=labels, y_scores=anomaly_scores)                          # best point-adjusted F1 over 100 thresholds (test labels)
Reconstruction (momentfm/models/moment.py `reconstruction`): per-window RevIN with the input mask, patch length 8,
T5 encoder, de-normalised output. Here each lookback window (L bars, F features as C channels) is left-padded to 512
with input_mask = 0 on the padding; score = mean over channels of the squared error at the LAST bar.

L8: the Time Series Pile (HF AutonLab/Timeseries-PILE) is the pre-training corpus; MOMENT-1 was released with the
paper (arXiv 2402.03885, 6 Feb 2024; HF files updated May 2024). cutoff_date = '2024-02-01' is the *release* date, not
a proof of non-overlap: the pile's contents (UCR/UEA, Monash, TSB-UAD anomaly archives, ...) and their collection
dates must be checked against each fold's test start before claiming a leak-free zero-shot result.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import torch

from .common import ScoreToTradeBaseline, register
from ..data.pit import AsOfStamped

try:                                                          # optional dependency: pip install momentfm
    from momentfm import MOMENTPipeline                       # noqa: F401
    _HAS_MOMENT = True
    _IMPORT_ERROR = ""
except Exception as _e:                                       # noqa: BLE001
    MOMENTPipeline = None
    _HAS_MOMENT = False
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

INSTALL_LINE = "pip install momentfm   # then the checkpoint AutonLab/MOMENT-1-large is fetched from Hugging Face on first use"


class _ModelStamp(AsOfStamped):
    pass


@register("cg_moment")
class MomentZeroShotBaseline(ScoreToTradeBaseline):
    name = "cg_moment"
    paper = "Goswami, Szafer, Choudhry, Cai, Li, Dubrawski. MOMENT: A Family of Open Time-series Foundation Models. ICML 2024"
    official_repo = "https://github.com/moment-timeseries-foundation-model/moment"
    status = "official" if _HAS_MOMENT else "wrapper-needed"
    year = 2024
    cutoff_date = "2024-02-01"          # MOMENT-1 release (arXiv 2402.03885, 6 Feb 2024); Time Series Pile contents to be checked per fold
    conventions = {
        "L1": "official: per-window reversible instance normalisation inside the model (moment.py reconstruction(): "
              "`x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode='norm')`), i.e. window-internal (F_t-measurable); "
              "the tutorial's AnomalyDetectionDataset applies its own scaler (momentfm/data/anomaly_detection_dataset.py - statistics "
              "not verified here). causal: scaffold StreamingNormaliser + the model's own RevIN.",
        "L2": "official: no operating threshold - the tutorial reports adjbestf1 (momentfm/utils/anomaly_detection_metrics.py ~L61-77): "
              "100 thresholds between min and max of the TEST scores, best point-adjusted F1 kept. published here: best_f1_test "
              "(the scaffold's best-F1 threshold on the test labels, without PA); causal: delayed-ACI gate.",
        "L3": "official: best-over-thresholds on the test labels (L2 above); zero-shot otherwise. causal: none.",
        "L4": "official: point-adjust inside adjbestf1 (adjust_predicts ~L15-58). causal: none.",
        "L5": "not applicable.",
        "L6": "not applicable.",
        "L7": "official: detection only, every point of 512-step windows scored (the notebook de-duplicates overlapping windows). Here: "
              "sliding window ending at t left-padded to 512 with input_mask 0, score at the last bar, fill at t + latency.",
        "L8": "official: pre-trained on the Time Series Pile (HF AutonLab/Timeseries-PILE); release 2024-02 (cutoff_date). The pile's "
              "contents / collection dates vs each fold's test start must be verified by the protocol runner before a zero-shot claim.",
    }
    unspecified = [
        "Window padding: lookback L < 512 -> left zero-padding with input_mask = 0 (the tutorial feeds full 512-step windows).",
        "Channels: F features are passed as C independent channels (channel-independent model); the score averages the per-channel squared error.",
        "Score position: last bar only (the tutorial scores every point).",
        "No fine-tuning (zero-shot), model in eval mode; cfg['model']['hf_name'] selects the checkpoint (default AutonLab/MOMENT-1-large).",
        "The test-split scaler of the tutorial dataset class is not replicated (scaffold L1 instead).",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons, device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "best_f1_test")
        cfg.setdefault("published_norm", "train")
        super().__init__(cfg, dims, horizons, device)
        self.cutoff_date = str(cfg.get("cutoff_date", type(self).cutoff_date))   # yaml value is live (L8 audit field of the run)
        m = dict(cfg.get("model", {}))
        self.hf_name = str(m.get("hf_name", "AutonLab/MOMENT-1-large"))
        self.seq_len = int(m.get("seq_len", 512))
        self.bs = int(m.get("batch_size", 64))
        self.model = None
        self.diag["official_import"] = "ok" if _HAS_MOMENT else _IMPORT_ERROR
        self._stamp = _ModelStamp()
        self.guard.register(self._stamp)

    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        if not _HAS_MOMENT:
            raise ImportError(f"cg_moment needs the `momentfm` package (status wrapper-needed): {INSTALL_LINE}  [{_IMPORT_ERROR}]")
        self.model = MOMENTPipeline.from_pretrained(self.hf_name, model_kwargs={"task_name": "reconstruction"})
        self.model.init()
        self.model = self.model.to(self.device).float().eval()
        self._stamp.stamp(int(t_end))                      # zero-shot: nothing is fitted on the fold; the stamp records the as-of use
        self.diag.update({"status": self.status, "hf_name": self.hf_name, "cutoff_date": self.cutoff_date})

    @torch.no_grad()
    def _score(self, X: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X, dtype=np.float32)
        if len(Xn) == 0:
            return np.zeros(0)
        if self.model is None:
            raise ImportError(f"cg_moment is not fitted / `momentfm` missing: {INSTALL_LINE}")
        B, L, Fd = Xn.shape
        out = np.empty(B)
        for i in range(0, B, self.bs):
            xb = torch.nan_to_num(torch.as_tensor(Xn[i : i + self.bs]), nan=0.0).permute(0, 2, 1)       # (b, C, L)
            b = xb.shape[0]
            x_enc = torch.zeros(b, Fd, self.seq_len)
            mask = torch.zeros(b, self.seq_len)
            x_enc[:, :, -L:] = xb[:, :, -self.seq_len :]
            mask[:, -L:] = 1.0
            res = self.model(x_enc=x_enc.to(self.device), input_mask=mask.to(self.device))
            rec = res.reconstruction
            out[i : i + b] = ((rec[:, :, -1] - x_enc.to(self.device)[:, :, -1]) ** 2).mean(1).cpu().numpy()
        out[np.isnan(Xn[:, -1]).any(-1)] = np.nan
        return out

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._stamp.check(int(batch["t"].min()))
        return super().predict(batch)

"""Shared scaffold for the B1-B10 wrappers: every baseline is a *score -> alarm -> trade* pipeline with
mode in {published, causal}. Subclasses implement the model part only:

    _fit_model(X_train: (n, L, F) float32, t_end: int) -> None      fit on causally/offline normalised windows
    _score(X: (B, L, F)) -> np.ndarray (B,)                           higher = more anomalous; NaN = abstain
    _direction(X, batch) -> np.ndarray (B,) in {-1, 0, +1}            optional; default = momentum on the last 3 bars
    _published_threshold(s_train, s_val, s_test, y_test) -> float     optional; default = `published_threshold` kind

Channel handling (which operator carries which leak, see CONVENTIONS.md):
    L1 normalisation   published: OfflineScaler on `published_norm` rows ('global' = whole panel, 'train' = train fold)
                       causal:    StreamingNormaliser (stats from bars < t, warm-up abstains)
    L2 threshold       published: `published_threshold` in {test_quantile, combined_quantile, train_mean_std,
                                  best_f1_test, train_quantile}, computed in `prepare_test()` from the test scores
                       causal:    gate calibrated on the validation fold, updated online (delayed ACI by default)
    L3 selection       published: `prepare_test` may use test labels (best_f1_test); causal: never
    L4 point-adjust    not a model property: applied by causalgate.eval.metrics under the published protocol
    L5 availability    not a model property: the dataset's PIT fields and the protocol runner decide the join
    L7 fill            not a model property: the protocol runner prices decisions at t (same bar) or t + latency
    L8 pre-trained     subclass declares `cutoff_date`; the protocol runner compares it with the fold's test start

The event backtester and the metric matrix are identical across methods because every wrapper returns the shared
out dict {"a_prob", "w", "hold_p", "kappa_p", "abstain"} (warn2trade.baselines.event_base).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader

from warn2trade.baselines.event_base import BaseBaseline, register_event_baseline
from ..data.pit import FiltrationGuard
from ..data.stream import StreamingNormaliser, OfflineScaler
from ..models.gate import make_gate, conformal_quantile, BaseGate

MODES = ("published", "causal")
PUBLISHED_THRESHOLDS = ("test_quantile", "combined_quantile", "train_mean_std", "best_f1_test", "train_quantile", "val_quantile")


class ScoreToTradeBaseline(BaseBaseline):
    name = "cg_base"
    paper = ""
    official_repo = ""
    status = "reimplemented"            # official | reimplemented | reimplemented-fallback | wrapper-needed
    year = 0
    conventions: Dict[str, str] = {}    # L1..L8 -> what the original does; mirrored in CONVENTIONS.md
    cutoff_date: Optional[str] = None   # L8: training-data cutoff of any pre-trained component (ISO date) or None
    unspecified: List[str] = []         # choices the paper leaves open and how we fixed them
    # model-side one-switch channels this scaffold implements (norm_mode / thr_mode / published_threshold). The protocol
    # runner records a channel the class does not list as 'not applicable' instead of stamping the run 'published'.
    supported_switches = ("L1", "L2", "L3")
    # P-O conventions: the class-level source of truth for the 'published' protocol (overridden by the baseline yaml's
    # `published_norm` / `published_threshold`, see ProtocolRunner.resolve_published). Subclasses that set them via
    # cfg.setdefault in __init__ are honoured too, because the runner no longer overwrites the keys.
    published_norm_default = "global"
    published_threshold_default = "test_quantile"

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        super().__init__(cfg, dims, horizons, device)
        self.mode = cfg.get("mode", "causal")
        assert self.mode in MODES, self.mode
        # finer switches used by the one-switch protocols: normalisation and threshold can be toggled independently
        self.norm_mode = cfg.get("norm_mode", "offline" if self.mode == "published" else "streaming")     # offline | streaming
        self.thr_mode = cfg.get("thr_mode", "published" if self.mode == "published" else "gate")          # published | gate
        assert self.norm_mode in ("offline", "streaming") and self.thr_mode in ("published", "gate")
        self.panel = cfg.get("panel")
        self.L = int(cfg.get("lookback", 64))
        self.latency = int(cfg.get("latency", 1))
        self.hold_index = int(cfg.get("hold_index", 2))
        self.h = int(self.horizons[self.hold_index])
        self.ret_col = int(cfg.get("ret_col", 0))
        self.bars_per_year = int(cfg.get("bars_per_year", 252))
        self.direction_sign = float(cfg.get("direction_sign", 1.0))     # +1 momentum, -1 reversal (rule policy)
        self.alpha = cfg.get("alpha", 0.05)
        self.published_norm = cfg.get("published_norm", self.published_norm_default)
        self.published_threshold = cfg.get("published_threshold", self.published_threshold_default)
        assert self.published_threshold in PUBLISHED_THRESHOLDS, self.published_threshold
        self.anomaly_ratio = cfg.get("anomaly_ratio", None)             # published quantile level; None -> test label ratio
        self.k_std = float(cfg.get("k_std", 3.0))                       # train_mean_std rule: thr = mean + k * std
        self.guard = FiltrationGuard(strict=(self.mode == "causal"))
        ncfg = dict(cfg.get("normaliser", {}))
        self.norm = StreamingNormaliser(ncfg.get("mode", "welford"), int(ncfg.get("warmup", 60)), float(ncfg.get("halflife", 120.0)))
        self.offline_scaler = OfflineScaler()
        gcfg = dict(cfg.get("gate", {"type": "delayed_aci"}))
        n_assets = int(self.panel.N) if self.panel is not None else int(cfg.get("n_assets", 1))
        self.gate: BaseGate = make_gate(gcfg, self.bars_per_year, n_assets, self.h, self.latency, cfg.get("cost_bps"),
                                        label_delay=int(max(self.horizons)) + self.latency)
        if "alpha" in cfg and cfg["alpha"] is not None:
            self.gate.alpha = float(cfg["alpha"])
        self.alpha = float(self.gate.alpha)
        self.guard.register(self.norm); self.guard.register(self.gate)
        self._Z: Optional[np.ndarray] = None
        self._thr_published = float("inf")
        self._s_train: Optional[np.ndarray] = None
        self._s_val: Optional[np.ndarray] = None
        self._last_t = -1
        self.diag: Dict[str, Any] = {}

    # ------------------------------------------------------------------ hooks
    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        raise NotImplementedError

    def _score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _direction(self, X: np.ndarray, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        return np.sign(np.nan_to_num(X[:, -3:, self.ret_col]).sum(-1)) * self.direction_sign

    def _published_threshold(self, s_train, s_val, s_test, y_test) -> float:
        kind = self.published_threshold
        ratio = self.anomaly_ratio if self.anomaly_ratio is not None else (float(np.nanmean(y_test)) if y_test is not None and len(y_test) else 0.01)
        ratio = float(np.clip(ratio, 1e-4, 0.5))
        fin = lambda a: np.asarray(a, dtype=np.float64)[np.isfinite(np.asarray(a, dtype=np.float64))] if a is not None else np.zeros(0)
        if kind == "test_quantile":
            return float(np.quantile(fin(s_test), 1 - ratio)) if fin(s_test).size else float("inf")
        if kind == "combined_quantile":
            comb = np.concatenate([fin(s_train), fin(s_test)])
            return float(np.quantile(comb, 1 - ratio)) if comb.size else float("inf")
        if kind == "train_mean_std":
            st = fin(s_train)
            return float(st.mean() + self.k_std * st.std()) if st.size else float("inf")
        if kind == "train_quantile":
            return conformal_quantile(fin(s_train), ratio)
        if kind == "val_quantile":
            return conformal_quantile(fin(s_val) if fin(s_val).size else fin(s_train), ratio)
        if kind == "best_f1_test":
            from warn2trade.backtest.metrics import best_f1
            st = np.asarray(s_test, dtype=np.float64); ok = np.isfinite(st)
            if y_test is None or ok.sum() == 0 or np.asarray(y_test)[ok].sum() == 0:
                return float("inf")
            # best_f1 (precision_recall_curve) alarms on s >= thr while predict() alarms on s > thr: step one ulp down so
            # the realised alarm set is exactly {s >= thr_best} (the quantile kinds keep the official strict '>').
            return float(np.nextafter(best_f1(np.asarray(y_test)[ok], st[ok])[1], -np.inf))
        raise ValueError(kind)

    # ------------------------------------------------------------------ features
    def _panel_Z(self) -> np.ndarray:
        if self._Z is None:
            X = self.panel.price_feat.astype(np.float64)
            if self.norm_mode == "offline":
                rows = None if self.published_norm == "global" else self.cfg.get("norm_rows")
                self.offline_scaler.fit(X, rows)
                self._Z = self.offline_scaler.transform(X).astype(np.float32)
            else:
                self._Z = self.norm.transform_series(X).astype(np.float32)
        return self._Z

    def _windows(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        if self.panel is not None:
            Z = self._panel_Z()
            t = batch["t"].cpu().numpy(); i = batch["asset"].cpu().numpy()
            idx = t[:, None] - self.L + 1 + np.arange(self.L)[None, :]
            return Z[idx, i[:, None]]
        x = batch["price"].cpu().numpy().astype(np.float64)
        if self.norm_mode == "offline":
            return x.astype(np.float32)                                  # builder features taken as-is (already offline-normalised)
        out = np.empty_like(x)
        for b in range(x.shape[0]):
            out[b] = StreamingNormaliser(self.norm.mode, min(self.norm.warmup, self.L // 2), self.norm.halflife).transform_series(x[b])
        return out.astype(np.float32)

    def _collect(self, loader: DataLoader):
        Xs, ys, ts = [], [], []
        for b in loader:
            Xs.append(self._windows(b)); ys.append(b["y"].cpu().numpy()); ts.append(b["t"].cpu().numpy())
        if not Xs:
            return np.zeros((0, self.L, self.dims["price"]), np.float32), np.zeros(0), np.zeros(0, dtype=int)
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts)

    # ------------------------------------------------------------------ fit / prepare_test / predict
    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        X, y, t = self._collect(train)
        t_end = int(t.max()) if len(t) else -1
        self._fit_model(X, t_end)
        self._s_train = self._score(X) if len(X) else np.zeros(0)
        Xv, yv, tv = self._collect(val) if val is not None else (X, y, t)
        self._s_val = self._score(Xv) if len(Xv) else self._s_train
        t_cal = int(tv.max()) if len(tv) else t_end
        src = val if val is not None else train
        lav = self._label_avail_of(src) if len(yv) else None
        self.gate.calibrate(self._s_val, t_end=t_cal, labels=yv if len(yv) else None, label_avail=lav)
        self.diag.update({"mode": self.mode, "norm_mode": self.norm_mode, "thr_mode": self.thr_mode, "train_windows": int(len(X)), "scorer_t_end": t_end, "gate_as_of": int(self.gate.as_of), "alpha": self.alpha})
        self._last_t = -1

    def _avail_label(self) -> Optional[np.ndarray]:
        if self.panel is None:
            return None
        from ..data.pit import get_avail
        return get_avail(self.panel, "avail_label")

    def _label_avail_of(self, loader: DataLoader) -> Optional[np.ndarray]:
        """Availability bar of each row's label (same order as _collect); None if the panel carries no PIT field."""
        A = self._avail_label()
        if A is None:
            return None
        out = [A[b["t"].cpu().numpy(), b["asset"].cpu().numpy()] for b in loader]
        return np.concatenate(out) if out else np.zeros(0, dtype=int)

    def prepare_test(self, test: DataLoader) -> None:
        """Published threshold only: read the threshold off the test scores / labels (L2, L3). The gate never looks."""
        if self.thr_mode != "published":
            return
        Xt, yt, _ = self._collect(test)
        st = self._score(Xt) if len(Xt) else np.zeros(0)
        self._thr_published = self._published_threshold(self._s_train, self._s_val, st, yt)
        self.diag.update({"published_threshold": self.published_threshold, "thr": float(self._thr_published), "test_ratio": float(np.mean(yt)) if len(yt) else float("nan")})

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        t = batch["t"].cpu().numpy(); i = batch["asset"].cpu().numpy()
        X = self._windows(batch)
        s = self._score(X)
        B = len(t)
        alarm = np.zeros(B, dtype=bool); a_prob = np.zeros(B); thr_row = np.full(B, np.inf)
        if self.thr_mode == "published":
            thr = self._thr_published
            thr_row[:] = thr
            fin = np.isfinite(s)
            alarm[fin] = s[fin] > thr
            ref = self._s_train if self._s_train is not None and len(self._s_train) else s
            ref = ref[np.isfinite(ref)]
            if ref.size:
                a_prob[fin] = np.searchsorted(np.sort(ref), s[fin]) / ref.size
        else:
            if int(t.min()) < self._last_t:                    # strictly earlier bar = a new pass (a bar may straddle two batches)
                self.gate.reset_stream()
            y = batch["y"].cpu().numpy()
            A = self._avail_label()
            y_av = A[t, i] if A is not None else None
            # fallback reference for gates without a rolling buffer (fixed / val_quantile): the frozen calibration
            # scores (validation fold, else train fold) -- past-only, fixed at fit time
            ref = self._s_val if self._s_val is not None and len(self._s_val) else self._s_train
            ref = np.sort(ref[np.isfinite(ref)]) if ref is not None else np.zeros(0)
            for k in np.lexsort((i, t)):
                # the label is passed only with its availability bar (the gate queues it until then); never same-bar
                if y_av is not None:
                    al, q = self.gate.step(float(s[k]), int(t[k]), float(y[k]), int(y_av[k]))
                else:
                    al, q = self.gate.step(float(s[k]), int(t[k]))
                alarm[k] = al
                thr_row[k] = q if np.isfinite(q) else np.inf
                buf = np.asarray(getattr(self.gate, "buf", []), dtype=np.float64)
                if buf.size and np.isfinite(s[k]):
                    a_prob[k] = float((buf < s[k]).mean())
                elif ref.size and np.isfinite(s[k]):
                    a_prob[k] = float(np.searchsorted(ref, s[k], side="left") / ref.size)
            self._last_t = int(t.max())
        d = self._direction(X, batch)
        dev = self.device
        a_t = torch.as_tensor(alarm, dtype=torch.float32, device=dev)
        w = a_t * torch.as_tensor(np.nan_to_num(d), dtype=torch.float32, device=dev) * self.w_max
        return {"a_prob": torch.as_tensor(a_prob, dtype=torch.float32, device=dev), "w": w, "abstain": 1.0 - a_t,
                "hold_p": self._hold_fixed(B, self.hold_index), "kappa_p": self._immediate(B),
                "score": torch.as_tensor(np.nan_to_num(s, nan=-1.0), dtype=torch.float32, device=dev),
                "scored": torch.as_tensor(np.isfinite(s), dtype=torch.bool, device=dev),
                # per-row threshold actually applied (NaN-free; +inf where the gate cannot alarm)
                "thr": torch.as_tensor(np.nan_to_num(thr_row, nan=np.inf), dtype=torch.float64, device=dev)}


def register(name: str):
    """Tolerant registration under the cg_ namespace (first registration wins)."""
    return register_event_baseline(name)

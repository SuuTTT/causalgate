"""CausalGate: streaming normaliser -> causal scorer -> delayed-ACI alarm gate -> Warn2Trade policy head, emitting the
shared event-baseline `out` dict so that warn2trade.backtest.engine.EventBacktester and the metric matrix are reused
verbatim.

    out = {"a_prob": (B,), "w": (B,), "hold_p": (B, H), "kappa_p": (B, n_exec), "abstain": (B,)}

Causality by construction (Filtration Guard):
    * features are standardised by a causal sweep (stats from bars < t; warm-up bars abstain);
    * the scorer is fitted on the purged train fold and stamped as-of its last bar;
    * the gate is calibrated on the validation fold and updated online with feedback delayed by tau = h + latency;
      labels never reach the gate before their availability bar (panel.avail_label, fallback t + max(h) + latency):
      calibration rows whose label is not yet available at the calibration end are not label-filtered, and online
      the gate is label-free unless cfg['gate']['online_labels'] is true, in which case each label is released to the
      gate (calib='normal_only' only) at its availability bar and the gate is stamped with it;
    * scorer refits: 'static' (default), 'periodic' (every refit_every decision bars, on the trailing past windows
      ending <= the decision bar - refit_purge) or 'streaming' (one prequential SGD step per batch); a new pass
      restores the post-fit scorer as well as the post-calibration gate;
    * every predict() call processes rows in time order and the gate / scorer are `check`ed at each decision bar.
    * A second pass over the same test window (e.g. re-pricing at another cost level) restarts the gate from its
      post-calibration snapshot, so the batch backtest and a bar-by-bar replay coincide (Prop. 1, DG = 0).

Policy heads: 'momentum' (sign of the last three standardised returns on alarm, fixed holding horizon) or
'learned' (warn2trade.models.policy.RiskAwarePolicy trained with the differentiable event PnL on the train fold).
Panel-level normalisation (cfg['panel']) is preferred; without a panel the window itself is normalised causally.
Protocol switches (same keys as causalgate/baselines/common.py): cfg['norm_mode'] in {streaming (default), offline};
offline = OfflineScaler fitted on the whole panel (published_norm='global') or on cfg['norm_rows'] (the L1 channel,
used by the no_fg ablation); then the guard is non-strict and logs the violation instead of raising.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from warn2trade.baselines.event_base import BaseBaseline, register_event_baseline
from warn2trade.backtest.costs import CostModel
from warn2trade.backtest.engine import event_pnl
from warn2trade.models.policy import RiskAwarePolicy
from warn2trade.models.encoders import liq_transform
from ..data.pit import FiltrationGuard
from ..data.stream import StreamingNormaliser, OfflineScaler
from .scorer import make_scorer
from .gate import make_gate, conformal_quantile


@register_event_baseline("causalgate")
class CausalGate(BaseBaseline):
    name = "causalgate"
    paper = "CausalGate (this paper): streaming normalisation + causal scorer + delayed-ACI alarm gate + policy head."
    status = "implemented"

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        super().__init__(cfg, dims, horizons, device)
        self.panel = cfg.get("panel")
        self.L = int(cfg.get("lookback", 64))
        self.latency = int(cfg.get("latency", 1))
        self.hold_index = int(cfg.get("hold_index", 2))
        self.h = int(self.horizons[self.hold_index])
        self.bars_per_year = int(cfg.get("bars_per_year", 252))
        self.ret_col = int(cfg.get("ret_col", 0))
        n_assets = int(self.panel.N) if self.panel is not None else int(cfg.get("n_assets", 1))
        self.hmax = int(max(self.horizons))
        self.label_delay = self.hmax + self.latency                             # fallback label availability lag
        self.norm_mode = cfg.get("norm_mode", "streaming")
        assert self.norm_mode in ("streaming", "offline"), self.norm_mode
        self.published_norm = cfg.get("published_norm", "global")
        self.norm_rows = cfg.get("norm_rows")
        self.mode = cfg.get("mode", "published" if self.norm_mode == "offline" else "causal")
        ncfg = dict(cfg.get("normaliser", {}))
        self.guard = FiltrationGuard(strict=(self.mode == "causal" and self.norm_mode == "streaming"))
        self.norm = StreamingNormaliser(ncfg.get("mode", "welford"), int(ncfg.get("warmup", 60)), float(ncfg.get("halflife", 120.0)))
        self.guard.register(self.norm)
        self.offline_scaler = OfflineScaler()
        if self.norm_mode == "offline":
            self.guard.register(self.offline_scaler)
        self.scorer = self.guard.register(make_scorer(dict(cfg.get("scorer", {"type": "tcn"})), dims["price"], self.L, device))
        self.refit_purge = int(dict(cfg.get("scorer", {})).get("refit_purge", 0))
        gcfg = dict(cfg.get("gate", {"type": "delayed_aci"}))
        self.online_labels = bool(gcfg.pop("online_labels", False))
        self.gate = self.guard.register(make_gate(gcfg, self.bars_per_year, n_assets, self.h, self.latency, cfg.get("cost_bps"),
                                                  label_delay=self.label_delay))
        self.alpha = self.gate.alpha
        pcfg = dict(cfg.get("policy", {}))
        self.policy_type = pcfg.get("type", "momentum")
        self.policy: Optional[RiskAwarePolicy] = None
        if self.policy_type == "learned":
            self.policy = RiskAwarePolicy(7, len(self.horizons), self.n_exec, self.w_max, int(pcfg.get("hidden", 64)), True, float(pcfg.get("dropout", 0.0))).to(device)
            self.policy_epochs, self.policy_lr = int(pcfg.get("epochs", 3)), float(pcfg.get("lr", 1e-3))
            self.lam_down = float(pcfg.get("lam_down", 1.0))
            self.cost_model = CostModel(float(cfg.get("cost_bps", 5.0)))
        self._Z: Optional[np.ndarray] = None
        self._last_t = -1
        self._fit_t_end = -1
        self._hist_X: Optional[np.ndarray] = None                                # past windows for periodic refits
        self._hist_t: Optional[np.ndarray] = None
        self._last_refit = -1
        self._refit_snap: Optional[Dict[str, Any]] = None
        self.diag: Dict[str, Any] = {}

    # ------------------------------------------------------------------ features
    def _panel_Z(self) -> np.ndarray:
        if self._Z is None:
            X = self.panel.price_feat.astype(np.float64)
            if self.norm_mode == "offline":
                rows = None if self.published_norm == "global" else self.norm_rows
                self.offline_scaler.fit(X, rows)
                self._Z = self.offline_scaler.transform(X).astype(np.float32)
            else:
                self._Z = self.norm.reset().transform_series(X).astype(np.float32)   # fresh causal sweep every rebuild
        return self._Z

    def _label_avail(self, t: np.ndarray, i: np.ndarray) -> np.ndarray:
        av = getattr(self.panel, "avail_label", None) if self.panel is not None else None
        if av is not None:
            return np.asarray(av)[t, i].astype(np.int64)
        return np.asarray(t, dtype=np.int64) + self.label_delay

    # ------------------------------------------------------------------ scorer refits (periodic / streaming)
    def _push_hist(self, X: np.ndarray, t: np.ndarray) -> None:
        keep = 2 * self.scorer.refit_window
        self._hist_X = X if self._hist_X is None else np.concatenate([self._hist_X, X])[-keep:]
        self._hist_t = t if self._hist_t is None else np.concatenate([self._hist_t, t])[-keep:]

    def _snap_refit(self) -> None:
        if self.scorer.refit_policy == "static":
            return
        st = {k: v for k, v in self.scorer.__dict__.items() if k != "guard"}
        self._refit_snap = copy.deepcopy({"scorer": st, "hist_X": self._hist_X, "hist_t": self._hist_t, "last_refit": self._last_refit})

    def _restore_refit(self) -> None:
        if self._refit_snap is None:
            return
        snap = copy.deepcopy(self._refit_snap)
        self.scorer.__dict__.update(snap["scorer"])
        self._hist_X, self._hist_t, self._last_refit = snap["hist_X"], snap["hist_t"], snap["last_refit"]

    def _periodic_refit(self, tk: int) -> bool:
        if self.scorer.refit_policy != "periodic" or tk - self._last_refit < self.scorer.refit_every or self._hist_t is None:
            return False
        ok = self._hist_t <= tk - self.refit_purge
        if not ok.any():
            return False
        Xp, tp = self._hist_X[ok][-self.scorer.refit_window:], self._hist_t[ok][-self.scorer.refit_window:]
        self.scorer.refit(Xp, int(tp.max()))
        self._last_refit = tk
        self.diag["n_refits"] = int(self.diag.get("n_refits", 0)) + 1
        return True

    def _windows(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        if self.panel is not None:
            Z = self._panel_Z()
            t = batch["t"].cpu().numpy(); i = batch["asset"].cpu().numpy()
            idx = t[:, None] - self.L + 1 + np.arange(self.L)[None, :]
            return Z[idx, i[:, None]]                                           # (B, L, F)
        x = batch["price"].cpu().numpy().astype(np.float64)
        if self.norm_mode == "offline":
            return x.astype(np.float32)                                         # builder features taken as-is
        out = np.empty_like(x)
        for b in range(x.shape[0]):                                             # window-local causal sweep
            out[b] = StreamingNormaliser(self.norm.mode, min(self.norm.warmup, self.L // 2), self.norm.halflife).transform_series(x[b])
        return out.astype(np.float32)

    @staticmethod
    def _mom3(X: np.ndarray, col: int) -> np.ndarray:
        return np.nan_to_num(X[:, -3:, col]).sum(-1)

    # ------------------------------------------------------------------ fit
    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        Xs: List[np.ndarray] = []; Bs: List[Dict[str, torch.Tensor]] = []; Ts: List[np.ndarray] = []
        t_max = -1
        for b in train:
            Xs.append(self._windows(b)); Bs.append(b); Ts.append(b["t"].cpu().numpy())
            t_max = max(t_max, int(b["t"].max()))
        X = np.concatenate(Xs) if Xs else np.zeros((0, self.L, self.dims["price"]), np.float32)
        self.scorer.fit(X, t_end=t_max)
        self._fit_t_end = t_max
        s_train = self.scorer.score(X) if len(X) else np.zeros(0)
        # ---- gate calibration on the validation fold (falls back to the train fold)
        s_cal, y_cal, a_cal, t_cal = [], [], [], t_max
        Xv: List[np.ndarray] = []; Tv: List[np.ndarray] = []
        src = val if val is not None else train
        for b in src:
            Xb = self._windows(b)
            tb, ib = b["t"].cpu().numpy(), b["asset"].cpu().numpy()
            s_cal.append(self.scorer.score(Xb)); y_cal.append(b["y"].cpu().numpy()); a_cal.append(self._label_avail(tb, ib))
            t_cal = max(t_cal, int(tb.max()))
            if val is not None and self.scorer.refit_policy == "periodic":
                Xv.append(Xb); Tv.append(tb)
        s_cal = np.concatenate(s_cal) if s_cal else s_train
        y_cal = np.concatenate(y_cal) if y_cal else None
        a_cal = np.concatenate(a_cal) if a_cal else None
        self.gate.calibrate(s_cal, t_end=t_cal, labels=y_cal, label_avail=a_cal)
        self._hist_X = self._hist_t = None
        if self.scorer.refit_policy == "periodic" and len(X):
            self._push_hist(X, np.concatenate(Ts))
            if Xv:
                self._push_hist(np.concatenate(Xv), np.concatenate(Tv))
        self._last_refit = t_max
        self.diag.update({"train_windows": int(len(X)), "calib_scores": int(np.isfinite(s_cal).sum()), "alpha": self.alpha,
                          "scorer_as_of": int(self.scorer.as_of), "gate_as_of": int(self.gate.as_of)})
        # ---- learned policy (utility-trained on the train fold with a train-fold fixed alarm threshold)
        if self.policy is not None and len(X):
            thr_train = conformal_quantile(s_train, self.alpha)
            opt = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
            self.policy.train()
            off = 0
            for _ in range(self.policy_epochs):
                off = 0
                for b, Xb in zip(Bs, Xs):
                    n = len(Xb)
                    s = s_train[off : off + n]; off += n
                    alarm = np.isfinite(s) & (s > thr_train)
                    bd = {k: v.to(self.device) for k, v in b.items()}
                    feats = self._policy_feats(s, alarm, Xb, bd)
                    pol = self.policy(feats)
                    a = torch.as_tensor(alarm, dtype=torch.float32, device=self.device)
                    out = {"w": pol["w"] * a, "hold_p": pol["hold_p"], "kappa_p": pol["kappa_p"]}
                    pnl = event_pnl(out, bd, self.horizons, self.latency, self.cost_model, hard=False)["pnl"] * 100.0
                    loss = -pnl.mean() + self.lam_down * (torch.clamp(-pnl, min=0.0) ** 2).mean()
                    if torch.isfinite(loss):
                        opt.zero_grad(); loss.backward(); opt.step()
            self.policy.eval()
        self._last_t = -1
        self._snap_refit()

    def _policy_feats(self, s: np.ndarray, alarm: np.ndarray, X: np.ndarray, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        s_t = torch.as_tensor(np.nan_to_num(s), dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(alarm, dtype=torch.float32, device=self.device)
        mom = torch.as_tensor(self._mom3(X, self.ret_col), dtype=torch.float32, device=self.device)
        return torch.cat([s_t[:, None], a_t[:, None], mom[:, None], liq_transform(batch["liq"]), torch.log(batch["sigma"][:, None] * 1e2 + 1e-4)], -1)

    # ------------------------------------------------------------------ predict (sequential, F_t-measurable)
    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        t = batch["t"].cpu().numpy(); i = batch["asset"].cpu().numpy()
        order = np.lexsort((i, t))
        if int(t.min()) < self._last_t:                                        # a strictly earlier bar = a new pass: replay from the calibration snapshot
            self.gate.reset_stream()
            self._restore_refit()
        X = self._windows(batch)
        s_all = self.scorer.score(X)
        B = len(t)
        alarm = np.zeros(B, dtype=bool); thr = np.full(B, np.nan); a_prob = np.zeros(B)
        if self.online_labels:                                                 # labels reach the gate only at avail_label
            y = batch["y"].cpu().numpy(); av = self._label_avail(t, i)
        for pos, k in enumerate(order):
            tk = int(t[k])
            if self._periodic_refit(tk):                                       # re-score this and the later rows of the batch
                rest = order[pos:]
                s_all[rest] = self.scorer.score(X[rest])
            self.scorer.check(tk)
            if self.norm_mode == "offline":
                self.offline_scaler.check(tk)                                  # L1: logged (non-strict guard)
            if self.online_labels:
                al, q = self.gate.step(float(s_all[k]), tk, float(y[k]), int(av[k]))
            else:
                al, q = self.gate.step(float(s_all[k]), tk)
            alarm[k], thr[k] = al, q
            buf = np.asarray(getattr(self.gate, "buf", []), dtype=np.float64)
            if buf.size and np.isfinite(s_all[k]):
                a_prob[k] = float((buf < s_all[k]).mean())
            elif np.isfinite(s_all[k]) and np.isfinite(q) and q not in (np.inf, -np.inf):
                a_prob[k] = float(1.0 / (1.0 + np.exp(-(s_all[k] - q))))
        self._last_t = int(t.max())
        if self.scorer.refit_policy == "streaming":
            self.scorer.update(X, int(t.max()))                               # windows end <= t.max: known after this batch
        elif self.scorer.refit_policy == "periodic":
            self._push_hist(X, t)
        dev = self.device
        a_t = torch.as_tensor(alarm, dtype=torch.float32, device=dev)
        if self.policy is not None:
            pol = self.policy(self._policy_feats(s_all, alarm, X, {k: v.to(dev) for k, v in batch.items()}))
            w = pol["w"] * a_t
            hold_p, kappa_p = pol["hold_p"], pol["kappa_p"]
        else:
            mom = torch.as_tensor(np.sign(self._mom3(X, self.ret_col)), dtype=torch.float32, device=dev)
            w = a_t * mom * self.w_max
            hold_p, kappa_p = self._hold_fixed(B, self.hold_index), self._immediate(B)
        return {"a_prob": torch.as_tensor(a_prob, dtype=torch.float32, device=dev), "w": w, "abstain": 1.0 - a_t,
                "hold_p": hold_p, "kappa_p": kappa_p, "score": torch.as_tensor(np.nan_to_num(s_all, nan=-1.0), dtype=torch.float32, device=dev),
                "scored": torch.as_tensor(np.isfinite(s_all), device=dev),
                "thr": torch.as_tensor(np.nan_to_num(thr, nan=np.inf, posinf=1e9), dtype=torch.float32, device=dev)}

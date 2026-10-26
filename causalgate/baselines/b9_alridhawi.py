"""B9 Al Ridhawi, Haj Ali, Al Osman, "Adaptive Regime-Aware Stock Price Prediction Using Autoencoder-Gated Dual Node
Transformers with Reinforcement Learning Control", arXiv:2603.19136 (2026). status = reimplemented-fallback
(the SAC controller is replaced by a deterministic bounded update driven by the same reward; see `unspecified`).

From the paper (arxiv.org/html/2603.19136v2):
  * Sec. III-B, Eq. 3: expanding-window z-scores x~_{i,t} = (x_{i,t} - mu_{1:t}) / sigma_{1:t} within the training period;
    validation / test use the frozen statistics of the FULL TRAINING period -> L1 is train-fitted (not global).
  * Sec. III-C: autoencoder FC(64) - FC(32) - z (32) with a symmetric decoder, trained for 20 epochs only on 'stable'
    training days (VIX below the 75th percentile of its training-period distribution); anomaly score
    e_t = ||x_t - x^_t||^2; threshold tau initialised at the 95th percentile of the TRAINING-set reconstruction errors.
  * Sec. III-D: dual node transformers (6 layers, d_model 512, 8 heads, causal mask) for the normal and the event
    pathway; blended forecast y^ = alpha_t y^_normal + (1 - alpha_t) y^_event.
  * Sec. III-E: SAC controller, state s_t = [e_t, mean e over the last 5 days, sigma_t, RMSE_{t-1}, DA_{t-1},
    alpha_{t-1}, tau_{t-1}], action [delta tau, delta alpha] in [-0.1, 0.1]^2, reward
    r_t = -RMSE_t - 0.5 (1 - DA_t) - 0.1 |delta tau| computed on the previous day's realised close (one-day delay,
    Sec. IV-H); the policy is frozen on the test period (state-dependent, parameter-static inference).
  * Targets: close at horizons {1, 5, 20}; split 1982-2010 / 2011-2016 / 2017-2025; 20 S&P 500 stocks; no trading
    rule, Sharpe or PnL is reported (the trade loop is the scaffold's).

Scaffold mapping: score for the gate = e_t (published mode: e_t - (tau_t - tau_0) so that the scalar published
threshold tau_0 reproduces the controller's time-varying rule e_t > tau_t); direction = sign of the blended one-step
forecast.
"""
from __future__ import annotations
import copy
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.pit import AsOfStamped
from .common import ScoreToTradeBaseline, register
from .b1_cpd_dmn import RowAwareMixin, native_threshold_requested



class _AlRidhawiModel(AsOfStamped):
    """Fitted AE + dual predictors; stamped at the end of the training fold."""


class _AlRidhawiThreshold(AsOfStamped):
    """tau_0 = 95th percentile of the training-set reconstruction errors; stamped at the end of the training fold."""


class _AE(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], latent: int):
        super().__init__()
        h1, h2 = int(hidden[0]), int(hidden[1])
        self.e1, self.e2, self.ez = nn.Linear(in_dim, h1), nn.Linear(h1, h2), nn.Linear(h2, latent)
        self.d1, self.d2, self.dx = nn.Linear(latent, h2), nn.Linear(h2, h1), nn.Linear(h1, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ez(F.relu(self.e2(F.relu(self.e1(x)))))
        return self.dx(F.relu(self.d2(F.relu(self.d1(z)))))               # linear output layer (Sec. III-C)


class _GRUHead(nn.Module):
    """Stand-in for a 6-layer node transformer: a causal GRU over the window (the recurrence is causal by construction)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.out(h[:, -1]).squeeze(-1)


class _Controller:
    """Deterministic, bounded, gradient-free stand-in for the SAC controller (Sec. III-E): per asset it keeps
    (tau, alpha) and the previous forecast, computes the paper's reward with a one-day delay from the realised return,
    and applies actions |delta tau| <= 0.1 tau_0 and |delta alpha| <= 0.1 : alpha hill-climbs on the delayed reward,
    tau tracks the recent (last `recent` days) 95th percentile of e_t. `online=False` freezes both after calibration."""

    def __init__(self, tau0: float, alpha0: float = 0.5, eta_alpha: float = 0.05, k: int = 5, recent: int = 60, online: bool = True):
        self.tau0, self.alpha0, self.eta, self.k, self.recent, self.online = float(tau0), float(alpha0), float(eta_alpha), int(k), int(recent), bool(online)
        self.state: Dict[int, Dict[str, Any]] = {}
        self._snap: Optional[Dict[int, Dict[str, Any]]] = None
        self.frozen = False

    def _new(self) -> Dict[str, Any]:
        return {"tau": self.tau0, "alpha": self.alpha0, "prev_f": None, "prev_reward": None, "prev_dalpha": self.eta, "prev_dtau": 0.0,
                "hist": deque(maxlen=self.k), "recent": deque(maxlen=self.recent)}

    def snapshot(self) -> None:
        self._snap = copy.deepcopy(self.state)

    def restore(self) -> None:
        if self._snap is not None:
            self.state = copy.deepcopy(self._snap)

    def step(self, asset: int, e: float, r_real: float, f_normal: float, f_event: float) -> Tuple[float, float, float, float]:
        st = self.state.setdefault(int(asset), self._new())
        reward = float("nan")
        if st["prev_f"] is not None and np.isfinite(r_real):
            rmse = abs(st["prev_f"] - r_real)
            da = 1.0 if np.sign(st["prev_f"]) == np.sign(r_real) else 0.0
            reward = -rmse - 0.5 * (1.0 - da) - 0.1 * abs(st["prev_dtau"]) / max(self.tau0, 1e-12)   # Sec. IV-H, delayed one day
        if self.online and not self.frozen:
            dalpha = 0.0
            if np.isfinite(reward) and st["prev_reward"] is not None:
                dalpha = float(np.clip(self.eta * np.sign(reward - st["prev_reward"]) * np.sign(st["prev_dalpha"] or 1.0), -0.1, 0.1))
                if dalpha == 0.0:
                    dalpha = self.eta
            st["alpha"] = float(np.clip(st["alpha"] + dalpha, 0.0, 1.0))
            st["prev_dalpha"] = dalpha
            if np.isfinite(e):
                st["recent"].append(float(e))
            if len(st["recent"]) >= 20:
                target = float(np.quantile(np.asarray(st["recent"]), 0.95))
                dtau = float(np.clip(target - st["tau"], -0.1 * self.tau0, 0.1 * self.tau0))
            else:
                dtau = 0.0
            st["tau"] = float(st["tau"] + dtau)
            st["prev_dtau"] = dtau
        if np.isfinite(reward):
            st["prev_reward"] = reward
        if np.isfinite(e):
            st["hist"].append(float(e))
        f = st["alpha"] * f_normal + (1.0 - st["alpha"]) * f_event
        st["prev_f"] = float(f) if np.isfinite(f) else None
        return st["tau"], st["alpha"], float(f), reward


@register("cg_alridhawi")
class AlRidhawiBaseline(RowAwareMixin, ScoreToTradeBaseline):
    name = "cg_alridhawi"
    paper = "Al Ridhawi, Haj Ali, Al Osman, Adaptive Regime-Aware Stock Price Prediction Using AE-Gated Dual Node Transformers with RL Control, arXiv:2603.19136 (2026)"
    official_repo = ""
    status = "reimplemented-fallback"
    year = 2026
    cutoff_date = None
    conventions = {
        "L1": "Train-fitted, not global: expanding-window z-scores inside the training period and the FULL-TRAINING-PERIOD mean / std frozen for "
              "validation and test (Sec. III-B, Eq. 3). published mode: `published_norm: train` (OfflineScaler on the train fold rows); "
              "causal mode: streaming (expanding) z-score, i.e. the paper's own training-time rule applied throughout.",
        "L2": "tau_0 = 95th percentile of the TRAINING-set reconstruction errors (Sec. III-C), then moved by the controller with "
              "|delta tau| <= 0.1 using a reward on the previous day's realised close (one-day delay, Sec. III-E / IV-H): causal.",
        "L3": "Chronological split 1982-2010 / 2011-2016 / 2017-2025 (Sec. IV); early stopping on validation loss (AE); the SAC policy is "
              "frozen on the test period (state-dependent inference only). No test-set selection reported.",
        "L4": "No anomaly labels; e_t > tau_t routes the regime. Metrics are MAPE / RMSE / DA / Theil's U / CTR on prices (Sec. IV).",
        "L5": "Unspecified for the event-pathway context c_t (sentiment spike from a BERT model over X posts 2007-2025, earnings features whose "
              "source and availability time are not stated, VIX regime terciles); the availability time relative to the forecast is not stated.",
        "L6": "20 named S&P 500 stocks over 1982-2025 (16 from 1982, 4 from their IPO): today's constituents, i.e. survivorship-biased universe (not discussed).",
        "L7": "Forecast of the close at t + h made with data up to the close of t; no trading rule, no execution timing, no costs. "
              "The protocol runner prices the scaffold's trades at t (published) or t + latency (causal).",
        "L8": "BERT fine-tuned on the Market Sentiment Evaluation dataset (sentiment features for 2007+): training cutoff of that model not stated; "
              "not used here (no text), so cutoff_date = None.",
    }
    unspecified = [
        "SAC controller replaced by a deterministic bounded gradient-free update of (tau, alpha) driven by the same delayed reward "
        "r_t = -RMSE_t - 0.5 (1 - DA_t) - 0.1 |delta tau|: alpha hill-climbs on the reward with |delta alpha| <= 0.1, tau tracks the recent "
        "95th percentile of e_t with |delta tau| <= 0.1 tau_0 (fallback marked in `status`).",
        "Online adaptation on the test period: the paper freezes the SAC parameters but recomputes actions from the state; cfg "
        "`controller_online` = True (default, the paper's state-dependent inference) keeps updating (tau, alpha) on test, False freezes both "
        "at their end-of-validation values (the alternative reading).",
        "VIX is not available: the 'stable day' filter uses the realised volatility of the last 20 bars of the asset's own returns as a proxy "
        "(stable = below the 75th percentile of the training rows).",
        "Dual node transformers (6 layers, d 512) replaced by two small causal GRUs on the window; the event pathway's context vector c_t "
        "(sentiment spike, earnings, VIX regime, cross-asset stress) is replaced by the realised-volatility proxy channel (L5 unspecified).",
        "The AE reconstructs the last `ae_bars` bars of the scaffold-normalised features (the paper: the 17-dimensional daily feature vector).",
        "Forecast target = next-bar log return scaled by the window's return std (the paper predicts normalised closes at h in {1, 5, 20}); "
        "direction = sign of the blended one-step forecast.",
        "tau_0 uses all training rows (the paper does not say whether the 95th percentile is over all or only the stable training days).",
        "alpha_0 = 0.5 (initial blending weight not stated).",
        "No early stopping / validation-loss model selection in the smoke config (2 epochs).",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_norm", "train")
        cfg.setdefault("published_threshold", "train_quantile")
        super().__init__(cfg, dims, horizons, device)
        self._init_rows()
        a = dict(cfg.get("ae", {}))
        self.ae_bars, self.ae_hidden, self.ae_latent = int(a.get("bars", 5)), tuple(a.get("hidden", (16, 8))), int(a.get("latent", 4))
        p = dict(cfg.get("predictor", {}))
        self.pred_hidden = int(p.get("hidden", 8))
        self.epochs, self.lr, self.batch_size = int(cfg.get("epochs", 2)), float(cfg.get("lr", 1e-3)), int(cfg.get("batch_size", 256))
        self.vol_bars, self.stable_q, self.tau_q = int(cfg.get("vol_bars", 20)), float(cfg.get("stable_quantile", 0.75)), float(cfg.get("tau_quantile", 0.95))
        c = dict(cfg.get("controller", {}))
        self.ctrl_cfg = {"alpha0": float(c.get("alpha0", 0.5)), "eta_alpha": float(c.get("eta_alpha", 0.05)), "k": int(c.get("k", 5)),
                         "recent": int(c.get("recent", 60)), "online": bool(cfg.get("controller_online", True))}
        self.model_op = self._register_op(_AlRidhawiModel())
        self.thr_op = self._register_op(_AlRidhawiThreshold())
        self.ae: Optional[_AE] = None
        self.pred_normal: Optional[_GRUHead] = None
        self.pred_event: Optional[_GRUHead] = None
        self.ctrl: Optional[_Controller] = None
        self.tau0 = float("inf")
        self.vol_thr = float("inf")
        self._ctrl_last_t = -1
        self._last_dir: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ pieces
    def _ae_input(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(X[:, -self.ae_bars :, :]).reshape(len(X), -1).astype(np.float32)

    def _vol_proxy(self, X: np.ndarray) -> np.ndarray:
        R = self._raw_returns(X)
        return np.nan_to_num(R[:, -self.vol_bars :]).std(1)

    def _window_sigma(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(np.nan_to_num(self._raw_returns(X)).std(1), 1e-4)

    def _event_input(self, X: np.ndarray, vol: np.ndarray) -> np.ndarray:
        reg = np.broadcast_to((vol / max(self.vol_thr, 1e-12))[:, None, None], (len(X), X.shape[1], 1))
        return np.concatenate([np.nan_to_num(X), reg.astype(np.float32)], -1).astype(np.float32)

    def _errors(self, X: np.ndarray) -> np.ndarray:
        self.ae.eval()
        xin = torch.as_tensor(self._ae_input(X), device=self.device)
        with torch.no_grad():
            e = ((self.ae(xin) - xin) ** 2).sum(-1).cpu().numpy()        # e_t = ||x_t - x^_t||^2 (Sec. III-C2)
        return e.astype(np.float64)

    def _forecasts(self, X: np.ndarray, vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.pred_normal.eval(); self.pred_event.eval()
        with torch.no_grad():
            fn = self.pred_normal(torch.as_tensor(np.nan_to_num(X).astype(np.float32), device=self.device)).cpu().numpy()
            fe = self.pred_event(torch.as_tensor(self._event_input(X, vol), device=self.device)).cpu().numpy()
        return fn.astype(np.float64), fe.astype(np.float64)

    def _train_net(self, net: nn.Module, x: np.ndarray, target: Optional[np.ndarray]) -> None:
        n = len(x)
        if n < 8:
            return
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        xt = torch.as_tensor(x, device=self.device)
        yt = None if target is None else torch.as_tensor(target.astype(np.float32), device=self.device)
        g = torch.Generator().manual_seed(int(self.cfg.get("seed", 0)))
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for a in range(0, n, self.batch_size):
                idx = perm[a : a + self.batch_size]
                out = net(xt[idx])
                loss = ((out - xt[idx]) ** 2).mean() if yt is None else ((out - yt[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    # ------------------------------------------------------------------ hooks
    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        rows = self._rows_for(X)
        ok = np.isfinite(X).all((1, 2))
        vol = self._vol_proxy(X)
        self.vol_thr = float(np.quantile(vol[ok], self.stable_q)) if ok.any() else float("inf")
        stable = ok & (vol < self.vol_thr)
        torch.manual_seed(int(self.cfg.get("seed", 0)))
        Fdim = X.shape[-1]
        self.ae = _AE(self.ae_bars * Fdim, self.ae_hidden, self.ae_latent).to(self.device)
        self.pred_normal = _GRUHead(Fdim, self.pred_hidden).to(self.device)
        self.pred_event = _GRUHead(Fdim + 1, self.pred_hidden).to(self.device)
        self._train_net(self.ae, self._ae_input(X[stable]), None)                      # AE on stable training days only
        sig = self._window_sigma(X)
        y = (rows["fut_ret"][:, 0] if rows is not None else np.zeros(len(X))) / sig       # vol-scaled next-bar return
        self._train_net(self.pred_normal, np.nan_to_num(X[stable]).astype(np.float32), y[stable])
        self._train_net(self.pred_event, self._event_input(X[ok], vol[ok]), y[ok])
        e = self._errors(X[ok]) if ok.any() else np.zeros(0)
        self.tau0 = float(np.quantile(e, self.tau_q)) if e.size else float("inf")       # tau_0: 95th pct of training errors
        self.ctrl = _Controller(self.tau0, **self.ctrl_cfg)
        self._ctrl_last_t = -1
        self.model_op.stamp(t_end + 1)                                                # forecasters read r_{t_end+1} (fut_ret[:, 0])
        self.thr_op.stamp(t_end)                                                      # tau_0 / AE: windows up to t_end only
        self.diag.update({"tau0": self.tau0, "vol_thr": self.vol_thr, "n_stable": int(stable.sum()), "controller_online": self.ctrl_cfg["online"]})

    def _score(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n == 0:
            self._last_dir = np.zeros(0)
            return np.zeros(0)
        rows = self._rows_for(X)
        ok = np.isfinite(X).all((1, 2))
        e = np.full(n, np.nan); fn = np.zeros(n); fe = np.zeros(n)
        vol = self._vol_proxy(X)
        if ok.any():
            e[ok] = self._errors(X[ok])
            fn[ok], fe[ok] = self._forecasts(X[ok], vol[ok])
        r_real = np.nan_to_num(self._raw_returns(X)[:, -1]) / self._window_sigma(X)
        if rows is not None:
            t, asset = rows["t"], rows["asset"]
            if int(t.min()) < self._ctrl_last_t:
                self.ctrl.restore()                                                     # a new pass over earlier bars (replay)
            order = np.lexsort((asset, t))
        else:
            t, asset, order = np.arange(n), np.zeros(n, dtype=int), np.arange(n)
        tau = np.empty(n); alpha = np.empty(n); f = np.empty(n)
        for k in order:
            tau[k], alpha[k], f[k], _ = self.ctrl.step(int(asset[k]), float(e[k]), float(r_real[k]), float(fn[k]), float(fe[k]))
        if rows is not None:
            self._ctrl_last_t = int(t.max())                                       # last bar of THIS batch (B4 convention)
        self._last_dir = np.sign(f)
        self.diag.update({"alpha_last": float(alpha[order[-1]]), "tau_last": float(tau[order[-1]])})
        return e - (tau - self.tau0) if self._native_thr() else e        # adaptive tau only under the as-published threshold

    def fit(self, train, val=None) -> None:
        super().fit(train, val)
        self.ctrl.snapshot()                                                           # end-of-validation controller state

    def _native_thr(self) -> bool:
        return native_threshold_requested(self)

    def prepare_test(self, test) -> None:
        if self.thr_mode != "published":
            return
        self.ctrl.restore()
        super().prepare_test(test)
        self.ctrl.restore()
        self._ctrl_last_t = -1                                                         # first predict batch continues from the snapshot

    def _direction(self, X: np.ndarray, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        if self._last_dir is None or len(self._last_dir) != len(X):
            self._score(X)
        return np.nan_to_num(self._last_dir)

    def _published_threshold(self, s_train, s_val, s_test, y_test) -> float:
        if not self._native_thr():                                                     # protocol-requested kind (L2 / L3 switch)
            self.diag["published_rule"] = self.published_threshold
            if self.published_threshold in ("test_quantile", "combined_quantile", "best_f1_test") and self._rows is not None and len(self._rows["t"]):
                self.thr_op.stamp(int(self._rows["t"].max()))                          # the threshold reads the test fold
            return super()._published_threshold(s_train, s_val, s_test, y_test)
        self.diag["published_rule"] = "tau0_adaptive"
        self.thr_op.stamp(int(self.diag.get("scorer_t_end", -1)))
        return self.tau0

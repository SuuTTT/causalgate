"""Alarm gates: fixed conformal quantile, POT/EVT, adaptive conformal inference (ACI, Gibbs & Candès 2021) and
delayed-feedback ACI (El Halabi & Brandt 2026, arXiv:2609.07251) with tau = horizon + latency.

The gate controls the *long-run alarm rate* at a budget alpha derived from the cost model:
    alpha = annual_cost_budget / (cost_per_trade * decisions_per_year)            (plan Sec. 3.3)
ACI recursion:  alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1[alarm_t]
Threshold:      q_t = conformal (1 - alpha_t)-quantile of the calibration window of past scores
                (ceil((n+1)(1-alpha_t))-th order statistic; +inf when alpha_t <= 0, -inf when alpha_t >= 1).
Delayed ACI:    the feedback of decision t (its alarm indicator, and the label y_t needed for a "normal-only"
                calibration set) becomes usable only at t + tau; the recursion and the calibration window are
                updated when it arrives. Prop. 4: |mean alarm rate - alpha| <= C(tau, gamma, T).
Every gate is an AsOfStamped operator; `calibrate()` stamps it at the end of the calibration range and
`reset_stream()` restores that state so that a replay (or a second pricing pass) starts from the same point.

Labels (calib='normal_only' only; calib='all' never reads a label). A label y_t is usable only from its availability
bar avail_t (event end, AAER release, t + max horizon + latency, ...), never at the decision bar t:
    * calibrate(scores, t_end, labels, label_avail): a calibration row is filtered on its label only when
      label_avail <= t_end; rows whose label is not yet available at t_end are kept as unlabelled scores.
    * step(s, t, y, y_avail): the score waits in a label queue and enters the calibration window at y_avail (if its
      label is 0); the gate is stamped with y_avail when it consumes the label (y_avail <= t, so the stamp is causal).
      If y is given without y_avail, avail = t + `label_delay`; when `label_delay` is unknown (None) the label is
      ignored and the score is treated as unlabelled.
"""
from __future__ import annotations
import copy
from collections import deque
import heapq
import itertools
import warnings
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..data.pit import AsOfStamped


def alpha_from_cost_budget(cost_bps_per_trade: float, annual_budget_bps: float, bars_per_year: int, n_assets: int = 1,
                           min_alpha: float = 0.0, max_alpha: float = 0.5) -> float:
    """Allowed alarms per year = B / c; decisions per year = bars_per_year * n_assets; alpha = their ratio, bounded to
    [min_alpha, max_alpha] (a budget <= 0 gives alpha = 0, i.e. no alarms). A binding bound raises a warning, so the
    realised budget never silently differs from the configured one."""
    c = max(float(cost_bps_per_trade), 1e-9)
    decisions = max(float(bars_per_year) * max(int(n_assets), 1), 1.0)
    raw = (float(annual_budget_bps) / c) / decisions
    out = float(np.clip(raw, min_alpha, max_alpha))
    if out != raw:
        warnings.warn(f"alpha_from_cost_budget: budget-implied alpha {raw:.4g} clipped to {out:.4g} "
                      f"(B={annual_budget_bps}, c={cost_bps_per_trade}, decisions/yr={decisions:.0f})")
    return out


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th order statistic of `values` (finite-sample valid); +inf / -inf outside (0, 1)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0 or alpha <= 0:
        return float("inf")
    if alpha >= 1:
        return float("-inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(v, k - 1)[k - 1])


def alarm_rate_error(alarms: np.ndarray, alpha: float) -> float:
    a = np.asarray(alarms, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(abs(a.mean() - alpha)) if a.size else float("nan")


# --------------------------------------------------------------------------------------- base
class BaseGate(AsOfStamped):
    name = "base"

    def __init__(self, alpha: float = 0.05, calib: str = "all"):
        assert calib in ("all", "normal_only"), calib
        self.alpha = float(alpha)
        self.calib = calib
        self.label_delay: Optional[int] = None          # fallback availability lag when step() gets y without y_avail
        self._snapshot: Optional[Dict[str, Any]] = None
        self.last_t = -1
        self.alpha_used = float(alpha)                  # level that produced the latest decision
        self._lab: List[Tuple[int, int, float, float]] = []   # heap of (avail, seq, score, label)
        self._seq = 0

    # ---- label handling (calib='normal_only')
    def _calib_select(self, scores: np.ndarray, t_end: int, labels: Optional[np.ndarray], label_avail: Optional[np.ndarray]) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        if labels is None or self.calib != "normal_only":
            return s
        y = np.asarray(labels)
        known = np.ones(len(s), dtype=bool) if label_avail is None else (np.asarray(label_avail) <= int(t_end))
        return s[~known | (y == 0)]

    def _admit(self, s: float) -> None:            # overridden by windowed gates
        pass

    def _offer(self, s: float, t: int, y: Optional[float], y_avail: Optional[int]) -> None:
        """Route a finite score to the calibration window, through the label queue when its label matters."""
        if not np.isfinite(s):
            return
        if self.calib != "normal_only" or y is None or not np.isfinite(y):
            self._admit(s); return
        a = y_avail if y_avail is not None else (None if self.label_delay is None else int(t) + int(self.label_delay))
        if a is None:
            self._admit(s); return                      # availability unknown -> label ignored
        heapq.heappush(self._lab, (int(max(a, t)), self._seq, float(s), float(y)))
        self._seq += 1

    def _release_labels(self, t: int) -> None:
        while self._lab and self._lab[0][0] <= t:
            a, _, s, y = heapq.heappop(self._lab)
            self.stamp(a)                               # consumed a label known at a <= t
            if y == 0:
                self._admit(s)

    def calibrate(self, scores: np.ndarray, t_end: int, labels: Optional[np.ndarray] = None) -> "BaseGate":
        raise NotImplementedError

    def step(self, s: float, t: int, y: Optional[float] = None, y_avail: Optional[int] = None) -> Tuple[bool, float]:
        raise NotImplementedError

    def _take_snapshot(self) -> None:
        self._snapshot = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in ("_snapshot", "guard")}

    def reset_stream(self) -> None:
        if self._snapshot is None:
            raise RuntimeError(f"{type(self).__name__}.reset_stream() called before calibrate(): no calibration snapshot to restore")
        self.__dict__.update({k: copy.deepcopy(v) for k, v in self._snapshot.items()})

    def run(self, scores: np.ndarray, t0: int = 0, labels: Optional[np.ndarray] = None,
            label_avail: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Sequential replay over a (T,) or (T, N) score array (assets at the same bar are processed in order).
        alpha_path[t] is the level used for the first decision of bar t."""
        S = np.asarray(scores, dtype=np.float64)
        flat = S.ndim == 1
        S2 = S[:, None] if flat else S
        Y2 = None if labels is None else (np.asarray(labels)[:, None] if flat else np.asarray(labels))
        A2 = None if label_avail is None else (np.asarray(label_avail)[:, None] if flat else np.asarray(label_avail))
        T, N = S2.shape
        alarm = np.full((T, N), np.nan)
        thr = np.full((T, N), np.nan)
        apath = np.full(T, np.nan)
        for t in range(T):
            for i in range(N):
                a, q = self.step(S2[t, i], t0 + t, None if Y2 is None else Y2[t, i], None if A2 is None else int(A2[t, i]))
                alarm[t, i], thr[t, i] = (float(a) if np.isfinite(S2[t, i]) else np.nan), q
                if i == 0:
                    apath[t] = self.alpha_used
        out = {"alarm": alarm[:, 0] if flat else alarm, "thr": thr[:, 0] if flat else thr, "alpha_path": apath}
        out["realised_rate"] = float(np.nanmean(out["alarm"]))
        out["rate_error"] = alarm_rate_error(out["alarm"], self.alpha)
        return out


# --------------------------------------------------------------------------------------- fixed quantile
class FixedQuantileGate(BaseGate):
    """Conformal (1 - alpha)-quantile of the calibration scores (all of them for calib='all', the default; the
    label-0 ones whose label is available at t_end for 'normal_only'); never updated (the P-W / offline convention
    when the calibration set is the train/val fold; the L2 leak when it is the test fold)."""
    name = "fixed"

    def __init__(self, alpha: float = 0.05, calib: str = "all"):
        super().__init__(alpha, calib)
        self.thr = float("inf")

    def calibrate(self, scores, t_end, labels=None, label_avail=None):
        s = self._calib_select(scores, t_end, labels, label_avail)
        self.thr = conformal_quantile(s, self.alpha)
        self.stamp(int(t_end))
        self._take_snapshot()
        return self

    def step(self, s, t, y=None, y_avail=None):
        self.check(t)
        self.last_t = int(t)
        return bool(np.isfinite(s) and s > self.thr), self.thr


# --------------------------------------------------------------------------------------- POT / EVT
class POTGate(BaseGate):
    """Peaks-over-threshold (Siffer et al., KDD 2017 'SPOT' style): a GPD is fitted on the exceedances of the
    calibration window over its (1 - u)-quantile and the alarm threshold is the GPD quantile at tail probability
    alpha; refitted every `refit_every` bars on the rolling window. calib='all' (default, SPOT-like) admits every finite
    score; 'normal_only' admits label-0 scores once their label is available (never on the same-bar label).
    When alpha is not inside the fitted tail (alpha * n >= n_u) the empirical conformal quantile is used."""
    name = "pot"

    def __init__(self, alpha: float = 0.05, u: float = 0.9, window: int = 1000, refit_every: int = 20, calib: str = "all"):
        super().__init__(alpha, calib)
        self.u, self.window, self.refit_every = float(u), int(window), int(refit_every)
        self.buf: Deque[float] = deque(maxlen=self.window)
        self.thr = float("inf")
        self._since = 0

    def _fit(self) -> None:
        from scipy.stats import genpareto
        v = np.asarray([x for x in self.buf if np.isfinite(x)])
        if v.size < 30:
            self.thr = conformal_quantile(v, self.alpha)
            return
        thr_u = float(np.quantile(v, self.u))
        exc = v[v > thr_u] - thr_u
        n, nu = v.size, exc.size
        if nu < 10 or self.alpha * n >= nu:              # alpha outside the fitted tail: GPD not valid there
            self.thr = conformal_quantile(v, self.alpha)
            return
        try:
            xi, _, beta = genpareto.fit(exc, floc=0.0)
            ratio = (self.alpha * n / nu)
            if abs(xi) < 1e-6:
                self.thr = thr_u - beta * np.log(ratio)
            else:
                self.thr = thr_u + beta / xi * (ratio ** (-xi) - 1.0)
            if not np.isfinite(self.thr):
                self.thr = conformal_quantile(v, self.alpha)
        except Exception:
            self.thr = conformal_quantile(v, self.alpha)

    def _admit(self, s: float) -> None:
        self.buf.append(float(s))

    def calibrate(self, scores, t_end, labels=None, label_avail=None):
        s = self._calib_select(scores, t_end, labels, label_avail)
        s = s[np.isfinite(s)]
        self.buf.clear(); self._lab = []; self._since = 0; self.last_t = -1; self.thr = float("inf")
        self.buf.extend([float(x) for x in s[-self.window :]])
        self._fit()
        self.stamp(int(t_end))
        self._take_snapshot()
        return self

    def step(self, s, t, y=None, y_avail=None):
        self.check(t)
        self.last_t = int(t)
        self._release_labels(int(t))
        self.alpha_used = self.alpha
        alarm = bool(np.isfinite(s) and s > self.thr)
        self._offer(s, int(t), y, y_avail)
        self._since += 1
        if self._since >= self.refit_every:
            self._fit()
            self._since = 0
        return alarm, self.thr


# --------------------------------------------------------------------------------------- ACI
class ACIGate(BaseGate):
    """Adaptive conformal alarm gate with immediate feedback (err_t = alarm_t known at t)."""
    name = "aci"

    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, window: int = 500, calib: str = "all", clip: bool = False):
        super().__init__(alpha, calib)
        self.gamma, self.window, self.clip = float(gamma), int(window), bool(clip)
        self.buf: Deque[float] = deque(maxlen=self.window)
        self.alpha_t = float(alpha)
        self.thr = float("inf")

    def calibrate(self, scores, t_end, labels=None, label_avail=None):
        s = self._calib_select(scores, t_end, labels, label_avail)
        s = s[np.isfinite(s)]
        self.buf.clear(); self._lab = []; self.last_t = -1
        self.buf.extend([float(x) for x in s[-self.window :]])
        self.alpha_t = self.alpha
        self.thr = conformal_quantile(np.asarray(self.buf), self.alpha_t)
        self.stamp(int(t_end))
        self._take_snapshot()
        return self

    def _admit(self, s: float) -> None:
        self.buf.append(float(s))

    def _update_alpha(self, err: float) -> None:
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
        if self.clip:
            self.alpha_t = float(np.clip(self.alpha_t, 0.0, 1.0))

    def step(self, s, t, y=None, y_avail=None):
        self.check(t)
        self.last_t = int(t)
        self._release_labels(int(t))
        self.alpha_used = self.alpha_t
        self.thr = conformal_quantile(np.asarray(self.buf), self.alpha_t)
        alarm = bool(np.isfinite(s) and s > self.thr)
        if np.isfinite(s):
            self._update_alpha(float(alarm))
            self._offer(s, int(t), y, y_avail)
        return alarm, self.thr


class DelayedACIGate(ACIGate):
    """ACI whose feedback for decision t (alarm indicator and, for normal-only calibration, the label y_t) is only
    usable at t + tau. Decisions wait in a pending queue; when their feedback matures the alpha recursion is applied
    and the score enters the calibration window (for 'normal_only': at max(t + tau, label availability), via the
    label queue). tau = horizon + latency in the trading setting."""
    name = "delayed_aci"

    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, window: int = 500, tau: int = 6, calib: str = "all", clip: bool = False):
        super().__init__(alpha, gamma, window, calib, clip)
        self.tau = int(tau)
        self.pending: Deque[Tuple[int, float, float, Optional[float], Optional[int]]] = deque()

    def calibrate(self, scores, t_end, labels=None, label_avail=None):
        self.pending.clear()
        return super().calibrate(scores, t_end, labels, label_avail)

    def _release(self, t: int) -> None:
        while self.pending and self.pending[0][0] + self.tau <= t:
            t0, s, err, y, a = self.pending.popleft()
            self._update_alpha(err)
            self._offer(s, t0, y, a)

    def step(self, s, t, y=None, y_avail=None):
        self.check(t)
        self.last_t = int(t)
        self._release(int(t))
        self._release_labels(int(t))
        self.alpha_used = self.alpha_t
        self.thr = conformal_quantile(np.asarray(self.buf), self.alpha_t)
        alarm = bool(np.isfinite(s) and s > self.thr)
        if np.isfinite(s):
            self.pending.append((int(t), float(s), float(alarm), y, y_avail))
        return alarm, self.thr


GATES = {"fixed": FixedQuantileGate, "pot": POTGate, "aci": ACIGate, "delayed_aci": DelayedACIGate}


def make_gate(cfg: Dict[str, Any], bars_per_year: int = 252, n_assets: int = 1, horizon: int = 5, latency: int = 1,
              cost_bps: Optional[float] = None, label_delay: Optional[int] = None) -> BaseGate:
    """Build a gate from a config dict. alpha is read directly or derived from {budget: {annual_bps, cost_bps}}.
    `label_delay` (e.g. max(horizons) + latency) is the fallback label availability lag for step(y) without y_avail."""
    kind = cfg.get("type", "delayed_aci")
    if "alpha" in cfg and cfg["alpha"] is not None:
        alpha = float(cfg["alpha"])
    else:
        b = cfg.get("budget") or {}
        alpha = alpha_from_cost_budget(cost_bps if cost_bps is not None else b.get("cost_bps", 10.0), b.get("annual_bps", 500.0), bars_per_year, n_assets)
    kw: Dict[str, Any] = {"alpha": alpha, "calib": cfg.get("calib") or "all"}
    if kind in ("aci", "delayed_aci"):
        kw.update(gamma=float(cfg.get("gamma", 0.01)), window=int(cfg.get("window", 500)), clip=bool(cfg.get("clip", False)))
    if kind == "delayed_aci":
        tau = cfg.get("tau")
        kw["tau"] = int(tau) if tau is not None else int(horizon + latency)
    if kind == "pot":
        kw.update(u=float(cfg.get("u", 0.9)), window=int(cfg.get("window", 1000)), refit_every=int(cfg.get("refit_every", 20)))
    g = GATES[kind](**kw)
    ld = cfg.get("label_delay", label_delay)
    g.label_delay = None if ld is None else int(ld)
    return g

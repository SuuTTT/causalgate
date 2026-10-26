"""B10 Konczal & Poloczanski, "Anomaly detection in European cryptocurrency exchange-traded products",
arXiv:2608.09576 (2026). status = reimplemented (XGBoost / LightGBM replaced by sklearn HistGradientBoosting).

From the paper (arxiv.org/html/2608.09576): 1-minute log returns of four crypto ETPs on Xetra / Nasdaq Stockholm,
Jan 2024 - Dec 2025, chronological 80 / 20 split, trained once, no retraining.
  * EVT benchmark (Sec. on the POT / GPD benchmark, Eq. 3): threshold u = 95th percentile of the TRAINING-sample losses
    L_t = -r_t (the 5th percentile of training returns); a GPD is fitted by maximum likelihood to the training
    exceedances L_t - u; with the empirical exceedance rate zeta_u ~ 0.05 the tail quantile at p = 0.01 is
        q_p = u + beta / xi [(p / zeta_u)^(-xi) - 1]   (xi != 0),      q_p = u + beta log(zeta_u / p)   (xi = 0),
    and bar t is flagged (D^POT) when r_t < -q_p.
  * Indicator rules: cross-venue divergence D^CV (rolling OLS without intercept over n = 1440 bars, residual beyond
    its rolling 99th percentile), no-recovery D^NR (r_t below the rolling 1 % quantile AND the cumulative return over
    the NEXT K = 10 bars < 0.3 |r_t| -- defined with future bars, hence a label (L4), never a feature), momentum
    reversal D^MR (r_t below the rolling 1 % quantile AND sum of the previous 5 returns > 0); D^ANY = OR of all.
  * Prediction: one-bar-ahead classification of each indicator with logistic regression / random forest / XGBoost /
    LightGBM with class weights on 34 features (rolling std, mean, skew, drawdown, ATR, RSI over 5 / 15 / 30; high-low
    range, volatility ratio std_10 / std_120, 120-bar z-score, MA deviation; effective spread, relative volume,
    relative spread, Amihud, Kyle lambda, order imbalance at t-1 and their 5-bar means); AUC-ROC is the headline
    metric; precision / recall are reported at the F1-optimal threshold computed on the TEST set ("descriptive", not
    out-of-sample). No feature normalisation is stated. No trading rule, PnL or Sharpe is reported.

Scaffold mapping: score for the gate = predicted probability that the next bar is an anomaly (D^`target`);
direction = -1 (the paper studies left-tail drops; a short on a predicted crash is OUR trade loop, not theirs);
published threshold = F1-optimal on the test indicator labels (what the paper does), causal = delayed-ACI gate.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import torch
from scipy import stats

from ..data.pit import AsOfStamped
from .common import ScoreToTradeBaseline, register
from .b1_cpd_dmn import RowAwareMixin

TARGETS = ("pot", "mr", "nr", "any")


class _EVTThreshold(AsOfStamped):
    """GPD tail quantile; stamped at the end of the sample it was fitted on (train fold, or the panel end for full_sample)."""


class _KonczalClassifier(AsOfStamped):
    """Fitted GBM / logistic classifier; stamped at the end of the training fold."""


class _KonczalPublishedThreshold(AsOfStamped):
    """F1-optimal probability threshold on the test indicator labels; stamped at the test end (published mode only)."""


def gpd_tail_quantile(losses: np.ndarray, u_q: float = 0.95, p: float = 0.01) -> Tuple[float, Dict[str, float]]:
    """Eq. 3 of Konczal & Poloczanski: POT threshold u = u_q-quantile of the losses, GPD(xi, beta) by ML on the exceedances."""
    L = np.asarray(losses, dtype=np.float64)
    L = L[np.isfinite(L)]
    if L.size < 50:
        return float("nan"), {"n": int(L.size)}
    u = float(np.quantile(L, u_q))
    exc = L[L > u] - u
    if exc.size < 10:
        return float("nan"), {"n": int(L.size), "n_exc": int(exc.size)}
    zeta = exc.size / L.size
    try:
        xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
    except Exception:
        xi, beta = 0.0, float(exc.mean())
    if abs(xi) < 1e-6:
        q = u + beta * np.log(zeta / p)
    else:
        q = u + beta / xi * ((p / zeta) ** (-xi) - 1.0)
    if not np.isfinite(q):
        q = float(np.quantile(L, 1.0 - p))
    return float(q), {"u": u, "xi": float(xi), "beta": float(beta), "zeta": float(zeta), "n": int(L.size), "n_exc": int(exc.size)}


def _rsi(r: np.ndarray) -> np.ndarray:
    g, l = np.clip(r, 0, None).mean(1), np.clip(-r, 0, None).mean(1)
    return 100.0 - 100.0 / (1.0 + g / (l + 1e-12))


@register("cg_konczal_evt")
class KonczalEVTBaseline(RowAwareMixin, ScoreToTradeBaseline):
    name = "cg_konczal_evt"
    paper = "Konczal, Poloczanski, Anomaly detection in European cryptocurrency exchange-traded products, arXiv:2608.09576 (2026)"
    official_repo = ""
    status = "reimplemented"
    year = 2026
    cutoff_date = None
    conventions = {
        "L1": "No feature normalisation stated (tree ensembles are scale-invariant; the EVT threshold is on raw returns). This wrapper reads the "
              "raw feature window, so the scaffold's L1 switch is inert; the 'logistic' classifier uses a train-fitted standardiser.",
        "L2": "EVT: u = 95th percentile of TRAINING losses, GPD by ML on TRAINING exceedances, p = 0.01 tail quantile (Eq. 3): train-fitted "
              "in both modes (what the paper does); cfg `evt_sample: full_sample` is the one-switch leaky variant (published mode only). "
              "Classifier: the F1-optimal probability threshold is computed on the TEST set (stated as descriptive): `best_f1_test` in published mode.",
        "L3": "Chronological 80 / 20 split, hyper-parameters fixed across instruments and targets, trained once, no retraining; the F1-optimal "
              "threshold uses test labels (L3 under the published protocol).",
        "L4": "No-recovery indicator D^NR is defined with the next K = 10 bars: it is a label only (never a feature). No point-adjust.",
        "L5": "Prices, volumes and quoted spreads at 1-minute resolution; features use bars strictly before the predicted bar.",
        "L6": "Four ETPs chosen ex post (VanEck BTC / ETH on Xetra, Virtune BTC / ETH on Nasdaq Stockholm); inactive bars removed.",
        "L7": "No trading rule / execution timing / costs: the trade loop (alarm at t -> short, fill at t + latency, hold h) is ours.",
        "L8": "No pre-trained component.",
    }
    unspecified = [
        "XGBoost / LightGBM are not installed: sklearn HistGradientBoostingClassifier(class_weight='balanced') is the GBM stand-in; "
        "`clf: logistic` gives the logistic-regression variant; random forest is not wired.",
        "Rolling 1 % quantile over n = 1440 bars (for D^MR / D^NR) is computed over the look-back window (L bars) because the scaffold only "
        "exposes the window; the 120-bar z-score and std_120 are likewise clipped to the window.",
        "30 of the 34 features are computable from [log_ret, log_vol, hl_range, log_spread]: Kyle lambda and order imbalance (need signed "
        "order flow / bid-ask volumes) are omitted together with their 5-bar means; the cross-venue indicator D^CV needs a second venue and is skipped.",
        "ATR is the mean high-low range over the window (no previous-close term); Amihud = |r| / volume relative to its window mean.",
        "EVT fitted per asset on the training rows (the paper fits per instrument); `evt_pooled: true` pools the assets.",
        "The F1-optimal published threshold is chosen against the paper's indicator labels of the test rows (`f1_labels: indicator`); "
        "`planted` uses the dataset labels instead; if the test fold has no positive indicator the train-quantile threshold is used.",
        "The classifier target is D^`target` (default `any`); class weights = balanced (the paper's weights are not given).",
        "Direction -1 on a predicted crash and the hold / fill rules are ours (the paper reports no trades).",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "best_f1_test")
        super().__init__(cfg, dims, horizons, device)
        self._init_rows()
        self.evt_sample = str(cfg.get("evt_sample", "train"))
        assert self.evt_sample in ("train", "full_sample"), self.evt_sample
        assert not (self.evt_sample == "full_sample" and self.mode == "causal"), "full_sample GPD is the published-mode switch only"
        self.evt_pooled = bool(cfg.get("evt_pooled", False))
        self.u_q, self.p_tail, self.q_roll = float(cfg.get("u_quantile", 0.95)), float(cfg.get("p_tail", 0.01)), float(cfg.get("rolling_quantile", 0.01))
        self.K, self.recovery_frac = int(cfg.get("K", 10)), float(cfg.get("recovery_frac", 0.3))
        self.target = str(cfg.get("target", "any"))
        assert self.target in TARGETS, self.target
        self.clf_kind = str(cfg.get("clf", "hgb"))
        self.max_iter, self.max_depth = int(cfg.get("max_iter", 60)), int(cfg.get("max_depth", 3))
        self.f1_labels = str(cfg.get("f1_labels", "indicator"))
        names = list(self.panel.meta.get("feat_names", [])) if self.panel is not None else []
        col = dict(cfg.get("cols", {}))
        pick = lambda key, name, default: int(col.get(key, names.index(name) if name in names else default))
        self.c_ret, self.c_vol, self.c_rng, self.c_spr = self.ret_col, pick("vol", "log_vol", 1), pick("range", "hl_range", 2), pick("spread", "log_spread", 3)
        self.evt_op = self._register_op(_EVTThreshold())
        self.clf_op = self._register_op(_KonczalClassifier())
        self.thr_op = self._register_op(_KonczalPublishedThreshold())
        self.q_p: Dict[int, float] = {}
        self.clf = None
        self.scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._train_pos_rate = float("nan")

    # ------------------------------------------------------------------ features (bars <= t of the window)
    def _features(self, raw: np.ndarray) -> np.ndarray:
        r, v, hl, sp = (raw[..., self.c_ret], raw[..., self.c_vol], raw[..., self.c_rng], raw[..., self.c_spr])
        n, L = r.shape
        p = np.cumsum(r, 1)
        cols = []
        for w in (5, 15, 30):
            w = min(w, L)
            rw = r[:, -w:]
            cols += [rw.std(1), rw.mean(1), np.nan_to_num(stats.skew(rw, axis=1, bias=True)), p[:, -w:].max(1) - p[:, -1], hl[:, -w:].mean(1), _rsi(rw)]
        wl = min(120, L)
        cols += [hl[:, -1], r[:, -10:].std(1) / (r[:, -wl:].std(1) + 1e-12), (r[:, -1] - r[:, -wl:].mean(1)) / (r[:, -wl:].std(1) + 1e-12),
                 p[:, -1] - p[:, -min(20, L) :].mean(1)]
        vol_lin = np.exp(np.clip(v, -30, 30))
        amihud = np.abs(r) / (vol_lin + 1e-12)
        micro = [sp, v - v.mean(1, keepdims=True), sp - sp.mean(1, keepdims=True), amihud / (amihud.mean(1, keepdims=True) + 1e-12)]
        for m in micro:
            cols += [m[:, -2], m[:, -6:-1].mean(1)]                                       # at t-1 and its 5-bar mean
        F = np.stack(cols, 1).astype(np.float64)
        return F

    # ------------------------------------------------------------------ indicators (labels for the next bar)
    def _indicators(self, raw: np.ndarray, fut: np.ndarray, asset: np.ndarray) -> Dict[str, np.ndarray]:
        r = raw[..., self.c_ret]
        r_next = fut[:, 0]
        q_roll = np.quantile(r, self.q_roll, axis=1)
        qp = np.asarray([self.q_p.get(int(a), self.q_p.get(-1, np.nan)) for a in asset])
        pot = r_next < -qp
        below = r_next < q_roll
        mr = below & (r[:, -5:].sum(1) > 0)
        K = min(self.K, fut.shape[1] - 1)
        nr = below & (fut[:, 1 : 1 + K].sum(1) < self.recovery_frac * np.abs(r_next))
        out = {"pot": pot, "mr": mr, "nr": nr}
        out["any"] = pot | mr | nr
        return {k: v.astype(np.float64) for k, v in out.items()}

    def _fit_evt(self, rows: Dict[str, np.ndarray], t_end: int) -> None:
        self.q_p = {}
        info = {}
        if self.evt_sample == "full_sample" and self.panel is not None:
            R = self.panel.price_feat[:, :, self.c_ret].astype(np.float64)                 # whole panel (leaky one-switch variant)
            groups = {-1: R.ravel()} if self.evt_pooled else {i: R[:, i] for i in range(R.shape[1])}
            self.evt_op.stamp(int(R.shape[0] - 1))
        else:
            r_t = rows["raw"][:, -1, self.c_ret]
            groups = {-1: r_t} if self.evt_pooled else {int(a): r_t[rows["asset"] == a] for a in np.unique(rows["asset"])}
            self.evt_op.stamp(int(t_end))
        for key, r in groups.items():
            q, inf = gpd_tail_quantile(-np.asarray(r), self.u_q, self.p_tail)
            self.q_p[key] = q
            info[key] = inf
        if not self.evt_pooled and -1 not in self.q_p:
            self.q_p[-1] = float(np.nanmedian([v for v in self.q_p.values()])) if self.q_p else float("nan")
        self.diag["evt"] = {"sample": self.evt_sample, "q_p": {k: float(v) for k, v in self.q_p.items()}, "fit": info}

    # ------------------------------------------------------------------ hooks
    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        rows = self._rows_for(X)
        assert rows is not None, "B10 needs the batch rows (raw window + future returns) to build the EVT labels"
        self._fit_evt(rows, t_end)
        F = self._features(rows["raw"])
        y = self._indicators(rows["raw"], rows["fut_ret"], rows["asset"])[self.target]
        ok = np.isfinite(F).all(1) & np.isfinite(y)
        F, y = F[ok], y[ok]
        self._train_pos_rate = float(y.mean()) if len(y) else float("nan")
        if len(y) >= 20 and 0 < y.sum() < len(y):
            if self.clf_kind == "logistic":
                from sklearn.linear_model import LogisticRegression
                mu, sd = F.mean(0), np.maximum(F.std(0), 1e-9)
                self.scaler = (mu, sd)
                self.clf = LogisticRegression(class_weight="balanced", max_iter=300).fit((F - mu) / sd, y)
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.clf = HistGradientBoostingClassifier(max_iter=self.max_iter, max_depth=self.max_depth, learning_rate=0.1,
                                                          class_weight="balanced", early_stopping=False, random_state=int(self.cfg.get("seed", 0))).fit(F, y)
        else:
            self.clf = None                                                     # degenerate fold: fall back to the EVT rule itself
        reach = 1 + (min(self.K, rows["fut_ret"].shape[1] - 1) if self.target in ("nr", "any") else 0)
        self.clf_op.stamp(t_end + reach)                                        # labels read fut_ret up to bar t_end + reach
        self.diag.update({"train_pos_rate": self._train_pos_rate, "n_features": int(F.shape[1]), "clf": self.clf_kind if self.clf is not None else "evt_rule"})

    def _score(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros(0)
        rows = self._rows_for(X)
        raw = rows["raw"] if rows is not None else np.nan_to_num(X.astype(np.float64))
        F = self._features(raw)
        ok = np.isfinite(F).all(1)
        s = np.full(n, np.nan)
        if self.clf is not None and ok.any():
            Fo = F[ok] if self.scaler is None else (F[ok] - self.scaler[0]) / self.scaler[1]
            s[ok] = self.clf.predict_proba(Fo)[:, 1]
        elif ok.any():                                                          # EVT rule on the current bar as the score
            qp = np.asarray([self.q_p.get(int(a), self.q_p.get(-1, np.nan)) for a in (rows["asset"] if rows is not None else np.full(n, -1))])
            s[ok] = (-raw[ok, -1, self.c_ret]) / np.where(np.isfinite(qp[ok]), qp[ok], np.nan)
        return s

    def _direction(self, X: np.ndarray, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        return -np.ones(len(X))                                                 # short on a predicted left-tail drop (our rule)

    def _published_threshold(self, s_train, s_val, s_test, y_test) -> float:
        rows = self._rows
        t_test_end = int(rows["t"].max()) if rows is not None and len(rows["t"]) else -1
        kind = self.published_threshold
        if kind != "best_f1_test":                                              # protocol-requested kind honoured as is
            if kind in ("test_quantile", "combined_quantile"):
                self.thr_op.stamp(t_test_end)                                   # the threshold reads the test fold (L2)
            self.diag["published_threshold_labels"] = "n/a (%s)" % kind
            return super()._published_threshold(s_train, s_val, s_test, y_test)
        y_ind = None
        if self.f1_labels == "indicator" and rows is not None and len(rows["t"]) == len(s_test):
            y_ind = self._indicators(rows["raw"], rows["fut_ret"], rows["asset"])[self.target]
        y_use = y_ind if y_ind is not None else y_test
        if y_use is not None and len(y_use) and np.nansum(y_use) > 0:
            self.thr_op.stamp(t_test_end)
            thr = super()._published_threshold(s_train, s_val, s_test, y_use)
            self.diag["published_threshold_labels"] = "indicator" if y_ind is not None else "planted"
            return thr
        self.published_threshold = "train_quantile"                             # best_f1_test without any positive label
        try:
            thr = super()._published_threshold(s_train, s_val, s_test, y_use)
        finally:
            self.published_threshold = kind
        self.diag["published_threshold_labels"] = "none -> train_quantile"
        return thr

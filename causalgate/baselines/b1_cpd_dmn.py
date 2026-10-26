"""B1 CPD-DMN: Wood, Roberts, Zohren, "Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning
and Changepoint Detection", J. Financial Data Science 2022 (arXiv:2105.13727). status = reimplemented.

Two modules, as in the paper:
  (1) GP change-point detection on a look-back window (LBW) of returns standardised *within the window* (Eq. 2):
      Matern-3/2 GP (Eq. 4, nlml Eq. 7) versus the change-point kernel k_C = k1 * sigma(x) sigma(x') +
      k2 * (1 - sigma(x)) (1 - sigma(x')) with sigma(x) = 1 / (1 + exp(-s (x - c))) (Eq. 9); severity and location
      (Eq. 10): nu = 1 - 1 / (1 + exp(-(nlml_C - nlml_M))), gamma = (c - (t - l)) / l.
      The type-II ML over the kernel hyper-parameters is done on a fixed grid with the signal variance profiled out
      (closed form), which makes the whole computation one batched quadratic form per window instead of one
      L-BFGS run per window (gpflow in the official code). No subsampling of windows is needed.
  (2) Deep Momentum Network head: LSTM over per-bar features (vol-normalised returns at scales {1, 21, 63, 126, 252}
      clipped to the look-back, MACD(8,24), (16,48), (32,96), CPD severity and location), time-distributed tanh
      output = position in (-1, 1) (Eq. 13), trained with the Sharpe loss (Eq. 14) on the captured returns
      R_{t+1} = X_t * sigma_tgt / sigma_t * r_{t+1} (Eq. 11; sigma_t = 60-day EWM std, sigma_tgt = 15 %).

Scaffold mapping: score for the gate = CPD severity nu; direction = DMN position (|direction| <= 1, the scaffold
multiplies by w_max). The paper has no alarm threshold (positions every day): in published mode the threshold is
-inf (`always_in_market`), in causal mode the delayed-ACI gate decides when the position is taken.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..data.pit import AsOfStamped
from .common import ScoreToTradeBaseline, register

def native_threshold_requested(bl, native_kind: str = "train_quantile") -> bool:
    """True when the protocol asks for the baseline's OWN (as-published) threshold rule rather than an explicit scaffold kind.

    ProtocolRunner marks the P-O row with cfg['_published_conventions'] (every other row carries cfg['_applied_channels']
    only); other explicit flags (`threshold_source` / `threshold_request` == 'published', `as_published`) are honoured.
    Without any flag (direct construction): the runner passes the baseline yaml's `published_threshold` through only when the protocol threshold is
    'published' (P-O) and otherwise writes the requested kind (test_quantile for causal+L2 / P-W / offline_all, best_f1_test for
    L3), so the native rule applies iff the kind is still the wrapper's own default `native_kind`."""
    cfg = getattr(bl, "cfg", {}) or {}
    if getattr(bl, "thr_mode", "gate") != "published":
        return False
    if "_published_conventions" in cfg:                                 # ProtocolRunner: P-O row (resolve_published)
        return True
    if "_applied_channels" in cfg:                                      # ProtocolRunner: any other protocol / one-switch row
        return False
    for key in ("threshold_source", "threshold_request"):
        if key in cfg:
            return str(cfg[key]) == "published"
    if "as_published" in cfg:
        return bool(cfg["as_published"])
    return getattr(bl, "published_threshold", None) == native_kind



# ======================================================================================= shared helper
class RowAwareMixin:
    """The scaffold hands `_score` only X (B, L, F). This mixin keeps the rows behind those windows (t, asset, label,
    the *raw* feature window and the in-fold future returns) so that cross-sectional models (B8), sequential controllers
    (B9), in-fold training targets (B1, B10) and raw-return thresholds (B10) can be built without touching common.py.
    `_windows` is the single point where every path (fit/_collect, prepare_test, predict) sees the batch, so the rows
    stashed there are always aligned with the X that follows. It also stamps/checks the fitted model operators so that
    the Filtration Guard log covers the model, not only the gate."""

    def _init_rows(self) -> None:
        self._rows: Optional[Dict[str, np.ndarray]] = None
        self._row_buf: Optional[List[Dict[str, np.ndarray]]] = None
        self._ops: List[AsOfStamped] = []

    def _register_op(self, op: AsOfStamped) -> AsOfStamped:
        self.guard.register(op)
        self._ops.append(op)
        return op

    def _stamp_ops(self, t_end: int) -> None:
        for op in self._ops:
            op.stamp(int(t_end))

    def _rows_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        B = int(batch["t"].shape[0])
        raw = batch["price"].cpu().numpy().astype(np.float64) if "price" in batch else np.full((B, self.L, self.dims["price"]), np.nan)
        fut = batch["fut_ret"].cpu().numpy().astype(np.float64) if "fut_ret" in batch else np.zeros((B, 1))
        return {"t": batch["t"].cpu().numpy().astype(np.int64), "asset": batch["asset"].cpu().numpy().astype(np.int64),
                "y": batch["y"].cpu().numpy().astype(np.float64), "raw": raw, "fut_ret": fut}

    def _windows(self, batch):
        X = super()._windows(batch)
        rows = self._rows_from_batch(batch)
        if self._row_buf is not None:
            self._row_buf.append(rows)
        else:
            self._rows = rows
        return X

    def _collect(self, loader):
        self._row_buf = []
        try:
            out = super()._collect(loader)
            buf = self._row_buf
            self._rows = {k: np.concatenate([r[k] for r in buf]) for k in buf[0]} if buf else None
        finally:
            self._row_buf = None
        return out

    def _rows_for(self, X: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        r = self._rows
        return r if (r is not None and len(r["t"]) == len(X)) else None

    def _raw_returns(self, X: np.ndarray) -> np.ndarray:
        """Raw log-returns of the windows when the panel carries raw features, else the scaffold-normalised column."""
        rows = self._rows_for(X)
        raw_ok = rows is not None and (self.panel is None or bool(self.panel.meta.get("raw_features", True)))
        return rows["raw"][..., self.ret_col] if raw_ok else X[..., self.ret_col].astype(np.float64)

    def predict(self, batch):
        t0 = int(batch["t"].min())
        for op in self._ops:
            op.check(t0)
        return super().predict(batch)


class _CPDDMNModel(AsOfStamped):
    """Fitted state of B1 (feature scaler + LSTM); stamped at t_end + 1, the last bar whose return the Sharpe loss read."""


# ======================================================================================= GP change-point module
def matern32(x: np.ndarray, ell: float) -> np.ndarray:
    d = np.abs(x[:, None] - x[None, :]) * np.sqrt(3.0) / float(ell)
    return (1.0 + d) * np.exp(-d)


class GPChangePointScorer:
    """Eq. 4 / 7 / 9 / 10 of Wood et al. on a fixed hyper-parameter grid.

    For a zero-mean GP with covariance sigma_h^2 (K_theta + rho I) the ML signal variance is q / m with q = y' K~^{-1} y,
    so nlml(theta, rho) = m/2 log(q/m) + 1/2 log|K~| + m/2 (1 + log 2 pi) (profile likelihood). K~^{-1} and log|K~|
    depend only on the grid point, so every window costs one (m*m)-dimensional dot product per grid point: a single
    GEMM for all windows. Grid: Matern = lengthscales x noise ratios; change-point = locations x steepness x
    (lengthscale_1, lengthscale_2) x output-scale ratio of k2/k1 x noise ratios."""

    def __init__(self, lbw: int = 21, lengthscales: Sequence[float] = (2.0, 8.0), noise_ratios: Sequence[float] = (0.3, 1.0, 3.0),
                 scale_ratios: Sequence[float] = (0.25, 1.0, 4.0), steepness: Sequence[float] = (1.0,), loc_step: int = 1):
        self.lbw = int(lbw)
        x = np.arange(self.lbw, dtype=np.float64)
        I = np.eye(self.lbw)
        Minv, Mld = [], []
        for ell in lengthscales:
            for rho in noise_ratios:
                K = matern32(x, ell) + rho * I
                Minv.append(np.linalg.inv(K).ravel()); Mld.append(np.linalg.slogdet(K)[1])
        self.M_inv, self.M_logdet = np.stack(Minv), np.asarray(Mld)
        Cinv, Cld, Cloc = [], [], []
        for c in range(1, self.lbw - 1, int(loc_step)):                 # c in (t - l, t) open interval (Sec. II)
            for s in steepness:
                sig = 1.0 / (1.0 + np.exp(-float(s) * (x - c)))
                S1, S2 = sig[:, None] * sig[None, :], (1 - sig)[:, None] * (1 - sig)[None, :]
                for e1 in lengthscales:
                    k1 = matern32(x, e1)
                    for e2 in lengthscales:
                        k2 = matern32(x, e2)
                        for a2 in scale_ratios:
                            base = k1 * S1 + a2 * k2 * S2
                            for rho in noise_ratios:
                                K = base + rho * I
                                Cinv.append(np.linalg.inv(K).ravel()); Cld.append(np.linalg.slogdet(K)[1]); Cloc.append(c)
        self.C_inv, self.C_logdet, self.C_loc = np.stack(Cinv), np.asarray(Cld), np.asarray(Cloc, dtype=np.float64)
        self.n_grid = (int(self.M_inv.shape[0]), int(self.C_inv.shape[0]))

    def _nlml(self, YY: np.ndarray, Kinv: np.ndarray, logdet: np.ndarray) -> np.ndarray:
        m = self.lbw
        q = np.maximum(YY @ Kinv.T, 1e-12)                                      # (n, G) quadratic forms
        return 0.5 * m * np.log(q / m) + 0.5 * logdet[None, :] + 0.5 * m * (1.0 + np.log(2 * np.pi))

    def score(self, R: np.ndarray, chunk: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """R: (n, >= lbw) returns; the last `lbw` bars of each row are standardised (Eq. 2) and scored.
        Returns (severity nu, location gamma), NaN where the window is not finite or has zero variance."""
        R = np.asarray(R, dtype=np.float64)[:, -self.lbw :]
        n = R.shape[0]
        nu, gam = np.full(n, np.nan), np.full(n, np.nan)
        ok = np.isfinite(R).all(1)
        sd = R.std(1)
        ok &= sd > 1e-12
        if not ok.any():
            return nu, gam
        Y = (R[ok] - R[ok].mean(1, keepdims=True)) / sd[ok][:, None]
        out_nu, out_g = [], []
        for a in range(0, Y.shape[0], chunk):
            y = Y[a : a + chunk]
            YY = (y[:, :, None] * y[:, None, :]).reshape(len(y), -1)
            nM = self._nlml(YY, self.M_inv, self.M_logdet).min(1)
            nC_all = self._nlml(YY, self.C_inv, self.C_logdet)
            j = nC_all.argmin(1)
            nC = nC_all[np.arange(len(y)), j]
            out_nu.append(1.0 - 1.0 / (1.0 + np.exp(-np.clip(nC - nM, -50, 50))))        # Eq. 10
            out_g.append(self.C_loc[j] / self.lbw)                                            # Eq. 10: (c - (t - l)) / l
        nu[ok], gam[ok] = np.concatenate(out_nu), np.concatenate(out_g)
        return nu, gam


# ======================================================================================= DMN features
def _ewm(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    m = x[:, 0].copy()
    out[:, 0] = m
    for j in range(1, x.shape[1]):
        m = (1.0 - alpha) * m + alpha * x[:, j]
        out[:, j] = m
    return out


def _expanding_std(x: np.ndarray, min_periods: int = 5) -> np.ndarray:
    n = np.arange(1, x.shape[1] + 1, dtype=np.float64)[None, :]
    c1, c2 = np.cumsum(x, 1), np.cumsum(x * x, 1)
    var = np.maximum(c2 / n - (c1 / n) ** 2, 0.0)
    sd = np.sqrt(var)
    sd[:, : min_periods - 1] = np.nan
    return sd


def dmn_features(R: np.ndarray, nu: np.ndarray, gamma: np.ndarray, scales: Sequence[int], macd_pairs: Sequence[Sequence[int]],
                 vol_span: int, bars_per_year: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bar DMN inputs from a window of raw returns R (n, L): returns the feature tensor (n, L, F) and the
    annualised ex-ante volatility (n, L) used by the Sharpe loss. Every quantity at bar j uses bars <= j of the window."""
    n, L = R.shape
    R = np.nan_to_num(R)
    alpha = 2.0 / (min(int(vol_span), L) + 1.0)
    m1, m2 = _ewm(R, alpha), _ewm(R * R, alpha)
    sig = np.sqrt(np.maximum(m2 - m1 * m1, 1e-12))
    sig_lag = np.concatenate([sig[:, :1], sig[:, :-1]], 1)                 # sigma_{t-1} applied at t (official: .shift(1))
    sig_lag = np.maximum(sig_lag, 1e-6)
    cum = np.cumsum(R, 1)
    feats = []
    for s in scales:
        s = int(s)
        if s > L - 1:
            continue
        f = np.zeros((n, L))
        f[:, s - 1] = cum[:, s - 1]
        if s < L:
            f[:, s:] = cum[:, s:] - cum[:, :-s]
        f = f / (sig_lag * np.sqrt(s))
        f[:, : s - 1] = 0.0
        feats.append(f)
    p = cum                                                                 # log price relative to the window start
    for S, Lg in macd_pairs:
        macd = _ewm(p, 1.0 / float(S)) - _ewm(p, 1.0 / float(Lg))        # halflife = log(.5)/log(1-1/S) <=> alpha = 1/S
        q = macd / np.maximum(_expanding_std(p), 1e-8)                     # official: rolling(63).std -> clipped to the window
        sgn = q / np.maximum(_expanding_std(q), 1e-8)                      # official: rolling(252).std -> clipped to the window
        feats.append(np.nan_to_num(sgn))
    feats.append(np.broadcast_to(np.nan_to_num(nu, nan=0.5)[:, None], (n, L)).copy())
    feats.append(np.broadcast_to(np.nan_to_num(gamma, nan=0.5)[:, None], (n, L)).copy())
    F = np.stack(feats, -1).astype(np.float32)
    return np.nan_to_num(F), sig_lag * np.sqrt(float(bars_per_year))


class _DMN(nn.Module):
    def __init__(self, f: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(f, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.drop(x))
        return torch.tanh(self.out(self.drop(h))).squeeze(-1)             # (B, L) positions in (-1, 1), Eq. 13


def sharpe_loss(pos: torch.Tensor, nxt: torch.Tensor, sig_ann: torch.Tensor, vol_target: float, bars_per_year: int) -> torch.Tensor:
    """Eq. 14 on the captured returns of Eq. 11 over all (asset, time) pairs of the minibatch."""
    R = pos * (vol_target / sig_ann) * nxt
    return -np.sqrt(float(bars_per_year)) * R.mean() / torch.sqrt(R.var(unbiased=False) + 1e-9)


# ======================================================================================= baseline
@register("cg_cpd_dmn")
class CPDDMNBaseline(RowAwareMixin, ScoreToTradeBaseline):
    name = "cg_cpd_dmn"
    paper = "Wood, Roberts, Zohren, Slow Momentum with Fast Reversion, JFDS 2022, arXiv:2105.13727"
    official_repo = "github.com/kieranjwood/slow-momentum-fast-reversion (CPD) ; github.com/kieranjwood/trading-momentum-transformer (DMN)"
    status = "reimplemented"
    year = 2022
    cutoff_date = None
    conventions = {
        "L1": "Causal. CPD input standardised within the LBW window (Eq. 2; changepoint_detection.py: StandardScaler().fit(Y_data) on the window); "
              "DMN inputs vol-scaled by a 60-day EWM std lagged one day (classical_strategies.py: VOL_LOOKBACK=60, .shift(1)) and "
              "winsorised at 5 EWM std with a 252-day half-life (App. B); final StandardScaler fitted on the TRAINING split only "
              "(mom_trans/model_inputs.py ModelFeatures.set_scalers). No full-sample statistic -> the scaffold's L1 switch is inert for B1.",
        "L2": "No alarm threshold: a position in (-1, 1) is taken every day (Sec. IV-A, Eq. 13). published mode: threshold = -inf "
              "(always in market); causal mode: delayed-ACI gate on the CPD severity (our anomaly-to-trade reading).",
        "L3": "Validation = last 10 % of the training window per asset (Sec. IV-A); 50 random-search trials over Exhibit 7, LBW in "
              "{10, 21, 63, 126, 252} picked by validation loss; early stopping patience 25 (App. B). No test-set selection.",
        "L4": "No labels, no point-adjust (strategy returns only). Our planted labels are used for AUC-PR reporting only.",
        "L5": "Prices only (Pinnacle CLC continuous futures); no availability lag issue.",
        "L6": "Fixed universe of 50 liquid continuous futures 1990-2020 (Sec. IV-B, App. A); mild selection on liquidity/ data completeness (<10 % missing).",
        "L7": "Position X_t formed at close t earns r_{t+1} = p_{t+1}/p_t - 1 (Eq. 11): decision and fill at the same close (same-bar fill). "
              "Transaction costs not in the loss; App. C Eq. 15 charges C * sigma_tgt |X_t/sigma_t - X_{t-1}/sigma_{t-1}| for C in 0..5 bps. "
              "The protocol runner prices decisions at t (published) or t + latency (causal).",
        "L8": "No pre-trained component.",
    }
    unspecified = [
        "Kernel hyper-parameters are maximised over a fixed grid with the signal variance profiled out (closed form) instead of gpflow's L-BFGS-B "
        "(Eq. 7 is still the objective); steepness s is on the grid rather than optimised; the paper fixes the initialisation c = t - l/2, s = 1.",
        "The CPD features nu, gamma are computed once at the window end and broadcast to every LSTM step (the paper precomputes them per day); "
        "the LSTM sequence length equals the scaffold look-back (paper: 63) and CPD LBW = 21 (paper: tuned in {10, 21, 63, 126, 252}).",
        "Return scales > look-back - 1 are dropped; the MACD normalisers (rolling 63 / 252-day std) and the EWM volatility (span 60) are "
        "truncated to the look-back window (expanding statistics inside the window) because the scaffold only exposes the window.",
        "Location score follows Eq. 10 (gamma = (c - (t - l)) / l); the official code exports the complement (t - c) / l (changepoint_detection.py).",
        "Winsorisation at 5 EWM std (252-day half-life) replaced by clipping the train-standardised features at +-5.",
        "Sharpe loss over all (asset, bar) pairs of the minibatch (keras batch loss); next-bar target at the last window step = in-fold future return.",
        "The sigma_tgt / sigma_t leverage of Eq. 11 enters the loss only; the traded weight is the tanh position times w_max (scaffold cap).",
        "No random search / early stopping in the smoke config (epochs fixed by cfg); the smoke config trains 2 epochs.",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons: Sequence[int], device: torch.device):
        super().__init__(cfg, dims, horizons, device)
        self._init_rows()
        c = dict(cfg.get("cpd", {}))
        self.cpd = GPChangePointScorer(int(c.get("lbw", 21)), tuple(c.get("lengthscales", (2.0, 8.0))), tuple(c.get("noise_ratios", (0.3, 1.0, 3.0))),
                                       tuple(c.get("scale_ratios", (0.25, 1.0, 4.0))), tuple(c.get("steepness", (1.0,))), int(c.get("loc_step", 1)))
        d = dict(cfg.get("dmn", {}))
        self.hidden, self.dropout = int(d.get("hidden", 8)), float(d.get("dropout", 0.1))
        self.epochs, self.lr, self.batch_size = int(d.get("epochs", 2)), float(d.get("lr", 1e-3)), int(d.get("batch_size", 256))
        self.vol_span, self.vol_target = int(d.get("vol_span", 60)), float(d.get("vol_target", 0.15))
        self.scales = [int(s) for s in d.get("return_scales", (1, 21, 63, 126, 252))]
        self.macd_pairs = [tuple(int(v) for v in p) for p in d.get("macd_pairs", ((8, 24), (16, 48), (32, 96)))]
        self.winsor = float(d.get("winsor", 5.0))
        self.always_in_market = bool(cfg.get("always_in_market", True))
        self.model_op = self._register_op(_CPDDMNModel())
        self.net: Optional[_DMN] = None
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self._last_pos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ features
    def _features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = self._raw_returns(X)
        nu, gam = self.cpd.score(R)
        F, sig_ann = dmn_features(R, nu, gam, self.scales, self.macd_pairs, self.vol_span, self.bars_per_year)
        return F, sig_ann, nu, gam

    def _standardise(self, F: np.ndarray) -> np.ndarray:
        Z = (F - self.feat_mean) / self.feat_std
        return np.clip(Z, -self.winsor, self.winsor).astype(np.float32)

    def _positions(self, Fz: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for a in range(0, len(Fz), 1024):
                out.append(self.net(torch.as_tensor(Fz[a : a + 1024], device=self.device)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.L))

    # ------------------------------------------------------------------ hooks
    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        rows = self._rows_for(X)
        F, sig_ann, nu, gam = self._features(X)
        R = self._raw_returns(X)
        fut0 = rows["fut_ret"][:, 0] if rows is not None else np.zeros(len(X))
        nxt = np.concatenate([np.nan_to_num(R[:, 1:]), np.nan_to_num(fut0)[:, None]], 1)   # r_{j+1} for every window step
        flat = F.reshape(-1, F.shape[-1])
        self.feat_mean = flat.mean(0, keepdims=False)[None, None, :]
        self.feat_std = np.maximum(flat.std(0)[None, None, :], 1e-6)
        Fz = self._standardise(F)
        torch.manual_seed(int(self.cfg.get("seed", 0)))
        self.net = _DMN(F.shape[-1], self.hidden, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n = len(Fz)
        if n >= 8:
            xt = torch.as_tensor(Fz, device=self.device)
            nt = torch.as_tensor(nxt, dtype=torch.float32, device=self.device)
            st = torch.as_tensor(np.maximum(sig_ann, 1e-4), dtype=torch.float32, device=self.device)
            g = torch.Generator().manual_seed(int(self.cfg.get("seed", 0)))
            self.net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                for a in range(0, n, self.batch_size):
                    idx = perm[a : a + self.batch_size]
                    if len(idx) < 4:
                        continue
                    loss = sharpe_loss(self.net(xt[idx]), nt[idx], st[idx], self.vol_target, self.bars_per_year)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    opt.step()
        self._stamp_ops(t_end + 1)                                          # Sharpe-loss target r_{t_end+1} = fut_ret[:, 0] of the last row
        self.diag.update({"cpd_grid": self.cpd.n_grid, "dmn_features": int(F.shape[-1]), "train_nu_mean": float(np.nanmean(nu)) if len(nu) else float("nan")})

    def _score(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            self._last_pos = np.zeros(0)
            return np.zeros(0)
        F, _, nu, _ = self._features(X)
        self._last_pos = self._positions(self._standardise(F))[:, -1] if self.net is not None else np.zeros(len(X))
        return nu

    def _direction(self, X: np.ndarray, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        if self._last_pos is None or len(self._last_pos) != len(X):
            self._score(X)
        return np.clip(np.nan_to_num(self._last_pos), -1.0, 1.0)

    def _published_threshold(self, s_train, s_val, s_test, y_test) -> float:
        if self.always_in_market and native_threshold_requested(self):     # protocol-requested kinds (L2 / L3 switch) are honoured
            return float("-inf")                                            # the paper trades every day (no threshold)
        return super()._published_threshold(s_train, s_val, s_test, y_test)

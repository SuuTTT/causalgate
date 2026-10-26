"""Point-in-time (PIT) availability for a Panel, the PITView, and the Filtration Guard primitives.

Every datum x carries an availability time a(x) >= its event time t: a bar close is knowable at t, a tweet at its
publish bar, an AAER label at its release date, an index constituent at the announcement. The decision filtration is
F_t = sigma({x : a(x) <= t}). A pipeline is causal iff every operator it applies at t (including fitted parameters and
model-selection decisions) is F_t-measurable (docs/causalgate_research_plan.md, Sec. 2.1).

This module adds *optional* availability arrays to a Panel instance without touching warn2trade/data/schema.py
(append-only rule of the shared checkout): they are attached as instance attributes and listed in
panel.meta["pit_fields"]. Builders that do not emit them get `default_availability` (everything knowable at its bar,
labels knowable after the impact horizon).

PIT fields (bar-index units, int64; a value >= T means "never knowable inside the panel"):
    avail_feat   (T, N)       bar at which the row of price_feat / mid / liq became knowable (default t)
    avail_text   (T, N, M)    bar at which each document became knowable (publish time -> first bar close at or after it)
    avail_label  (T, N)       bar at which the label became knowable (event end, AAER release, pump peak, ...)
    avail_graph  (T, N, K)    bar at which the neighbour relation became knowable (co-mention, constituent change)
    universe     (T, N) bool  as-of membership: the asset was in the tradable universe at t *as known at t*
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import numpy as np

from warn2trade.data.schema import Panel

PIT_FIELDS = ("avail_feat", "avail_text", "avail_label", "avail_graph", "universe")
NEVER = np.iinfo(np.int64).max // 4      # sentinel for "not knowable within the panel"


# --------------------------------------------------------------------------------------- attach / default
def get_avail(panel: Panel, name: str) -> Optional[np.ndarray]:
    return getattr(panel, name, None)


def attach_availability(panel: Panel, **fields: np.ndarray) -> Panel:
    """Validate shapes and attach availability arrays as optional Panel fields (recorded in meta['pit_fields'])."""
    T, N = panel.mid.shape
    for k, v in fields.items():
        if k not in PIT_FIELDS:
            raise KeyError(f"unknown PIT field {k!r}; allowed: {PIT_FIELDS}")
        if v is None:
            continue
        v = np.asarray(v)
        if k == "avail_text":
            assert panel.text is not None and v.shape == panel.text.shape[:3], (k, v.shape)
        elif k == "avail_graph":
            assert panel.graph is not None and v.shape == panel.graph.shape[:3], (k, v.shape)
        else:
            assert v.shape == (T, N), (k, v.shape)
        if k == "universe":
            v = v.astype(bool)
        else:
            v = v.astype(np.int64)
            if k != "avail_text" and k != "avail_graph":
                tt = np.arange(T)[:, None]
                assert (v >= tt).all(), f"{k}: availability before the event bar"
        setattr(panel, k, v)
    panel.meta["pit_fields"] = sorted(set(panel.meta.get("pit_fields", [])) | {k for k, v in fields.items() if v is not None})
    return panel


def default_availability(panel: Panel, label_lag: Optional[int] = None, text_lag: int = 0, graph_lag: int = 0) -> Panel:
    """Fill in missing PIT fields with the conservative defaults: rows knowable at their bar, docs at bar + text_lag,
    labels at bar + label_lag (default: max(max impact horizon, label_rule horizon + lead), because a forward-looking
    label is only knowable once the horizon has elapsed), universe = asset exists (mid not NaN)."""
    T, N = panel.mid.shape
    tt = np.arange(T, dtype=np.int64)[:, None]
    f: Dict[str, Any] = {}
    if get_avail(panel, "avail_feat") is None:
        f["avail_feat"] = np.broadcast_to(tt, (T, N)).copy()
    if get_avail(panel, "avail_label") is None:
        if label_lag is not None:
            lag = int(label_lag)
        else:
            # a lead-extended rule label (label[t] = 1 if an onset occurs within `lead` bars, the onset itself being
            # defined on the next `horizon` bars) is only knowable at t + lead + horizon; take the larger of that and
            # the panel's max impact horizon
            lag = int(max(panel.meta.get("horizons", [1])))
            rule = panel.meta.get("label_rule")
            if isinstance(rule, dict) and rule.get("horizon") is not None:
                lag = max(lag, int(rule.get("horizon", 0)) + int(rule.get("lead", 0) or 0))
        f["avail_label"] = np.broadcast_to(tt + lag, (T, N)).copy()
    if panel.text is not None and get_avail(panel, "avail_text") is None:
        f["avail_text"] = np.broadcast_to(tt[:, :, None] + text_lag, panel.text.shape[:3]).copy()
    if panel.graph is not None and get_avail(panel, "avail_graph") is None:
        f["avail_graph"] = np.broadcast_to(tt[:, :, None] + graph_lag, panel.graph.shape[:3]).copy()
    if get_avail(panel, "universe") is None:
        f["universe"] = ~np.isnan(panel.mid)
    return attach_availability(panel, **f)


def clone_panel(panel: Panel) -> Panel:
    """Deep copy of all arrays, PIT fields included (used by the perturbation families)."""
    kw = {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in panel.__dict__.items() if k not in PIT_FIELDS}
    new = Panel(**kw)
    for k in PIT_FIELDS:
        v = get_avail(panel, k)
        if v is not None:
            setattr(new, k, v.copy())
    return new


# --------------------------------------------------------------------------------------- PITView
class PITView:
    """Read-only view of a Panel as known at bar t: rows after t are cut, and every datum with availability > t is
    masked (NaN for floats, False for masks, 0 for labels) even when its event bar is <= t (late labels, documents
    published after their content date, back-filled constituents)."""

    def __init__(self, panel: Panel, t: int):
        default_availability(panel)
        self.p, self.t = panel, int(t)

    # ---- helpers
    def _rows(self, arr: np.ndarray) -> np.ndarray:
        return arr[: self.t + 1]

    def _mask_rows(self, arr: np.ndarray, avail: np.ndarray, fill) -> np.ndarray:
        a = self._rows(arr).copy()
        late = self._rows(avail) > self.t
        if a.ndim > late.ndim:
            late = late[(...,) + (None,) * (a.ndim - late.ndim)]
            late = np.broadcast_to(late, a.shape)
        a[late] = fill
        return a

    # ---- fields
    @property
    def times(self) -> np.ndarray:
        return self._rows(self.p.times)

    @property
    def mid(self) -> np.ndarray:
        return self._mask_rows(self.p.mid, self.p.avail_feat, np.nan)

    @property
    def price_feat(self) -> np.ndarray:
        return self._mask_rows(self.p.price_feat.astype(np.float64), self.p.avail_feat, np.nan)

    @property
    def liq(self) -> np.ndarray:
        return self._mask_rows(self.p.liq.astype(np.float64), self.p.avail_feat, np.nan)

    @property
    def tradable(self) -> np.ndarray:
        return self._rows(self.p.tradable)

    @property
    def labels(self) -> np.ndarray:
        """Float array: NaN where the label is not yet knowable at t."""
        return self._mask_rows(self.p.labels.astype(np.float64), self.p.avail_label, np.nan)

    @property
    def text(self) -> Optional[np.ndarray]:
        if self.p.text is None:
            return None
        return self._mask_rows(self.p.text, self.p.avail_text, 0.0)

    @property
    def text_mask(self) -> Optional[np.ndarray]:
        if self.p.text is None:
            return None
        return self._mask_rows(self.p.text_mask, self.p.avail_text, False)

    @property
    def graph_mask(self) -> Optional[np.ndarray]:
        if self.p.graph is None:
            return None
        return self._mask_rows(self.p.graph_mask, self.p.avail_graph, False)

    @property
    def universe(self) -> np.ndarray:
        return self.p.universe[self.t]

    def knowable_label_rows(self) -> np.ndarray:
        """(t+1, N) bool: label rows that are already knowable at t (for delayed-feedback calibration)."""
        return self._rows(self.p.avail_label) <= self.t


# --------------------------------------------------------------------------------------- Filtration Guard
class FiltrationError(RuntimeError):
    """Raised when an operator stamped with as-of time a is used at a decision time t < a."""


class AsOfStamped:
    """Mixin for every fitted operator (normaliser, scaler, threshold, PCA, embedding index, checkpoint).

    `as_of` is the largest bar index whose data influenced the fitted state (-1 = unfitted / stateless).
    `check(t)` raises FiltrationError when the operator is used before its as-of time."""
    as_of: int = -1
    guard: Optional["FiltrationGuard"] = None

    def stamp(self, t: int) -> None:
        self.as_of = max(int(self.as_of), int(t))
        if self.guard is not None:
            self.guard.note(self)

    def check(self, t: int) -> None:
        if self.guard is not None:
            self.guard.record_use(self, t)
        if int(self.as_of) > int(t) and (self.guard is None or self.guard.strict):
            raise FiltrationError(f"{type(self).__name__} fitted as of bar {self.as_of} used at decision bar {t}")


@dataclass
class FiltrationGuard:
    """Registry of stamped operators + a log of (operator, t) uses; `assert_causal()` replays the log."""
    strict: bool = True
    ops: List[AsOfStamped] = field(default_factory=list)
    uses: List[Any] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def register(self, op: AsOfStamped) -> AsOfStamped:
        op.guard = self
        if op not in self.ops:
            self.ops.append(op)
        return op

    def note(self, op: AsOfStamped) -> None:
        if op not in self.ops:
            self.ops.append(op)

    def record_use(self, op: AsOfStamped, t: int) -> None:
        self.uses.append((type(op).__name__, int(op.as_of), int(t)))

    def assert_causal(self) -> None:
        bad = [u for u in self.uses if u[1] > u[2]]
        self.violations = [f"{n} as_of={a} used at t={t}" for n, a, t in bad]
        if bad and self.strict:
            raise FiltrationError("; ".join(self.violations))

    def report(self) -> Dict[str, Any]:
        return {"n_ops": len(self.ops), "n_uses": len(self.uses), "violations": list(self.violations),
                "ops": [{"op": type(o).__name__, "as_of": int(o.as_of)} for o in self.ops]}


# --------------------------------------------------------------------------------------- future perturbation
PERTURBATION_FAMILIES = ("noise", "shuffle", "splice", "delete")


def _future_rows_mask(avail: np.ndarray, t: int) -> np.ndarray:
    return avail > t


def perturb_future(panel: Panel, t: int, family: str, rng: np.random.Generator, block: int = 5, noise_scale: float = 1.0) -> Panel:
    """Return a copy of `panel` in which every datum with availability > t is perturbed by `family`.

    Covered fields: mid (log-return space, so the level path stays continuous at t), price_feat, liq and lob (rows with
    avail_feat > t), tradable and universe (rows > t, every family), labels / bars_to_peak (avail_label > t), documents
    with their text_age (avail_text > t), graph relations (avail_graph > t). Documents and labels whose *event bar* is
    <= t but whose availability is > t are perturbed too: that is exactly what a label-time or content-date join reads
    illegally. Anything with availability <= t is bit-identical to the original.

    `splice` replays a past segment of the *same* panel (rows [1, t], wrapped) after t; for labels / documents / graph
    relations it draws replacement values from data already knowable at t (bootstrap), so it differs from `shuffle`.
    A back-filled universe row with event bar <= t needs its own availability array to be testable; universe rows
    are treated as knowable at their bar here.
    """
    assert family in PERTURBATION_FAMILIES, family
    default_availability(panel)
    q = clone_panel(panel)
    T, N = q.mid.shape
    fut = np.arange(T) > t                                     # rows strictly after t
    n_fut = int(fut.sum())
    if n_fut == 0:
        return q
    # --- price path (log-return space), features, liquidity, LOB: rows whose avail_feat > t
    lr = np.zeros_like(q.mid)
    lr[1:] = np.log(q.mid[1:]) - np.log(q.mid[:-1])
    past_lr = lr[1 : t + 1]
    sd = np.nanstd(past_lr, axis=0) if past_lr.shape[0] > 2 else np.ones(N) * 0.01
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.01)
    feat_rows = _future_rows_mask(q.avail_feat, t)             # (T, N)
    lob = q.lob
    if lob is not None:
        lob_sd = np.nanstd(lob[: t + 1].reshape(-1, lob.shape[-1]).astype(np.float64), axis=0) if t >= 1 else np.ones(lob.shape[-1])
        lob_sd = np.where(np.isfinite(lob_sd) & (lob_sd > 0), lob_sd, 1.0)
    if family == "noise":
        lr[t + 1 :] += rng.standard_normal((n_fut, N)) * sd * noise_scale
        q.price_feat[feat_rows] += (rng.standard_normal(q.price_feat[feat_rows].shape) * noise_scale).astype(q.price_feat.dtype)
        q.liq[feat_rows] *= np.exp(rng.standard_normal(q.liq[feat_rows].shape) * 0.1).astype(q.liq.dtype)
        if lob is not None:
            lob[feat_rows] += (rng.standard_normal(lob[feat_rows].shape) * lob_sd * noise_scale).astype(lob.dtype)
        q.tradable[t + 1 :] ^= rng.random((n_fut, N)) < 0.5
        q.universe[t + 1 :] ^= rng.random((n_fut, N)) < 0.5
    elif family in ("shuffle", "splice"):
        src = (t + 1 + _block_permutation(n_fut, block, rng)) if family == "shuffle" else _splice_source(T, t, n_fut, rng)
        lr[t + 1 :] = lr[src]
        for arr in (q.price_feat, q.liq, lob, q.tradable, q.universe):
            if arr is not None:
                arr[t + 1 :] = arr[src]
    elif family == "delete":
        lr[t + 1 :] = 0.0
        q.price_feat[t + 1 :] = np.nan
        q.liq[t + 1 :] = q.liq[t]
        if lob is not None:
            if np.issubdtype(lob.dtype, np.floating):
                lob[t + 1 :] = np.nan
            else:
                lob[t + 1 :] = lob[t]
        q.tradable[t + 1 :] = q.tradable[t]
        q.universe[t + 1 :] = q.universe[t]
    q.mid[t + 1 :] = q.mid[t] * np.exp(np.cumsum(lr[t + 1 :], axis=0))
    # --- labels with availability > t (includes rows <= t with late release)
    lab_late = _future_rows_mask(q.avail_label, t)
    if lab_late.any():
        if family == "noise":
            flip = rng.random(lab_late.sum()) < 0.5
            vals = q.labels[lab_late].copy()
            vals[flip] = 1 - vals[flip]
            q.labels[lab_late] = vals
        elif family == "shuffle":
            q.labels[lab_late] = rng.permutation(q.labels[lab_late])
        elif family == "splice":
            known = q.labels[~lab_late]
            q.labels[lab_late] = rng.choice(known, size=int(lab_late.sum())) if known.size else rng.permutation(q.labels[lab_late])
        else:
            q.labels[lab_late] = 0
        btp = q.bars_to_peak
        btp[lab_late & (q.labels == 0)] = np.nan
    # --- documents with availability > t (embedding and age move together)
    if q.text is not None:
        doc_late = _future_rows_mask(q.avail_text, t) & q.text_mask
        if doc_late.any():
            k = int(doc_late.sum())
            age = q.text_age
            if family == "noise":
                q.text[doc_late] += (rng.standard_normal(q.text[doc_late].shape) * noise_scale * np.std(q.text[q.text_mask]) if q.text_mask.any() else 0).astype(q.text.dtype)
                if age is not None:
                    age[doc_late] = (age[doc_late] + rng.integers(1, 6, size=k)).astype(age.dtype)
            elif family in ("shuffle", "splice"):
                early = (~_future_rows_mask(q.avail_text, t)) & q.text_mask
                if family == "splice" and early.any():
                    idx = rng.integers(0, int(early.sum()), size=k)
                    q.text[doc_late] = q.text[early][idx]
                    if age is not None:
                        age[doc_late] = age[early][idx]
                else:
                    perm = rng.permutation(k)
                    q.text[doc_late] = q.text[doc_late][perm]
                    if age is not None:
                        age[doc_late] = age[doc_late][perm]
            else:
                q.text[doc_late] = 0.0
                q.text_mask[doc_late] = False
                if age is not None:
                    age[doc_late] = 0
    # --- graph relations with availability > t
    if q.graph is not None:
        g_late = _future_rows_mask(q.avail_graph, t) & q.graph_mask
        if g_late.any():
            k = int(g_late.sum())
            if family == "noise":
                q.graph[g_late] += (rng.standard_normal(q.graph[g_late].shape) * noise_scale).astype(q.graph.dtype)
            elif family in ("shuffle", "splice"):
                early = (~_future_rows_mask(q.avail_graph, t)) & q.graph_mask
                if family == "splice" and early.any():
                    q.graph[g_late] = q.graph[early][rng.integers(0, int(early.sum()), size=k)]
                else:
                    q.graph[g_late] = q.graph[g_late][rng.permutation(k)]
            else:
                q.graph_mask[g_late] = False
    return q


def perturb_fill_gap(panel: Panel, t: int, latency: int, rng: np.random.Generator, delta: Optional[float] = None) -> Panel:
    """L7 probe: perturb only the returns of the bars strictly between the decision bar t and the fill bar t+latency
    (a common log shift applied to every price from t+1 on). A pipeline that fills at t + latency has a decision-t PnL
    that is invariant to this shift; a same-bar fill (fill at the close of t) is not."""
    q = clone_panel(panel)
    T = q.mid.shape[0]
    if t + latency >= T:
        return q
    sd = np.nanstd(np.log(q.mid[1 : t + 1]) - np.log(q.mid[: t]), axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.01)
    d = (rng.standard_normal(q.mid.shape[1]) * sd * 3.0) if delta is None else np.full(q.mid.shape[1], delta)
    q.mid[t + 1 :] *= np.exp(d)[None, :]
    if q.price_feat.shape[-1] > 0:
        # the return feature of bar t+1 moves with the price shift (feature column 0 is the log return by convention
        # of the planted builder; real builders declare meta['ret_col'])
        col = int(q.meta.get("ret_col", 0))
        q.price_feat[t + 1, :, col] += d.astype(q.price_feat.dtype)
    return q


def _block_permutation(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / block))
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in order])
    if n > 1 and np.array_equal(idx, np.arange(n)):
        idx = np.roll(idx, block)
    return idx


def _splice_source(T: int, t: int, n_fut: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a past segment (wrapped) of length n_fut taken from rows [1, t] of the same panel."""
    if t >= 2:
        start = int(rng.integers(1, max(2, t)))
        src = 1 + (start - 1 + np.arange(n_fut)) % t
    else:
        # too little past to splice from: replay the future rows in reverse order (still a perturbation)
        src = np.arange(t + 1, t + 1 + n_fut)[::-1]
    return src

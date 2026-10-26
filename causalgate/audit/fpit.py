"""Future-Perturbation Invariance Test (FPIT): the black-box causality audit.

For random probe times t and perturbation families P in {noise, shuffle, splice, delete} applied to every datum with
availability > t (prices in log-return space, late labels, late documents, late graph relations), recompute the
pipeline's score s_t, threshold q_t, alarm z_t and weight w_t. Any change is a violation: a causal pipeline is
F_t-measurable, so its decision-time outputs cannot depend on data outside F_t (Prop. 5: soundness).

A fifth family, `fill_gap`, probes execution timing (L7): the returns of the bars strictly between the decision bar t
and the fill bar t + latency are shifted; the realised PnL of decision t must be invariant. A same-bar fill is not.

Localisation: if the pipeline returns one (T, N) array per operator (in `pipeline.operators` order), the first
operator whose output at t changed is reported. FPIT complements Fonseca's static type-and-effect checker
(arXiv:2607.04958) for opaque PyTorch / LLM pipelines; its power on planted leaks is measured in E4.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence
import numpy as np

from warn2trade.data.schema import Panel
from ..data.pit import perturb_future, perturb_fill_gap, PERTURBATION_FAMILIES, default_availability

DECISION_KEYS = ("score", "thr", "alarm", "weight")


class AuditablePipeline(Protocol):
    operators: Sequence[str]
    latency: int

    def run(self, panel: Panel) -> Dict[str, np.ndarray]: ...


@dataclass
class FPITResult:
    n_probes: int
    families: List[str]
    violation_rate: float                       # share of (probe, family) pairs with any decision-time change
    violation_rate_by_family: Dict[str, float]
    violation_rate_by_key: Dict[str, float]     # share of pairs in which score / thr / alarm / weight changed
    mean_abs_delta: Dict[str, float]            # mean |delta| of each decision-time quantity over violating pairs
    first_violating_operator: Dict[str, int]    # Counter of the earliest operator whose output at t changed
    fill_gap_violation_rate: float              # L7 probe (NaN if not run)
    passed: Optional[bool]                      # None = inconclusive (no probe pair, or L7 requested but not run)
    l7_tested: bool = False
    fill_gap_requested: bool = False
    details: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {"n_probes": self.n_probes, "violation_rate": self.violation_rate, "by_family": self.violation_rate_by_family,
                "by_key": self.violation_rate_by_key, "mean_abs_delta": self.mean_abs_delta,
                "first_violating_operator": dict(self.first_violating_operator),
                "fill_gap_violation_rate": self.fill_gap_violation_rate, "l7_tested": self.l7_tested,
                "fill_gap_requested": self.fill_gap_requested,
                "fill_gap": (("violation" if self.fill_gap_violation_rate > 0 else "no violation") if self.l7_tested else "not run"),
                "passed": self.passed}


def _delta(a: np.ndarray, b: np.ndarray, t: int) -> np.ndarray:
    x = np.atleast_1d(np.asarray(a, dtype=np.float64)[t])
    y = np.atleast_1d(np.asarray(b, dtype=np.float64)[t])
    same = (x == y) | (np.isnan(x) & np.isnan(y))   # equal values, matching +-inf and NaN-vs-NaN are unchanged
    with np.errstate(invalid="ignore"):
        d = np.abs(x - y)
    d = np.where(same, 0.0, d)
    d[np.isnan(d)] = np.inf                     # NaN on one side only, or +inf vs -inf = a change
    return d


def fpit(pipeline: AuditablePipeline, panel: Panel, probe_times: Sequence[int], families: Sequence[str] = PERTURBATION_FAMILIES,
         seed: int = 0, tol: float = 1e-9, fill_gap: bool = True, base: Optional[Dict[str, np.ndarray]] = None,
         keep_details: bool = True) -> FPITResult:
    """Run FPIT on `pipeline` over `probe_times`. `base` lets the caller pass a cached unperturbed run."""
    default_availability(panel)
    rng = np.random.default_rng(seed)
    base = base if base is not None else pipeline.run(panel)
    ops = [o for o in getattr(pipeline, "operators", ()) if o in base]
    fam_hits: Counter = Counter()
    fam_n: Counter = Counter()
    key_hits: Counter = Counter()
    deltas: Dict[str, List[float]] = {k: [] for k in DECISION_KEYS}
    first_ops: Counter = Counter()
    details: List[Dict[str, Any]] = []
    n_pairs = 0
    for t in probe_times:
        t = int(t)
        for fam in families:
            pert = perturb_future(panel, t, fam, rng)
            out = pipeline.run(pert)
            changed = {}
            for k in DECISION_KEYS:
                if k in base and k in out:
                    d = _delta(base[k], out[k], t)
                    changed[k] = float(np.nanmax(d)) if d.size else 0.0
            viol = any(v > tol for v in changed.values())
            n_pairs += 1
            fam_n[fam] += 1
            if viol:
                fam_hits[fam] += 1
                for k, v in changed.items():
                    if v > tol:
                        key_hits[k] += 1
                        deltas[k].append(v)
                first = None
                for o in ops:
                    if np.nanmax(_delta(base[o], out[o], t)) > tol:
                        first = o
                        break
                first_ops[first or "unlocalised"] += 1
            if keep_details:
                details.append({"t": t, "family": fam, "violation": bool(viol), "max_delta": changed})
    fg_rate = float("nan")
    l7_tested = False
    if fill_gap and "pnl" in base and len(probe_times):
        # the L7 probe only has power where the pipeline actually trades at t: keep the trading probes and, if fewer
        # than len(probe_times) of them trade, add the nearest trading bars inside the probed range
        fg_probes = list(probe_times)
        if "weight" in base:
            w = np.asarray(base["weight"], dtype=np.float64)
            trades_at = np.nan_to_num(w).reshape(w.shape[0], -1).any(axis=1) if w.ndim > 1 else np.nan_to_num(w) != 0
            lo, hi = int(min(probe_times)), int(max(probe_times)) + 1
            fg_probes = [int(t) for t in probe_times if trades_at[int(t)]]
            extra = [int(t) for t in np.arange(lo, hi) if trades_at[t] and int(t) not in fg_probes]
            fg_probes = (fg_probes + extra)[: max(len(probe_times), 1)]
        hits = 0
        for t in fg_probes:
            t = int(t)
            pert = perturb_fill_gap(panel, t, int(getattr(pipeline, "latency", 1)), rng)
            out = pipeline.run(pert)
            d = _delta(base["pnl"], out["pnl"], t)
            hit = bool(np.nanmax(d) > tol) if d.size else False
            hits += int(hit)
            if keep_details:
                details.append({"t": t, "family": "fill_gap", "violation": hit, "max_delta": {"pnl": float(np.nanmax(d)) if d.size else 0.0}})
        fg_rate = hits / len(fg_probes) if fg_probes else float("nan")
        l7_tested = bool(fg_probes)
    rate = sum(fam_hits.values()) / n_pairs if n_pairs else float("nan")
    if n_pairs == 0 and not l7_tested:
        passed: Optional[bool] = None
    elif (n_pairs and rate > 0.0) or (l7_tested and fg_rate > 0.0):
        passed = False
    elif n_pairs == 0 or (fill_gap and not l7_tested):
        passed = None                                       # nothing contradicts causality, but a requested channel was not tested
    else:
        passed = True
    res = FPITResult(
        n_probes=len(probe_times), families=list(families), violation_rate=float(rate),
        violation_rate_by_family={f: fam_hits[f] / max(fam_n[f], 1) for f in families},
        violation_rate_by_key={k: key_hits[k] / max(n_pairs, 1) for k in DECISION_KEYS},
        mean_abs_delta={k: (float(np.mean([v for v in deltas[k] if np.isfinite(v)])) if any(np.isfinite(deltas[k])) else 0.0) for k in DECISION_KEYS},
        first_violating_operator=dict(first_ops), fill_gap_violation_rate=float(fg_rate),
        passed=passed, l7_tested=l7_tested, fill_gap_requested=bool(fill_gap), details=details)
    return res


def sample_probe_times(t_lo: int, t_hi: int, n: int, seed: int = 0, near: Optional[Sequence[int]] = None,
                       frac_near: float = 0.5, span: int = 3) -> np.ndarray:
    """n distinct probe bars in [t_lo, t_hi). With `near` (e.g. labelled event onsets) a share `frac_near` of the probes
    is drawn from the `span` bars before those anchors ("stratified" probing): leaks that are sparse in time, such as
    a label-time join, are only visible at bars where a late datum is being read, so uniform probes have low per-probe
    power there (Prop. 5 then needs many probes); event-adjacent probes restore it."""
    rng = np.random.default_rng(seed)
    pool = np.arange(t_lo, t_hi)
    n = min(n, max(len(pool), 1))
    chosen: List[int] = []
    if near is not None and len(near) and frac_near > 0:
        anchors = np.asarray([a for a in near if t_lo + span <= a < t_hi], dtype=int)
        if len(anchors):
            k = min(int(round(frac_near * n)), len(anchors))
            for a in rng.choice(anchors, size=k, replace=False):
                chosen.append(int(a - rng.integers(0, span + 1)))
    chosen = sorted(set(chosen))                # nearby anchors can collide: dedupe before topping up to n
    rest = [t for t in pool if t not in set(chosen)]
    m = n - len(chosen)
    if m > 0 and rest:
        chosen += [int(t) for t in rng.choice(rest, size=min(m, len(rest)), replace=False)]
    return np.array(sorted(set(chosen)), dtype=int)


def detection_power(p_single: float, n_probes: int) -> float:
    """Prop. 5: P[detect after n probes] >= 1 - (1 - p)^n for per-probe detection probability p."""
    return float(1.0 - (1.0 - p_single) ** n_probes)

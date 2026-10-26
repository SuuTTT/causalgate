"""Leakage Audit Card (plan Sec. 3.4): one card per (method, dataset), assembled from the component audits.

    1. leakage premium per channel (one-switch) and total        -> eval.metrics.leakage_premium / channel_premiums
    2. FPIT violation rate and localisation                        -> audit.fpit
    3. date-shift probe profile and collapse flag                   -> audit.date_shift
    4. martingale-difference placebo                                -> audit.placebo
    5. Sharpe plausibility (PSR / DSR / haircut, flags)             -> audit.plausibility
    6. pre-trained component check (cutoff date vs fold test start) -> here
The card renders to JSON (results/causalgate/cards/<method>__<dataset>.json) and Markdown (for the supplement).
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import numpy as np


def _f(x, nd=3):
    try:
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))
    except Exception:
        return str(x)


@dataclass
class LeakageAuditCard:
    method: str
    dataset: str
    protocol_hash: str = ""
    seeds: list = field(default_factory=list)
    leakage_premium: Dict[str, float] = field(default_factory=dict)        # LP_SR, LP_AUC_PR, ...
    channel_premium: Dict[str, float] = field(default_factory=dict)        # L1..L8 -> LP_SR^{(l)}
    fpit: Dict[str, Any] = field(default_factory=dict)                     # summary() of FPITResult
    date_shift: Dict[str, Any] = field(default_factory=dict)
    placebo: Dict[str, Any] = field(default_factory=dict)
    plausibility: Dict[str, Any] = field(default_factory=dict)
    pretrained: Dict[str, Any] = field(default_factory=dict)               # cutoff_date, test_start, ok
    deployability_gap: Dict[str, float] = field(default_factory=dict)
    alarm_rate: Dict[str, float] = field(default_factory=dict)             # alpha, realised, error
    notes: list = field(default_factory=list)

    # ---- verdict
    def flags(self) -> Dict[str, bool]:
        f = {}
        _vr = self.fpit.get("violation_rate"); _fg = self.fpit.get("fill_gap_violation_rate")
        f["fpit_violation"] = bool((_vr is not None and np.isfinite(float(_vr)) and float(_vr) > 0.0)
                                   or (_fg is not None and np.isfinite(float(_fg)) and float(_fg) > 0.0))
        f["date_shift_collapse"] = bool(self.date_shift.get("collapse_flag", False))
        f["placebo_leak"] = bool(self.placebo.get("placebo_leak_flag", False) or self.plausibility.get("placebo_leak", False))
        f["implausible_sharpe"] = bool(self.plausibility.get("implausible_sharpe", False))
        f["pretrained_after_test_start"] = bool(self.pretrained.get("ok") is False)
        f["pretrained_unverified"] = bool(self.pretrained.get("ok") is None and self.pretrained.get("cutoff_date") is not None)
        vr = self.fpit.get("violation_rate", float("nan"))
        f["fpit_not_run"] = bool(not self.fpit or not self.fpit.get("n_probes") or vr is None or not np.isfinite(float(vr)))
        f["l7_not_run"] = bool(self.fpit.get("fill_gap_requested", False) and not self.fpit.get("l7_tested", False))
        lp = self.leakage_premium.get("LP_SR", float("nan"))
        f["positive_leakage_premium"] = bool(np.isfinite(lp) and lp > 0)
        dg = self.deployability_gap.get("DG_SR", float("nan"))
        f["nonzero_deployability_gap"] = bool(np.isfinite(dg) and abs(dg) > 1e-6)
        return f

    def verdict(self) -> str:
        f = self.flags()
        hard = f["fpit_violation"] or f["pretrained_after_test_start"] or f["nonzero_deployability_gap"]
        soft = f["date_shift_collapse"] or f["placebo_leak"] or f["implausible_sharpe"]
        if hard:
            return "NOT CAUSAL"
        if soft:
            return "SUSPECT (statistical audit flags; no constructive violation found)"
        missing = self.incomplete()
        if missing:
            return "INCOMPLETE (audit not run / unverified: " + ", ".join(missing) + ")"
        return "CAUSAL (no violation found by the audit)"

    def incomplete(self) -> list:
        f = self.flags(); m = []
        if f["fpit_not_run"]:
            m.append("FPIT")
        if f["l7_not_run"]:
            m.append("fill-gap (L7)")
        for name in ("date_shift", "placebo", "plausibility"):
            if not getattr(self, name):
                m.append(name)
        if f["pretrained_unverified"]:
            m.append("pre-trained cutoff")
        return m

    # ---- io
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["flags"] = self.flags(); d["verdict"] = self.verdict()
        return d

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.method}__{self.dataset}.json")
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        with open(path[:-5] + ".md", "w") as fh:
            fh.write(self.to_markdown())
        return path

    def to_markdown(self) -> str:
        f = self.flags()
        L = [f"# Leakage Audit Card: {self.method} on {self.dataset}", "",
             f"Protocol hash `{self.protocol_hash or 'n/a'}` | seeds {self.seeds or 'n/a'} | **Verdict: {self.verdict()}**", "",
             "## 1. Leakage premium (P-O minus P-C)", "",
             "| metric | LP |", "|---|---|"]
        L += [f"| {k} | {_f(v)} |" for k, v in self.leakage_premium.items()]
        L += ["", "| channel | LP_SR (one-switch) |", "|---|---|"]
        L += [f"| {k} | {_f(v)} |" for k, v in self.channel_premium.items()]
        L += ["", "## 2. FPIT", "", f"violation rate {_f(self.fpit.get('violation_rate'))} over {self.fpit.get('n_probes', 'n/a')} probes; "
              f"fill-gap (L7) {_f(self.fpit.get('fill_gap_violation_rate')) if self.fpit.get('l7_tested', True) else 'not run'}; first violating operator {self.fpit.get('first_violating_operator', {})}; "
              f"by family {self.fpit.get('by_family', {})}", ""]
        L += ["## 3. Date-shift probe", "", f"profile {self.date_shift.get('profile', {})}; peek gain {_f(self.date_shift.get('peek_gain'))}; "
              f"delay decay {_f(self.date_shift.get('delay_decay'))}; collapse ratio {_f(self.date_shift.get('collapse_ratio'))} -> flag {f['date_shift_collapse']}", ""]
        L += ["## 4. Martingale-difference placebo", "", f"real SR {_f(self.placebo.get('real'))}; placebo mean {_f(self.placebo.get('placebo_mean'))}, "
              f"p99 {_f(self.placebo.get('placebo_p99'))}, share positive {_f(self.placebo.get('placebo_frac_positive'))}, p-value {_f(self.placebo.get('placebo_p_value'))} -> flag {f['placebo_leak']}", ""]
        L += ["## 5. Sharpe plausibility", "", f"SR {_f(self.plausibility.get('SR'))}, PSR {_f(self.plausibility.get('PSR'))}, DSR {_f(self.plausibility.get('DSR'))} "
              f"(trials {self.plausibility.get('n_trials', 'n/a')}), haircut SR {_f(self.plausibility.get('SR_haircut'))} -> implausible {f['implausible_sharpe']}", ""]
        L += ["## 6. Pre-trained components", "", f"{self.pretrained if self.pretrained else 'none declared'}", ""]
        L += ["## Deployability gap and alarm rate", "", f"{self.deployability_gap} | {self.alarm_rate}", ""]
        if self.notes:
            L += ["## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(L) + "\n"


def pretrained_check(cutoff_date: Optional[str], test_start_date: Optional[str]) -> Dict[str, Any]:
    """L8: a pre-trained component is admissible for a fold iff its training cutoff < the fold's test start."""
    if cutoff_date is None:
        return {"cutoff_date": None, "test_start": test_start_date, "ok": True, "note": "no pre-trained component declared"}
    if test_start_date is None:
        return {"cutoff_date": cutoff_date, "test_start": None, "ok": None, "note": "fold test start unknown (bar-indexed panel)"}
    ok = str(cutoff_date) < str(test_start_date)
    return {"cutoff_date": cutoff_date, "test_start": test_start_date, "ok": bool(ok), "note": "" if ok else "cutoff after test start: L8 leak unless the checkpoint is replaced"}

#!/usr/bin/env python3
"""E4 on the planted testbed (A2): FPIT power / false-positive rate per planted leak and per probe budget, plus the
date-shift and placebo diagnostics of the reference pipeline. Ground truth is known (the switches), so this gives the
audit's ROC.

    python3 scripts/cg_planted_leaks.py --seeds 42,2024,2025,2026 --n-probes 4,8,16 --out results/causalgate/planted
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warn2trade.utils import DATASETS
import causalgate.data.planted  # noqa: F401
from causalgate.data.planted import ReferencePipeline, LeakSwitches
from causalgate.audit.fpit import fpit, sample_probe_times, detection_power
from causalgate.audit.date_shift import date_shift_profile
from causalgate.audit.placebo import placebo_distribution
from warn2trade.backtest import metrics as WM

LEAKS = ("none", "global_norm", "test_quantile_threshold", "same_bar_fill", "label_time_join")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,2024,2025,2026")
    ap.add_argument("--n-probes", default="4,8,16")
    ap.add_argument("--T", type=int, default=2500)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n-placebo", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "causalgate", "planted"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        p = DATASETS.get("cg_planted")(dict(name="cg_planted", seed=seed, T=args.T, N=args.N)).build_panel()
        test_start = int(0.65 * p.T)
        onsets = [e["onset_t"] for e in p.meta["events"] if test_start + 10 <= e["onset_t"] < p.T - 40]
        for leak in LEAKS:
            sw = LeakSwitches(**({leak: True} if leak != "none" else {}))
            pipe = ReferencePipeline(sw, latency=1, hold=5, alpha=0.05, warmup=60, window=250, test_start=test_start)
            base = pipe.run(p)
            for n in [int(x) for x in args.n_probes.split(",")]:
                for strategy in ("uniform", "stratified"):
                    probes = sample_probe_times(test_start + 5, p.T - 40, n, seed=seed, near=onsets if strategy == "stratified" else None)
                    t0 = time.time()
                    res = fpit(pipe, p, probes, seed=seed, base=base, keep_details=False)
                    flagged = (res.passed is False)              # None = inconclusive (not a flag)
                    rows.append({"seed": seed, "leak": leak, "n_probes": n, "n_probes_drawn": int(len(probes)), "strategy": strategy, "violation_rate": res.violation_rate,
                                 "passed": res.passed, "l7_tested": res.l7_tested,
                                 "fill_gap_rate": res.fill_gap_violation_rate, "flagged": flagged, "correct": flagged == (leak != "none"),
                                 "first_op": max(res.first_violating_operator, key=res.first_violating_operator.get) if res.first_violating_operator else "",
                                 "sec": time.time() - t0})
            # date-shift and placebo diagnostics (n_probes-independent)
            lam = 0 if sw.same_bar_fill else 1
            ds = date_shift_profile(base["weight"][test_start:], p.mid[test_start:], lam, 5, 252)
            def run_fn(pn, pipe=pipe, lam=lam):
                o = pipe.run(pn); lp = np.log(pn.mid); r = np.zeros(pn.T); w = np.nan_to_num(o["weight"])
                for t in range(test_start, pn.T - lam - 5):
                    r[t + lam + 5] += np.nansum(w[t] * (lp[t + lam + 5] - lp[t + lam]))
                return WM.sharpe(r[test_start:] / pn.N, 252)
            pl = placebo_distribution(run_fn, p, n=args.n_placebo, seed=seed, keep_from=test_start - 1)
            rows.append({"seed": seed, "leak": leak, "n_probes": 0, "strategy": "diagnostics", "date_shift_collapse": ds["collapse_ratio"],
                         "date_shift_flag": ds["collapse_flag"], "placebo_mean": pl["placebo_mean"], "placebo_frac_pos": pl["placebo_frac_positive"],
                         "placebo_flag": pl["placebo_leak_flag"], "SR_real": pl["real"]})
            print(f"seed {seed} {leak:24s} FPIT {[(r['n_probes'], r['strategy'], round(r['violation_rate'], 2)) for r in rows if r['seed'] == seed and r['leak'] == leak and r['n_probes']]} "
                  f"| shift collapse {ds['collapse_ratio']:.2f} | placebo mean {pl['placebo_mean']:.2f}")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "fpit_power.csv"), index=False)
    fp = df[df["n_probes"] > 0]
    power = fp.groupby(["leak", "strategy", "n_probes"])["flagged"].mean().unstack("n_probes")
    power.to_csv(os.path.join(args.out, "fpit_power_table.csv"))
    print("\nFPIT flag rate (power for planted leaks, false-positive rate for 'none'):\n", power.round(2).to_string())
    summary = {"power_table": power.reset_index().to_dict(orient="records"), "prop5_bound_example": detection_power(0.5, 8)}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()

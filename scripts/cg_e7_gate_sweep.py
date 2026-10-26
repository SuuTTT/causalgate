#!/usr/bin/env python3
"""E7: alarm-rate control across regimes -- fixed quantile vs POT vs ACI vs delayed-ACI, tau and gamma sweeps.

Score streams come from (a) the synthetic variance-switch null (|N(0, sigma_t)|, sigma switching 1 -> 3 -> 1) and
(b) the planted panel scored by the reference causal pipeline (events included). Outputs:
    results/causalgate/e7/gate_sweep.csv    realised rate, |err|, post-switch rate per (stream, gate, tau, gamma, seed)
    results/causalgate/e7/gate_paths.csv    rolling alarm-rate paths for Fig 6

    python3 scripts/cg_e7_gate_sweep.py --seeds 42,2024,2025,2026
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warn2trade.utils import DATASETS
import causalgate.data.planted  # noqa: F401
from causalgate.data.planted import ReferencePipeline, LeakSwitches
from causalgate.models.gate import FixedQuantileGate, POTGate, ACIGate, DelayedACIGate


def streams(seed: int, T: int = 6000):
    rng = np.random.default_rng(seed)
    sig = np.ones(T); sig[T // 3 : 2 * T // 3] = 3.0
    yield "variance_switch", np.abs(rng.standard_normal(T) * sig), [T // 3, 2 * T // 3], None
    p = DATASETS.get("cg_planted")(dict(name="cg_planted", seed=seed, T=3000, N=8)).build_panel()
    s = ReferencePipeline(LeakSwitches(), warmup=60, window=250).run(p)["score"]
    regime = p.meta["regime"]
    sw = [int(t) for t in np.nonzero(np.diff(regime))[0] if t >= 600][:6]     # switches inside the streamed segment (after calibration)
    yield "planted_pooled", np.nanmean(s, axis=1), sw, p.labels.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,2024,2025,2026")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "causalgate", "e7"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, paths = [], []
    for seed in [int(x) for x in args.seeds.split(",")]:
        for name, s, switches, labels in streams(seed):
            n_cal = 500
            calib, stream = s[:n_cal], s[n_cal:]
            lab = None if labels is None else labels[n_cal:]
            cfgs = [("fixed", FixedQuantileGate(args.alpha), {}), ("pot", POTGate(args.alpha, window=1000, refit_every=20), {})]
            for gamma in (0.002, 0.005, 0.01, 0.02, 0.05):
                cfgs.append((f"aci_g{gamma}", ACIGate(args.alpha, gamma=gamma, window=500), {"gamma": gamma}))
                for tau in (1, 3, 6, 11, 21):
                    cfgs.append((f"delayed_aci_g{gamma}_tau{tau}", DelayedACIGate(args.alpha, gamma=gamma, window=500, tau=tau), {"gamma": gamma, "tau": tau}))
            for gname, g, meta in cfgs:
                g.calibrate(calib, t_end=n_cal - 1)
                r = g.run(stream, t0=n_cal, labels=lab)
                a = r["alarm"]
                post = [float(np.nanmean(a[sw - n_cal : sw - n_cal + 500])) for sw in switches if sw - n_cal + 500 <= len(a) and sw - n_cal >= 0]
                rows.append({"seed": seed, "stream": name, "gate": gname, "alpha": args.alpha, **meta, "realised": r["realised_rate"], "abs_err": r["rate_error"],
                             "post_switch_rate_mean": float(np.mean(post)) if post else np.nan, "post_switch_abs_err_max": float(max(abs(x - args.alpha) for x in post)) if post else np.nan})
                if seed == 42 and gname in ("fixed", "pot", "aci_g0.01", "delayed_aci_g0.01_tau6"):
                    roll = pd.Series(np.nan_to_num(a)).rolling(250, min_periods=50).mean().to_numpy()
                    for t in range(0, len(a), 10):
                        paths.append({"stream": name, "gate": gname, "t": n_cal + t, "rolling_rate": roll[t], "alpha": args.alpha, "switch_t": ";".join(str(int(x)) for x in switches) if switches else ""})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, "gate_sweep.csv"), index=False)
    pd.DataFrame(paths).to_csv(os.path.join(args.out, "gate_paths.csv"), index=False)
    summ = df.groupby(["stream", "gate"])[["realised", "abs_err", "post_switch_abs_err_max"]].mean().round(4)
    summ.to_csv(os.path.join(args.out, "gate_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()

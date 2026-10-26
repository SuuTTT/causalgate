#!/usr/bin/env python3
"""Produce one Leakage Audit Card per (method, dataset): FPIT (with localisation and the L7 fill-gap probe),
date-shift probe, martingale-difference placebo, Sharpe plausibility (DSR with the declared trial count), the
pre-trained cutoff check, and the leakage premium read from existing protocol runs when present.

    python3 scripts/cg_audit.py --dataset cg_planted --baseline causalgate --protocol causal --seed 42 --device cpu --smoke \
        --n-probes 12 --n-placebo 10 --max-folds 1
    python3 scripts/cg_audit.py --dataset cg_planted --baseline cg_damp --protocol published ...

Cards are written to results/causalgate/cards/<baseline>__<dataset>[__<protocol>].{json,md}. Trials for the DSR are
counted from the result directories of the same (dataset, baseline) unless --n-trials is given.
"""
from __future__ import annotations
import glob
import json
import os
import time
import numpy as np
from cg_common import base_parser, load_cfgs, build_panel, make_runner, protocol_from_args, import_all, log, RESULTS
from causalgate.audit.fpit import fpit, sample_probe_times
from causalgate.audit.adapter import EventBaselinePipeline
from causalgate.audit.date_shift import date_shift_profile
from causalgate.audit.placebo import placebo_distribution
from causalgate.audit.plausibility import sharpe_plausibility
from causalgate.audit.card import LeakageAuditCard, pretrained_check
from causalgate.eval.metrics import leakage_premium, channel_premiums, CHANNELS
from warn2trade.backtest import metrics as WM


def _read_agg(results: str, dataset: str, baseline: str, protocol: str, bps: str = "5"):
    vals = []
    for path in glob.glob(os.path.join(results, dataset, baseline, protocol, "seed*", "fold*.json")):
        r = json.load(open(path))
        if bps in r.get("eval", {}):
            vals.append(r["eval"][bps])
    if not vals:
        return None
    keys = set().union(*[set(v) for v in vals])
    return {k: float(np.nanmean([v[k] for v in vals if k in v and isinstance(v[k], (int, float))])) for k in keys}


def main():
    p = base_parser("CausalGate leakage audit card")
    p.add_argument("--n-probes", type=int, default=12)
    p.add_argument("--n-placebo", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--families", default="noise,shuffle,splice,delete")
    p.add_argument("--no-refit", action="store_true", help="FPIT reuses the fitted model (tests test-time operators only)")
    p.add_argument("--placebo-method", default="sign_flip", choices=("sign_flip", "block_bootstrap", "garch_sign"))
    p.add_argument("--sr-var", type=float, default=None,
                   help="variance of the ANNUALISED Sharpe across the declared trials (DSR); default: estimated from the "
                        "seed runs under the results tree, else 0.25 (annualised SR std 0.5)")
    args = p.parse_args()
    import_all()
    protocol = protocol_from_args(args)
    cfg, ds_cfg, bcfg = load_cfgs(args)
    builder, panel = build_panel(ds_cfg)
    runner = make_runner(cfg, ds_cfg, builder, panel, args)
    fold = runner.folds(protocol, cfg["split"].get("max_folds") or 1)[0]
    from causalgate.eval.protocols import apply_data_switches
    panel_v = apply_data_switches(panel, protocol, fold)
    cls = runner.__class__
    from warn2trade.utils.registry import BASELINES
    bl_cls = BASELINES.get(args.baseline)
    bl_cfg = runner.baseline_cfg(bcfg, protocol, fold, panel_v, bl_cls)
    bl_cfg.pop("panel", None)
    hold_index = int(bl_cfg.get("hold_index", 2))
    pipe = EventBaselinePipeline(args.baseline, bl_cfg, fold, runner.horizons, runner.lookback, runner.latency, runner.exec_max, hold_index,
                                 runner.device, args.seed, refit=not args.no_refit, fill=protocol.fill)
    t0 = time.time()
    base = pipe.run(panel_v)
    log.info(f"base run {time.time() - t0:.1f}s")
    onsets = [int(e["onset_t"]) for e in panel.meta.get("events", []) if isinstance(e, dict) and "onset_t" in e]
    t_lo, t_hi = int(fold.test[0]) + 5, int(fold.test[-1]) - max(runner.horizons) - runner.latency - 5
    probes = sample_probe_times(t_lo, t_hi, args.n_probes, seed=args.seed, near=onsets if onsets else None)
    t0 = time.time()
    fp = fpit(pipe, panel_v, probes, families=tuple(args.families.split(",")), seed=args.seed, base=base, keep_details=False)
    log.info(f"FPIT {fp.summary()} ({time.time() - t0:.0f}s)")
    # date shift on the frozen decisions of the test window
    te0, te1 = int(fold.test[0]), int(fold.test[-1]) + 1
    hold = int(runner.horizons[hold_index])
    lam = 0 if protocol.fill == "same_bar" else runner.latency
    ds = date_shift_profile(base["weight"][te0:te1], panel_v.mid[te0:te1], lam, hold, runner.bars_per_year)
    # placebo: unchanged pipeline on martingale-difference surrogates of the test window (past kept intact)
    def run_fn(pn):
        o = pipe.run(pn)
        lp = np.log(pn.mid); r = np.zeros(pn.T); w = np.nan_to_num(o["weight"])
        for t in range(te0, pn.T - lam - hold):
            r[t + lam + hold] += np.nansum(w[t] * (lp[t + lam + hold] - lp[t + lam]))
        return WM.sharpe(r[te0:] / pn.N, runner.bars_per_year)
    t0 = time.time()
    pl = placebo_distribution(run_fn, panel_v, n=args.n_placebo, seed=args.seed, method=args.placebo_method, keep_from=te0 - 1)
    log.info(f"placebo real={pl['real']:.2f} mean={pl['placebo_mean']:.2f} frac+={pl['placebo_frac_positive']:.2f} ({time.time() - t0:.0f}s)")
    # plausibility on the real bar returns
    lp = np.log(panel_v.mid); r = np.zeros(panel_v.T); w = np.nan_to_num(base["weight"])
    for t in range(te0, panel_v.T - lam - hold):
        r[t + lam + hold] += np.nansum(w[t] * (lp[t + lam + hold] - lp[t + lam]))
    n_trials = args.n_trials or max(1, len(glob.glob(os.path.join(args.out, args.dataset, args.baseline, "*", "seed*"))))
    sr_var, sr_var_src = args.sr_var, "--sr-var"
    if sr_var is None:
        trial_srs = []
        for path in glob.glob(os.path.join(args.out, args.dataset, args.baseline, "*", "seed*", "fold*.json")):
            try:
                v = json.load(open(path)).get("eval", {}).get("5", {}).get("SR")
            except Exception:
                v = None
            if isinstance(v, (int, float)) and np.isfinite(v):
                trial_srs.append(float(v))
        if len(trial_srs) >= 3:
            sr_var, sr_var_src = float(np.var(trial_srs, ddof=1)), f"estimated from {len(trial_srs)} runs"
        else:
            sr_var, sr_var_src = 0.25, "default (annualised SR std 0.5)"
    plaus = sharpe_plausibility(r[te0:] / panel_v.N, runner.bars_per_year, n_trials, sr_var, pl["placebo"])
    # leakage premium from existing runs (if any)
    m_pub = _read_agg(args.out, args.dataset, args.baseline, "published"); m_c = _read_agg(args.out, args.dataset, args.baseline, "causal")
    lp_tab = leakage_premium(m_pub, m_c) if (m_pub and m_c) else {}
    one = {ch: _read_agg(args.out, args.dataset, args.baseline, f"causal+{ch}") for ch in CHANNELS}
    one = {k: v for k, v in one.items() if v}
    ch_tab = channel_premiums(m_c, one) if (m_c and one) else {}
    test_start = runner.folds(protocol, 1)[0].test[0]
    from causalgate.eval.protocols import _times_to_date
    pre = pretrained_check(getattr(bl_cls, "cutoff_date", None), _times_to_date(panel.times, int(test_start)))
    card = LeakageAuditCard(method=args.baseline, dataset=args.dataset, protocol_hash=protocol.hash(), seeds=[args.seed], leakage_premium=lp_tab,
                            channel_premium=ch_tab, fpit=fp.summary(), date_shift=ds, placebo={k: v for k, v in pl.items() if k != "placebo"},
                            plausibility=plaus, pretrained=pre,
                            alarm_rate={"alpha": float(bl_cfg.get("gate", {}).get("alpha", float("nan")) if isinstance(bl_cfg.get("gate"), dict) else float("nan")),
                                        "realised": float(np.nanmean(base["alarm"][te0:te1]))},
                            notes=[f"protocol {protocol.name} ({protocol.offline_channels or 'no offline channel'})", f"fold {fold.k}, probes {list(map(int, probes))}",
                                   f"FPIT refit={'no' if args.no_refit else 'yes'}", f"placebo method {args.placebo_method}, n={args.n_placebo}", f"DSR trials {n_trials}, annualised SR variance {sr_var:.3g} ({sr_var_src})",
                                   f"FPIT probes requested {args.n_probes}, drawn {len(probes)}"])
    out_dir = os.path.join(args.out, "cards")
    path = card.save(out_dir)
    if protocol.name != "causal":
        os.replace(path, path.replace(".json", f"__{protocol.name}.json")); os.replace(path[:-5] + ".md", path[:-5] + f"__{protocol.name}.md")
    log.info(f"verdict: {card.verdict()} -> {out_dir}")
    print(card.to_markdown())


if __name__ == "__main__":
    main()

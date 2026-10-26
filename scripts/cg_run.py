#!/usr/bin/env python3
"""Run one (dataset, baseline, protocol, seed) through the CausalGate protocol runner.

    python3 scripts/cg_run.py --freeze                                   # freeze protocol hashes (once, before test-fold runs;
                                                                         # refuses to overwrite a differing freeze without --force-refreeze)
    python3 scripts/cg_run.py --dataset cg_planted --baseline causalgate --protocol causal --seed 42 --device cpu --max-folds 1 --smoke
    python3 scripts/cg_run.py --dataset cg_stocknet --baseline cg_damp --protocol published --seed 2024

Writes results/causalgate/<dataset>/<baseline>/<protocol>/seed<k>/fold<j>.json (+ trades csv at the training cost).
"""
from __future__ import annotations
import os
import time
from cg_common import base_parser, load_cfgs, build_panel, make_runner, protocol_from_args, import_all, save_json, log, BASELINES


def main():
    p = base_parser("CausalGate single run")
    p.add_argument("--adia", action="store_true", help="also run the P-C/ADIA detection-only replay on each fold (always on for causal_adia)")
    args = p.parse_args()
    status = import_all(verbose=False)
    protocol = protocol_from_args(args)
    cfg, ds_cfg, bcfg = load_cfgs(args)
    if args.baseline not in BASELINES.keys():
        raise SystemExit(f"baseline {args.baseline!r} not registered; available: {[k for k in BASELINES.keys() if k.startswith('cg_') or k == 'causalgate']}\nmodule status: {status}")
    builder, panel = build_panel(ds_cfg)
    runner = make_runner(cfg, ds_cfg, builder, panel, args)
    alpha = bcfg.get("gate", {}).get("alpha") if isinstance(bcfg.get("gate"), dict) else None
    t0 = time.time()
    res = runner.run(args.baseline, bcfg, protocol, cfg["split"].get("max_folds"), tag=args.tag, alpha=alpha)
    if args.adia or protocol.detection_only:
        from causalgate.eval.protocols import run_adia
        for f, fr in zip(runner.folds(protocol, cfg["split"].get("max_folds")), res["folds"]):
            # the protocol-switched panel the baseline was fitted on (L5 / L6 views), not the raw panel
            fr["adia"] = run_adia(fr["_baseline"], fr.get("_panel", panel), f, cfg["data"]["lookback"], runner.horizons, runner.latency, runner.exec_max)
    a = res["agg"]
    log.info(f"[{args.dataset} | {args.baseline} | {protocol.name} | seed {args.seed}] folds={a['n_folds']} "
             f"SR={a.get('SR', float('nan')):.2f} AUC_PR={a.get('AUC_PR', float('nan')):.3f} F1={a.get('F1_headline', float('nan')):.3f} "
             f"TS_AUC={a.get('TS_AUC', float('nan')):.3f} VUS_PR={a.get('VUS_PR', float('nan')):.3f} alarm={a.get('alarm_rate', float('nan')):.3f} "
             f"trades={a.get('n_trades', float('nan')):.0f} ({time.time() - t0:.0f}s)")
    d = os.path.join(args.out, args.dataset, args.baseline, args.tag or protocol.name, f"seed{args.seed}")
    save_json({"agg": a, "protocol": res["protocol"], "protocol_hash": res["protocol_hash"], "config": cfg, "dataset_config": {k: v for k, v in ds_cfg.items()},
               "baseline_config": {k: v for k, v in bcfg.items() if k != "panel"}, "module_status": status,
               "adia": [fr.get("adia") for fr in res["folds"]]}, os.path.join(d, "summary.json"))


if __name__ == "__main__":
    main()

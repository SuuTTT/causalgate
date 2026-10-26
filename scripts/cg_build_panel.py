#!/usr/bin/env python3
"""cq job: build one dataset panel and write it (pickled, PIT fields included) to $CQ_OUTPUT_DIR under the exact file
name scripts/cg_common.build_panel looks up in <root>/cache/panels, plus a small summary.json.

    python scripts/cg_build_panel.py --dataset stocknet
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cg_common import import_all, load_cfgs, build_panel

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
a = ap.parse_args()
out = os.environ.get("CQ_OUTPUT_DIR", "results/causalgate/panels")
os.makedirs(out, exist_ok=True)
os.environ["CG_PANEL_CACHE_DIR"] = out
import_all()
cfg, ds_cfg, _ = load_cfgs(argparse.Namespace(dataset=a.dataset, baseline="causalgate", smoke=False, override=[], max_folds=None))
t0 = time.time()
print(f"0/1 building {a.dataset}", flush=True)
_, p = build_panel(ds_cfg)
s = {"dataset": a.dataset, "T": p.T, "N": p.N, "n_feat": int(p.price_feat.shape[-1]), "text": None if p.text is None else list(p.text.shape),
     "pos_rate": float(np.nanmean(p.labels)), "pit_fields": p.meta.get("pit_fields"), "layout": p.meta.get("layout"), "sec": time.time() - t0,
     "files": sorted(f for f in os.listdir(out) if f.endswith(".pkl"))}
json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=2, default=str)
print(json.dumps(s, default=str), flush=True)
print("1/1 done", flush=True)

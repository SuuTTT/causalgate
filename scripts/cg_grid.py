#!/usr/bin/env python3
"""Parallel, resumable CausalGate grid driver (one process per run, single-threaded BLAS).

    python3 scripts/cg_grid.py --datasets cg_stocknet,cg_crypto_pd --baselines all --protocols main,one_switch \
        --seeds 42,2024,2025,2026 --workers 48 --shard 0/2 --python /path/to/python

A job is skipped when its summary.json already exists (resume after interruption). Each job writes its own log under
<out>/logs/; failures are listed in <out>/logs/failed_<shard>.txt and never retried silently. Jobs are ordered
cheapest-first per dataset so partial results arrive early. Nothing here changes the protocol definitions: every job
goes through scripts/cg_run.py, which verifies the frozen protocol hashes.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_BASELINES = ["causalgate", "cg_damp", "cg_memstream", "cg_konczal_evt", "cg_stage", "cg_catch", "cg_cpd_dmn",
                 "cg_alridhawi", "cg_anomaly_transformer", "cg_dcdetector", "cg_moment"]
MAIN = ["published", "walkforward", "causal"]
ONE_SWITCH = [f"causal+L{i}" for i in range(1, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--baselines", default="all")
    ap.add_argument("--protocols", default="main", help="comma list of protocol names, or the groups main / one_switch")
    ap.add_argument("--seeds", default="42,2024,2025,2026")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i (for several hosts)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "causalgate"))
    ap.add_argument("--extra", default="", help="extra args passed to cg_run.py (e.g. '--max-folds 3')")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bls = ALL_BASELINES if a.baselines == "all" else a.baselines.split(",")
    prots = []
    for p in a.protocols.split(","):
        prots += MAIN if p == "main" else ONE_SWITCH if p == "one_switch" else [p]
    jobs = [(d, b, p, int(s)) for d in a.datasets.split(",") for s in a.seeds.split(",") for p in prots for b in bls]
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = jobs[i::n]
    logs = os.path.join(a.out, "logs"); os.makedirs(logs, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(os.path.join(a.out, j[0], j[1], j[2], f"seed{j[3]}", "summary.json"))]
    print(f"[cg_grid] shard {a.shard}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run on {a.workers} workers", flush=True)
    if a.dry_run:
        for j in todo[:20]:
            print("  ", j)
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONWARNINGS="ignore")

    def run(j):
        d, b, p, s = j
        tag = f"{d}__{b}__{p.replace('+', '_')}__s{s}"
        cmd = [a.python, "-W", "ignore", os.path.join(ROOT, "scripts", "cg_run.py"), "--dataset", d, "--baseline", b, "--protocol", p,
               "--seed", str(s), "--device", "cpu", "--out", a.out] + a.extra.split()
        t0 = time.time()
        with open(os.path.join(logs, tag + ".log"), "w") as fh:
            rc = subprocess.call(["nice", "-n", "10"] + cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        return j, rc, time.time() - t0

    done = failed = 0
    t_start = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex, open(os.path.join(logs, f"failed_{i}of{n}.txt"), "a") as ff:
        for j, rc, dt in (f.result() for f in cf.as_completed([ex.submit(run, j) for j in todo])):
            done += 1
            if rc != 0:
                failed += 1; ff.write(" ".join(map(str, j)) + f" rc={rc}\n"); ff.flush()
            if done % 10 == 0 or rc != 0:
                print(f"[cg_grid] {done}/{len(todo)} done, {failed} failed, last {j} rc={rc} {dt:.0f}s, elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)
    print(f"[cg_grid] finished: {done} run, {failed} failed", flush=True)


if __name__ == "__main__":
    main()

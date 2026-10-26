#!/usr/bin/env python3
"""Collect results/causalgate/<dataset>/<baseline>/<protocol>/seed*/fold*.json into the paper tables.

    python3 scripts/cg_make_tables.py --results results/causalgate --bps 5 --fmt markdown

Tables (written to <results>/tables/ as csv + md; placeholders are NOT invented -- missing cells stay empty):
    E1  leakage premium: per (dataset, baseline) mean and sd over (seed, fold) under P-O, P-W, P-C, and LP = P-O - P-C
        (sd of LP over paired (seed, fold) differences); Kendall tau P-O vs P-C. All at the headline cost --bps.
    E2  channel decomposition: causal+Lk minus causal (L1..L7; L8 is an audit-card flag), at --bps
    E3  causal ranking at EVERY cost level found in the results (long format, `bps` column)
    E6  refit policy (ablation tags refit_*), E7 alarm-rate control (gate_* tags), Abl. the remaining ablations
    DG  prequential-replay check: share of causal runs whose fresh, refitted model replayed bar by bar on the panel
        truncated at each bar reproduces the batch decisions (DG = 0), plus mean DG_SR
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional
from causalgate.eval.metrics import kendall_tau_rankings
from causalgate.eval.protocols import ONE_SWITCH

CHANNELS = tuple(ONE_SWITCH)          # L1..L7 (L8 is an audit-card flag, never a run)

KEYS = ("SR", "ARR", "MDD", "AUC_PR", "AUC_ROC", "F1_headline", "PA_f1", "f1_@thr", "EAR", "FDP", "TS_AUC", "VUS_PR", "event_f1", "alarm_rate", "n_trades", "wlt", "Turnover")


def load(results: str, bps: Optional[float] = None) -> pd.DataFrame:
    """One row per (run, fold, cost level); `bps` restricts to one level (None = every level in the json)."""
    rows = []
    for path in glob.glob(os.path.join(results, "*", "*", "*", "seed*", "fold*.json")):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        rp = r.get("replay") or {}
        for key, ev in r.get("eval", {}).items():
            try:
                b = float(key)
            except ValueError:
                continue
            if bps is not None and abs(b - bps) > 1e-9:
                continue
            row = {"dataset": r["dataset"], "baseline": r["baseline"], "protocol": r["protocol"], "tag": os.path.basename(os.path.dirname(os.path.dirname(path))),
                   "seed": r["seed"], "fold": r["fold"], "bps": b, "mode": r.get("mode"), "status": r.get("status"),
                   "dg_zero": r.get("deployability_gap_zero") if rp.get("kind") == "prequential_refit_bar_by_bar" else None,
                   "DG_SR": rp.get("DG_SR", np.nan), "hash": r.get("protocol_hash")}
            row.update({k: ev.get(k, np.nan) for k in KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def mean_sd(df: pd.DataFrame, by, key) -> pd.DataFrame:
    g = df.groupby(by)[key]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.count()}).reset_index()


def fmt_cell(m, s):
    return "" if not np.isfinite(m) else (f"{m:.2f} ± {s:.2f}" if np.isfinite(s) else f"{m:.2f}")


def table_e1(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """mean and sd over (seed, fold) per protocol; LP / LP_W from paired (seed, fold) differences."""
    d = df[df["protocol"].isin(["published", "walkforward", "causal"])]
    idx = ["dataset", "baseline"]
    piv = d.pivot_table(index=idx + ["seed", "fold"], columns="protocol", values=key, aggfunc="mean")
    for p in ("published", "walkforward", "causal"):
        if p not in piv:
            piv[p] = np.nan
    piv["LP"] = piv["published"] - piv["causal"]
    piv["LP_W"] = piv["walkforward"] - piv["causal"]
    g = piv.groupby(level=idx)
    out = pd.concat({"mean": g.mean(), "sd": g.std(), "n": g.count()}, axis=1)
    cols = []
    for c in ("published", "walkforward", "causal", "LP", "LP_W"):
        out[c] = out[("mean", c)]; out[f"{c}_sd"] = out[("sd", c)]; out[f"{c}_n"] = out[("n", c)]
        cols += [c, f"{c}_sd", f"{c}_n"]
    out = out[cols]; out.columns = cols
    return out.reset_index()


def table_e2(df: pd.DataFrame, key: str) -> pd.DataFrame:
    base = df[df["protocol"] == "causal"].groupby(["dataset", "baseline"])[key].mean()
    rows = []
    for ch in CHANNELS:
        sw = df[df["protocol"] == f"causal+{ch}"].groupby(["dataset", "baseline"])[key].mean()
        lp = (sw - base).rename(ch)
        rows.append(lp)
    out = pd.concat(rows, axis=1) if rows else pd.DataFrame()
    return out.reset_index() if len(out) else out


def rank_tau(df: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for ds, d in df.groupby("dataset"):
        a = d[d["protocol"] == "published"].groupby("baseline")[key].mean().to_dict()
        b = d[d["protocol"] == "causal"].groupby("baseline")[key].mean().to_dict()
        kt = kendall_tau_rankings(a, b)
        rows.append({"dataset": ds, **kt})
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "_no runs yet_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if not np.isfinite(v) else floatfmt.format(v))
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "causalgate"))
    ap.add_argument("--bps", type=float, default=5.0)
    ap.add_argument("--fmt", default="markdown")
    args = ap.parse_args()
    df_all = load(args.results, None)
    df = df_all[np.isclose(df_all["bps"], args.bps)] if len(df_all) else df_all
    out_dir = os.path.join(args.results, "tables")
    os.makedirs(out_dir, exist_ok=True)
    if len(df) == 0:
        print("no results under", args.results)
        open(os.path.join(out_dir, "README.md"), "w").write("No runs yet.\n")
        return
    df.to_csv(os.path.join(out_dir, "all_folds.csv"), index=False)
    main_runs = df[df["tag"] == df["protocol"]]                       # exclude ablation tags from the main tables
    md = [f"# CausalGate tables (headline cost {args.bps:g} bps; E1: mean, sd, n over seeds x folds; E3: every cost level)", ""]
    for key in ("SR", "AUC_PR", "F1_headline", "EAR"):
        e1 = table_e1(main_runs, key); e1.to_csv(os.path.join(out_dir, f"E1_{key}.csv"), index=False)
        md += [f"## E1 leakage premium -- {key}", "", to_md(e1)]
    tau = rank_tau(main_runs, "SR"); tau.to_csv(os.path.join(out_dir, "E1_kendall_tau_SR.csv"), index=False)
    md += ["## E1 rank agreement P-O vs P-C (Kendall tau on SR)", "", to_md(tau)]
    for key in ("SR", "AUC_PR"):
        e2 = table_e2(main_runs, key)
        if len(e2):
            e2.to_csv(os.path.join(out_dir, f"E2_{key}.csv"), index=False)
        md += [f"## E2 channel decomposition (causal+Lk minus causal) -- {key}", "", to_md(e2)]
    causal = main_runs[main_runs["protocol"] == "causal"]
    all_main = df_all[df_all["tag"] == df_all["protocol"]]
    c_all = all_main[all_main["protocol"] == "causal"]
    e3 = c_all.groupby(["bps", "dataset", "baseline"])[["SR", "AUC_PR", "EAR", "F1_headline", "TS_AUC", "VUS_PR", "alarm_rate", "n_trades"]].mean().reset_index()
    e3.to_csv(os.path.join(out_dir, "E3_causal.csv"), index=False)
    md += ["## E3 honest ranking under P-C (every cost level)", "", to_md(e3)]
    abl = df[df["tag"].str.startswith("abl_")]
    if len(abl):
        a = abl.groupby(["dataset", "tag"])[["SR", "AUC_PR", "alarm_rate", "n_trades"]].mean().reset_index()
        a.to_csv(os.path.join(out_dir, "ablations.csv"), index=False)
        md += ["## Ablations (E6 refit_*, E7 gate_*, others)", "", to_md(a)]
    dg = causal.groupby(["dataset", "baseline"]).agg(
        share_DG_zero=("dg_zero", lambda s: float(np.mean([bool(v) for v in s if v is not None])) if s.notna().any() else np.nan),
        DG_SR=("DG_SR", "mean")).reset_index()
    md += ["## Deployability gap: share of causal runs whose prequential replay (fresh refitted model, panel truncated at "
           "each bar) reproduces the batch decisions; runs without that replay are left empty", "", to_md(dg)]
    adia = []
    for path in glob.glob(os.path.join(args.results, "*", "*", "*", "seed*", "summary.json")):
        r = json.load(open(path))
        vals = [x["TS_AUC_adia"] for x in r.get("adia", []) if x]
        if vals:
            adia.append({"path": os.path.relpath(path, args.results), "TS_AUC_adia": float(np.mean(vals))})
    if adia:
        md += ["## P-C/ADIA detection-only replay (TS-AUC)", "", to_md(pd.DataFrame(adia))]
    text = "\n".join(md)
    open(os.path.join(out_dir, "tables.md"), "w").write(text)
    print(text if args.fmt == "markdown" else df.to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper figures from results/causalgate/ (matplotlib; nothing is drawn from numbers that do not exist on disk).

    python3 scripts/cg_figures.py --results results/causalgate --out results/causalgate/figures

Fig 1 motivation       same detector, offline (published) vs causal equity curve        <- trades_fold*.csv of both protocols
Fig 2 LP heatmap       datasets x baselines, LP_SR                                         <- tables/E1_SR.csv
Fig 3 channel bars     one-switch LP per channel, per dataset family                       <- tables/E2_SR.csv
Fig 4 bump chart       method rank P-O -> P-C per dataset                                  <- tables/E1_SR.csv
Fig 5 FPIT power       flag rate vs probe budget per planted leak                          <- planted/fpit_power.csv
Fig 6 alarm tracking   realised alarm rate vs alpha across regimes (E7)                    <- e7/gate_sweep.csv
Fig 7 date-shift       Sharpe vs shift k for honest vs leaky pipelines                     <- cards/*.json
Fig 8 audit card       rendered markdown of one card (text figure)                         <- cards/*.md
Figures whose inputs are missing are skipped with a message.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _skip(name, why):
    print(f"[skip] {name}: {why}")


def fig_lp_heatmap(tables, out):
    p = os.path.join(tables, "E1_SR.csv")
    if not os.path.exists(p):
        return _skip("fig2 LP heatmap", "tables/E1_SR.csv missing")
    df = pd.read_csv(p)
    if "LP" not in df or df["LP"].notna().sum() == 0:
        return _skip("fig2 LP heatmap", "no LP values yet")
    piv = df.pivot(index="dataset", columns="baseline", values="LP")
    fig, ax = plt.subplots(figsize=(1.2 * piv.shape[1] + 2, 0.6 * piv.shape[0] + 2))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto", vmin=-np.nanmax(np.abs(piv.to_numpy(dtype=float))), vmax=np.nanmax(np.abs(piv.to_numpy(dtype=float))))
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="leakage premium (Sharpe, P-O minus P-C)")
    ax.set_title("E1: leakage premium")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_lp_heatmap.pdf")); plt.close(fig)


def fig_channel_bars(tables, out):
    p = os.path.join(tables, "E2_SR.csv")
    if not os.path.exists(p):
        return _skip("fig3 channel bars", "tables/E2_SR.csv missing")
    df = pd.read_csv(p)
    chans = [c for c in df.columns if c.startswith("L")]
    if not chans or df[chans].notna().sum().sum() == 0:
        return _skip("fig3 channel bars", "no one-switch runs yet")
    m = df.groupby("dataset")[chans].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    m.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("LP_SR^(l) (causal+L_l minus causal)"); ax.set_title("E2: channel decomposition")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_channel_bars.pdf")); plt.close(fig)


def fig_bump(tables, out):
    p = os.path.join(tables, "E1_SR.csv")
    if not os.path.exists(p):
        return _skip("fig4 bump chart", "tables/E1_SR.csv missing")
    df = pd.read_csv(p)
    if df["published"].notna().sum() == 0 or df["causal"].notna().sum() == 0:
        return _skip("fig4 bump chart", "need both P-O and P-C runs")
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = df[df["dataset"] == ds].dropna(subset=["published", "causal"])
        r0 = d["published"].rank(ascending=False); r1 = d["causal"].rank(ascending=False)
        for b, a, c in zip(d["baseline"], r0, r1):
            ax.plot([0, 1], [a, c], marker="o"); ax.text(1.02, c, b, fontsize=7, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["P-O", "P-C"]); ax.invert_yaxis(); ax.set_title(ds); ax.set_ylabel("rank (SR)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_rank_bump.pdf")); plt.close(fig)


def fig_fpit_power(results, out):
    p = os.path.join(results, "planted", "fpit_power.csv")
    if not os.path.exists(p):
        return _skip("fig5 FPIT power", "planted/fpit_power.csv missing (run scripts/cg_planted_leaks.py)")
    df = pd.read_csv(p); df = df[df["n_probes"] > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for (leak, strat), d in df.groupby(["leak", "strategy"]):
        g = d.groupby("n_probes")["flagged"].mean()
        ax.plot(g.index, g.values, marker="o", ls="-" if strat == "stratified" else "--", label=f"{leak} ({strat})")
    ax.set_xlabel("probe budget n"); ax.set_ylabel("flag rate (power; FPR for 'none')"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=6)
    ax.set_title("E4: FPIT power on planted leaks")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_fpit_power.pdf")); plt.close(fig)


def fig_alarm_tracking(results, out):
    p = os.path.join(results, "e7", "gate_paths.csv")
    if not os.path.exists(p):
        return _skip("fig6 alarm tracking", "e7/gate_paths.csv missing (run scripts/cg_e7_gate_sweep.py)")
    df = pd.read_csv(p)
    if "stream" not in df:
        df["stream"] = "stream"
    streams = list(df.groupby("stream"))
    fig, axes = plt.subplots(len(streams), 1, figsize=(7, 3.2 * len(streams)), squeeze=False)
    alpha = float(df["alpha"].iloc[0])
    for ax, (sname, ds) in zip(axes[:, 0], streams):
        for g, d in ds.groupby("gate"):                      # one line per (stream, gate): streams are never joined
            d = d.sort_values("t")
            ax.plot(d["t"], d["rolling_rate"], label=g)
        ax.axhline(alpha, color="k", ls=":", label="alpha")
        if "switch_t" in ds:                                # this stream's own switches (';'-joined lists accepted)
            sw = set()
            for v in ds["switch_t"].dropna().unique():
                for x in str(v).split(";"):
                    try:
                        sw.add(float(x))
                    except ValueError:
                        pass
            for x in sorted(sw):
                ax.axvline(x, color="grey", lw=0.5)
        ax.set_xlabel("bar"); ax.set_ylabel("rolling alarm rate (250 bars)"); ax.legend(fontsize=7); ax.set_title(f"E7: alarm-rate control -- {sname}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_alarm_tracking.pdf")); plt.close(fig)


def fig_date_shift(results, out):
    cards = glob.glob(os.path.join(results, "cards", "*.json"))
    if not cards:
        return _skip("fig7 date shift", "no audit cards yet")
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in cards:
        r = json.load(open(c))
        prof = r.get("date_shift", {}).get("profile", {})
        if prof:
            ks = sorted(int(k) for k in prof); ax.plot(ks, [prof[str(k)] if str(k) in prof else prof[k] for k in ks], marker="o", label=f"{r['method']}@{r['dataset']}")
    ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("signal shift k (bars; <0 peeks, >0 delays)"); ax.set_ylabel("Sharpe"); ax.legend(fontsize=6)
    ax.set_title("Date-shift probe profiles")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_date_shift.pdf")); plt.close(fig)


def fig_motivation(results, out):
    runs = glob.glob(os.path.join(results, "*", "*", "published", "seed42", "trades_fold0.csv"))
    for pub in runs:
        cau = pub.replace("/published/", "/causal/")
        if not os.path.exists(cau):
            continue
        a, b = pd.read_csv(pub), pd.read_csv(cau)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for df, lab in ((a, "P-O (as published)"), (b, "P-C (causal)")):
            d = df[df["w"] != 0].sort_values("exit_t")
            ax.plot(d["exit_t"], d["pnl"].cumsum(), label=lab)
        ax.set_xlabel("bar"); ax.set_ylabel("cumulative net PnL (log units)"); ax.legend()
        parts = pub.split(os.sep); ax.set_title(f"Fig 1: {parts[-5]} / {parts[-4]}")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig1_motivation_{parts[-5]}_{parts[-4]}.pdf")); plt.close(fig)
        return
    _skip("fig1 motivation", "need a published and a causal run of the same (dataset, baseline) with seed 42")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "causalgate"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    tables = os.path.join(args.results, "tables")
    fig_motivation(args.results, out); fig_lp_heatmap(tables, out); fig_channel_bars(tables, out); fig_bump(tables, out)
    fig_fpit_power(args.results, out); fig_alarm_tracking(args.results, out); fig_date_shift(args.results, out)
    cards = glob.glob(os.path.join(args.results, "cards", "*.md"))
    if cards:
        import shutil; shutil.copy(cards[0], os.path.join(out, "fig8_audit_card_example.md"))
    print("figures in", out, ":", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()

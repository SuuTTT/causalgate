"""Event-level detection metrics pooled across folds, with case-level bootstrap confidence intervals.

Unit of analysis = a regulatory case (event id), not a bar: bars inside one case are strongly dependent, and the number
of cases (not bars) drives statistical power. Scores are rank-normalised within each (fold, seed) because different
folds are scored by different models.
    event score      max normalised score over the case's pre-peak warning rows (bars_to_peak >= 0)
    event detected   any pre-peak row with raw score >= that fold's validation threshold
    lead             bars_to_peak at the first fired pre-peak row (larger = earlier)
    negatives        rows with no event (y == 0), grouped into asset x `block` bar blocks for the bootstrap
    event_AUROC      P(event score > negative row score)       event_AP   AP of events (one positive each) vs negative rows
    FA_per_1k        fired negative rows per 1000 negative rows
Bootstrap: resample cases and negative blocks with replacement (same resample for every seed and, in paired mode, for
both methods); report percentile CIs. Seeds are averaged inside each resample.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _prep(trades: pd.DataFrame, block: int) -> pd.DataFrame:
    d = trades.copy()
    d["pct"] = d.groupby(["fold", "seed"])["score"].rank(pct=True)
    d["fired"] = d["score"] >= d["thr_val"]
    d["nblock"] = d["asset"].astype(np.int64) * 1_000_000 + (d["t"] // max(block, 1)).astype(np.int64)
    return d


def _per_seed_tables(d: pd.DataFrame):
    out = {}
    pre = d[(d["y"] == 1) & (d["event_id"] >= 0) & (d["bars_to_peak"] >= 0)]
    neg = d[(d["y"] == 0)]
    for seed, g in pre.groupby("seed"):
        g = g.sort_values("t")
        ev = g.groupby("event_id").agg(s=("pct", "max"), det=("fired", "any"))
        first = g[g["fired"]].groupby("event_id")["bars_to_peak"].first()
        ev["lead"] = first.reindex(ev.index)
        ng = neg[neg["seed"] == seed]
        out[seed] = (ev, ng[["pct", "fired", "nblock"]])
    return out


def _metrics(ev: pd.DataFrame, neg: pd.DataFrame) -> Dict[str, float]:
    if len(ev) == 0 or len(neg) == 0:
        return {"event_AUROC": np.nan, "event_AP": np.nan, "event_recall": np.nan, "FA_per_1k": np.nan, "lead_median": np.nan}
    y = np.r_[np.ones(len(ev)), np.zeros(len(neg))]
    s = np.r_[ev["s"].to_numpy(), neg["pct"].to_numpy()]
    return {"event_AUROC": float(roc_auc_score(y, s)), "event_AP": float(average_precision_score(y, s)),
            "event_recall": float(ev["det"].mean()), "FA_per_1k": float(1000 * neg["fired"].mean()),
            "lead_median": float(np.nanmedian(ev["lead"])) if ev["lead"].notna().any() else np.nan}


def _resample(tables, ev_ids, blocks, rng):
    e_s = rng.choice(ev_ids, size=len(ev_ids), replace=True) if len(ev_ids) else ev_ids
    b_s = rng.choice(blocks, size=len(blocks), replace=True) if len(blocks) else blocks
    res = []
    for seed, (ev, neg) in tables.items():
        ev_b = ev.reindex(e_s).dropna(subset=["s"])
        cnt = pd.Series(b_s).value_counts()
        neg_b = neg.merge(cnt.rename("w"), left_on="nblock", right_index=True)
        neg_b = neg_b.loc[neg_b.index.repeat(neg_b["w"])]
        res.append(_metrics(ev_b, neg_b))
    return pd.DataFrame(res).mean().to_dict()


def event_level_summary(trades: pd.DataFrame, n_boot: int = 500, block: int = 20, seed: int = 0, ci: float = 0.95) -> Dict[str, float]:
    d = _prep(trades, block)
    tables = _per_seed_tables(d)
    if not tables:
        return {"n_events": 0}
    point = pd.DataFrame([_metrics(*v) for v in tables.values()]).mean().to_dict()
    ev_ids = np.unique(np.concatenate([v[0].index.to_numpy() for v in tables.values()]))
    blocks = np.unique(d.loc[d["y"] == 0, "nblock"].to_numpy())
    rng = np.random.default_rng(seed)
    boots = pd.DataFrame([_resample(tables, ev_ids, blocks, rng) for _ in range(n_boot)]) if n_boot > 0 else pd.DataFrame()
    a = (1 - ci) / 2
    out = {"n_events": int(len(ev_ids)), "n_neg_rows": int((d["y"] == 0).sum() / max(d["seed"].nunique(), 1)), "n_seeds": int(d["seed"].nunique())}
    for k, v in point.items():
        out[k] = v
        if len(boots) and k in boots:
            out[f"{k}_lo"], out[f"{k}_hi"] = float(boots[k].quantile(a)), float(boots[k].quantile(1 - a))
    return out


def paired_event_delta(trades_a: pd.DataFrame, trades_b: pd.DataFrame, metric: str = "event_AUROC", n_boot: int = 1000,
                       block: int = 20, seed: int = 0, ci: float = 0.95) -> Dict[str, float]:
    """A - B on the same cases and negative blocks (paired case bootstrap). Requires identical folds / seeds / labels."""
    da, db = _prep(trades_a, block), _prep(trades_b, block)
    ta, tb = _per_seed_tables(da), _per_seed_tables(db)
    common = sorted(set(ta) & set(tb))
    if not common:
        return {"delta": float("nan")}
    ta, tb = {s: ta[s] for s in common}, {s: tb[s] for s in common}
    ev_ids = np.unique(np.concatenate([ta[s][0].index.to_numpy() for s in common]))
    blocks = np.unique(da.loc[da["y"] == 0, "nblock"].to_numpy())
    pa = pd.DataFrame([_metrics(*v) for v in ta.values()]).mean()[metric]
    pb = pd.DataFrame([_metrics(*v) for v in tb.values()]).mean()[metric]
    deltas = []
    for i in range(n_boot):
        rng_a, rng_b = np.random.default_rng(seed + i), np.random.default_rng(seed + i)
        deltas.append(_resample(ta, ev_ids, blocks, rng_a)[metric] - _resample(tb, ev_ids, blocks, rng_b)[metric])
    deltas = np.asarray(deltas)
    a = (1 - ci) / 2
    return {"metric": metric, "A": float(pa), "B": float(pb), "delta": float(pa - pb), "lo": float(np.nanquantile(deltas, a)),
            "hi": float(np.nanquantile(deltas, 1 - a)), "p_one_sided": float((1 + (deltas <= 0).sum()) / (len(deltas) + 1)),
            "n_events": int(len(ev_ids))}


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return float(np.sqrt(max(0.0, (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / max(n_pos * n_neg, 1))))


def minimum_detectable_auroc_delta(auc: float, n_events: int, n_neg_eff: int, corr: float = 0.5, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    """Paired-comparison MDE for AUROC at 80% power, two-sided 5%: (z_a + z_b) * SE * sqrt(2 (1 - corr))."""
    se = hanley_mcneil_se(auc, n_events, n_neg_eff)
    return float((z_alpha + z_power) * se * np.sqrt(2 * (1 - corr)))

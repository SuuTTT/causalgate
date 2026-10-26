"""Rolling (walk-forward) experiment driver: folds x protocols x label budgets x seeds -> tidy metric rows.

Per fold (train window [lo, as_of], validation, test):
  1. supervision: K nested cases per class available under the protocol at as_of (budget seed fixed across methods),
     confident normals, block-level calibration set C0 (semi_dataset.make_labeled_arrays / SemiEventDataset)
  2. fit: the semi-supervised model (trainer-flag baselines are the same network with switches) or a score-only baseline
  3. scores on validation and test; threshold chosen on VALIDATION (best F1 if it holds >= min_val_events cases, else
     the fire quantile of validation scores); the test fold never selects anything
  4. detection metrics against EVALUATION labels = the oracle anchor raster (eval.labels: anchors | rule | union)
  5. trading, identical for every detector: 'adapter' (shared score-to-trade rule) and, for models with a learned policy,
     'policy'; each priced over the cost grid and run through the netted portfolio backtest + long-only de-risk overlay
     (benchmark-adjusted) with timing-permutation nulls at the default cost
Rows carry K requested, K_eff (cases actually used), n_available, capped classes, thr source and event counts.
"""
from __future__ import annotations
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from ..data.schema import Panel
from ..data.splits import WalkForwardSplitter, Fold
from ..data.labels import realized_vol
from ..data.semi_dataset import LabelBudget, SemiEventDataset, make_labeled_arrays, eval_label_arrays
from ..data.weak_labelers import LabelingFunction, build_vote_matrix
from ..utils.registry import MODELS
from ..utils.seed import set_seed
from ..semi.trainer import SemiSupervisedTrainer
from .costs import CostModel
from .engine import EventBacktester
from .portfolio_eval import evaluate_portfolio, deflated_sharpe_from_moments
from . import metrics as M

KEY_METRICS = ("AP", "AUROC", "R@K", "F1_val", "event_recall", "WAD_mean", "net_SR", "net_SR_perm_p", "ovl_IR", "ovl_IR_perm_p",
               "ovl_active_ARR", "net_DSR", "K_eff", "n_test_events", "PLCA_pos", "gate_FDR_emp")


def _k_label(k: Optional[int]) -> str:
    return "Full" if k is None else str(int(k))


class RollingExperiment:
    def __init__(self, panel: Panel, anchor_table: pd.DataFrame, cfg: Dict[str, Any], lfs: Sequence[LabelingFunction],
                 logger: Optional[logging.Logger] = None, device: str = "cpu"):
        self.p, self.table, self.cfg, self.lfs, self.device = panel, anchor_table, cfg, list(lfs), device
        self.log = logger or logging.getLogger("warn2trade.rolling")
        self.votes = build_vote_matrix(panel, self.lfs)
        self.lf_lookahead = np.array([lf.looks_forward for lf in self.lfs], dtype=int)
        sp = dict(cfg["splits"])
        self.d = dict(cfg["data"])
        self.lb = dict(cfg.get("label_budget", {}))
        self.ev = dict(cfg.get("eval", {}))
        self.horizons = list(cfg["model"].get("horizons") or panel.meta.get("horizons"))
        self.hmax = max(self.horizons) + int(self.d.get("latency", 1)) + int(self.d.get("exec_max", 4))
        need = max([int(x) for x in self.lf_lookahead] + [self.hmax])
        if int(sp.get("purge", 0)) < need:
            self.log.warning("splits.purge=%s < max(LF look-ahead, horizon+latency+exec)=%s: raising purge", sp.get("purge", 0), need)
            sp["purge"] = need
        self.splitter = WalkForwardSplitter(panel.T, sp["train"], sp["val"], sp["test"], sp.get("step"), sp.get("purge", 0),
                                            sp.get("embargo", 0), sp.get("anchored", False), sp.get("max_folds"))
        bt = dict(cfg.get("backtest", {}))
        self.default_bps = float(bt.get("commission_bps", 5.0))
        self.cost = CostModel(self.default_bps, bt.get("use_half_spread", True), bt.get("impact_eta", 0.1), bt.get("delay_zeta", 0.05), bt.get("notional", 1.0))
        self.bps_grid = sorted(set([float(b) for b in bt.get("bps_grid", [1, 5, 10, 15, 30])] + [self.default_bps]))
        self.bars_per_year = int(panel.meta.get("bars_per_year", 252))
        self.eval_arrays = eval_label_arrays(panel, anchor_table, self.ev.get("labels", "anchors"), int(self.lb.get("lead", 3)), self.horizons)
        from ..data.semi_dataset import panel_cache
        self.sigma = panel_cache(panel, self.horizons, int(self.d.get("vol_window", 60)))["sigma"]
        self.no_labels = np.full(panel.mid.shape, -1, dtype=np.int8)

    # ------------------------------------------------------------------ orchestration
    def run(self, seeds: Sequence[int], k_values: Sequence[Optional[int]], protocols: Sequence[str] = ("release",)) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows, scores = [], []
        n_total, n_done = len(list(self.splitter)) * len(protocols) * len(k_values) * len(seeds), 0
        prog = os.environ.get("CQ_OUTPUT_DIR")
        for fold in self.splitter:
            for protocol in protocols:
                for k in k_values:
                    for seed in seeds:
                        t0 = time.time()
                        r, sc = self.run_fold(fold, int(seed), k, protocol)
                        rows.extend(r)
                        scores.append(sc)
                        self.log.info("fold %d | %s | K=%s | seed %d | %.1fs", fold.k, protocol, _k_label(k), seed, time.time() - t0)
                        n_done += 1
                        print(f"progress {n_done}/{n_total}", flush=True)
                        if prog:                                       # cq stall detection: n/N line in the progress file
                            with open(os.path.join(prog, "progress.txt"), "a") as fh:
                                fh.write(f"{n_done}/{n_total}\n")
        rows = pd.DataFrame(rows)
        return self.add_dsr(rows), (pd.concat(scores, ignore_index=True) if scores else pd.DataFrame())

    def _ds(self, idx: np.ndarray, y_l: np.ndarray, rng, train: bool) -> SemiEventDataset:
        return SemiEventDataset(self.p, idx, int(self.d["lookback"]), self.horizons, self.votes, y_l, int(self.d.get("latency", 1)),
                                int(self.d.get("exec_max", 4)), int(self.d.get("vol_window", 60)),
                                self.d.get("unlabeled_subsample") if train else None, rng, self.lf_lookahead,
                                float(self.lb.get("n_cal", 0.1)) if train else 0, int(self.lb.get("cal_block", 20)), self.eval_arrays)

    def run_fold(self, fold: Fold, seed: int, k: Optional[int], protocol: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        set_seed(seed)
        budget = LabelBudget(k, int(self.lb.get("n_neg_per_pos", 20)), int(self.lb.get("min_neg", 50)), int(self.lb.get("neg_margin", 20)),
                             int(self.lb.get("lead", 3)), int(self.lb.get("budget_seed", 7)), protocol)
        lab = make_labeled_arrays(self.p, self.table, int(fold.train[-1]), budget, self.votes, int(fold.train[0]), neg_seed=seed)
        rng = np.random.default_rng(seed)
        ds_tr, ds_va, ds_te = self._ds(fold.train, lab["y_l"], rng, True), self._ds(fold.val, self.no_labels, rng, False), self._ds(fold.test, self.no_labels, rng, False)
        latency = int(self.d.get("latency", 1))
        baseline = self.cfg.get("baseline")
        method = baseline or self.cfg["model"].get("name", "warn2trade")
        fit = self._fit_baseline(baseline, ds_tr) if baseline else self._fit_model(ds_tr, seed)
        s_tr, s_va, s_te = fit["score"](ds_tr), fit["score"](ds_va), fit["score"](ds_te)
        n_bad = int(sum((~np.isfinite(x)).sum() for x in (s_tr, s_va, s_te)))
        if n_bad:
            self.log.warning("%d non-finite detector scores replaced by the minimum finite score (reported as n_nonfinite_scores)", n_bad)
            fin = np.concatenate([x[np.isfinite(x)] for x in (s_tr, s_va, s_te)])
            lo = float(fin.min()) if len(fin) else 0.0
            s_tr, s_va, s_te = (np.where(np.isfinite(x), x, lo) for x in (s_tr, s_va, s_te))
        y_va, eid_va = ds_va.flat_eval("y"), ds_va.flat_eval("event_id")
        y_te, eid_te, btp_te = ds_te.flat_eval("y"), ds_te.flat_eval("event_id"), ds_te.flat_eval("bars_to_peak")
        thr, thr_src, n_val_ev = self._threshold(s_va, y_va, eid_va)
        meta = {"method": method, "fold": fold.k, "seed": seed, "K": _k_label(k), "protocol": protocol, "K_eff": lab["budget"]["k_eff_cases"],
                "n_available": lab["budget"]["n_available_cases"], "capped": lab["budget"]["capped"], "thr_source": thr_src,
                "test_t0": int(fold.test[0]), "test_t1": int(fold.test[-1])}
        score_df = pd.DataFrame({"t": ds_te.index[:, 0], "asset": ds_te.index[:, 1], "score": s_te, "thr_val": thr, "y": y_te,
                                 "event_id": eid_te, "bars_to_peak": btp_te, **{kk: vv for kk, vv in meta.items() if kk not in ("capped",)}})
        det = self._detection(score_df, latency)
        det.update({"K_eff": float(lab["budget"]["k_eff_cases"]), "n_available": float(lab["budget"]["n_available_cases"]),
                    "n_val_events": float(n_val_ev), "n_test_events": float(len(np.unique(eid_te[(y_te == 1) & (eid_te >= 0)]))),
                    "thr_from_val": float(thr_src == "val_f1"), "n_nonfinite_scores": float(n_bad), **fit.get("extra", {})})
        rows = [{**meta, "rule": "detection", "bps": self.default_bps, "metric": m, "value": v} for m, v in det.items()]
        rules = ["adapter"] + (["policy"] if fit.get("policy") else [])
        for rule in rules:
            for bps in self.bps_grid:
                cm = self.cost.with_bps(bps)
                trades = fit["policy"](ds_te, cm) if rule == "policy" else self._adapter_trades(ds_te, s_te, s_tr, thr, cm, latency)
                trades["score"] = s_te
                m = self._trading(trades, fold, thr, cm, latency, seed, with_null=(bps == self.default_bps), overlay=(rule == "adapter"))
                rows.extend({**meta, "rule": rule, "bps": bps, "metric": name, "value": val} for name, val in m.items())
        return rows, score_df

    # ------------------------------------------------------------------ fitting
    def _fit_model(self, ds_tr: SemiEventDataset, seed: int) -> Dict[str, Any]:
        model_cfg = dict(self.cfg["model"])
        model_cfg["horizons"] = self.horizons
        model = MODELS.get(model_cfg.get("name", "warn2trade"))(self.p.dims(), model_cfg)
        trainer = SemiSupervisedTrainer(model, {**self.cfg, "seed": seed}, self.cost, self.device, self.log)
        trainer.fit(ds_tr)
        rep = trainer.pseudo_report
        extra = {k: float(rep[k]) for k in ("PLCA_pos", "PLCA_neg", "gate_FDR_emp", "gate_recall", "n_bh_selected", "gate_k_min", "n_cal_trimmed", "validity_slack") if k in rep and rep[k] is not None}
        extra["lm_flipped"] = float(bool(rep.get("lm_flipped", False)))
        out = {"score": lambda ds: trainer.predict_scores(ds), "extra": extra}
        if bool(self.ev.get("policy", True)):
            out["policy"] = lambda ds, cm: trainer.collect_trades(ds, EventBacktester(self.horizons, int(self.d.get("latency", 1)), cm))
        return out

    def _fit_baseline(self, name: str, ds_tr: SemiEventDataset) -> Dict[str, Any]:
        from ..baselines import SEMI_BASELINES
        base = SEMI_BASELINES.get(name)(dict(self.cfg.get("baseline_cfg", {})))
        base.fit(ds_tr)
        return {"score": base.score, "extra": {}}

    def _adapter_trades(self, ds, s_te, s_tr, thr, cm, latency) -> pd.DataFrame:
        from ..baselines.base import ScoreToTradeAdapter
        hold = int(self.ev.get("hold_h", self.horizons[min(3, len(self.horizons) - 1)]))
        ad = ScoreToTradeAdapter(self.horizons, latency, cm, thr, float(self.ev.get("w", 0.5)), hold, self.ev.get("direction", "fade"), s_tr)
        return ad.collect(ds, s_te, int(self.d.get("batch_size", 256)))

    # ------------------------------------------------------------------ evaluation
    def _threshold(self, s_va: np.ndarray, y_va: np.ndarray, eid_va: np.ndarray):
        n_ev = int(len(np.unique(eid_va[(y_va == 1) & (eid_va >= 0)])))
        if n_ev >= int(self.ev.get("min_val_events", 3)) and 0 < y_va.sum() < len(y_va):
            _, thr = M.best_f1(y_va, s_va)
            return float(thr), "val_f1", n_ev
        return float(np.quantile(s_va, float(self.ev.get("fire_quantile", 0.99)))), "val_quantile", n_ev

    def _detection(self, sc: pd.DataFrame, latency: int) -> Dict[str, float]:
        y, s, thr = sc["y"].to_numpy(), sc["score"].to_numpy(), float(sc["thr_val"].iloc[0])
        k = max(1, int(float(self.ev.get("k_frac", 0.01)) * len(y)))
        out = {"AP": M.auc_pr(y, s), "AUROC": M.auc_roc(y, s), "P@K": M.precision_at_k(y, s, k), "R@K": M.recall_at_k(y, s, k),
               "pos_rate": float(y.mean())}
        f = M.f1_at(y, s, thr)
        out.update({"F1_val": f["f1"], "precision_val": f["precision"], "recall_val": f["recall"], "fire_rate": float((s >= thr).mean())})
        wad = M.warning_to_alpha_delay(sc.assign(w=0.0), thr, latency)
        out.update({"WAD_mean": wad["WAD_mean"], "WAD_ahead_rate": wad["WAD_ahead_rate"], "event_recall": wad["event_recall"]})
        return out

    def _trading(self, trades: pd.DataFrame, fold: Fold, thr: float, cm: CostModel, latency: int, seed: int, with_null: bool, overlay: bool) -> Dict[str, float]:
        traded = trades["w"].to_numpy() != 0
        y, pnl = trades["y"].to_numpy(), trades["pnl"].to_numpy()
        out = {"trade_rate": float(traded.mean()), "FDP": M.false_discovery_penalty(pnl, traded, y), "EAR": M.executable_alpha_ratio(pnl, traded, y),
               "ACR": M.alpha_capture_ratio(trades), "frozen_rate": float(trades["frozen"].to_numpy()[traded].mean()) if traded.any() else float("nan")}
        n_bars = int(fold.test[-1] - fold.test[0] + 1 + self.hmax)
        pf = evaluate_portfolio(self.p, trades, int(fold.test[0]), n_bars, thr, cm, latency, self.bars_per_year, self.sigma,
                                {**self.ev, "borrow_bps_annual": self.ev.get("borrow_bps_annual", 300.0)}, seed, with_null)
        if not overlay:
            pf = {k: v for k, v in pf.items() if not (k.startswith("ovl_") or k.startswith("bench_"))}
        out.update(pf)
        return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else v) for k, v in out.items()}

    # ------------------------------------------------------------------ aggregation
    def add_dsr(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Deflated Sharpe for every (run, rule) at the default cost, deflating against all configurations in `rows`."""
        if rows.empty:
            return rows
        key = ["method", "rule", "protocol", "K", "fold", "seed", "bps"]
        sub = rows[(rows["bps"] == self.default_bps) & rows["metric"].isin(["net_sr_bar", "net_n_bars", "net_skew", "net_kurt"])]
        if sub.empty:
            return rows
        wide = sub.pivot_table(index=key, columns="metric", values="value").reset_index()
        trials = wide["net_sr_bar"].to_numpy()
        add = []
        for _, r in wide.iterrows():
            dsr = deflated_sharpe_from_moments(r.get("net_sr_bar", np.nan), r.get("net_n_bars", np.nan), r.get("net_skew", np.nan), r.get("net_kurt", np.nan), trials)
            base = rows[(rows[key[0]] == r[key[0]]) & (rows["rule"] == r["rule"]) & (rows["protocol"] == r["protocol"]) & (rows["K"] == r["K"])
                        & (rows["fold"] == r["fold"]) & (rows["seed"] == r["seed"]) & (rows["bps"] == r["bps"])].iloc[0].to_dict()
            add.append({**base, "metric": "net_DSR", "value": dsr})
        return pd.concat([rows, pd.DataFrame(add)], ignore_index=True)

    @staticmethod
    def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
        g = rows.groupby(["method", "rule", "protocol", "K", "bps", "metric"])["value"]
        return g.agg(["mean", "std", "count"]).reset_index()

    @staticmethod
    def cost_curve(agg: pd.DataFrame, metric: str = "net_SR", K: str = "10") -> pd.DataFrame:
        sub = agg[(agg["metric"] == metric) & (agg["K"] == K)]
        rows = []
        for (method, rule, protocol), g in sub.groupby(["method", "rule", "protocol"]):
            rows.append({"method": method, "rule": rule, "protocol": protocol, "metric": metric, "K": K, **M.cost_resilience(g["bps"].to_numpy(), g["mean"].to_numpy())})
        return pd.DataFrame(rows)

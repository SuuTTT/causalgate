"""Regression tests for the review fixes in causalgate/eval/{protocols,metrics}.py, causalgate/baselines/common.py and
scripts/cg_common.py (review group C). Each test fails on the pre-fix behaviour. Plain-script runner:

    python3 tests/test_causalgate.py --device cpu -k review_fixes_eval
"""
from __future__ import annotations
import os
import sys
import tempfile
import types
import warnings
from dataclasses import replace
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from causalgate.eval import metrics as CM

PLANTED = dict(name="cg_planted", seed=1, T=1800, N=6)
SPLIT = {"train": 900, "val": 200, "test": 500, "embargo": 10}
BCFG = {"hold_index": 2, "n_exec": 4, "w_max": 1.0, "gate": {"alpha": 0.05, "gamma": 0.02, "window": 500}}


def _planted():
    import causalgate.data.planted  # noqa: F401
    from warn2trade.utils import DATASETS
    return DATASETS.get("cg_planted")(PLANTED).build_panel()


def _register():
    """cg_absz_rf (causal |z| scorer) and cg_absz_rf_leak (reads the panel 3 bars AHEAD of the decision bar)."""
    from causalgate.baselines.common import ScoreToTradeBaseline, register
    from warn2trade.utils import BASELINES
    if "cg_absz_rf" in BASELINES.keys():
        return
    @register("cg_absz_rf")
    class AbsZ(ScoreToTradeBaseline):
        name = "cg_absz_rf"
        def _fit_model(self, X, t_end): pass
        def _score(self, X):
            return np.abs(np.nan_to_num(X[:, -1, 0])) + np.where(np.isnan(X[:, -1, 0]), np.nan, 0.0)

    @register("cg_absz_rf_leak")
    class AbsZLeak(AbsZ):
        name = "cg_absz_rf_leak"
        def _windows(self, batch):
            t = batch["t"].cpu().numpy(); i = batch["asset"].cpu().numpy()
            P = self.panel.price_feat
            tf = np.minimum(t + 3, P.shape[0] - 1)                       # look-ahead: 3 bars into the future
            X = np.zeros((len(t), self.L, P.shape[-1]), np.float32)
            X[:, -1, :] = np.nan_to_num(P[tf, i]); X[:, -1, 0] = np.abs(np.nan_to_num(P[tf, i, 0])) + 1e-3
            return X


def _runner(p, out=None, **d):
    from causalgate.eval.protocols import ProtocolRunner
    data = {"lookback": 48, "latency": 1, "exec_max": 4, "vol_window": 60}; data.update(d)
    return ProtocolRunner(p, list(p.meta["horizons"]), 252, data, dict(SPLIT), {}, torch.device("cpu"),
                          out or tempfile.mkdtemp(dir=os.environ.get("CG_TEST_TMP")), 42, "cg_planted")


# ------------------------------------------------------------------------------------------------ metrics
def test_review_fixes_eval_soft_ap_ties_match_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(60) < 0.3).astype(int); y[0] = 1
        s = rng.integers(0, 4, 60).astype(float)                      # heavy ties
        assert abs(CM._soft_ap(y.astype(float), s) - average_precision_score(y, s)) < 1e-12
        perm = rng.permutation(60)                                    # row (time) order inside a tie group is irrelevant
        assert abs(CM._soft_ap(y[perm].astype(float), s[perm]) - CM._soft_ap(y.astype(float), s)) < 1e-12
    y = np.array([0, 1, 0, 1, 0, 0]); s = np.zeros(6)                # constant score -> AP = base rate
    assert abs(CM._soft_ap(y.astype(float), s) - 2 / 6) < 1e-12


def test_review_fixes_eval_vus_pr_grid_and_default_buffer():
    y = np.zeros(200); y[[50, 120]] = 1                               # 1-bar events (typical panel label)
    s = np.zeros(200); s[[51, 121]] = 1.0                             # alarms one bar beside each event
    ap = CM._soft_ap(y, s)
    assert CM.vus_pr(y, s) > ap + 0.05, "1-bar events must still get a buffer volume"
    # each half-width once: with max_buffer=0 and 1 the result is plain AP only for max_buffer=0
    assert abs(CM.vus_pr(y, s, max_buffer=0) - ap) < 1e-12
    assert CM.vus_pr(y, s, max_buffer=1, n_buffers=2) > ap + 0.05


def test_review_fixes_eval_leakage_premium_f1_key_and_collapse_flag():
    lp = CM.leakage_premium({"F1_headline": 0.9, "F1_best": 0.5, "SR": 1.0}, {"F1_headline": 0.4, "F1_best": 0.5, "SR": 0.5})
    assert "LP_F1_headline" in lp and abs(lp["LP_F1_headline"] - 0.5) < 1e-12 and "LP_F1_best" not in lp
    # delay side collapses, peeking does not help -> flagged (old rule: collapse_ratio = 1 - 0.5 = 0.5 -> not flagged)
    d = CM.date_shift_asymmetry({-2: 0.4, -1: 0.5, 0: 1.0, 1: 0.1, 2: 0.0})
    assert d["collapse_flag"] and d["delay_collapse"] > 0.75 and d["peek_ratio"] == 0.5
    # honest one-bar edge that improves with peeking -> not flagged
    d = CM.date_shift_asymmetry({-1: 2.0, 0: 1.0, 1: 0.1})
    assert not d["collapse_flag"]
    d = CM.date_shift_asymmetry({0: 1.0, 1: 0.0, 2: 0.0})          # no peek shifts -> untested, never flagged
    assert not d["collapse_flag"] and d["peek_tested"] is False


def test_review_fixes_eval_pa_event_per_asset_and_scored_mask():
    from causalgate.eval.protocols import ProtocolRunner, PRESETS
    # two assets, 4 bars; asset 0 events at t=1,2 (alarm at t=2); asset 1 events at t=0,1 (missed)
    t = np.repeat(np.arange(4), 2); a = np.tile([0, 1], 4)
    Y = {(0, 0): 0, (1, 0): 1, (2, 0): 1, (3, 0): 0, (0, 1): 1, (1, 1): 1, (2, 1): 0, (3, 1): 0}
    y = np.array([Y[(tt, aa)] for tt, aa in zip(t, a)])
    sc = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.1, 0.1, 0.1])        # t-major rows
    tr = pd.DataFrame({"t": t, "asset": a, "y": y, "score": sc})
    alarm = (sc > 0.5).astype(np.float32)
    outs = [{"abstain": torch.as_tensor(1 - alarm), "scored": torch.ones(8, dtype=torch.bool)}]
    m = ProtocolRunner._extra_metrics(None, tr, outs, PRESETS["published"])
    # own alarms, per asset: asset0 segment hit (2 TP), asset1 missed (2 FN) -> PA P=1, R=0.5, F1=2/3
    # (the flat t-major vector joins both assets into one segment -> F1 = 1)
    assert m["n_events"] == 2 and abs(m["PA_f1_@thr"] - 2 / 3) < 1e-12
    assert abs(m["F1_headline"] - m["PA_f1_@thr"]) < 1e-12
    Yp = np.zeros((4, 2), int); Sp = np.zeros((4, 2)); Mp = np.ones((4, 2), bool)
    for (tt, aa), v in Y.items():
        Yp[tt, aa] = v
    Sp[t, a] = sc
    e = CM.pa_event_panel(Yp, Sp, Mp, 0.5)
    assert abs(e["event_recall"] - 0.5) < 1e-12 and abs(e["PA_recall"] - 0.5) < 1e-12
    flat = CM.f1_with_optional_pa(y, sc, 0.5, pa=True)                # the old (wrong) flat computation
    assert flat["recall"] == 1.0
    # unscored rows are excluded from the alarm-rate denominator (old mask used a_prob > -1, always true)
    scored = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    outs = [{"abstain": torch.as_tensor(1 - alarm), "scored": torch.as_tensor(scored)}]
    m = ProtocolRunner._extra_metrics(None, tr, outs, PRESETS["causal"])
    assert abs(m["alarm_rate"] - 1 / 6) < 1e-12 and abs(m["scored_frac"] - 0.75) < 1e-12
    assert abs(m["F1_headline"] - m["f1_@thr"]) < 1e-12


# ------------------------------------------------------------------------------------------------ common.py
def test_review_fixes_eval_best_f1_test_uses_ge():
    from causalgate.baselines.common import ScoreToTradeBaseline
    from warn2trade.backtest.metrics import best_f1
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 300).astype(float); y = (rng.random(300) < 0.2).astype(int) | (s >= 4).astype(int)
    me = types.SimpleNamespace(published_threshold="best_f1_test", anomaly_ratio=None, k_std=3.0)
    thr = ScoreToTradeBaseline._published_threshold(me, s, s, s, y)
    thr_best = best_f1(y, s)[1]
    assert np.array_equal(s > thr, s >= thr_best), "published alarm set must equal the best-F1 set {s >= thr}"


def test_review_fixes_eval_fixed_gate_a_prob_not_degenerate():
    from causalgate.eval.protocols import PRESETS
    _register()
    p = _planted()
    R = _runner(p)
    pr = replace(PRESETS["causal"], name="causal_fixedgate", threshold="val_quantile")
    f = R.folds(pr, 1)[0]
    r = R.run_fold("cg_absz_rf", BCFG, pr, f, save=False, replay_check=False)
    a = r["_trades"]["score"].to_numpy()
    assert a.max() > 0 and len(np.unique(a)) > 10, "a_prob must rank under the fixed-quantile gate"
    assert abs(r["eval"]["5"]["AUC_ROC"] - 0.5) > 1e-6
    bl = r["_baseline"]; b = next(iter(R._loaders(r["_panel"], f)["test"]))
    o = bl.predict(b)
    assert "thr" in o and torch.isfinite(o["thr"]).any() and not torch.isnan(o["thr"]).any()


# ------------------------------------------------------------------------------------------------ protocol runner
def test_review_fixes_eval_published_conventions_and_not_applicable():
    from causalgate.eval.protocols import PRESETS, one_switch
    from warn2trade.utils import BASELINES
    _register()
    p = _planted()
    R = _runner(p)
    f = R.folds(PRESETS["published"], 1)[0]
    cls = BASELINES.get("cg_absz_rf")
    # (a) baseline yaml says train-fitted -> train-fold rows, not the preset's global scaler
    cfg = R.baseline_cfg(dict(BCFG, published_norm="train", published_threshold="combined_quantile"), PRESETS["published"], f, p, cls)
    assert cfg["published_norm"] == "rows" and np.array_equal(cfg["norm_rows"], f.train) and cfg["published_threshold"] == "combined_quantile"
    assert cfg["_published_conventions"]["norm_source"] == "baseline yaml" and cfg["mode"] == "published"
    # (b) nothing declared -> keys left unset so the class default (cfg.setdefault in __init__) applies
    cfg = R.baseline_cfg(dict(BCFG), PRESETS["published"], f, p, cls)
    assert "published_norm" not in cfg and "published_threshold" not in cfg and np.array_equal(cfg["norm_rows"], f.train)
    # (c) class attribute published_protocol wins over the yaml
    sub = type("PubCls", (cls,), {"published_protocol": {"normalisation": "global", "threshold": "test_quantile"}})
    cfg = R.baseline_cfg(dict(BCFG, published_norm="train"), PRESETS["published"], f, p, sub)
    assert cfg["published_norm"] == "global" and cfg["norm_rows"] is None and cfg["published_threshold"] == "test_quantile"
    # (d) a class without model-side switches (CausalGate) is not stamped 'published' for L1 / L3
    nos = type("NoSwitch", (cls,), {"supported_switches": ()})
    for ch in ("L1", "L3"):
        cfg = R.baseline_cfg(dict(BCFG), one_switch(ch), f, p, nos)
        assert cfg["mode"] == "causal" and cfg["_not_applicable_channels"] == [ch] and cfg["norm_mode"] == "streaming" and cfg["thr_mode"] == "gate"


def test_review_fixes_eval_l6_uses_membership_at_t_only():
    from causalgate.eval.protocols import PRESETS, apply_data_switches
    from causalgate.data.pit import default_availability
    p = _planted()
    default_availability(p)
    R = _runner(p)
    f = R.folds(PRESETS["causal"], 1)[0]
    t_rm = int(f.test[100])
    uni = np.ones((p.T, p.N), bool); uni[t_rm:, 0] = False             # asset 0 leaves the universe at t_rm
    p.universe = uni
    q = apply_data_switches(p, PRESETS["causal"], f)
    assert np.isfinite(q.mid[t_rm:, 0]).all(), "prices must not be masked"
    ds = R._loaders(q, f)["test"].dataset
    keep = {(int(a), int(b)) for a, b in ds.index}
    assert (t_rm - 1, 0) in keep, "in-universe decision at t_rm-1 must survive (old: dropped via future NaN mid)"
    assert (t_rm, 0) not in keep and (t_rm + 5, 0) not in keep and (t_rm, 1) in keep


def test_review_fixes_eval_prequential_replay_detects_lookahead():
    from causalgate.eval.protocols import PRESETS
    _register()
    p = _planted()
    R = _runner(p, replay_max_bars=25)
    f = R.folds(PRESETS["causal"], 1)[0]
    ok = R.run_fold("cg_absz_rf", BCFG, PRESETS["causal"], f, save=False)
    assert ok["replay"]["kind"] == "prequential_refit_bar_by_bar" and ok["replay"]["n_bars"] == 25
    assert ok["deployability_gap_zero"] is True
    bad = R.run_fold("cg_absz_rf_leak", BCFG, PRESETS["causal"], f, save=False)
    assert bad["deployability_gap_zero"] is False and bad["replay_max_abs_dw"] > 0, "a 3-bar look-ahead must break the replay"


def test_review_fixes_eval_adia_truth_is_onset():
    from causalgate.eval.protocols import _onset_truth
    p = _planted()
    lead = int(p.meta["lead"])
    ev = [e for e in p.meta["events"] if 200 < e["onset_t"] < p.T - 50]
    e = ev[0]; o, a = int(e["onset_t"]), int(e["asset"])
    T0 = o - 2 * lead - 1
    tr = _onset_truth(p, T0, o + 5)
    assert p.labels[o - 1, a] == 1, "warning label is on before the onset (lead)"
    assert tr[o - 1 - T0, a] == 0 and tr[o - T0, a] == 1


# ------------------------------------------------------------------------------------------------ freeze / scripts
def test_review_fixes_eval_freeze_refuses_overwrite_and_dataset_prefix():
    from causalgate.eval.protocols import freeze, read_frozen
    d = tempfile.mkdtemp(dir=os.environ.get("CG_TEST_TMP"))
    path = os.path.join(d, "PROTOCOL_HASH")
    h = freeze(path)
    stamp = open(path).read()
    assert freeze(path) == h and open(path).read() == stamp, "unchanged freeze must leave the file untouched"
    lines = open(path).read().splitlines(); lines[1] = lines[1].split()[0] + " deadbeefdeadbeef"
    open(path, "w").write("\n".join(lines) + "\n")
    try:
        freeze(path); raise AssertionError("must refuse to overwrite a differing freeze")
    except RuntimeError:
        pass
    freeze(path, force=True)
    assert read_frozen(path) == h and os.path.exists(path + ".history")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import argparse
    from cg_common import load_cfgs
    a = argparse.Namespace(dataset="cg_stocknet", baseline="causalgate", smoke=False, override=[], max_folds=None)
    cfg, ds_cfg, _ = load_cfgs(a)
    assert ds_cfg.get("name") is not None

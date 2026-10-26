"""Re-implemented baselines B1 CPD-DMN, B8 STAGE, B9 Al Ridhawi, B10 Konczal-EVT (causalgate/baselines/b1_cpd_dmn.py,
b8_stage.py, b9_alridhawi.py, b10_konczal_evt.py) on the planted panel, one purged walk-forward fold, both modes:

    causal     delayed-ACI gate {alpha 0.05, gamma 0.02, window 500}; Filtration Guard must stay clean
    published  the baseline's own threshold rule (B1: always in market, B8: Eq. 20 reading, B9: tau_0, B10: best-F1 on test)

Collected by tests/test_causalgate.py (plain-script runner, `-k baselines_reimpl`); also pytest-compatible.
Configs come from configs/causalgate/baselines/*.yaml (smoke sizes).
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from warn2trade.utils import DATASETS, set_seed
import causalgate.data.planted  # noqa: F401  registers cg_planted
from warn2trade.data import EventDataset, build_loader, WalkForwardSplitter
from warn2trade.backtest import CostModel, EventBacktester
from warn2trade.train import evaluate_outputs
from causalgate.baselines.common import MODES
from causalgate.baselines.b1_cpd_dmn import CPDDMNBaseline
from causalgate.baselines.b8_stage import STAGEBaseline, READINGS
from causalgate.baselines.b9_alridhawi import AlRidhawiBaseline
from causalgate.baselines.b10_konczal_evt import KonczalEVTBaseline

DEVICE = "cpu"
LOOKBACK, LATENCY = 48, 1
GATE = {"type": "delayed_aci", "alpha": 0.05, "gamma": 0.02, "window": 500}
CFG_DIR = os.path.join(ROOT, "configs", "causalgate", "baselines")
BASELINES = {"cpd_dmn": CPDDMNBaseline, "stage": STAGEBaseline, "alridhawi": AlRidhawiBaseline, "konczal_evt": KonczalEVTBaseline}
BUDGET_S = 25.0
_CACHE = {}


def _setup():
    if "p" not in _CACHE:
        set_seed(42)
        p = DATASETS.get("cg_planted")({"name": "cg_planted", "seed": 1, "T": 1800, "N": 6}).build_panel()
        horizons = list(p.meta["horizons"])
        fold = next(iter(WalkForwardSplitter(p.T, 900, 200, 500, purge=16, embargo=10, max_folds=1)))
        mk = lambda idx: EventDataset(p, idx, LOOKBACK, horizons, LATENCY, 4, 60)
        # batch = multiple of N (kept for determinism); the scaffold now resets the gate stream only when a batch starts at a
        # bar strictly earlier than the previous batch's last bar, so straddling bars no longer trigger a reset
        bs = 510 if p.N and 510 % p.N == 0 else 512 - (512 % p.N)
        ld = {k: build_loader(mk(getattr(fold, k)), bs, shuffle=(k == "train"), seed=42) for k in ("train", "val", "test")}
        _CACHE.update(p=p, horizons=horizons, fold=fold, ld=ld)
    return _CACHE


def _cfg(name, mode, **over):
    with open(os.path.join(CFG_DIR, name + ".yaml")) as f:
        cfg = yaml.safe_load(f)
    c = _setup()
    cfg.update(mode=mode, panel=c["p"], lookback=LOOKBACK, latency=LATENCY, norm_rows=c["fold"].train, seed=0)
    if mode == "causal":
        cfg["gate"] = dict(GATE)
    cfg.update(over)
    return cfg


def _run(name, mode, **over):
    c = _setup()
    cfg = _cfg(name, mode, **over)
    set_seed(0)
    bl = BASELINES[name](cfg, c["p"].dims(), c["horizons"], torch.device(DEVICE))
    t0 = time.time()
    bl.fit(c["ld"]["train"], c["ld"]["val"])
    bl.prepare_test(c["ld"]["test"])
    bt = EventBacktester(c["horizons"], LATENCY, CostModel(5.0))
    outs, frames = [], []
    for b in c["ld"]["test"]:
        o = bl.predict(b)
        outs.append({k: v.cpu().numpy() for k, v in o.items()})
        frames.append(bt.collect(o, b))
    elapsed = time.time() - t0
    import pandas as pd
    trades = pd.concat(frames, ignore_index=True)
    m = evaluate_outputs(trades, 252, len(c["fold"].test), int(c["fold"].test[0]), None, c["p"].N)
    return bl, outs, m, elapsed


def _check(bl, outs, m, elapsed, mode, label, rate_tol=0.03):
    horizons = _setup()["horizons"]
    assert set(outs[0]) >= {"a_prob", "w", "hold_p", "kappa_p", "abstain", "score"}, set(outs[0])
    for o in outs:
        B = len(o["w"])
        assert o["a_prob"].shape == (B,) and o["abstain"].shape == (B,) and o["score"].shape == (B,)
        assert o["hold_p"].shape == (B, len(horizons)) and o["kappa_p"].shape == (B, 4)
        assert np.all(np.abs(o["w"]) <= bl.w_max + 1e-6) and np.all(np.isfinite(o["w"]))
        assert np.all((o["abstain"] == 0) | (o["abstain"] == 1))
    score = np.concatenate([o["score"] for o in outs])
    scored = score != -1.0                                           # the scaffold maps NaN (abstain) to -1.0
    frac = float(scored.mean())
    assert frac >= 0.9, (label, mode, "finite score fraction", frac)
    alarm = np.concatenate([1.0 - o["abstain"] for o in outs])
    rate = float(alarm[scored].mean())
    assert m["n_trades"] > 0, (label, mode, "no trades")
    assert np.isfinite(m["AUC_PR"]), (label, mode, m["AUC_PR"])
    bl.guard.assert_causal()                                         # strict in causal mode (raises), log-only in published
    if mode == "causal":
        assert not bl.guard.violations, (label, bl.guard.violations)
        assert abs(rate - GATE["alpha"]) <= rate_tol, (label, "realised alarm rate", rate)
    assert elapsed <= BUDGET_S, (label, mode, f"{elapsed:.1f}s > {BUDGET_S}s")
    thr = bl.diag.get("thr", None)
    print(f"    {label:12s} {mode:9s} AUC_PR={m['AUC_PR']:.3f} alarm_rate={rate:.3f} trades={m['n_trades']:4d} SR={m['SR']:6.2f} "
          f"thr={'-' if thr is None else f'{thr:.4g}'} FG_violations={len(bl.guard.violations)} scored={frac:.2f} ({elapsed:.1f}s)")
    return rate


def _check_sheet(cls):
    assert set(cls.conventions) == {f"L{i}" for i in range(1, 9)}, cls.name
    assert all(isinstance(v, str) and len(v) > 20 for v in cls.conventions.values()), cls.name
    assert cls.status in ("official", "reimplemented", "reimplemented-fallback"), cls.status
    assert cls.unspecified and cls.paper and cls.year >= 2022, cls.name


# ----------------------------------------------------------------------------------------------- B1
def test_baselines_reimpl_b1_cpd_dmn():
    _check_sheet(CPDDMNBaseline)
    for mode in MODES:
        bl, outs, m, el = _run("cpd_dmn", mode)
        rate = _check(bl, outs, m, el, mode, "B1 cpd_dmn")
        assert bl.diag["cpd_grid"][1] > bl.diag["cpd_grid"][0] and bl.diag["dmn_features"] >= 4
        pos = np.concatenate([o["w"] for o in outs])
        assert np.unique(np.round(pos, 3)).size > 2, "the DMN head must output graded positions, not a sign"
        if mode == "published":
            assert bl.diag["thr"] == float("-inf") and rate == 1.0, (bl.diag["thr"], rate)      # always in market
        else:
            assert bl.model_op.as_of <= int(_setup()["fold"].train[-1]) + 1   # trained on the next-bar return of the last train bar


# ----------------------------------------------------------------------------------------------- B8
def test_baselines_reimpl_b8_stage_two_modes_and_three_readings():
    _check_sheet(STAGEBaseline)
    bl_pub = None
    for mode in MODES:
        bl, outs, m, el = _run("stage", mode)
        _check(bl, outs, m, el, mode, "B8 stage")
        if mode == "published":
            bl_pub = bl
    thr = bl_pub.thresholds_by_reading()
    assert set(thr) == set(READINGS) and all(np.isfinite(v) for v in thr.values()), thr
    vals = np.asarray([thr[r] for r in READINGS])
    assert np.all(np.abs(vals[:, None] - vals[None, :])[~np.eye(3, dtype=bool)] > 1e-9), ("readings must differ", thr)
    assert bl_pub.thr_op.as_of >= int(_setup()["fold"].test[0]), "test_fold reading is stamped at the test end"
    print("    B8 stage Eq. 20 thresholds by reading: " + ", ".join(f"{r}={thr[r]:.4g}" for r in READINGS))
    # the train_fold reading is the only one whose threshold operator stays before the test start
    bl_tr, outs_tr, m_tr, el_tr = _run("stage", "published", stage_reading="train_fold")
    _check(bl_tr, outs_tr, m_tr, el_tr, "published", "B8 stage/tr")
    assert bl_tr.thr_op.as_of <= int(_setup()["fold"].train[-1]) and not bl_tr.guard.violations


# ----------------------------------------------------------------------------------------------- B9
def test_baselines_reimpl_b9_alridhawi():
    _check_sheet(AlRidhawiBaseline)
    for mode in MODES:
        bl, outs, m, el = _run("alridhawi", mode)
        _check(bl, outs, m, el, mode, "B9 alridhawi")
        assert np.isfinite(bl.diag["tau0"]) and bl.diag["n_stable"] > 0
        assert 0.0 <= bl.diag["alpha_last"] <= 1.0
        if mode == "published":
            assert bl.diag["thr"] == bl.diag["tau0"] and bl.thr_op.as_of <= int(_setup()["fold"].train[-1])
            assert not bl.guard.violations, bl.guard.violations          # tau_0 is train-fitted: clean even as published
    bl_f, outs_f, m_f, el_f = _run("alridhawi", "causal", controller_online=False)
    _check(bl_f, outs_f, m_f, el_f, "causal", "B9 alrid/frz")


# ----------------------------------------------------------------------------------------------- B10
def test_baselines_reimpl_b10_konczal_evt():
    _check_sheet(KonczalEVTBaseline)
    for mode in MODES:
        bl, outs, m, el = _run("konczal_evt", mode)
        _check(bl, outs, m, el, mode, "B10 konczal")
        qp = bl.diag["evt"]["q_p"]
        assert all(np.isfinite(v) and v > 0 for v in qp.values()), qp
        assert bl.diag["clf"] == "hgb" and 0 < bl.diag["train_pos_rate"] < 0.5, bl.diag
        if mode == "published":
            assert bl.diag["published_threshold_labels"] == "indicator", bl.diag
            assert bl.guard.violations, "best-F1-on-test must show up in the Filtration Guard log"
    bl_fs, outs_fs, m_fs, el_fs = _run("konczal_evt", "published", evt_sample="full_sample")
    _check(bl_fs, outs_fs, m_fs, el_fs, "published", "B10 konc/fs")
    assert bl_fs.evt_op.as_of == _setup()["p"].T - 1

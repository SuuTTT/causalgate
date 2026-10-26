"""B2-B7 baseline smoke tests on the planted panel (collected by tests/test_causalgate.py):

    python3 tests/test_causalgate.py --device cpu -k baselines_official

For B2 (Anomaly Transformer), B3 (DAMP), B4 (MemStream), B5 (DCdetector) and B7 (CATCH) in both modes:
fit(train, val) -> prepare_test(test) -> predict over the test loader -> EventBacktester.collect; the out dict keys /
shapes, finiteness of the scores, the presence of trades and (causal mode) the absence of Filtration-Guard violations
are asserted; one line per (baseline, mode) prints AUC_PR (warn2trade.train.evaluate_outputs) and the realised alarm
rate. B6 (MOMENT) is asserted to be 'wrapper-needed' with an ImportError naming `momentfm` when the package is absent.
Hyper-parameters come from configs/causalgate/baselines/<name>.yaml with the `smoke:` block merged on top.
"""
from __future__ import annotations
import copy
import importlib
import os
import sys
import time
from typing import Any, Dict
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from warn2trade.utils import DATASETS, set_seed                      # noqa: E402
import causalgate.data.planted                                      # noqa: E402,F401  registers cg_planted
from warn2trade.data import EventDataset, build_loader, WalkForwardSplitter   # noqa: E402
from warn2trade.backtest import CostModel, EventBacktester          # noqa: E402
from warn2trade.train import evaluate_outputs                       # noqa: E402

DEVICE = torch.device("cpu")
PLANTED = dict(name="cg_planted", seed=1, T=1800, N=6)
LOOKBACK = 48
CFG_DIR = os.path.join(ROOT, "configs", "causalgate", "baselines")
BASELINES = [
    ("anomaly_transformer", "causalgate.baselines.b2_anomaly_transformer", "AnomalyTransformerBaseline", "cg_anomaly_transformer"),
    ("damp", "causalgate.baselines.b3_damp", "DampBaseline", "cg_damp"),
    ("memstream", "causalgate.baselines.b4_memstream", "MemStreamBaseline", "cg_memstream"),
    ("dcdetector", "causalgate.baselines.b5_dcdetector", "DCdetectorBaseline", "cg_dcdetector"),
    ("catch", "causalgate.baselines.b7_catch", "CatchBaseline", "cg_catch"),
]
GATE = {"type": "delayed_aci", "alpha": 0.05, "gamma": 0.02, "window": 500}
TIME_BUDGET_S = 25.0
_CACHE: Dict[str, Any] = {}


def _setup() -> Dict[str, Any]:
    if "panel" not in _CACHE:
        p = DATASETS.get("cg_planted")(PLANTED).build_panel()
        horizons = list(p.meta["horizons"]); latency = 1
        fold = next(iter(WalkForwardSplitter(p.T, 900, 200, 500, purge=16, embargo=10, max_folds=1)))
        mk = lambda idx: EventDataset(p, idx, LOOKBACK, horizons, latency, 4, 60)   # noqa: E731
        ld = {k: build_loader(mk(getattr(fold, k)), 256, shuffle=(k == "train"), seed=42) for k in ("train", "val", "test")}
        _CACHE.update(panel=p, horizons=horizons, latency=latency, fold=fold, ld=ld)
    return _CACHE


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load_cfg(name: str) -> Dict[str, Any]:
    with open(os.path.join(CFG_DIR, f"{name}.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    smoke = cfg.pop("smoke", {}) or {}
    cfg = _deep_merge(cfg, smoke)
    assert cfg["lookback"] == LOOKBACK, (name, cfg["lookback"])
    return cfg


def _run(name: str, mode: str) -> Dict[str, Any]:
    s = _setup()
    _, module, cls_name, reg_name = [b for b in BASELINES if b[0] == name][0]
    cls = getattr(importlib.import_module(module), cls_name)
    assert cls.name == reg_name and cls.status in ("official", "reimplemented", "reimplemented-fallback", "wrapper-needed"), (cls.name, cls.status)
    assert set(cls.conventions) == {f"L{i}" for i in range(1, 9)}, (name, sorted(cls.conventions))
    assert cls.paper and cls.official_repo and cls.year >= 2021 and isinstance(cls.unspecified, list) and cls.unspecified
    cfg = _load_cfg(name)
    p, horizons, latency, fold, ld = s["panel"], s["horizons"], s["latency"], s["fold"], s["ld"]
    cfg.update({"panel": p, "mode": mode, "latency": latency, "gate": dict(GATE), "norm_rows": fold.train})
    set_seed(0)
    bl = cls(cfg, p.dims(), horizons, DEVICE)
    assert bl.mode == mode and bl.gate.tau == horizons[bl.hold_index] + latency
    t0 = time.time()
    bl.fit(ld["train"], ld["val"])
    bl.prepare_test(ld["test"])
    bt = EventBacktester(horizons, latency, CostModel(5.0))
    outs, frames = [], []
    for b in ld["test"]:
        o = bl.predict(b)
        outs.append({k: v.cpu().numpy() for k, v in o.items()})
        frames.append(bt.collect(o, b))
    elapsed = time.time() - t0
    import pandas as pd
    trades = pd.concat(frames, ignore_index=True)
    # out dict contract
    assert set(outs[0]) >= {"a_prob", "w", "hold_p", "kappa_p", "abstain", "score"}, sorted(outs[0])
    B0 = len(outs[0]["w"])
    assert outs[0]["a_prob"].shape == (B0,) and outs[0]["w"].shape == (B0,) and outs[0]["abstain"].shape == (B0,)
    assert outs[0]["hold_p"].shape == (B0, len(horizons)) and outs[0]["kappa_p"].shape == (B0, 4)
    a_prob = np.concatenate([o["a_prob"] for o in outs]); w = np.concatenate([o["w"] for o in outs])
    score = np.concatenate([o["score"] for o in outs]); alarm = np.concatenate([1.0 - o["abstain"] for o in outs])
    assert np.isfinite(a_prob).all() and np.isfinite(w).all() and (np.abs(w) <= 1.0 + 1e-6).all()
    scored = score > -1.0                                             # -1 marks an abstained (NaN) score
    assert scored.mean() > 0.9, (name, mode, float(scored.mean()))
    assert np.isfinite(score[scored]).all()
    rate = float(alarm[scored].mean())
    m = evaluate_outputs(trades, 252, len(fold.test), int(fold.test[0]), None, p.N)
    assert m["n_trades"] > 0 and np.isfinite(m["AUC_PR"]), (name, mode, m["n_trades"], m["AUC_PR"])
    assert (w != 0).sum() == m["n_trades"]
    if mode == "causal":
        bl.guard.assert_causal()
        assert not bl.guard.violations, bl.guard.violations
        assert bl.diag["gate_as_of"] <= int(fold.val[-1]) and bl._stamp.as_of >= int(fold.train[0])
        assert 0.0 < rate < 0.5, rate
    else:
        assert np.isfinite(bl.diag["thr"]) and bl.diag["published_threshold"] == bl.published_threshold
    assert elapsed <= TIME_BUDGET_S, (name, mode, f"{elapsed:.1f}s > {TIME_BUDGET_S}s")
    status = getattr(bl, "status", cls.status)
    print(f"    {name:20s} {mode:9s} status={status:22s} AUC_PR={m['AUC_PR']:.3f} AUC_ROC={m['AUC_ROC']:.3f} alarm_rate={rate:.3f} "
          f"trades={m['n_trades']:4d} SR={m['SR']:6.2f} thr={bl.diag.get('thr', float('nan')):.4g} ({elapsed:.1f}s)")
    return {"metrics": m, "rate": rate, "status": status, "elapsed": elapsed, "baseline": bl}


def test_baselines_official_causal():
    for name, *_ in BASELINES:
        r = _run(name, "causal")
        assert r["status"] in ("reimplemented", "reimplemented-fallback", "official")


def test_baselines_official_published():
    expected_thr = {"anomaly_transformer": "combined_quantile", "damp": "train_quantile", "memstream": "train_quantile",
                    "dcdetector": "combined_quantile", "catch": "combined_quantile"}
    for name, *_ in BASELINES:
        r = _run(name, "published")
        assert r["baseline"].published_threshold == expected_thr[name], (name, r["baseline"].published_threshold)


def test_baselines_official_replay_is_deterministic():
    """A second pass over the same test window (gate + memory restarted from their snapshots) reproduces the decisions."""
    s = _setup()
    for name in ("memstream", "damp"):
        cfg = _load_cfg(name)
        cls = getattr(importlib.import_module([b for b in BASELINES if b[0] == name][0][1]), [b for b in BASELINES if b[0] == name][0][2])
        cfg.update({"panel": s["panel"], "mode": "causal", "latency": s["latency"], "gate": dict(GATE)})
        set_seed(0)
        bl = cls(cfg, s["panel"].dims(), s["horizons"], DEVICE)
        bl.fit(s["ld"]["train"], s["ld"]["val"])
        o1 = [{k: v.cpu().numpy() for k, v in bl.predict(b).items()} for b in s["ld"]["test"]]
        o2 = [{k: v.cpu().numpy() for k, v in bl.predict(b).items()} for b in s["ld"]["test"]]
        for a, b in zip(o1, o2):
            assert np.array_equal(a["w"], b["w"]) and np.array_equal(a["abstain"], b["abstain"]) and np.allclose(a["score"], b["score"]), name
        bl.guard.assert_causal()


def test_baselines_official_moment_wrapper():
    s = _setup()
    mod = importlib.import_module("causalgate.baselines.b6_moment")
    cls = mod.MomentZeroShotBaseline
    cfg = _load_cfg("moment")
    cfg.update({"panel": s["panel"], "mode": "causal", "latency": s["latency"], "gate": dict(GATE)})
    bl = cls(cfg, s["panel"].dims(), s["horizons"], DEVICE)
    assert bl.published_threshold == "best_f1_test" and cls.cutoff_date == "2024-02-01"
    assert set(cls.conventions) == {f"L{i}" for i in range(1, 9)} and "Time Series Pile" in cls.conventions["L8"]
    try:
        import momentfm  # noqa: F401
        has = True
    except Exception:  # noqa: BLE001
        has = False
    if has:
        assert cls.status == "official"
        print("    moment: momentfm installed -> status official (zero-shot fit not exercised by the smoke test)")
        return
    assert cls.status == "wrapper-needed", cls.status
    try:
        bl.fit(s["ld"]["train"], s["ld"]["val"])
        raise AssertionError("expected ImportError from cg_moment.fit without momentfm")
    except ImportError as e:
        assert "momentfm" in str(e) and "pip install momentfm" in str(e), str(e)
        print(f"    moment               wrapper-needed: {str(e)[:90]}...")

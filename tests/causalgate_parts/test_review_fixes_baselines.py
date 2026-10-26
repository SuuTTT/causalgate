"""Regression tests for the review fixes in causalgate/baselines/b*.py (B1, B4, B8, B9, B10).

    STAGE node features never wrap to future rows (node_bars > lookback)
    B9 controller restored only when a pass starts strictly before the previous batch's last bar
    B1 / B4 / B9 / B10 honour a protocol-requested published threshold kind (L2 / L3 one-switch, P-W, offline_all)
    B4 label-filtered initial memory and B9 adaptive-tau score keyed on the switches, not on cfg['mode']
    B1 / B9 / B10 label-trained operators stamped at the last bar whose future return they read

Collected by tests/test_causalgate.py (`-k review_fixes_baselines`).
"""
from __future__ import annotations
import os
import sys
import types
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from warn2trade.utils import DATASETS, set_seed
import causalgate.data.planted  # noqa: F401
from warn2trade.data import EventDataset, build_loader, WalkForwardSplitter
from causalgate.baselines.common import ScoreToTradeBaseline
from causalgate.baselines.b1_cpd_dmn import CPDDMNBaseline
from causalgate.baselines.b4_memstream import MemStreamBaseline
from causalgate.baselines.b8_stage import STAGEBaseline
from causalgate.baselines.b9_alridhawi import AlRidhawiBaseline
from causalgate.baselines.b10_konczal_evt import KonczalEVTBaseline

DEV = torch.device("cpu")
CFG_DIR = os.path.join(ROOT, "configs", "causalgate", "baselines")
_C = {}


def _setup(lookback=32):
    key = ("p", lookback)
    if key not in _C:
        set_seed(7)
        p = DATASETS.get("cg_planted")({"name": "cg_planted", "seed": 3, "T": 1000, "N": 4}).build_panel()
        hz = list(p.meta["horizons"])
        fold = next(iter(WalkForwardSplitter(p.T, 500, 120, 250, purge=16, embargo=10, max_folds=1)))
        mk = lambda idx: EventDataset(p, idx, lookback, hz, 1, 4, 60)
        ld = {k: build_loader(mk(getattr(fold, k)), 120, shuffle=False, seed=0) for k in ("train", "val", "test")}
        _C[key] = dict(p=p, hz=hz, fold=fold, ld=ld)
    return _C[key]


def _cfg(name, lookback=32, **over):
    with open(os.path.join(CFG_DIR, name + ".yaml")) as f:
        cfg = yaml.safe_load(f)
    sm = dict(cfg.pop("smoke", {}) or {})
    sm.pop("lookback", None)
    for k, v in sm.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    c = _setup(lookback)
    cfg.update(panel=c["p"], lookback=lookback, latency=1, norm_rows=c["fold"].train, seed=0)
    cfg.update(over)
    return cfg


def _make(cls, name, lookback=32, **over):
    c = _setup(lookback)
    return cls(_cfg(name, lookback, **over), c["p"].dims(), c["hz"], DEV)


L2 = dict(mode="published", norm_mode="streaming", thr_mode="published", published_threshold="test_quantile")   # causal + L2


# ------------------------------------------------------------------------------------------------ STAGE (critical)
def test_review_fixes_baselines_stage_no_wrap_to_future_rows():
    bl = _make(STAGEBaseline, "stage", lookback=16, mode="causal", node_bars=48)
    assert bl.node_bars == 16, bl.node_bars                                    # capped at the lookback
    p = _setup(16)["p"]
    Z = np.random.default_rng(0).normal(size=(p.T, p.N, 3)).astype(np.float32)
    bl._panel_Z = lambda: Z
    bl._adjacency = lambda t: np.eye(p.N, dtype=np.float32)
    bl.node_bars = 48                                                          # even if forced above t + 1: no wrap
    ts = np.array([16, 20, 40])
    a, _ = bl._cross_sections(ts)
    Z[-60:] += 100.0                                                           # perturb the panel end (future rows)
    b, _ = bl._cross_sections(ts)
    assert np.array_equal(a, b), "cross-sections at early bars read the last panel rows"
    assert np.all(a[0].reshape(p.N, 48, 3)[:, :48 - 17] == 0)                  # pre-sample bars zero-filled


# ------------------------------------------------------------------------------------------------ B9 controller replay
def test_review_fixes_baselines_b9_restore_only_on_new_pass():
    c = _setup()
    bl = _make(AlRidhawiBaseline, "alridhawi", mode="causal", gate={"type": "delayed_aci", "alpha": 0.05, "gamma": 0.02, "window": 500})
    bl.fit(c["ld"]["train"], c["ld"]["val"])
    bl.prepare_test(c["ld"]["test"])
    n = {"k": 0}
    orig = bl.ctrl.restore
    bl.ctrl.restore = lambda: (n.__setitem__("k", n["k"] + 1), orig())[1]
    s = []
    for rep in range(2):
        s.append(np.concatenate([bl.predict(b)["score"].cpu().numpy() for b in c["ld"]["test"]]))
    nb = len(c["ld"]["test"])
    assert nb >= 3 and n["k"] == 1, (nb, n["k"])                                # one restore at the start of the second pass only
    assert np.allclose(s[0], s[1]), "replay must reproduce the first pass"
    # published (as-published) mode: the first predict pass continues from the end-of-validation snapshot
    bp = _make(AlRidhawiBaseline, "alridhawi", mode="published")
    bp.fit(c["ld"]["train"], c["ld"]["val"]); bp.prepare_test(c["ld"]["test"])
    m = {"k": 0}; o2 = bp.ctrl.restore
    bp.ctrl.restore = lambda: (m.__setitem__("k", m["k"] + 1), o2())[1]
    for b in c["ld"]["test"]:
        bp.predict(b)
    assert m["k"] == 0, m["k"]


# ------------------------------------------------------------------------------------------------ threshold kind honoured
def test_review_fixes_baselines_requested_threshold_kind_is_honoured():
    rng = np.random.default_rng(1)
    s_tr, s_va, s_te = rng.normal(size=400), rng.normal(size=100), rng.normal(2.0, 1.0, size=300)
    y = (rng.random(300) < 0.05).astype(float)
    q = lambda bl: float(np.quantile(s_te, 1 - float(np.clip(bl.anomaly_ratio if bl.anomaly_ratio is not None else y.mean(), 1e-4, 0.5))))
    # B1: always_in_market only for the as-published threshold
    b1 = _make(CPDDMNBaseline, "cpd_dmn", **L2)
    assert np.isclose(b1._published_threshold(s_tr, s_va, s_te, y), q(b1))
    assert _make(CPDDMNBaseline, "cpd_dmn", mode="published")._published_threshold(s_tr, s_va, s_te, y) == float("-inf")
    # B4: beta only for the as-published threshold
    b4 = _make(MemStreamBaseline, "memstream", **L2); b4.core = types.SimpleNamespace(beta=-123.0)
    assert np.isclose(b4._published_threshold(s_tr, s_va, s_te, y), q(b4))
    b4n = _make(MemStreamBaseline, "memstream", mode="published"); b4n.core = types.SimpleNamespace(beta=-123.0)
    assert b4n._published_threshold(s_tr, s_va, s_te, y) == -123.0
    # B9: tau0 only for the as-published threshold
    b9 = _make(AlRidhawiBaseline, "alridhawi", **L2); b9.tau0 = -77.0
    assert np.isclose(b9._published_threshold(s_tr, s_va, s_te, y), q(b9))
    b9n = _make(AlRidhawiBaseline, "alridhawi", mode="published"); b9n.tau0 = -77.0
    assert b9n._published_threshold(s_tr, s_va, s_te, y) == -77.0
    # B10: test_quantile / combined_quantile not replaced by train_quantile
    b10 = _make(KonczalEVTBaseline, "konczal_evt", **L2)
    assert np.isclose(b10._published_threshold(s_tr, s_va, s_te, y), q(b10)) and b10.published_threshold == "test_quantile"
    b10c = _make(KonczalEVTBaseline, "konczal_evt", **{**L2, "published_threshold": "combined_quantile"})
    assert np.isclose(b10c._published_threshold(s_tr, s_va, s_te, y), ScoreToTradeBaseline._published_threshold(b10c, s_tr, s_va, s_te, y))
    # explicit runner flag wins
    b9f = _make(AlRidhawiBaseline, "alridhawi", mode="published", threshold_source="protocol"); b9f.tau0 = -77.0
    assert b9f._published_threshold(s_tr, s_va, s_te, y) != -77.0
    # ProtocolRunner markers: P-O row (resolve_published) -> native; any other row -> the requested kind
    b9r = _make(AlRidhawiBaseline, "alridhawi", mode="published", published_threshold="train_quantile", _applied_channels=["L2"]); b9r.tau0 = -77.0
    assert b9r._published_threshold(s_tr, s_va, s_te, y) != -77.0
    b9p = _make(AlRidhawiBaseline, "alridhawi", mode="published", _published_conventions={}, _applied_channels=["L1", "L2"]); b9p.tau0 = -77.0
    assert b9p._published_threshold(s_tr, s_va, s_te, y) == -77.0


# ------------------------------------------------------------------------------------------------ switches, not mode
def test_review_fixes_baselines_behaviour_keyed_on_switches():
    l1 = dict(mode="published", norm_mode="offline", thr_mode="gate", gate={"type": "delayed_aci"})   # causal + L1 only
    assert _make(MemStreamBaseline, "memstream", **l1).init_normal_only is False
    assert _make(MemStreamBaseline, "memstream", **L2).init_normal_only is False
    assert _make(MemStreamBaseline, "memstream", mode="published").init_normal_only is True              # as-published row
    assert _make(AlRidhawiBaseline, "alridhawi", **l1)._native_thr() is False                            # no tau shift on L1


# ------------------------------------------------------------------------------------------------ stamps include label reach
def test_review_fixes_baselines_label_trained_ops_stamped_at_label_reach():
    c = _setup()
    t_end = int(c["fold"].train[-1])
    b10 = _make(KonczalEVTBaseline, "konczal_evt", mode="causal", gate={"type": "delayed_aci", "alpha": 0.05, "gamma": 0.02, "window": 500})
    b10.fit(c["ld"]["train"], c["ld"]["val"])
    reach = 1 + (min(b10.K, max(c["hz"]) + 1 + 4 - 1) if b10.target in ("nr", "any") else 0)
    assert b10.clf_op.as_of >= t_end + 1, (b10.clf_op.as_of, t_end)
    assert b10.clf_op.as_of <= t_end + reach + 1
    b9 = _make(AlRidhawiBaseline, "alridhawi", mode="causal", gate={"type": "delayed_aci", "alpha": 0.05, "gamma": 0.02, "window": 500})
    b9.fit(c["ld"]["train"], c["ld"]["val"])
    assert b9.model_op.as_of == t_end + 1 and b9.thr_op.as_of == t_end, (b9.model_op.as_of, b9.thr_op.as_of, t_end)

"""Regression tests for the review fixes in causalgate/models/{causalgate,gate,scorer}.py and causalgate/data/stream.py.

Run:  python3 tests/test_causalgate.py --device cpu -k review_fixes_models
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from warn2trade.utils import DATASETS, set_seed
import causalgate.data.planted  # noqa: F401
from causalgate.models.gate import make_gate, alpha_from_cost_budget, POTGate, ACIGate, DelayedACIGate
from causalgate.data.stream import StreamingNormaliser

_P = None


def _panel():
    global _P
    if _P is None:
        _P = DATASETS.get("cg_planted")(dict(name="cg_planted", seed=3, T=1100, N=3)).build_panel()
    return _P


def _setup(gate, scorer=None, **extra):
    from warn2trade.data import EventDataset, build_loader, WalkForwardSplitter
    from causalgate.models.causalgate import CausalGate
    set_seed(0)
    p = _panel(); H = list(p.meta["horizons"])
    fold = next(iter(WalkForwardSplitter(p.T, 500, 150, 250, purge=max(H) + 5, embargo=5, max_folds=1)))
    mk = lambda idx: EventDataset(p, idx, 32, H, 1, 4, 60)
    ld = {k: build_loader(mk(getattr(fold, k)), 256, shuffle=False, seed=0) for k in ("train", "val", "test")}
    cfg = {"panel": p, "lookback": 32, "latency": 1, "hold_index": 2, "n_exec": 4, "w_max": 1.0, "bars_per_year": 252,
           "normaliser": {"mode": "welford", "warmup": 40}, "scorer": scorer or {"type": "lmp", "m": 6},
           "gate": gate, "policy": {"type": "momentum"}}
    cfg.update(extra)
    bl = CausalGate(cfg, p.dims(), H, torch.device("cpu"))
    bl.fit(ld["train"], ld["val"])
    return bl, ld


def _run(bl, batches):
    th, al = [], []
    for b in batches:
        o = bl.predict(b)
        th.append(o["thr"].numpy()); al.append(o["abstain"].numpy())
    return np.concatenate(th), np.concatenate(al)


def test_review_fixes_models_test_labels_never_move_the_gate():
    """Critical leak: replacing the test-fold labels (random / flipped) must leave thresholds and alarms bit-identical
    for every gate type and calib mode (old code: POT filtered on y_t; normal_only ACI admitted on y_t)."""
    rng = np.random.default_rng(0)
    for kind in ("fixed", "pot", "aci", "delayed_aci"):
        for calib in ("all", "normal_only"):
            g = {"type": kind, "alpha": 0.1, "calib": calib, "gamma": 0.02, "window": 300, "refit_every": 5}
            bl, ld = _setup(g)
            base = list(ld["test"])
            th0, al0 = _run(bl, base)
            for how in ("random", "flipped"):
                pert = []
                for b in base:
                    b2 = dict(b); y = b["y"].clone()
                    b2["y"] = (1 - y) if how == "flipped" else torch.as_tensor(rng.integers(0, 2, len(y)), dtype=y.dtype)
                    pert.append(b2)
                th1, al1 = _run(bl, pert)                           # new pass -> reset to the calibration snapshot
                assert np.array_equal(th0, th1) and np.array_equal(al0, al1), (kind, calib, how)


def test_review_fixes_models_labels_only_after_availability():
    """Gate-level: with calib='normal_only' a labelled score enters the window only at its avail bar; calibration rows
    whose label is unavailable at t_end are not label-filtered; consuming a label stamps the gate."""
    for G in (POTGate, ACIGate, DelayedACIGate):
        kw = {"tau": 2} if G is DelayedACIGate else {}
        g = G(alpha=0.1, calib="normal_only", **kw)
        s = np.arange(50, dtype=float); y = np.zeros(50); y[-5:] = 1
        g.calibrate(s, t_end=100, labels=y, label_avail=np.r_[np.full(45, 90), np.full(5, 200)])
        assert len(g.buf) == 50, (G.__name__, len(g.buf))   # last 5 labels not yet known at t_end -> kept
        n0 = len(g.buf)
        g.step(1e3, 101, y=0.0, y_avail=110)
        for t in range(102, 110):
            g.step(np.nan, t)
        assert len(g.buf) == n0, G.__name__                   # not admitted before avail
        g.step(np.nan, 110)
        assert len(g.buf) == n0 + 1 and g.as_of == 110, (G.__name__, len(g.buf), g.as_of)
    # label without avail and unknown delay -> ignored (score treated as unlabelled), never filtered on y_t
    g = POTGate(alpha=0.1, calib="normal_only"); g.calibrate(np.arange(50.0), t_end=0)
    g.step(5.0, 1, y=1.0)
    assert len(g.buf) == 51


def test_review_fixes_models_refit_policies():
    """streaming used to crash under no_grad; periodic was a no-op equal to static. Both must change the scorer,
    stay F_t-stamped, and replay identically on a second pass."""
    base_sc = {"type": "tcn", "epochs": 1, "hidden": 8, "n_layers": 2, "refit_every": 40, "refit_window": 400}
    g = {"type": "delayed_aci", "alpha": 0.1, "gamma": 0.02, "window": 300}
    res = {}
    for pol in ("static", "periodic", "streaming"):
        bl, ld = _setup(g, dict(base_sc, refit_policy=pol))
        w0 = [p.detach().clone() for p in bl.scorer.parameters()]
        test = list(ld["test"])
        sc = []
        for b in test:
            sc.append(bl.predict(b)["score"].numpy())
        changed = any(not torch.equal(a, p) for a, p in zip(w0, bl.scorer.parameters()))
        assert changed == (pol != "static"), (pol, changed)
        sc2 = np.concatenate([bl.predict(b)["score"].numpy() for b in test])
        assert np.array_equal(np.concatenate(sc), sc2), pol           # replay restores the post-fit scorer
        res[pol] = sc2
        if pol == "periodic":
            assert bl.diag.get("n_refits", 0) >= 2, bl.diag
    assert not np.array_equal(res["periodic"], res["static"]) and not np.array_equal(res["streaming"], res["static"])


def test_review_fixes_models_panel_Z_rebuild_and_offline_norm():
    """_panel_Z rebuilt after cache invalidation must equal the first causal sweep; norm_mode=offline (no_fg / L1)
    must actually switch the normalisation and log (not raise) the guard violation."""
    g = {"type": "delayed_aci", "alpha": 0.1}
    bl, ld = _setup(g)
    Z1 = bl._panel_Z().copy(); bl._Z = None; Z2 = bl._panel_Z()
    assert np.array_equal(Z1, Z2, equal_nan=True)
    bo, ldo = _setup(g, norm_mode="offline", published_norm="global", mode="published")
    Zo = bo._panel_Z()
    assert not np.array_equal(Z1, Zo, equal_nan=True) and np.isfinite(Zo[:40]).all()
    for b in ldo["test"]:
        bo.predict(b)
    bo.guard.assert_causal()
    assert any("OfflineScaler" in v for v in bo.guard.violations), bo.guard.violations[:3]


def test_review_fixes_models_minors():
    # EWMA debiasing: z right after warm-up has unit scale
    rng = np.random.default_rng(0)
    Z = StreamingNormaliser("ewma", 60, 120.0).transform_series(rng.standard_normal((400, 300)))
    sd = float(np.nanstd(Z[60:80]))
    assert 0.85 < sd < 1.2, sd
    # StreamingNormaliser.reset gives a fresh sweep
    n = StreamingNormaliser("welford", 5); x = rng.standard_normal((50, 2))
    a = n.transform_series(x); b = n.reset().transform_series(x)
    assert np.array_equal(a, b, equal_nan=True)
    # POT outside the fitted tail falls back to the empirical quantile
    s = rng.standard_normal(3000)
    g = POTGate(alpha=0.5, u=0.9, window=1000, refit_every=20).calibrate(s[:1000], t_end=999)
    r = g.run(s[1000:], t0=1000)["realised_rate"]
    assert abs(r - 0.5) < 0.03, r
    # calibrate clears the buffer; NaNs never enter the window
    g = ACIGate(alpha=0.1).calibrate(np.r_[np.arange(10.0), np.nan], t_end=10)
    g.calibrate(np.arange(5.0), t_end=12)
    assert len(g.buf) == 5
    p = POTGate().calibrate(np.r_[np.full(5, np.nan), np.arange(10.0)], t_end=3)
    assert len(p.buf) == 10
    # reset_stream before calibrate is loud
    try:
        ACIGate().reset_stream(); raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    # alpha_path holds the level used for the decision at t (identical convention for ACI and delayed ACI)
    a = ACIGate(alpha=0.1, gamma=0.05).calibrate(np.arange(100.0), t_end=0)
    out = a.run(rng.standard_normal(20) * 100, t0=1)
    assert out["alpha_path"][0] == 0.1
    # budget: null falls back; zero budget -> alpha 0 with a warning; clip is announced
    assert make_gate({"type": "aci", "budget": None}).alpha > 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert alpha_from_cost_budget(10, 0.0, 252) == 0.0 and alpha_from_cost_budget(10, 5000, 252) == 0.5
        assert sum("clipped" in str(x.message) for x in w) == 1, [str(x.message) for x in w]
    # fixed / POT honour calib (default 'all' no longer drops labelled anomalies)
    f = make_gate({"type": "fixed", "alpha": 0.5}).calibrate(np.arange(10.0), t_end=0, labels=np.r_[np.zeros(5), np.ones(5)])
    assert f.thr >= 5

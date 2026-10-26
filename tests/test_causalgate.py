"""CausalGate unit + smoke tests. Plain-script mode (system Python 3.9 has no pytest):

    python3 tests/test_causalgate.py --device cpu            # all tests
    python3 tests/test_causalgate.py --device cpu -k fpit    # substring filter

Every test is also pytest-compatible. Nothing here touches real data; the planted-leak panel is synthetic by design.
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warn2trade.utils import DATASETS, set_seed
import causalgate.data.planted  # noqa: F401  registers cg_planted
from causalgate.data.pit import (PITView, FiltrationError, AsOfStamped, FiltrationGuard, perturb_future, perturb_fill_gap,
                                 default_availability, get_avail)
from causalgate.data.planted import ReferencePipeline, LeakSwitches, expanding_zscore
from causalgate.audit.fpit import fpit, sample_probe_times, detection_power

DEVICE = "cpu"
PLANTED = dict(name="cg_planted", seed=1, T=1800, N=6)


def _planted():
    return DATASETS.get("cg_planted")(PLANTED).build_panel()


# ----------------------------------------------------------------------------------------------- deliverable 1
def test_planted_panel_valid_and_regimes():
    p = _planted()
    p.validate()
    assert p.meta["raw_features"] and p.meta["n_events"] >= 10, p.meta["n_events"]
    reg = p.meta["regime"]
    assert 0.05 < reg.mean() < 0.6, reg.mean()                       # both variance regimes visited
    for k in ("avail_feat", "avail_label", "avail_text", "universe"):
        assert get_avail(p, k) is not None, k
    e = p.meta["events"][0]
    assert p.avail_text[e["doc_row"], e["asset"], 0] == e["doc_avail"] > e["doc_row"]      # insider note published late
    assert p.avail_label[e["onset_t"], e["asset"]] == e["end_t"]                            # label knowable at event end
    hi_var = np.nanstd(p.price_feat[reg == 1, :, 0]); lo_var = np.nanstd(p.price_feat[reg == 0, :, 0])
    assert hi_var > 2.0 * lo_var, (hi_var, lo_var)


def test_pitview_masks_late_data():
    p = _planted()
    e = p.meta["events"][0]
    t_mid = e["doc_row"] + 1                       # after the content date, before publication
    v = PITView(p, t_mid)
    assert v.price_feat.shape[0] == t_mid + 1
    assert not v.text_mask[e["doc_row"], e["asset"], 0], "late document must be hidden before its availability"
    assert np.isnan(v.labels[e["onset_t"], e["asset"]]) if e["onset_t"] <= t_mid else True
    v2 = PITView(p, e["doc_avail"])
    assert v2.text_mask[e["doc_row"], e["asset"], 0], "document visible from its availability bar"
    v3 = PITView(p, e["end_t"])
    assert v3.labels[e["onset_t"], e["asset"]] == 1.0
    # universe as-of
    assert v.universe.shape == (p.N,) and v.universe.all()


def test_filtration_guard_raises():
    class Scaler(AsOfStamped):
        pass
    g = FiltrationGuard(strict=True)
    s = g.register(Scaler())
    s.stamp(500)
    s.check(600)                                   # fine: used after its as-of time
    try:
        s.check(400)
        raise AssertionError("expected FiltrationError")
    except FiltrationError:
        pass
    g2 = FiltrationGuard(strict=False)
    s2 = g2.register(Scaler()); s2.stamp(500); s2.check(400)
    g2.assert_causal()
    assert len(g2.violations) == 1 and "as_of=500" in g2.violations[0]


def test_expanding_zscore_is_causal():
    x = np.random.default_rng(0).standard_normal((300, 2))
    z = expanding_zscore(x, warmup=20)
    assert np.isnan(z[:19]).all() and np.isfinite(z[25:]).all()
    x2 = x.copy(); x2[200:] += 100.0
    z2 = expanding_zscore(x2, warmup=20)
    assert np.allclose(z[:200], z2[:200], equal_nan=True) and not np.allclose(z[200:], z2[200:])


def test_perturbation_preserves_past():
    p = _planted()
    rng = np.random.default_rng(0)
    t = 900
    for fam in ("noise", "shuffle", "splice", "delete"):
        q = perturb_future(p, t, fam, rng)
        assert np.array_equal(q.mid[: t + 1], p.mid[: t + 1]) and np.array_equal(q.price_feat[: t + 1], p.price_feat[: t + 1])
        assert np.isclose(q.mid[t], p.mid[t]).all()
        assert not np.array_equal(q.mid[t + 1 :], p.mid[t + 1 :]), fam
        early_docs = (p.avail_text <= t) & p.text_mask
        assert np.array_equal(q.text[early_docs], p.text[early_docs]), fam
        known = p.avail_label <= t
        assert np.array_equal(q.labels[known], p.labels[known]), fam
    q = perturb_fill_gap(p, t, 1, rng)
    assert np.array_equal(q.mid[: t + 1], p.mid[: t + 1]) and not np.allclose(q.mid[t + 1], p.mid[t + 1])
    assert np.allclose(np.log(q.mid[t + 3]) - np.log(q.mid[t + 2]), np.log(p.mid[t + 3]) - np.log(p.mid[t + 2]))


def test_fpit_flags_each_planted_leak_and_passes_causal():
    p = _planted()
    test_start = 1200
    onsets = [e["onset_t"] for e in p.meta["events"] if test_start + 10 <= e["onset_t"] < p.T - 40]
    probes = sample_probe_times(test_start + 5, p.T - 40, 12, seed=0, near=onsets, frac_near=0.5)
    uniform = sample_probe_times(test_start + 5, p.T - 40, 12, seed=0)
    expected_first = {"global_norm": "normaliser", "test_quantile_threshold": "threshold", "label_time_join": "text_join"}
    report = {}
    for leak in (None, "global_norm", "test_quantile_threshold", "same_bar_fill", "label_time_join"):
        sw = LeakSwitches(**({leak: True} if leak else {}))
        pipe = ReferencePipeline(sw, latency=1, hold=5, alpha=0.05, warmup=60, window=250, test_start=test_start)
        t0 = time.time()
        res = fpit(pipe, p, probes, seed=0, keep_details=False)
        report[leak or "causal"] = (res.violation_rate, res.fill_gap_violation_rate, dict(res.first_violating_operator), time.time() - t0)
        if leak is None:
            assert res.passed, res.summary()
        elif leak == "same_bar_fill":
            assert res.violation_rate == 0.0 and res.fill_gap_violation_rate > 0.5, res.summary()
        else:
            floor = 0.3 if leak == "label_time_join" else 0.5      # the join leak is sparse in time (only while a late doc is read)
            assert res.violation_rate > floor, (leak, res.summary())
            top = max(res.first_violating_operator, key=res.first_violating_operator.get)
            assert top == expected_first[leak], (leak, res.first_violating_operator)
    for k, (vr, fg, ops, sec) in report.items():
        print(f"    FPIT {k:24s} violation_rate={vr:.2f} fill_gap={fg:.2f} first_op={ops} ({sec:.1f}s)")
    # power comparison for the sparse leak: uniform vs event-adjacent probes (E4 ingredient)
    pipe = ReferencePipeline(LeakSwitches(label_time_join=True), latency=1, hold=5, alpha=0.05, warmup=60, window=250, test_start=test_start)
    r_uni = fpit(pipe, p, uniform, seed=0, keep_details=False, fill_gap=False)
    print(f"    FPIT label_time_join power: uniform probes {r_uni.violation_rate:.2f} vs stratified {report['label_time_join'][0]:.2f}; "
          f"Prop.5 bound after 12 uniform probes = {detection_power(max(r_uni.violation_rate, 1e-3), 12):.2f}")
    assert detection_power(0.5, 8) > 0.99


# ----------------------------------------------------------------------------------------------- deliverable 2
def test_stream_normalisers_causal():
    from causalgate.data.stream import Welford, EWMA, P2Quantile, StreamingNormaliser, OfflineScaler
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 3)) * np.array([1.0, 2.0, 0.5]) + np.array([0.0, 1.0, -1.0])
    w = Welford(3)
    for t in range(2000):
        w.update(x[t], t)
    assert np.allclose(w.mean, x.mean(0), atol=1e-9) and np.allclose(w.var, x.var(0, ddof=1), atol=1e-9) and w.as_of == 1999
    e = EWMA(3, halflife=50)
    for t in range(2000):
        e.update(x[t])
    assert np.all(np.isfinite(e.std)) and np.allclose(e.std, x.std(0), rtol=0.35)
    q = P2Quantile(3, 0.5)
    for t in range(2000):
        q.update(x[t])
    assert np.allclose(q.value, np.median(x, 0), atol=0.12), (q.value, np.median(x, 0))
    for mode in ("welford", "ewma", "robust"):
        Z = StreamingNormaliser(mode, warmup=30, halflife=100).transform_series(x)
        assert np.isnan(Z[:30]).all() and np.isfinite(Z[31:]).all(), mode
        x2 = x.copy(); x2[1000:] += 50.0
        Z2 = StreamingNormaliser(mode, warmup=30, halflife=100).transform_series(x2)
        assert np.allclose(Z[:1000], Z2[:1000], equal_nan=True), mode          # causal: future change leaves the past untouched
        assert abs(np.nanmean(Z[200:])) < 0.3 and 0.5 < np.nanstd(Z[200:]) < 1.5, mode
    sc = OfflineScaler().fit(x, rows=np.arange(2000))
    assert sc.as_of == 1999
    try:
        sc.transform(x[:10], t=10); raise AssertionError("expected FiltrationError")
    except FiltrationError:
        pass


def test_gate_alarm_rate_under_variance_switch():
    from causalgate.models.gate import FixedQuantileGate, ACIGate, DelayedACIGate, POTGate, alpha_from_cost_budget
    rng = np.random.default_rng(1)
    T, switch, alpha = 4000, 2000, 0.05
    sig = np.where(np.arange(T) < switch, 1.0, 3.0)
    s = np.abs(rng.standard_normal(T) * sig)
    calib, stream = s[:500], s[500:]
    rates = {}
    for name, g in (("fixed", FixedQuantileGate(alpha)), ("pot", POTGate(alpha, window=500, refit_every=20)),
                    ("aci", ACIGate(alpha, gamma=0.01, window=500)), ("delayed_aci", DelayedACIGate(alpha, gamma=0.01, window=500, tau=6))):
        g.calibrate(calib, t_end=499)
        r = g.run(stream, t0=500)
        post = float(np.nanmean(r["alarm"][switch - 500 :]))
        rates[name] = (r["realised_rate"], post, r["rate_error"])
        print(f"    gate {name:12s} realised={r['realised_rate']:.3f} post-switch={post:.3f} |err|={r['rate_error']:.3f} (alpha={alpha})")
    assert rates["fixed"][1] > 0.15, rates                       # a fixed threshold over-fires after the variance switch
    for name in ("aci", "delayed_aci"):
        assert rates[name][2] <= 0.02, (name, rates[name])        # long-run alarm rate within +-0.02 of alpha
        assert abs(rates[name][1] - alpha) <= 0.02, (name, rates[name])
    assert abs(alpha_from_cost_budget(10.0, 500.0, 252, 4) - 500.0 / 10.0 / (252 * 4)) < 1e-12


def test_scorers_fit_score_and_tcn_is_causal():
    import torch
    from causalgate.models.scorer import CausalTCN, make_scorer
    from causalgate.data.stream import streaming_zscore
    set_seed(0)
    p = _planted()
    Z = streaming_zscore(p.price_feat.astype(np.float64), "welford", 60).astype(np.float32)
    L = 32
    tt = np.arange(300, 900, 2); ii = np.arange(p.N)
    idx = tt[:, None] - L + 1 + np.arange(L)[None, :]
    X = np.concatenate([Z[idx, i] for i in ii])                                  # (n, L, F)
    net = CausalTCN(Z.shape[-1], hidden=16, n_layers=3).eval()        # eval: no dropout noise in the causality check
    x = torch.as_tensor(np.nan_to_num(X[:8]))
    x2 = x.clone(); x2[:, 20:] += 5.0
    with torch.no_grad():
        o1, o2 = net(x), net(x2)
    assert torch.allclose(o1[:, :20], o2[:, :20], atol=1e-6) and not torch.allclose(o1[:, 20:], o2[:, 20:])
    for kind in ("tcn", "ae", "lmp"):
        sc = make_scorer({"type": kind, "epochs": 2, "m": 8, "hidden": 16, "n_layers": 3}, Z.shape[-1], L, torch.device(DEVICE))
        sc.fit(X, t_end=899)
        s = sc.score(X[:64])
        assert s.shape == (64,) and np.isfinite(s).all() and sc.as_of == 899, kind
        try:
            sc.check(800); raise AssertionError("scorer used before its as-of time must raise")
        except FiltrationError:
            pass


def test_causalgate_end_to_end_alarm_rate_and_replay():
    import torch
    from warn2trade.data import EventDataset, build_loader, WalkForwardSplitter
    from warn2trade.backtest import CostModel, EventBacktester
    from warn2trade.train import evaluate_outputs
    from causalgate.models.causalgate import CausalGate
    set_seed(42)
    p = _planted()
    horizons = list(p.meta["horizons"]); latency = 1
    fold = next(iter(WalkForwardSplitter(p.T, 900, 200, 500, purge=max(horizons) + latency + 4, embargo=10, max_folds=1)))
    mk = lambda idx: EventDataset(p, idx, 48, horizons, latency, 4, 60)
    ld = {k: build_loader(mk(getattr(fold, k)), 512, shuffle=(k == "train"), seed=42) for k in ("train", "val", "test")}
    cfg = {"panel": p, "lookback": 48, "latency": latency, "hold_index": 2, "n_exec": 4, "w_max": 1.0, "bars_per_year": 252,
           "normaliser": {"mode": "welford", "warmup": 60}, "scorer": {"type": "tcn", "epochs": 2, "hidden": 16, "n_layers": 3},
           "gate": {"type": "delayed_aci", "alpha": 0.05, "gamma": 0.02, "window": 500}, "policy": {"type": "momentum"}}
    bl = CausalGate(cfg, p.dims(), horizons, torch.device(DEVICE))
    t0 = time.time()
    bl.fit(ld["train"], ld["val"])
    fit_s = time.time() - t0
    assert bl.gate.tau == horizons[2] + latency and bl.scorer.as_of <= int(fold.train[-1]) and bl.gate.as_of <= int(fold.val[-1])
    bt = EventBacktester(horizons, latency, CostModel(5.0))
    outs, frames = [], []
    for b in ld["test"]:
        o = bl.predict(b)
        outs.append({k: v.cpu().numpy() for k, v in o.items()})
        frames.append(bt.collect(o, b))
    import pandas as pd
    trades = pd.concat(frames, ignore_index=True)
    assert set(outs[0]) >= {"a_prob", "w", "hold_p", "kappa_p", "abstain"}
    assert outs[0]["hold_p"].shape[1] == len(horizons) and outs[0]["kappa_p"].shape[1] == 4
    alarm = np.concatenate([1.0 - o["abstain"] for o in outs])
    scored = np.concatenate([o["score"] for o in outs]) > -1.0
    rate = float(alarm[scored].mean())
    m = evaluate_outputs(trades, 252, len(fold.test), int(fold.test[0]), None, p.N)
    print(f"    CausalGate planted fold0: fit {fit_s:.1f}s, realised alarm rate {rate:.3f} (alpha 0.05), trades {m['n_trades']}, "
          f"AUC_PR {m['AUC_PR']:.3f}, SR {m['SR']:.2f}, FG violations {len(bl.guard.violations)}")
    assert abs(rate - 0.05) <= 0.02, rate
    assert m["n_trades"] > 0 and np.isfinite(m["AUC_PR"])
    # determinism: a second pass over the same window restarts the gate from its snapshot -> identical decisions
    # (the deployability-gap check proper is the prequential replay in ProtocolRunner.prequential_replay)
    outs2 = [{k: v.cpu().numpy() for k, v in bl.predict(b).items()} for b in ld["test"]]
    for o1, o2 in zip(outs, outs2):
        assert np.array_equal(o1["w"], o2["w"]) and np.array_equal(o1["abstain"], o2["abstain"])
    bl.guard.assert_causal()


# ----------------------------------------------------------------------------------------------- deliverable 5
class _AbsZ:
    """Trivial score-to-trade baseline used to exercise the protocol runner and the FPIT adapter (not a method)."""
    pass


def _register_absz():
    from causalgate.baselines.common import ScoreToTradeBaseline, register
    from warn2trade.utils import BASELINES
    if "cg_absz_test" in BASELINES.keys():
        return
    @register("cg_absz_test")
    class AbsZ(ScoreToTradeBaseline):
        name = "cg_absz_test"
        def _fit_model(self, X, t_end): pass
        def _score(self, X):
            return np.abs(np.nan_to_num(X[:, -1, 0])) + np.clip(np.nan_to_num(X[:, -1, 1]), 0, None) + np.where(np.isnan(X[:, -1, 0]), np.nan, 0.0)


def test_protocol_presets_hash_and_one_switch():
    from causalgate.eval.protocols import PRESETS, one_switch, all_protocols, Protocol
    assert PRESETS["causal"].offline_channels == [] and PRESETS["published"].offline_channels[:3] == ["L1", "L2", "L3"]
    assert set(PRESETS["offline_all"].offline_channels) == {f"L{i}" for i in range(1, 9)}
    for ch in ("L1", "L2", "L3", "L4", "L5", "L6", "L7"):
        assert one_switch(ch).offline_channels == [ch], ch
    assert "causal+L8" not in all_protocols(), "L8 is an audit-card flag, not a one-switch protocol"
    # the yaml mirrors must match the code (hash AND derived offline_channels)
    import yaml
    mdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "causalgate", "protocols")
    for k, v in all_protocols().items():
        y = yaml.safe_load(open(os.path.join(mdir, k.replace("+", "_") + ".yaml")))
        assert y["hash"] == v.hash() and list(y["offline_channels"]) == v.offline_channels, k
    h = {k: v.hash() for k, v in all_protocols().items()}
    assert len(set(h.values())) == len(h), "protocol hashes must be distinct"
    assert Protocol("x").hash() == PRESETS["causal"].hash(), "hash ignores the name"


def test_protocol_runner_offline_vs_causal_on_planted():
    import torch
    from causalgate.eval.protocols import ProtocolRunner, PRESETS, one_switch
    _register_absz()
    set_seed(42)
    p = _planted()
    H = list(p.meta["horizons"])
    out = tempfile.mkdtemp(prefix="cg_runner_test_", dir=os.environ.get("CG_TEST_TMP"))
    R = ProtocolRunner(p, H, 252, {"lookback": 48, "latency": 1, "exec_max": 4, "vol_window": 60, "replay_max_bars": 60}, {"train": 900, "val": 200, "test": 500, "embargo": 10},
                       {"commission_bps": 5.0}, torch.device(DEVICE), out, 42, "cg_planted")
    bcfg = {"hold_index": 2, "n_exec": 4, "w_max": 1.0, "gate": {"alpha": 0.05, "gamma": 0.02, "window": 500}}
    res = {}
    for name, pr in (("causal", PRESETS["causal"]), ("walkforward", PRESETS["walkforward"]), ("offline_all", PRESETS["offline_all"]), ("L7", one_switch("L7"))):
        r = R.run("cg_absz_test", bcfg, pr, max_folds=1, alpha=0.05)
        res[name] = r["agg"]; f0 = r["folds"][0]
        assert np.isfinite(r["agg"]["SR"]) and r["agg"]["n_trades"] > 0 and np.isfinite(r["agg"]["TS_AUC"]) and np.isfinite(r["agg"]["VUS_PR"]), name
        assert f0["mode"] == ("causal" if name in ("causal", "L7") else "published"), (name, f0["mode"])
        if name in ("causal", "L7"):
            assert f0["deployability_gap_zero"] is True and not f0["guard_violations"], name
        print(f"    runner {name:12s} SR={r['agg']['SR']:.2f} AUC_PR={r['agg']['AUC_PR']:.3f} alarm={r['agg'].get('alarm_rate', float('nan')):.3f} trades={r['agg']['n_trades']:.0f}")
    assert abs(res["causal"]["alarm_rate"] - 0.05) <= 0.02
    assert res["L7"]["SR"] > res["causal"]["SR"], "same-bar fill must carry a positive leakage premium on the planted panel"
    fp = os.path.join(out, "cg_planted", "cg_absz_test", "causal", "seed42", "fold0.json")
    import json
    saved = json.load(open(fp))
    assert saved["protocol_hash"] == PRESETS["causal"].hash() and saved["seed"] == 42 and saved["fold"] == 0 and saved["mode"] == "causal"
    assert saved["replay"]["kind"] == "prequential_refit_bar_by_bar" and saved["replay"]["n_bars"] == 60


def test_fpit_adapter_flags_offline_baseline_and_passes_causal():
    import torch
    from causalgate.eval.protocols import ProtocolRunner, PRESETS, one_switch
    from causalgate.audit.adapter import EventBaselinePipeline
    _register_absz()
    set_seed(42)
    p = _planted()
    H = list(p.meta["horizons"])
    R = ProtocolRunner(p, H, 252, {"lookback": 48, "latency": 1, "exec_max": 4}, {"train": 900, "val": 200, "test": 500, "embargo": 10}, {}, torch.device(DEVICE), os.environ.get("CG_TEST_TMP", tempfile.gettempdir()), 42, "cg_planted")
    fold = R.folds(PRESETS["causal"], 1)[0]
    from warn2trade.utils import BASELINES
    cls = BASELINES.get("cg_absz_test")
    onsets = [e["onset_t"] for e in p.meta["events"]]
    probes = sample_probe_times(int(fold.test[0]) + 5, int(fold.test[-1]) - 20, 6, seed=0, near=onsets)
    verdict = {}
    for name, pr in (("causal", PRESETS["causal"]), ("L1", one_switch("L1")), ("L2", one_switch("L2"))):
        cfg = R.baseline_cfg({"hold_index": 2, "n_exec": 4, "w_max": 1.0, "gate": {"alpha": 0.05, "gamma": 0.02, "window": 500}}, pr, fold, p, cls); cfg.pop("panel")
        pipe = EventBaselinePipeline("cg_absz_test", cfg, fold, H, 48, 1, 4, 2, torch.device(DEVICE), 42, refit=True)
        res = fpit(pipe, p, probes, families=("noise", "shuffle"), seed=0, keep_details=False)
        verdict[name] = (res.passed, res.violation_rate, dict(res.first_violating_operator))
        print(f"    FPIT via adapter {name:7s} passed={res.passed} violation_rate={res.violation_rate:.2f} first_op={dict(res.first_violating_operator)}")
    assert verdict["causal"][0], verdict
    assert not verdict["L1"][0] and max(verdict["L1"][2], key=verdict["L1"][2].get) == "normaliser", verdict
    assert not verdict["L2"][0] and max(verdict["L2"][2], key=verdict["L2"][2].get) == "threshold", verdict


# ----------------------------------------------------------------------------------------------- runner
def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-k", default=None, help="comma-separated substring filter on test names")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device
    import inspect, importlib, glob
    fns = [f for n, f in list(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    fns = sorted(fns, key=lambda f: f.__code__.co_firstlineno)
    parts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "causalgate_parts")
    for path in sorted(glob.glob(os.path.join(parts_dir, "test_*.py"))):      # sub-task test modules
        mod = importlib.import_module("tests.causalgate_parts." + os.path.basename(path)[:-3])
        part = [f for n, f in vars(mod).items() if n.startswith("test_") and inspect.isfunction(f)]
        fns += sorted(part, key=lambda f: f.__code__.co_firstlineno)
    if args.k:
        keys = [k for k in args.k.split(",") if k]
        fns = [f for f in fns if any(k in f.__name__ for k in keys)]
    set_seed(0)
    t_all = time.time()
    for f in fns:
        t0 = time.time()
        f()
        print(f"PASS {f.__name__} ({time.time() - t0:.1f}s)")
    print(f"{len(fns)} tests passed in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    _main()

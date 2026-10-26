"""Regression tests for the review fixes of the audit stack (perturb_future, adapter, fpit, plausibility, placebo,
date-shift, card, default availability). Synthetic planted panel only.

Run:  python3 tests/test_causalgate.py --device cpu -k review_audit
"""
from __future__ import annotations
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from warn2trade.utils import DATASETS, set_seed
import causalgate.data.planted  # noqa: F401
from causalgate.data.pit import perturb_future, default_availability, clone_panel, PERTURBATION_FAMILIES
from causalgate.audit.fpit import fpit, _delta, sample_probe_times

_P = dict(name="cg_planted", seed=1, T=1800, N=6)


def _panel(T=1800, N=6, lob=True):
    p = DATASETS.get("cg_planted")(dict(_P, T=T, N=N)).build_panel()
    if lob and p.lob is None:
        rng = np.random.default_rng(3)
        p.lob = rng.standard_normal((p.T, p.N, 4)).astype(np.float32)
    # non-constant tradability and as-of membership (shuffle / splice of a constant column is a no-op by construction)
    rng = np.random.default_rng(4)
    p.tradable = rng.random((p.T, p.N)) < 0.8
    default_availability(p)
    p.universe = rng.random((p.T, p.N)) < 0.7
    return p


def test_review_audit_perturb_soundness_every_field():
    p = _panel()
    default_availability(p)
    t = 900
    for fam in PERTURBATION_FAMILIES:
        q = perturb_future(p, t, fam, np.random.default_rng(1))
        # soundness: rows with availability <= t bit-identical, for every field
        for k in ("mid", "price_feat", "liq", "lob", "tradable", "universe"):
            a, b = getattr(p, k), getattr(q, k)
            known = p.avail_feat[: t + 1] <= t
            assert np.array_equal(a[: t + 1][known], b[: t + 1][known]), (fam, k)
        kl = p.avail_label <= t
        assert np.array_equal(p.labels[kl], q.labels[kl]) and np.array_equal(p.bars_to_peak[kl], q.bars_to_peak[kl], equal_nan=True), fam
        kd = p.avail_text <= t
        for k in ("text", "text_mask", "text_age"):
            assert np.array_equal(getattr(p, k)[kd], getattr(q, k)[kd]), (fam, k)
        if p.graph is not None:
            kg = p.avail_graph <= t
            assert np.array_equal(p.graph[kg], q.graph[kg]) and np.array_equal(p.graph_mask[kg], q.graph_mask[kg]), fam
        # power: every family perturbs the future of lob, tradable and universe (old code: lob never, universe only delete)
        assert not np.array_equal(p.lob[t + 1 :], q.lob[t + 1 :], equal_nan=True), fam
        assert not np.array_equal(p.tradable[t + 1 :], q.tradable[t + 1 :]), fam
        assert not np.array_equal(p.universe[t + 1 :], q.universe[t + 1 :]), fam
        late_docs = (p.avail_text > t) & p.text_mask
        if late_docs.sum() > 1 and fam == "noise":
            assert not np.array_equal(p.text_age[late_docs], q.text_age[late_docs]), fam
        if late_docs.sum() > 1 and fam == "delete":
            assert not q.text_mask[late_docs].any() and (q.text_age[late_docs] == 0).all(), fam
    # splice at t < 2 is not a no-op any more
    q = perturb_future(p, 1, "splice", np.random.default_rng(0))
    assert not np.array_equal(p.price_feat[2:], q.price_feat[2:], equal_nan=True)


def test_review_audit_universe_lob_leak_detected_by_every_family():
    p = _panel(T=600, N=4)

    class Peek:
        operators = ("policy",); latency = 1
        def __init__(self, field): self.f = field
        def run(self, pn):
            T, N = pn.mid.shape
            x = np.zeros((T, N))
            src = np.asarray(getattr(pn, self.f), dtype=np.float64)
            src = src.reshape(T, N, -1).sum(-1)
            x[:-1] = src[1:]                                    # reads bar t+1 at t
            return {"score": x, "weight": x, "policy": x}

    for field in ("lob", "universe", "tradable"):
        for fam in PERTURBATION_FAMILIES:
            r = fpit(Peek(field), p, [200, 300, 400], families=(fam,), seed=0, fill_gap=False)
            assert r.passed is False and r.violation_rate > 0, (field, fam)


def test_review_audit_delta_inf_and_scalar():
    a = np.array([[np.inf, 1.0, np.nan], [0.0, 0.0, 0.0]])
    assert _delta(a, a.copy(), 0).max() == 0.0
    b = a.copy(); b[0, 0] = -np.inf
    assert np.isinf(_delta(a, b, 0)[0])
    assert _delta(np.array([np.inf, 2.0]), np.array([np.inf, 2.0]), 0).tolist() == [0.0]


def test_review_audit_fpit_l7_not_run_and_empty_probes():
    p = _panel(T=400, N=3, lob=False)

    class NoPnl:
        operators = ("score",); latency = 1
        def run(self, pn):
            z = np.zeros(pn.mid.shape); return {"score": z, "weight": z}
    r = fpit(NoPnl(), p, [100, 150], families=("noise",), seed=0)
    assert r.passed is None and not r.l7_tested and r.summary()["fill_gap"] == "not run"
    r = fpit(NoPnl(), p, [], families=("noise",), seed=0)
    assert r.passed is None and not np.isfinite(r.violation_rate)

    class ZeroW:
        operators = ("score",); latency = 1
        def run(self, pn):
            z = np.zeros(pn.mid.shape); return {"score": z, "weight": z, "pnl": z}
    r = fpit(ZeroW(), p, [], families=("noise",), seed=0)                    # old: ValueError min() of empty
    assert r.passed is None


def test_review_audit_sample_probe_times_returns_n():
    near = list(range(100, 110))                               # dense anchors -> colliding draws
    for s in range(20):
        pr = sample_probe_times(50, 500, 12, seed=s, near=near, frac_near=1.0, span=3)
        assert len(pr) == 12 and len(set(pr.tolist())) == 12, (s, pr)


def _adapter_setup():
    import torch
    from causalgate.eval.protocols import ProtocolRunner, PRESETS
    from warn2trade.utils import BASELINES
    import tests.test_causalgate as TC
    TC._register_absz()
    set_seed(42)
    p = _panel(T=1300, N=4, lob=False)
    H = list(p.meta["horizons"])
    R = ProtocolRunner(p, H, 252, {"lookback": 32, "latency": 1, "exec_max": 4}, {"train": 600, "val": 150, "test": 350, "embargo": 10},
                       {}, torch.device("cpu"), os.environ.get("CG_TEST_TMP", tempfile.gettempdir()), 42, "cg_planted")
    fold = R.folds(PRESETS["causal"], 1)[0]
    return p, H, R, fold, BASELINES.get("cg_absz_test")


def test_review_audit_adapter_fill_threshold_refit():
    import torch
    from causalgate.eval.protocols import PRESETS, one_switch
    from causalgate.audit.adapter import EventBaselinePipeline
    p, H, R, fold, cls = _adapter_setup()
    bcfg = {"hold_index": 2, "n_exec": 4, "w_max": 1.0, "gate": {"alpha": 0.1, "gamma": 0.02, "window": 300}}
    probes = sample_probe_times(int(fold.test[0]) + 5, int(fold.test[-1]) - 20, 4, seed=0)
    res = {}
    for name, pr in (("causal", PRESETS["causal"]), ("L7", one_switch("L7"))):
        cfg = R.baseline_cfg(bcfg, pr, fold, p, cls); cfg.pop("panel")
        pipe = EventBaselinePipeline("cg_absz_test", cfg, fold, H, 32, 1, 4, 2, torch.device("cpu"), 42, refit=False, fill=pr.fill)
        base = pipe.run(p)
        te = np.asarray(fold.test)
        # gate-mode threshold is recorded (old: all NaN)
        assert np.isfinite(base["thr"][te]).any(), name
        # refit=False is reproducible
        again = pipe.run(p)
        for k in ("score", "thr", "weight"):
            assert np.array_equal(base[k], again[k], equal_nan=True), (name, k)
        res[name] = fpit(pipe, p, probes, families=("noise",), seed=0, base=base, keep_details=False)
    assert res["causal"].passed is True and res["causal"].fill_gap_violation_rate == 0.0, res["causal"].summary()
    assert res["L7"].l7_tested and res["L7"].fill_gap_violation_rate > 0 and res["L7"].passed is False, res["L7"].summary()


def test_review_audit_plausibility_haircut_and_dsr_units():
    from causalgate.audit.plausibility import sharpe_plausibility, haircut_t
    rng = np.random.default_rng(0)
    r = 0.01 + 0.01 * rng.standard_normal(2000)                # t-stat ~ 45: old haircut = inf
    o = sharpe_plausibility(r, 252, n_trials=10, sr_var_across_trials=0.25)
    assert np.isfinite(o["SR_haircut"]) and 0 < o["SR_haircut"] <= o["SR"]
    o1 = sharpe_plausibility(r, 252, n_trials=1)
    assert np.isclose(o1["SR_haircut"], o1["SR"], rtol=1e-6)
    assert np.isfinite(haircut_t(60.0, 100)) and 0 < haircut_t(60.0, 100) < 60.0 and haircut_t(0.5, 100) == 0.0
    # DSR units: annualised SR 3 on 1000 daily bars, 20 trials, annualised SR std 0.5 -> clearly significant
    r2 = 3.0 / np.sqrt(252) * 0.01 + 0.01 * rng.standard_normal(1000)
    r2 = (r2 - r2.mean()) / r2.std() * 0.01 + 3.0 / np.sqrt(252) * 0.01
    o2 = sharpe_plausibility(r2, 252, n_trials=20, sr_var_across_trials=0.25)
    assert o2["DSR"] > 0.9, o2                                 # old: per-bar 0.25 -> DSR ~ 0


def test_review_audit_placebo_surrogates():
    from causalgate.audit.placebo import surrogate_panel, placebo_distribution
    p = _panel(T=500, N=3, lob=False)
    p.mid = p.mid.astype(np.float64).copy()
    p.mid[200:260, 0] = np.nan                                 # suspension after keep_from
    p.mid[:150, 1] = np.nan                                    # listed after keep_from
    rng = np.random.default_rng(0)
    for m in ("sign_flip", "block_bootstrap", "garch_sign"):
        q = surrogate_panel(p, np.random.default_rng(1), m, keep_from=100)
        assert np.array_equal(np.isnan(q.mid), np.isnan(p.mid)), m
        assert np.array_equal(q.mid[:101], p.mid[:101], equal_nan=True), m
        assert np.isfinite(q.mid[260:, 0]).all() and np.isfinite(q.mid[150:, 1]).all(), m
    a = surrogate_panel(p, np.random.default_rng(5), "sign_flip", keep_from=100)
    b = surrogate_panel(p, np.random.default_rng(5), "garch_sign", keep_from=100)
    ra = np.abs(np.diff(np.log(a.mid[101:, 2]))); rb = np.abs(np.diff(np.log(b.mid[101:, 2])))
    rp = np.abs(np.diff(np.log(p.mid[101:, 2])))
    assert np.allclose(ra, rp) and not np.allclose(rb, rp)    # garch_sign is no longer sign_flip
    d = placebo_distribution(lambda pn: float(np.nanmean(np.diff(np.log(pn.mid[300:, 2])))) * 0 + 1.0, p, n=6, seed=0, keep_from=100)
    assert d["placebo_leak_flag"] is True and len(d["placebo"]) == 6
    d = placebo_distribution(lambda pn: float(np.nansum(np.diff(np.log(pn.mid[101:, 2])))), p, n=6, seed=0, keep_from=100)
    assert np.isfinite(d["real"]) and all(np.isfinite(d["placebo"]))


def test_review_audit_date_shift():
    from causalgate.audit.date_shift import shifted_bar_returns, date_shift_profile
    rng = np.random.default_rng(0)
    T, N = 600, 2
    lr = 0.01 * rng.standard_normal((T, N)); mid = 100 * np.exp(np.cumsum(lr, 0))
    w = np.zeros((T, N)); w[:-2] = np.sign(lr[2:])            # knows the return of the bar t+1 -> t+2 (latency 1, hold 1)
    r0 = shifted_bar_returns(w, mid, 1, 1, 0)
    assert np.isclose(r0.sum(), np.abs(lr[2:]).sum() / N)
    prof = date_shift_profile(w, mid, 1, 1, 252)
    assert prof["profile"][0] > 5 and prof["profile"][1] < 2 and prof["collapse_flag"] is True


def test_review_audit_card_incomplete_and_verdicts():
    from causalgate.audit.card import LeakageAuditCard
    c = LeakageAuditCard("m", "d")
    assert c.verdict().startswith("INCOMPLETE")
    ok = dict(fpit={"n_probes": 4, "violation_rate": 0.0, "fill_gap_violation_rate": 0.0, "l7_tested": True, "fill_gap_requested": True},
              date_shift={"collapse_flag": False}, placebo={"placebo_leak_flag": False}, plausibility={"implausible_sharpe": False},
              pretrained={"cutoff_date": None, "ok": True})
    assert LeakageAuditCard("m", "d", **ok).verdict().startswith("CAUSAL")
    bad = dict(ok, fpit=dict(ok["fpit"], violation_rate=float("nan")))
    assert LeakageAuditCard("m", "d", **bad).verdict().startswith("INCOMPLETE")
    nol7 = dict(ok, fpit=dict(ok["fpit"], l7_tested=False, fill_gap_violation_rate=float("nan")))
    c = LeakageAuditCard("m", "d", **nol7)
    assert "L7" in c.verdict() and "not run" in c.to_markdown()
    pre = dict(ok, pretrained={"cutoff_date": "2024-01-01", "ok": None})
    assert LeakageAuditCard("m", "d", **pre).verdict().startswith("INCOMPLETE")
    leak = dict(ok, fpit=dict(ok["fpit"], fill_gap_violation_rate=0.5))
    assert LeakageAuditCard("m", "d", **leak).verdict() == "NOT CAUSAL"


def test_review_audit_default_avail_label_lead():
    p = _panel(T=300, N=2, lob=False)
    q = clone_panel(p)
    for k in ("avail_label",):
        if hasattr(q, k):
            delattr(q, k)
    q.meta = dict(q.meta, horizons=[1, 2, 3], label_rule={"type": "abnormal_move_labels", "horizon": 5, "lead": 3})
    default_availability(q)
    assert (q.avail_label[:, 0] == np.arange(300) + 8).all()

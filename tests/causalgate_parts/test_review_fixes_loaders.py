"""Regression tests for the loader review fixes (findings_E). Every file written here is a FORMAT FIXTURE (placeholder
rows in the real raw layout; not data). Each test fails on the pre-fix behaviour.

Run:  python3 tests/test_causalgate.py --device cpu -k review_fixes_loaders
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from warn2trade.utils import DATASETS
from causalgate.data.pit import NEVER, PITView
from causalgate.data.loaders.common import (liquidity_fields, next_bar_index, rule_labels_and_peaks, event_table_labels,
                                            NAT_NS)
from causalgate.data.loaders.doc_cache import fill_text_slots
import causalgate.data.loaders.csi300_limit as L_csi
import test_loaders_constructed as TC          # fixture builders (format fixtures)


def test_review_fixes_loaders_liquidity_adv_is_causal():
    rng = np.random.default_rng(0)
    T, N, t0 = 80, 4, 40
    close = 50 * np.exp(np.cumsum(rng.standard_normal((T, N)) * 0.01, 0))
    vol = 1e6 * (1 + rng.random((T, N)))
    base = liquidity_fields(close, vol, close * 1.01, close * 0.99)
    vol2 = vol.copy(); vol2[t0 + 1:] *= 50.0                       # perturb the future only
    close2 = close.copy(); close2[t0 + 1:] *= 3.0
    pert = liquidity_fields(close2, vol2, close2 * 1.01, close2 * 0.99)
    assert np.array_equal(base[: t0 + 1], pert[: t0 + 1]), "liq[..., :t0] must not depend on bars after t0"


def test_review_fixes_loaders_fill_text_slots_knowable_first():
    T, N, M, D = 10, 1, 1, 3
    emb = np.eye(D, dtype=np.float32)[:2]
    rows = np.array([5, 5]); asset = np.array([0, 0])
    avail = np.array([5, 6])                       # doc 0 knowable at its row; doc 1 arrives after the close
    pub = np.array([100, 200])                     # doc 1 is newer
    out = fill_text_slots(T, N, M, emb, rows, avail, asset, pub)
    assert out["text_mask"][5, 0, 0] and out["avail_text"][5, 0, 0] == 5 and np.array_equal(out["text"][5, 0, 0], emb[0])
    assert out["stats"]["dropped_overflow"] == 1


def test_review_fixes_loaders_csi_run_end_one_bar_late():
    T, N = 20, 1
    d = np.zeros((T, N), dtype=np.int8); d[5:7, 0] = 1; d[18:, 0] = -1     # run 5..6 ends (observed) at 7; run 18..19 open
    ev = L_csi._runs_to_events(d)
    assert list(ev["end_t"]) == [7, 20] and list(ev["peak_t"]) == [6, 19]
    lab, btp, av = event_table_labels(T, N, ev, 2, [1, 2])
    assert av[6, 0] == 7 and av[3, 0] == 7 and btp[7, 0] == -1
    assert av[19, 0] == NEVER and av[16, 0] == NEVER, "a run still open at the panel end is never seen to end"


def test_review_fixes_loaders_next_bar_index_boundary_and_nat():
    bt = np.array([1000, 2000, 3000], dtype=np.int64)
    ev = np.array([2000, 2001, 1999, NAT_NS, 3001], dtype=np.int64)
    # convention: first close >= event (a doc stamped exactly at a close is knowable at that close), strict 1 ns either side
    assert next_bar_index(bt, ev).tolist() == [1, 2, 1, 3, 3]
    assert next_bar_index(bt, np.array([2000]), side="right").tolist() == [2]


def test_review_fixes_loaders_rule_label_avail_includes_lead():
    rng = np.random.default_rng(1)
    close = 50 * np.exp(np.cumsum(rng.standard_normal((120, 2)) * 0.01, 0))
    _, _, av = rule_labels_and_peaks(close, None, horizons=(1, 2, 3), horizon=5, lead=3)
    assert (av[:, 0] == np.arange(120) + 8).all()


def test_review_fixes_loaders_sp500_gaps_and_open_halt():
    root = TC._build_sp500_fixture()
    bp = os.path.join(root, "bars", "BBB.csv")
    b = pd.read_csv(bp); b.drop(index=range(85, 90)).to_csv(bp, index=False)           # day-2 gap minutes 10..14
    pd.DataFrame([{"ticker": "BBB", "halt_start": "2021-03-01 10:00:00", "halt_end": "2021-03-01 10:05:00"},
                  {"ticker": "CCC", "halt_start": "2021-03-01 10:30:00", "halt_end": ""},              # open-ended halt
                  {"ticker": "AAA", "halt_start": "garbage", "halt_end": "2021-03-01 10:05:00"}]) \
        .to_csv(os.path.join(root, "halts.csv"), index=False)
    p = DATASETS.get("cg_sp500_tick")({"root": root}).build_panel()
    times = pd.to_datetime(p.times, utc=True).tz_convert("America/New_York")
    day = np.asarray(times.strftime("%Y-%m-%d"))
    bbb, ccc, aaa = p.assets.index("BBB"), p.assets.index("CCC"), p.assets.index("AAA")
    d2 = np.flatnonzero(day == "2021-03-02")
    # gap rows: in session, mid forward-filled (finite) but no real bar -> untradable
    assert np.isfinite(p.mid[d2, bbb]).all()
    assert p.meta["n_gap_rows_untradable"] == 5
    gap_rows = d2[10:15]
    assert not p.tradable[gap_rows, bbb].any() and p.tradable[d2[5], bbb]
    # open-ended halt: CCC untradable from 10:30 through the end of 03-01, tradable again on 03-02 (outside the CB)
    t_h = (day == "2021-03-01") & (np.asarray(times.strftime("%H:%M")) >= "10:31")
    assert not p.tradable[t_h, ccc].any() and p.tradable[d2[0], ccc] and p.meta["n_halts_open_end"] == 1
    # unparseable start: skipped, not "everything before the end"
    assert p.meta["n_halts_bad_start"] == 1 and p.tradable[np.flatnonzero(day == "2021-03-01")[:20], aaa].all()


def test_review_fixes_loaders_edgar_nat_dropped():
    root = TC._build_edgar_fixture()
    fp = os.path.join(root, "filings.csv")
    f = pd.read_csv(fp)
    f = pd.concat([f, pd.DataFrame([{"ticker": "BBB", "form": "10-Q", "fyear": 2015, "accepted": "not-a-time", "sic": 7371, "emb_row": 1}])])
    f.to_csv(fp, index=False)
    p = DATASETS.get("cg_edgar")({"root": root}).build_panel()
    assert p.meta["n_filings_bad_ts"] == 1 and p.meta["n_filings"] == 4
    assert not p.meta["decision_mask"][0].any() and not p.text_mask[0].any()


def test_review_fixes_loaders_wsb_text_excludes_crawl_fields_and_nat():
    root, dates, bar = TC._build_wsb_fixture()
    p0 = DATASETS.get("cg_wsb_squeeze")({"root": root}).build_panel()
    rp = os.path.join(root, "reddit_wsb.csv")
    r = pd.read_csv(rp)
    r["score"] = r["score"] * 1000 + 7; r["comms_num"] = r["comms_num"] * 50 + 3          # crawl-time fields only
    r = pd.concat([r, pd.DataFrame([{"title": "$GME again", "score": 5, "id": "a4", "url": "u", "comms_num": 1,
                                     "created": np.nan, "body": "", "timestamp": ""}])], ignore_index=True)
    r.to_csv(rp, index=False)
    p1 = DATASETS.get("cg_wsb_squeeze")({"root": root}).build_panel()
    t = bar["2021-01-27"]
    assert np.array_equal(PITView(p0, t).text, PITView(p1, t).text), "causal text must not depend on crawl-time popularity"
    assert p1.meta["n_posts_bad_ts"] == 1 and not p1.text_mask[0].any()
    pc = DATASETS.get("cg_wsb_squeeze")({"root": root, "text_metadata": "crawl"}).build_panel()
    assert pc.text.shape[-1] == 4 and (pc.avail_text[pc.text_mask] == NEVER).all()

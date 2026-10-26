"""Format-fixture tests for the D4 / D6 / D7 / D9 / D10 CausalGate loaders.

EVERYTHING written to disk here is a FORMAT FIXTURE: a handful of rows in the *real raw file layout* (Yahoo / Qlib bin
/ Polygon csv / Kaggle WSB / FINRA / Elliptic++ column names) whose numbers are deterministic placeholders. They exist
only to exercise parsing, alignment and PIT stamping; they are NOT data and no result derived from them means anything.
Real data is never downloaded here; each loader must raise FileNotFoundError with its recipe on an empty root.

Run:  python3 tests/test_causalgate.py --device cpu -k loaders_constructed
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from warn2trade.utils import DATASETS
from causalgate.data.pit import NEVER, PITView, get_avail
import causalgate.data.loaders.edgar as L_edgar            # noqa: F401  registers cg_edgar
import causalgate.data.loaders.csi300_limit as L_csi       # noqa: F401  registers cg_csi300_limit
import causalgate.data.loaders.sp500_tick as L_sp          # noqa: F401  registers cg_sp500_tick
import causalgate.data.loaders.wsb_squeeze as L_wsb        # noqa: F401  registers cg_wsb_squeeze
import causalgate.data.loaders.fin_graph as L_fg           # noqa: F401  registers cg_fin_graph

_PREF = "/private/tmp/claude-501/-Users-zyy-package-anomalydetection-TradingSignal/cae6fc84-cde9-4354-ba01-4ae1653d3958/scratchpad"


def _fixture_dir(name: str) -> str:
    base = _PREF if os.path.isdir(_PREF) else tempfile.gettempdir()
    d = os.path.join(base, "cg_format_fixtures", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(d, "FORMAT_FIXTURE_NOT_DATA.txt"), "w") as fh:
        fh.write("placeholder rows in the real raw layout; exercises parsing only; not data\n")
    return d


def _empty_root(name: str) -> str:
    return _fixture_dir(name + "_empty")


def _walk(n: int, seed: int, p0: float = 50.0) -> np.ndarray:
    """Deterministic placeholder price path (format fixture, not data)."""
    rng = np.random.default_rng(seed)
    return p0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))


def _yahoo_csv(path: str, dates, close: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"), "Open": close * (1 + 0.002 * rng.standard_normal(len(close))),
                       "High": close * 1.01, "Low": close * 0.99, "Close": close, "Adj Close": close, "Volume": (1e6 * (1 + rng.random(len(close)))).astype(int)})
    df.to_csv(path, index=False)


def _check_ret_col(p) -> None:
    lr = np.full(p.mid.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr[1:] = np.log(p.mid[1:]) - np.log(p.mid[:-1])
    ok = np.isfinite(lr)
    assert p.meta["raw_features"] and p.meta["ret_col"] == 0 and p.meta["feat_names"][0] == "log_ret"
    assert np.allclose(p.price_feat[..., 0][ok], lr[ok], atol=1e-5), "column 0 must be the log return of mid"


def _check_pit_shapes(p, fields) -> None:
    T, N = p.mid.shape
    for f in fields:
        v = get_avail(p, f)
        assert v is not None, f"missing PIT field {f}"
        if f == "avail_text":
            assert v.shape == p.text.shape[:3]
        elif f == "avail_graph":
            assert v.shape == p.graph.shape[:3]
        else:
            assert v.shape == (T, N), (f, v.shape)
    assert set(p.meta["pit_fields"]) >= set(fields)
    for k in ("freq", "bars_per_year", "horizons", "feat_names", "pit_notes", "source", "licence"):
        assert k in p.meta, k


# ================================================================================================= D4 EDGAR
def _build_edgar_fixture() -> str:
    root = _fixture_dir("edgar")
    os.makedirs(os.path.join(root, "prices"))
    dates = pd.bdate_range("2015-01-02", periods=150)
    for k, tk in enumerate(("AAA", "BBB", "CCC")):
        _yahoo_csv(os.path.join(root, "prices", f"{tk}.csv"), dates, _walk(150, 10 + k), 20 + k)
    pd.DataFrame([
        {"ticker": "AAA", "form": "10-K", "fyear": 2014, "accepted": "2015-03-02 16:30:00", "sic": 7372, "emb_row": 0},   # after the close
        {"ticker": "BBB", "form": "10-K", "fyear": 2014, "accepted": "2015-03-02 09:15:00", "sic": 7371, "emb_row": 1},   # before the close
        {"ticker": "CCC", "form": "10-Q", "fyear": 2015, "accepted": "2015-05-05 12:00:00", "sic": 2834, "emb_row": 2},
        {"ticker": "AAA", "form": "10-Q", "fyear": 2015, "accepted": "2015-05-08 17:00:00", "sic": 7372, "emb_row": 3},
        {"ticker": "AAA", "form": "8-K", "fyear": 2015, "accepted": "2015-05-09 17:00:00", "sic": 7372, "emb_row": 3},    # filtered out
    ]).to_csv(os.path.join(root, "filings.csv"), index=False)
    pd.DataFrame([
        {"ticker": "AAA", "fyear": 2014, "misstate": 1, "aaer_release_date": "2015-06-15"},   # released inside the panel
        {"ticker": "BBB", "fyear": 2014, "misstate": 0, "aaer_release_date": ""},
        {"ticker": "CCC", "fyear": 2015, "misstate": 1, "aaer_release_date": ""},             # fraud, release date unknown -> NEVER
    ]).to_csv(os.path.join(root, "aaer.csv"), index=False)
    np.save(os.path.join(root, "embeddings.npy"), np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32))
    return root


def test_loaders_constructed_edgar_recipe_and_pit():
    try:
        DATASETS.get("cg_edgar")({"root": _empty_root("edgar")}).build_panel()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "recipe" in str(e).lower() and "aaer" in str(e).lower()
    # compact EDGAR header datetime is parsed as ET
    ns = L_edgar._parse_ts(pd.Series(["20150227163105"]), "America/New_York")[0]
    assert pd.Timestamp(ns, unit="ns", tz="UTC").tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S") == "2015-02-27 16:31:05"
    root = _build_edgar_fixture()
    p = DATASETS.get("cg_edgar")({"root": root, "horizons": [1, 3, 5, 10]}).build_panel()
    p.validate()
    _check_ret_col(p)
    _check_pit_shapes(p, ["avail_feat", "avail_text", "avail_label", "avail_graph", "universe"])
    T, N = p.mid.shape
    assert T == 150 and p.assets == ["AAA", "BBB", "CCC"]
    dates = pd.DatetimeIndex(pd.to_datetime(p.times, utc=True).tz_convert("America/New_York").date)
    bar = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    a, b, cc = 0, 1, 2
    t_aaa = bar["2015-03-03"]                              # accepted 16:30 on 03-02 -> next session close
    t_bbb = bar["2015-03-02"]                              # accepted 09:15 -> same session close
    assert p.meta["decision_mask"][t_aaa, a] and p.meta["decision_mask"][t_bbb, b] and p.meta["n_filings"] == 4
    assert p.labels[t_aaa, a] == 1 and p.labels[t_bbb, b] == 0
    assert p.avail_label[t_aaa, a] == bar["2015-06-15"] + 1 > t_aaa, "date-only AAER release -> next session close, not the filing"
    t_ccc = bar["2015-05-05"]
    assert p.labels[t_ccc, cc] == 1 and p.avail_label[t_ccc, cc] >= T and p.meta["n_fraud_without_release"] == 1
    assert p.avail_label[t_bbb, b] == t_bbb + 10                      # clean label: t + max(horizon)
    assert p.text_mask[t_aaa, a, 0] and p.avail_text[t_aaa, a, 0] == t_aaa and p.text_age[t_aaa, a, 0] == 0
    assert not p.text_mask[t_aaa - 1, a].any(), "text must not appear before the acceptance bar"
    # SIC-2 graph: AAA (73xx) <-> BBB (73xx); CCC (28xx) has no neighbour
    assert p.graph_mask[t_aaa, a, 0] and p.graph_mask[t_aaa, b, 0] and not p.graph_mask[t_aaa, cc].any()
    assert np.allclose(p.graph[t_aaa, a, 0], np.nan_to_num(p.price_feat[t_aaa, b])) and p.avail_graph[t_aaa, a, 0] == t_aaa
    # PITView: before the release the fraud label is hidden, after it visible
    assert np.isnan(PITView(p, bar["2015-06-01"]).labels[t_aaa, a]) and np.isnan(PITView(p, bar["2015-06-15"]).labels[t_aaa, a]) and PITView(p, bar["2015-06-15"] + 1).labels[t_aaa, a] == 1.0
    assert p.universe.all() and p.tradable.all()


# ================================================================================================= D6 CSI 300
def _write_qlib_bin(path: str, start_idx: int, vals: np.ndarray) -> None:
    np.concatenate([np.array([start_idx], dtype="<f4"), vals.astype("<f4")]).astype("<f4").tofile(path)


def _build_csi_qlib_fixture():
    root = _fixture_dir("csi300_qlib")
    q = os.path.join(root, "qlib_cn")
    os.makedirs(os.path.join(q, "calendars")); os.makedirs(os.path.join(q, "instruments"))
    cal = pd.bdate_range("2021-01-04", periods=150)
    with open(os.path.join(q, "calendars", "day.txt"), "w") as fh:
        fh.write("\n".join(cal.strftime("%Y-%m-%d")) + "\n")
    add_date = cal[70].strftime("%Y-%m-%d")
    with open(os.path.join(q, "instruments", "csi300.txt"), "w") as fh:
        fh.write(f"SH600000\t{cal[0].strftime('%Y-%m-%d')}\t2021-12-31\n")
        fh.write(f"SZ000001\t{cal[0].strftime('%Y-%m-%d')}\t2021-12-31\n")
        fh.write(f"SZ300750\t{add_date}\t2021-12-31\n")                     # ChiNext name added mid-panel (L6)
    rng = np.random.default_rng(3)
    for k, inst in enumerate(("SH600000", "SZ000001", "SZ300750")):
        close = np.round(_walk(150, 30 + k, 10.0), 2).astype(np.float32).astype(np.float64)   # the .bin is float32
        if inst == "SH600000":                                              # two consecutive limit-up closes (10 %)
            close[50] = L_csi._round_half_up_2(close[49] * 1.10)
            close[51] = L_csi._round_half_up_2(close[50] * 1.10)
        if inst == "SZ300750":                                              # one limit-down close (20 %, ChiNext after 2020-08-24)
            close[90] = L_csi._round_half_up_2(close[89] * 0.80)
        d = os.path.join(q, "features", inst.lower()); os.makedirs(d)
        _write_qlib_bin(os.path.join(d, "close.day.bin"), 0, close)
        _write_qlib_bin(os.path.join(d, "open.day.bin"), 0, close * (1 + 0.001 * rng.standard_normal(150)))
        _write_qlib_bin(os.path.join(d, "high.day.bin"), 0, close * 1.01)
        _write_qlib_bin(os.path.join(d, "low.day.bin"), 0, close * 0.99)
        _write_qlib_bin(os.path.join(d, "volume.day.bin"), 0, 1e5 * (1 + rng.random(150)))
        _write_qlib_bin(os.path.join(d, "factor.day.bin"), 0, np.ones(150))
    return root, cal, add_date


def _build_csi_csv_fixture():
    root = _fixture_dir("csi300_csv")
    cal = pd.bdate_range("2021-01-04", periods=60)
    rows = []
    for k, code in enumerate(("600000.SH", "000001.SZ")):
        close = np.round(_walk(60, 40 + k, 20.0), 2)
        pre = np.r_[np.nan, close[:-1]]
        up, dn = np.floor(pre * 1.10 * 100 + 0.5) / 100, np.floor(pre * 0.90 * 100 + 0.5) / 100
        if code == "000001.SZ":
            close[20] = dn[20]                                             # limit-down close, exact up/down columns
        for t in range(60):
            rows.append({"ts_code": code, "trade_date": cal[t].strftime("%Y%m%d"), "open": close[t], "high": close[t] * 1.01,
                         "low": close[t] * 0.99, "close": close[t], "pre_close": pre[t], "vol": 1000 + t, "amount": 20000 + t,
                         "up_limit": up[t], "down_limit": dn[t]})
    pd.DataFrame(rows).to_csv(os.path.join(root, "bars.csv"), index=False)
    pd.DataFrame([{"ts_code": "600000.SH", "in_date": "2021-01-04", "out_date": ""}, {"ts_code": "000001.SZ", "in_date": "2021-02-01", "out_date": ""}]) \
        .to_csv(os.path.join(root, "members.csv"), index=False)
    return root, cal


def test_loaders_constructed_csi300_limit_recipe_and_pit():
    try:
        DATASETS.get("cg_csi300_limit")({"root": _empty_root("csi300")}).build_panel()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "recipe" in str(e).lower() and "qlib" in str(e).lower()
    root, cal, add_date = _build_csi_qlib_fixture()
    p = DATASETS.get("cg_csi300_limit")({"root": root, "lead": 3}).build_panel()
    p.validate()
    _check_ret_col(p)
    _check_pit_shapes(p, ["avail_feat", "avail_label", "universe"])
    T, N = p.mid.shape
    assert T == 150 and p.assets == ["SH600000", "SZ000001", "SZ300750"] and p.meta["limit_mode"] == "exact"
    i0, i2 = 0, 2
    assert p.meta["limit_direction"][50, i0] == 1 and p.meta["limit_direction"][51, i0] == 1 and p.meta["limit_direction"][90, i2] == -1
    assert not p.tradable[50, i0] and not p.tradable[51, i0] and not p.tradable[90, i2] and p.tradable[49, i0]
    assert p.labels[47:52, i0].all() and not p.labels[46, i0] and not p.labels[52, i0]
    assert p.bars_to_peak[50, i0] == 1 and p.bars_to_peak[47, i0] == 4 and p.bars_to_peak[90, i2] == 0
    assert p.avail_label[47, i0] == 52 and p.avail_label[50, i0] == 52 and p.avail_label[90, i2] == 91   # run end seen one bar later
    assert p.avail_label[10, i0] == 10 + max(p.meta["horizons"])
    assert p.meta["n_limit_events"] == 2 and len(p.meta["events"]) == 2
    # universe as-of: the ChiNext name enters on its start date, never before
    t_add = int(np.flatnonzero(cal == pd.Timestamp(add_date))[0])
    assert not p.universe[: t_add, i2].any() and p.universe[t_add:, i2].all() and p.universe[:, i0].all()
    assert PITView(p, t_add - 1).universe[i2] == False and PITView(p, t_add).universe[i2] == True   # noqa: E712
    # CSV export layout with exact up/down limit columns
    root2, cal2 = _build_csi_csv_fixture()
    p2 = DATASETS.get("cg_csi300_limit")({"root": root2}).build_panel()
    p2.validate(); _check_ret_col(p2)
    assert p2.meta["layout"] == "csv" and p2.meta["limit_mode"] == "columns"
    j = p2.assets.index("000001.SZ")
    assert p2.meta["limit_direction"][20, j] == -1 and not p2.tradable[20, j] and p2.labels[17:21, j].all()
    assert not p2.universe[:19, j].any() and p2.universe[20:, j].all()


# ================================================================================================= D7 S&P 500 minute bars
def _build_sp500_fixture():
    root = _fixture_dir("sp500_tick")
    os.makedirs(os.path.join(root, "bars"))
    starts = []
    for day in ("2021-03-01", "2021-03-02"):
        starts += list(pd.date_range(f"{day} 09:30:00", periods=75, freq="1min"))       # bar START times, naive ET
    starts = pd.DatetimeIndex(starts)
    n = len(starts)
    rng = np.random.default_rng(7)
    for k, tk in enumerate(("AAA", "BBB", "CCC")):
        close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.0005))
        if tk == "AAA":                                                   # planted mini flash crash on day 1 at bar 30
            base = close[30]
            close[29] = base * 0.999                                                        # up-tick into bar 30: the run starts at 30
            close[31], close[32], close[33] = base * 0.996, base * 0.992, base * 0.988      # -1.2 % monotone over 3 bars
            close[34], close[35] = base * 0.993, base * 0.997                                # reverts > 50 % within 2 bars
            close[36:] = close[35] * close[36:] / close[36]
        df = pd.DataFrame({"timestamp": starts.strftime("%Y-%m-%d %H:%M:%S"), "open": close, "high": close * 1.001, "low": close * 0.999,
                           "close": close, "volume": (1000 * (1 + rng.random(n))).astype(int), "vwap": close, "trades": 10})
        df.to_csv(os.path.join(root, "bars", f"{tk}.csv"), index=False)
    pd.DataFrame([{"ticker": "BBB", "halt_start": "2021-03-01 10:00:00", "halt_end": "2021-03-01 10:05:00"}]) \
        .to_csv(os.path.join(root, "halts.csv"), index=False)
    pd.DataFrame([{"start": "2021-03-02 10:20:00", "end": "2021-03-02 10:25:00"}]).to_csv(os.path.join(root, "circuit_breakers.csv"), index=False)
    pd.DataFrame([{"ticker": "AAA", "add_date": "2020-01-01", "remove_date": ""}, {"ticker": "BBB", "add_date": "2020-01-01", "remove_date": ""},
                  {"ticker": "CCC", "add_date": "2021-03-02", "remove_date": ""}]).to_csv(os.path.join(root, "constituents.csv"), index=False)
    return root


def test_loaders_constructed_sp500_tick_recipe_and_pit():
    try:
        DATASETS.get("cg_sp500_tick")({"root": _empty_root("sp500")}).build_panel()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "recipe" in str(e).lower() and "polygon" in str(e).lower()
    pq_root = _fixture_dir("sp500_parquet_only")
    os.makedirs(os.path.join(pq_root, "bars"))
    open(os.path.join(pq_root, "bars", "AAA.parquet"), "wb").close()
    try:
        DATASETS.get("cg_sp500_tick")({"root": pq_root}).build_panel()
        raise AssertionError("expected FileNotFoundError for parquet-only input")
    except FileNotFoundError as e:
        assert "pyarrow" in str(e) and "csv" in str(e)
    root = _build_sp500_fixture()
    p = DATASETS.get("cg_sp500_tick")({"root": root, "lead": 3, "post_halt_minutes": 5}).build_panel()
    p.validate()
    _check_ret_col(p)
    _check_pit_shapes(p, ["avail_feat", "avail_label", "universe"])
    T, N = p.mid.shape
    assert T == 150 and p.assets == ["AAA", "BBB", "CCC"]
    et = pd.to_datetime(p.times, utc=True).tz_convert("America/New_York")
    assert et[0].strftime("%H:%M") == "09:31" and et[74].strftime("%H:%M") == "10:45", "timestamp = bar start; close = +1 min"
    ev = p.meta["events"]
    assert len(ev) >= 1 and (ev["asset"] == 0).any(), ev
    e = ev[ev["asset"] == 0].iloc[0]
    assert int(e.onset_t) == 30 and int(e.peak_t) == 33 and int(e.end_t) in (34, 35) and int(e.direction) == -1
    assert p.labels[27:34, 0].all() and not p.labels[26, 0] and p.bars_to_peak[30, 0] == 3 and p.bars_to_peak[33, 0] == 0
    assert p.avail_label[30, 0] == int(e.end_t) > int(e.peak_t), "mini-flash-crash label knowable only at the reversion bar"
    assert p.avail_label[100, 1] == 100 + max(p.meta["horizons"])
    # halt 10:00-10:05 (+5 min) on BBB: closes 10:00 .. 10:10 untradable  (bar index 29 = 10:00 close .. 39 = 10:10 close)
    hal = [i for i in range(75) if not p.tradable[i, 1]]
    assert hal == list(range(29, 40)), hal
    assert p.tradable[:, 0][:75].all() and p.tradable[28, 1] and p.tradable[40, 1]
    # circuit breaker on day 2 10:20-10:25 (+5): all assets untradable for closes 10:20 .. 10:30
    cb = [i for i in range(75, 150) if not p.tradable[i].any()]
    assert cb == list(range(75 + 49, 75 + 60)), cb
    # universe as-of: CCC joins on day 2
    assert not p.universe[:75, 2].any() and p.universe[75:, 2].all() and p.universe[:, 0].all()


# ================================================================================================= D9 WSB squeeze
def _build_wsb_fixture():
    root = _fixture_dir("wsb_squeeze")
    os.makedirs(os.path.join(root, "prices"))
    dates = pd.bdate_range("2020-10-01", periods=150)                       # spans the 2021-01-28..02-04 restriction window
    bar = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    for k, tk in enumerate(("AMC", "GME", "XYZ")):
        close = _walk(150, 50 + k, 20.0)
        if tk == "GME":
            t0 = bar["2021-01-25"]
            close[t0 : t0 + 6] = close[t0 - 1] * np.array([1.2, 1.5, 1.9, 2.2, 2.0, 1.7])    # > 50 % over 5 days
            close[t0 + 6 :] = close[t0 + 5] * close[t0 + 6 :] / close[t0 + 6]
        _yahoo_csv(os.path.join(root, "prices", f"{tk}.csv"), dates, close, 60 + k)
    pd.DataFrame([
        {"settlementDate": "2020-12-31", "symbolCode": "GME", "currentShortPositionQuantity": 60_000_000, "previousShortPositionQuantity": 55_000_000},
        {"settlementDate": "2021-01-15", "symbolCode": "GME", "currentShortPositionQuantity": 62_000_000, "previousShortPositionQuantity": 60_000_000},
        {"settlementDate": "2021-01-15", "symbolCode": "AMC", "currentShortPositionQuantity": 5_000_000, "previousShortPositionQuantity": 4_000_000},
    ]).to_csv(os.path.join(root, "short_interest.csv"), index=False)
    pd.DataFrame([{"symbol": "GME", "float": 50_000_000}, {"symbol": "AMC", "float": 100_000_000}, {"symbol": "XYZ", "float": 10_000_000}]) \
        .to_csv(os.path.join(root, "float.csv"), index=False)
    created = int(pd.Timestamp("2021-01-26 21:37:41", tz="UTC").timestamp())        # after the 01-26 close -> 01-27 bar
    pd.DataFrame([
        {"title": "GME to the moon $GME", "score": 1200, "id": "a1", "url": "https://reddit.com/r/wallstreetbets/a1", "comms_num": 300,
         "created": created, "body": "diamond hands", "timestamp": "2021-01-26 21:37:41"},
        {"title": "AMC and GME both", "score": 10, "id": "a2", "url": "https://reddit.com/r/wallstreetbets/a2", "comms_num": 2,
         "created": created + 3600, "body": "", "timestamp": "2021-01-26 22:37:41"},
        {"title": "nothing to see here", "score": 1, "id": "a3", "url": "https://reddit.com/r/wallstreetbets/a3", "comms_num": 0,
         "created": created + 7200, "body": "", "timestamp": "2021-01-26 23:37:41"},
    ]).to_csv(os.path.join(root, "reddit_wsb.csv"), index=False)
    return root, dates, bar


def test_loaders_constructed_wsb_squeeze_recipe_and_pit():
    try:
        DATASETS.get("cg_wsb_squeeze")({"root": _empty_root("wsb")}).build_panel()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "recipe" in str(e).lower() and "kaggle" in str(e).lower()
    root, dates, bar = _build_wsb_fixture()
    p = DATASETS.get("cg_wsb_squeeze")({"root": root}).build_panel()
    p.validate()
    _check_ret_col(p)
    _check_pit_shapes(p, ["avail_feat", "avail_text", "avail_label", "universe"])
    T, N = p.mid.shape
    assert T == 150 and p.assets == ["AMC", "GME", "XYZ"] and p.meta["feat_names"][-1] == "si_ratio"
    g = p.assets.index("GME")
    si = p.meta["si_table"]
    r0 = si[(si["asset"] == g)].sort_values("settle_bar").iloc[0]
    assert int(r0.settle_bar) == bar["2020-12-31"] and int(r0.pub_bar) > int(r0.settle_bar)       # published ~2 weeks later
    # causal (publication) join: nothing public before the publication bar, then 1.2
    assert np.isnan(p.price_feat[int(r0.settle_bar), g, -1]) and np.isnan(p.price_feat[int(r0.pub_bar) - 1, g, -1])
    assert np.isclose(p.price_feat[int(r0.pub_bar), g, -1], 1.2) and (p.avail_feat == np.arange(T)[:, None]).all()
    assert np.isclose(p.meta["si_content_dated"][int(r0.settle_bar), g], 1.2) and p.meta["si_avail"][int(r0.settle_bar), g] == int(r0.pub_bar)
    # leaky (settlement) join stamps the row with the publication bar
    q = DATASETS.get("cg_wsb_squeeze")({"root": root, "si_join": "settlement"}).build_panel()
    assert np.isclose(q.price_feat[int(r0.settle_bar), g, -1], 1.2) and q.avail_feat[int(r0.settle_bar), g] == int(r0.pub_bar) > int(r0.settle_bar)
    assert np.isnan(PITView(q, int(r0.settle_bar)).price_feat[int(r0.settle_bar), g, -1])
    # squeeze event on GME: onset within the planted run, label knowable at onset + peak_window
    ev = p.meta["events"]
    assert len(ev) == 1 and int(ev.iloc[0].asset) == g
    e = ev.iloc[0]
    t_on = int(e.onset_t)
    assert bar["2021-01-25"] <= t_on <= bar["2021-01-29"] and int(e.peak_t) >= t_on and int(e.end_t) == t_on + 10
    assert p.avail_label[t_on, g] == int(e.end_t) > t_on and p.labels[t_on - 5, g] == 1 and p.bars_to_peak[t_on, g] == int(e.peak_t) - t_on
    # posts: creation 21:37 UTC on 01-26 (16:37 ET, after the close) -> available at the 01-27 close; GME and AMC co-mention
    t_doc = bar["2021-01-27"]
    assert p.text_mask[t_doc, g, 0] and p.avail_text[t_doc, g, 0] == t_doc and not p.text_mask[t_doc - 1, g].any()
    assert p.text_mask[t_doc, p.assets.index("AMC")].sum() == 1 and p.text_mask[t_doc, g].sum() == 2 and not p.text_mask[:, p.assets.index("XYZ")].any()
    assert p.meta["text_source"] == "metadata_pit" and p.text.shape[-1] == 2
    # broker restriction window
    for d in ("2021-01-28", "2021-01-29", "2021-02-01", "2021-02-04"):
        assert not p.tradable[bar[d], g] and not p.tradable[bar[d], p.assets.index("AMC")] and p.tradable[bar[d], p.assets.index("XYZ")]
    assert p.tradable[bar["2021-01-27"], g] and p.tradable[bar["2021-02-05"], g]


# ================================================================================================= D10 Elliptic++ + BTC
def _build_fin_graph_fixture(use_calendar: bool = False):
    root = _fixture_dir("fin_graph" + ("_cal" if use_calendar else ""))
    os.makedirs(os.path.join(root, "prices")); os.makedirs(os.path.join(root, "ellipticpp"))
    opens = pd.date_range("2016-01-01 00:00:00", periods=150, freq="1h", tz="UTC")       # bar OPEN times (unix s)
    close = _walk(150, 70, 430.0)
    rng = np.random.default_rng(8)
    pd.DataFrame({"timestamp": (opens.astype("int64") // 10**9), "open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
                  "volume": 100 * (1 + rng.random(150))}).to_csv(os.path.join(root, "prices", "BTC.csv"), index=False)
    # 12 transactions: txs 1-6 in step 1 (hours 10..12), txs 7-12 in step 2 (hours 40..41); tx 3 and tx 9 illicit
    tx = pd.DataFrame({"txId": range(1, 13), "Time step": [1] * 6 + [2] * 6, "in_BTC_total": rng.random(12) * 10, "Local_feature_1": rng.random(12)})
    tx.to_csv(os.path.join(root, "ellipticpp", "txs_features.csv"), index=False)
    pd.DataFrame({"txId": range(1, 13), "class": ["2", "2", "1", "unknown", "2", "3", "2", "2", "1", "2", "unknown", "2"]}) \
        .to_csv(os.path.join(root, "ellipticpp", "txs_classes.csv"), index=False)
    pd.DataFrame({"txId1": [1, 2, 3, 3, 7, 8, 9, 5], "txId2": [2, 3, 4, 5, 8, 9, 10, 7]}).to_csv(os.path.join(root, "ellipticpp", "txs_edgelist.csv"), index=False)
    hours = [10, 10, 11, 11, 12, 12, 40, 40, 40, 41, 41, 41]
    if use_calendar:
        pd.DataFrame([{"time_step": 1, "start": str(opens[9]), "end": str(opens[12] + pd.Timedelta(minutes=30))},
                      {"time_step": 2, "start": str(opens[39]), "end": str(opens[41] + pd.Timedelta(minutes=30))}]) \
            .to_csv(os.path.join(root, "ellipticpp", "timestep_calendar.csv"), index=False)
    else:
        pd.DataFrame({"txId": range(1, 13), "timestamp": [str(opens[h] + pd.Timedelta(minutes=20)) for h in hours]}) \
            .to_csv(os.path.join(root, "ellipticpp", "tx_times.csv"), index=False)
    return root


def test_loaders_constructed_fin_graph_recipe_and_pit():
    try:
        DATASETS.get("cg_fin_graph")({"root": _empty_root("fin_graph")}).build_panel()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "recipe" in str(e).lower() and "ellipticplusplus" in str(e).lower()
    root = _build_fin_graph_fixture()
    lag = 24
    p = DATASETS.get("cg_fin_graph")({"root": root, "label_lag_bars": lag, "illicit_share_thr": 0.0}).build_panel()
    p.validate()
    _check_ret_col(p)
    _check_pit_shapes(p, ["avail_feat", "avail_label", "avail_graph", "universe"])
    T, N = p.mid.shape
    assert T == 150 and N == 1 and p.assets == ["BTC"] and p.graph.shape == (150, 1, 2, 6) and p.meta["label_lag_bars"] == lag
    tt = np.arange(T)
    assert (p.avail_graph[:, 0, 0] == tt).all() and (p.avail_graph[:, 0, 1] == tt + lag).all()
    assert (p.avail_label[:, 0] == tt + lag).all() and (p.avail_label[:, 0] > tt).all(), "retrospective labels: avail = t + lag"
    # tx at 10:20 lands on the 11:00 close = bar index 10 (close of bar opened at 10:00)
    bars_with_tx = set(np.flatnonzero(p.graph_mask[:, 0, 0]).tolist())
    assert bars_with_tx == {10, 11, 12, 40, 41}, bars_with_tx
    assert p.labels[11, 0] == 1 and p.labels[40, 0] == 1 and p.labels[10, 0] == 0 and p.labels[12, 0] == 0 and p.meta["n_label_bars"] == 2
    assert np.isclose(p.graph[11, 0, 1, 0], 1.0) and np.isclose(p.graph[40, 0, 1, 0], 1 / 3)   # illicit share of class-known txs
    assert np.isclose(p.graph[10, 0, 1, 1], 0.5)                       # bar 10: tx 2 has illicit neighbour 3 -> 1 of 2 txs
    assert np.isclose(p.graph[10, 0, 0, 0], np.log1p(2)) and np.isclose(p.graph[10, 0, 0, 1], np.log1p(1))   # txs 1,2 + internal edge 1->2
    v = PITView(p, 20)
    assert not v.graph_mask[11, 0, 1] and v.graph_mask[11, 0, 0] and np.isnan(v.labels[11, 0])     # label-derived slot hidden before the lag
    v2 = PITView(p, 11 + lag)
    assert v2.graph_mask[11, 0, 1] and v2.labels[11, 0] == 1.0
    # timestep_calendar fallback: every tx of a step lands on the step-end bar (conservative)
    root2 = _build_fin_graph_fixture(use_calendar=True)
    q = DATASETS.get("cg_fin_graph")({"root": root2, "label_lag_bars": lag}).build_panel()
    q.validate()
    assert set(np.flatnonzero(q.graph_mask[:, 0, 0]).tolist()) == {12, 41} and "timestep_calendar" in q.meta["time_source"]

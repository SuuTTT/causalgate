"""Tests for the five public CausalGate loaders (D1 StockNet, D2 FI-2010 / LOBSTER, D3 FNSPID, D5 crypto_pd, D8 Qlib).

Collected by tests/test_causalgate.py (plain-script runner; filter with `-k loaders_public`).

FORMAT FIXTURES, NOT DATA. Every test writes a handful of hand-made rows in the exact file layout of the real source
(2-3 assets x ~120 bars, seeded random walks) to a temp dir under the session scratchpad, solely to exercise parsing and
point-in-time alignment. Nothing here is a sample of StockNet / FI-2010 / FNSPID / La Morgia / Qlib data, and nothing is
downloaded. data_raw/ is never read.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from warn2trade.utils import DATASETS
from causalgate.data.pit import get_avail, NEVER
from causalgate.data.loaders import stocknet, fi2010, fnspid, crypto_pd, qlib_alpha158  # noqa: F401  (registers cg_*)
from causalgate.data.loaders.doc_cache import build_doc_cache, cache_path, read_doc_cache

SCRATCH = os.environ.get("CLAUDE_SCRATCHPAD") or "/private/tmp/claude-501/-Users-zyy-package-anomalydetection-TradingSignal/cae6fc84-cde9-4354-ba01-4ae1653d3958/scratchpad"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOADERS = ("cg_stocknet", "cg_fi2010", "cg_fnspid", "cg_crypto_pd", "cg_qlib_alpha158")
META_KEYS = ("freq", "bars_per_year", "horizons", "raw_features", "ret_col", "feat_names", "source", "licence", "pit_notes", "required_layout")


# ----------------------------------------------------------------------------------------------- helpers
def _fixture_root(name: str) -> str:
    base = SCRATCH
    try:
        os.makedirs(base, exist_ok=True)
        if not os.access(base, os.W_OK):
            raise OSError
    except OSError:
        base = tempfile.gettempdir()
    d = os.path.join(base, "cg_loader_format_fixtures", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _bdays(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _ohlcv(rng: np.random.Generator, n: int, p0: float):
    """Seeded random-walk rows (a *format* fixture: plausible-looking numbers, not market data)."""
    ret = rng.standard_normal(n) * 0.01
    close = p0 * np.exp(np.cumsum(ret))
    opn = close * np.exp(rng.standard_normal(n) * 0.004)
    high = np.maximum(opn, close) * (1 + np.abs(rng.standard_normal(n)) * 0.004)
    low = np.minimum(opn, close) * (1 - np.abs(rng.standard_normal(n)) * 0.004)
    vol = (rng.lognormal(12, 0.4, n)).astype(int)
    return opn, high, low, close, vol


def _build(name: str, cfg: dict):
    return DATASETS.get(name)(dict(name=name, **cfg)).build_panel()


def _common_checks(p, n_pit_fields):
    p.validate()
    T, N = p.mid.shape
    for k in META_KEYS:
        assert k in p.meta, k
    assert p.meta["raw_features"] is True and p.meta["ret_col"] == 0
    assert len(p.meta["feat_names"]) == p.price_feat.shape[-1] and p.meta["feat_names"][0] == "log_ret"
    assert set(p.meta["pit_notes"]) >= {"avail_feat", "avail_label", "universe"}, p.meta["pit_notes"].keys()
    # column 0 is the raw log return of mid (no normalisation inside the loader)
    lr = np.full((T, N), np.nan)
    lr[1:] = np.log(p.mid[1:]) - np.log(p.mid[:-1])
    ok = np.isfinite(lr) & np.isfinite(p.price_feat[..., 0])
    assert ok.sum() > 0 and np.allclose(p.price_feat[..., 0][ok], lr[ok], atol=1e-5), "column 0 must equal the log return of mid"
    assert np.isfinite(lr).sum() == ok.sum(), "log return feature missing where mid exists"
    # PIT fields
    for k in n_pit_fields:
        assert get_avail(p, k) is not None, k
    assert set(p.meta["pit_fields"]) >= set(n_pit_fields), p.meta["pit_fields"]
    tt = np.arange(T)[:, None]
    assert p.avail_feat.shape == (T, N) and (p.avail_feat == tt).all()
    assert p.avail_label.shape == (T, N) and (p.avail_label >= tt).all()
    assert p.universe.shape == (T, N) and p.universe.dtype == bool
    assert p.tradable.dtype == bool and (~p.tradable[np.isnan(p.mid)]).all(), "NaN mid must be untradable"
    # labels / bars_to_peak consistency
    assert set(np.unique(p.labels)) <= {0, 1}
    btp, lab = p.bars_to_peak, p.labels
    if p.meta["label_rule"]["type"] == "event_table":        # exact peaks: btp kept (negative) after the peak while labels are 0
        assert np.isfinite(btp[lab == 1]).all() and (btp[lab == 1] >= 0).all(), "labelled bars lie at or before the peak"
        assert (btp[np.isfinite(btp) & (lab == 0)] < 0).all(), "unlabelled bars with a finite bars_to_peak are after the peak"
    else:
        assert np.isnan(btp[lab == 0]).all(), "bars_to_peak must be NaN where labels == 0"
    if p.text is not None:
        assert p.avail_text.shape == p.text.shape[:3] and p.text_mask.shape == p.text.shape[:3]
        masked = p.text_mask
        rows = np.broadcast_to(np.arange(T)[:, None, None], masked.shape)
        assert (p.avail_text[masked] >= rows[masked]).all(), "a document must not be knowable before its row"
        # a document with no overlap with the pre-cutoff vocabulary embeds to zero under a point-in-time embedder (reported in
        # the cache meta as n_zero_embeddings); most fixture documents overlap, so most placed embeddings are non-zero
        assert (np.linalg.norm(p.text[masked], axis=-1) > 0).mean() >= 0.5, "placed documents mostly carry a non-zero embedding"
        assert (~p.text_mask | (p.text_age == 0)).all()


# ----------------------------------------------------------------------------------------------- (a) missing raw files
def test_loaders_public_missing_root_raises_with_recipe():
    root = _fixture_root("empty_root")
    for name in LOADERS:
        try:
            _build(name, {"root": root})
            raise AssertionError(f"{name}: expected FileNotFoundError on an empty root")
        except FileNotFoundError as e:
            msg = str(e)
            assert "recipe" in msg.lower() and "cg_prepare_data" in msg, (name, msg[:200])
            assert root in msg, (name, msg[:200])
    # nothing was created inside the empty root (no fabrication)
    assert os.listdir(root) == []


# ----------------------------------------------------------------------------------------------- D1 StockNet
def _write_stocknet_fixture(root: str, n: int = 120):
    rng = np.random.default_rng(1)
    dates = _bdays("2014-01-02", n)
    for k, tk in enumerate(("AAA", "BBB", "CCC")):
        o, h, l, c, v = _ohlcv(rng, n, 50.0 + 20 * k)
        pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Open": o.round(2), "High": h.round(2), "Low": l.round(2), "Close": c.round(2),
                      "Adj Close": (c * 0.98).round(4), "Volume": v}).to_csv(os.path.join(root, "stocknet", "price", "raw", f"{tk}.csv"), index=False)
    # tweets: preprocessed JSON lines (text = token list, created_at in Twitter format, UTC)
    def tweet(day: str, hhmm: str, text: str, tk: str = "AAA"):
        d = pd.Timestamp(f"{day} {hhmm}", tz="UTC")
        return {"text": text.split(), "user_id_str": "1234567", "created_at": d.strftime("%a %b %d %H:%M:%S +0000 %Y")}, day, tk
    recs = [tweet("2014-01-06", "15:00", "$AAA breaks out on heavy volume today"),          # 10:00 ET, in session -> same close
            tweet("2014-01-06", "21:30", "$AAA earnings call after the bell beat estimates"),  # 16:30 ET, after close -> next session
            tweet("2014-01-06", "18:10", "watching $AAA into the close"),
            tweet("2014-01-11", "12:00", "weekend read on $AAA supply chain"),               # Saturday -> Monday row, Monday close
            tweet("2014-01-07", "14:05", "$BBB upgrade this morning", "BBB"),
            tweet("2014-01-07", "16:45", "$BBB volume spike midday", "BBB"),
            tweet("2014-03-03", "15:30", "$CCC guidance cut shares slide", "CCC")]
    for rec, day, tk in recs:
        d = os.path.join(root, "stocknet", "tweet", "preprocessed", tk)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, day), "a") as f:
            f.write(json.dumps(rec) + "\n")
    return dates


def test_loaders_public_stocknet_format_fixture():
    root = _fixture_root("stocknet")
    os.makedirs(os.path.join(root, "stocknet", "price", "raw"))
    dates = _write_stocknet_fixture(root)
    # cache built through the real CLI (hash_tfidf fitted on the 5 tweets before the cutoff)
    cmd = [sys.executable, os.path.join(REPO, "scripts", "cg_prepare_data.py"), "--dataset", "stocknet", "--root", root,
           "--embed", "--embedder", "hash_tfidf", "--cutoff", "2014-01-10", "--dim", "8"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    cache = read_doc_cache(cache_path(root, "stocknet"))
    assert cache is not None and cache.n == 7 and cache.meta["embedder"] == "hash_tfidf" and "n_zero_embeddings" in cache.meta
    assert cache.meta["cutoff_date"] == "2014-01-10" and cache.meta["n_docs_before_cutoff"] == 5 and cache.meta["embedder_training_cutoff"] == "2014-01-10"
    p = _build("cg_stocknet", {"root": root, "text_slots": 4, "labels": {"window": 30}})
    _common_checks(p, ("avail_feat", "avail_text", "avail_label", "universe"))
    assert p.T == 120 and p.assets == ["AAA", "BBB", "CCC"] and p.meta["feat_names"] == ["log_ret", "log_volume", "hl_range", "gap"]
    assert cache.dim == 4                                               # SVD rank is capped at n_fit_docs - 1 (5 pre-cutoff tweets)
    assert p.price_feat.shape == (120, 3, 4) and p.text.shape == (120, 3, 4, cache.dim)
    row = lambda day: int(np.searchsorted(dates.values, np.datetime64(day)))
    r6, r7, r13 = row("2014-01-06"), row("2014-01-07"), row("2014-01-13")
    assert p.text_mask[r6, 0].sum() == 3 and p.text_mask[r7, 1].sum() == 2 and p.text_mask[r13, 0].sum() == 1
    av = p.avail_text[r6, 0][p.text_mask[r6, 0]]
    assert sorted(av.tolist()) == [r6, r6, r6 + 1], av                 # two in-session tweets same close, after-close -> next session
    assert (np.linalg.norm(p.text[r6, 0][p.text_mask[r6, 0]], axis=-1) > 0).all()   # pre-cutoff documents are never zero
    assert p.avail_text[r13, 0][p.text_mask[r13, 0]][0] == r13          # Saturday tweet: Monday row, knowable at Monday close
    assert p.meta["text_placement"]["placed"] == 7 and p.meta["text_placement"]["n_after_close"] == 1
    assert (p.avail_label == np.arange(120)[:, None] + 10).all()         # t + max(5, max(horizons)=10)
    assert p.universe.all() and p.tradable.all()
    assert "MIT" in p.meta["licence"] and "yumoxu" in p.meta["source"]
    assert p.times[r6] == pd.Timestamp("2014-01-06 16:00", tz="America/New_York").value
    # text-free path when the cache is absent
    os.remove(cache_path(root, "stocknet"))
    q = _build("cg_stocknet", {"root": root, "labels": {"window": 30}})
    q.validate()
    assert q.text is None and q.meta["text"] == "missing cache" and "cg_prepare_data" in q.meta["text_recipe"]
    assert np.array_equal(q.mid, p.mid, equal_nan=True)


# ----------------------------------------------------------------------------------------------- D2 FI-2010 / LOBSTER
def _fi2010_matrix(rng: np.random.Generator, n: int, level: float, tick: float = 1e-4, wide_at=None, noquote_at=None):
    """(149, n) matrix in the FI-2010 layout; DecPre-like scale (prices ~ level)."""
    mid = level * np.exp(np.cumsum(rng.standard_normal(n) * 0.0005))
    mid = np.round(mid / tick) * tick
    half = tick * rng.integers(1, 3, n)
    ask1, bid1 = mid + half, mid - half
    m = np.zeros((149, n))
    for i in range(10):
        m[4 * i] = ask1 + i * tick                 # ask_px_i
        m[4 * i + 1] = rng.integers(100, 2000, n)  # ask_sz_i
        m[4 * i + 2] = bid1 - i * tick             # bid_px_i
        m[4 * i + 3] = rng.integers(100, 2000, n)  # bid_sz_i
    if wide_at is not None:
        m[0, wide_at] = m[2, wide_at] + 30 * tick  # spread 30 ticks > 20
    if noquote_at is not None:
        m[3, noquote_at] = 0.0                     # bid size 0 at level 1
    m[40:144] = rng.standard_normal((104, n))      # hand-crafted features (ignored)
    m[144:149] = rng.integers(1, 4, (5, n))        # trend labels in {1,2,3}
    return m


def test_loaders_public_fi2010_format_fixture():
    root = _fixture_root("fi2010")
    rng = np.random.default_rng(2)
    d = os.path.join(root, "fi2010", "BenchmarkDatasets", "NoAuction", "3.NoAuction_DecPre")
    os.makedirs(os.path.join(d, "NoAuction_DecPre_Training")); os.makedirs(os.path.join(d, "NoAuction_DecPre_Testing"))
    train = np.concatenate([_fi2010_matrix(rng, 100, 0.25, wide_at=40, noquote_at=41), _fi2010_matrix(rng, 100, 0.12), _fi2010_matrix(rng, 100, 0.40)], 1)
    np.savetxt(os.path.join(d, "NoAuction_DecPre_Training", "Train_Dst_NoAuction_DecPre_CF_7.txt"), train, fmt="%.6g")
    for k, lvl in zip((7, 8, 9), (0.40, 0.12, 0.25)):       # day 8 continues the 0.40 segment, days 9/10 are new segments
        np.savetxt(os.path.join(d, "NoAuction_DecPre_Testing", f"Test_Dst_NoAuction_DecPre_CF_{k}.txt"), _fi2010_matrix(rng, 20, lvl), fmt="%.6g")
    zs = os.path.join(root, "fi2010", "BenchmarkDatasets", "NoAuction", "1.NoAuction_Zscore", "NoAuction_Zscore_Training")
    os.makedirs(zs)
    np.savetxt(os.path.join(zs, "Train_Dst_NoAuction_ZScore_CF_7.txt"), rng.standard_normal((149, 30)), fmt="%.4f")   # must be ignored
    p = _build("cg_fi2010", {"root": root, "labels": {"window": 30, "horizon": 10, "lead": 3}})
    _common_checks(p, ("avail_feat", "avail_label", "universe"))
    assert p.N == 5 and p.T == 120 and p.meta["segments"][2][1] - p.meta["segments"][2][0] == 120, p.meta["segments"]
    assert all("decpre" in f.lower() for f in p.meta["files"]) and len(p.meta["files"]) == 4 and p.meta["warnings"] == []
    assert "decpre" in p.meta["pit_notes"]["normalisation"].lower()
    assert p.lob.shape == (120, 5, 40) and p.meta["lob_layout"]["per_level"] == ["bid_px", "bid_sz", "ask_px", "ask_sz"]
    fin = np.isfinite(p.mid)
    assert np.allclose(p.mid[fin], 0.5 * (p.lob[..., 0] + p.lob[..., 2])[fin], atol=1e-6)      # mid from level-1 bid/ask
    assert (p.lob[..., 2] >= p.lob[..., 0])[fin].all()                                           # ask >= bid
    assert np.isnan(p.mid[20:, 3]).all() and np.isnan(p.mid[20:, 4]).all() and not p.universe[20:, 3].any()
    assert p.universe[:20, 3].all() and p.universe[:100, 0].all() and not p.universe[100:, 0].any()
    assert not p.tradable[40, 0] and not p.tradable[41, 0] and p.tradable[42, 0] and p.tradable[:40, 0].all()
    assert p.meta["fi2010_trend_labels"].shape == (120, 5, 5) and set(np.unique(p.meta["fi2010_trend_labels"][:100, 0])) == {1, 2, 3}
    assert (p.meta["fi2010_trend_labels"][100:, 0] == 0).all()
    assert p.meta["fi2010_trend_horizons"] == [10, 20, 30, 50, 100] and abs(p.meta["tick_size"][0] - 1e-4) < 1e-9
    assert p.meta["feat_names"] == ["log_ret", "log_depth_l1", "log_spread_bps", "ofi_l1", "imbalance_l1", "imbalance_l10"]
    assert np.isfinite(p.price_feat[1:100, 0, 3]).all() and np.isnan(p.price_feat[0, 0, 3])       # OFI needs t-1
    assert (p.avail_label == np.arange(120)[:, None] + 100).all()
    assert p.liq.shape == (120, 5, 3) and (p.liq[fin][:, 0] > 0).all()
    # explicit segment list and the ZScore-only refusal
    q = _build("cg_fi2010", {"root": root, "segmentation": "list", "segments": [[0, 100, "K"], [100, 200, "O"]], "labels": {"window": 30}})
    assert q.assets == ["K", "O"] and q.T == 100
    shutil.rmtree(d)
    try:
        _build("cg_fi2010", {"root": root})
        raise AssertionError("ZScore-only tree must be refused")
    except FileNotFoundError as e:
        assert "ZScore" in str(e) or "DecPre" in str(e)


def test_loaders_public_fi2010_lobster_variant_fixture():
    root = _fixture_root("lobster")
    rng = np.random.default_rng(3)
    d = os.path.join(root, "lobster")
    os.makedirs(d)
    for tk, lvl in (("AAA", 585.0), ("BBB", 32.0)):
        n = 150
        m = _fi2010_matrix(rng, n, lvl, tick=0.01)[:40].T * 1e4          # LOBSTER: 40 columns, prices in 1e-4 USD
        m[:, 1::2] = m[:, 1::2] / 1e4                                      # sizes are not scaled
        pd.DataFrame(m).to_csv(os.path.join(d, f"{tk}_2012-06-21_34200000_57600000_orderbook_10.csv"), header=False, index=False)
        t = 34200.0 + np.cumsum(rng.exponential(0.04, n))
        msg = pd.DataFrame({0: t, 1: rng.integers(1, 6, n), 2: rng.integers(1, 10 ** 6, n), 3: rng.integers(1, 500, n), 4: (lvl * 1e4), 5: rng.choice([-1, 1], n)})
        msg.to_csv(os.path.join(d, f"{tk}_2012-06-21_34200000_57600000_message_10.csv"), header=False, index=False)
    p = _build("cg_fi2010", {"root": root, "variant": "lobster", "bar_ms": 100, "labels": {"window": 20, "horizon": 5, "lead": 2}, "horizons": [5, 10, 20]})
    _common_checks(p, ("avail_feat", "avail_label", "universe"))
    assert p.N == 2 and p.assets[0].startswith("AAA_2012-06-21") and p.meta["variant"] == "lobster" and p.meta["tick_size"] == [0.01, 0.01]
    assert 20 < p.T < 150, p.T                                             # 100 ms buckets merged consecutive events
    assert 500 < np.nanmedian(p.mid[:, 0]) < 700 and 25 < np.nanmedian(p.mid[:, 1]) < 40   # USD, not 1e-4 USD
    assert len(p.meta["event_times_s"]) == 2 and p.meta["event_times_s"][0] is not None
    assert p.tradable[np.isfinite(p.mid)].mean() > 0.9


# ----------------------------------------------------------------------------------------------- D3 FNSPID
def test_loaders_public_fnspid_format_fixture():
    root = _fixture_root("fnspid")
    rng = np.random.default_rng(4)
    pdir = os.path.join(root, "fnspid", "Stock_price", "full_history")
    ndir = os.path.join(root, "fnspid", "Stock_news")
    os.makedirs(pdir); os.makedirs(ndir)
    dates = _bdays("2015-01-02", 120)
    for k, tk in enumerate(("AAA", "BBB")):
        o, h, l, c, v = _ohlcv(rng, 120, 30.0 + 10 * k)
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": o, "high": h, "low": l, "close": c, "adj close": c * 0.99, "volume": v}
                     ).to_csv(os.path.join(pdir, f"{tk}.csv"), index=False)
    cols = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author", "Article", "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]
    arts = [("2015-03-03 14:00:00 UTC", "AAA beats on revenue", "AAA"),            # 09:00 ET -> same close
            ("2015-03-03 22:00:00 UTC", "AAA guides lower after the close", "AAA"),   # 17:00 ET -> next session
            ("2015-03-07 10:00:00 UTC", "AAA weekend feature on supply chain", "AAA"),  # Saturday -> Monday
            ("2015-03-03 15:30:00 UTC", "BBB names new chief executive", "BBB"),
            ("2015-03-04 13:00:00 UTC", "ZZZ not in the panel", "ZZZ")]
    rows = [[d, t, s, "https://example.invalid/" + s, "Fixture Wire", "nobody", t + " body text", t + " summary", "", "", ""] for d, t, s in arts]
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(ndir, "nasdaq_exteral_data.csv"), index=False)
    docs = pd.DataFrame(list(fnspid.iter_documents(root, {"tickers": ["AAA", "BBB"]})))
    assert len(docs) == 4 and set(docs["asset"]) == {"AAA", "BBB"}
    docs_all = pd.DataFrame(list(fnspid.iter_documents(root, {"tickers": ["AAA", "BBB", "ZZZ"]})))
    build_doc_cache(cache_path(root, "fnspid"), docs_all, "fnspid", "2015-03-05", dim=8)
    p = _build("cg_fnspid", {"root": root, "tickers": ["AAA", "BBB"], "text_slots": 3, "labels": {"window": 30}})
    _common_checks(p, ("avail_feat", "avail_text", "avail_label", "universe"))
    assert p.T == 120 and p.assets == ["AAA", "BBB"] and p.text.shape == (120, 2, 3, 3)   # rank capped at n_fit_docs - 1 = 3
    row = lambda day: int(np.searchsorted(dates.values, np.datetime64(day)))
    r3, r9 = row("2015-03-03"), row("2015-03-09")
    assert p.text_mask[r3, 0].sum() == 2 and p.text_mask[r3, 1].sum() == 1 and p.text_mask[r9, 0].sum() == 1
    assert sorted(p.avail_text[r3, 0][p.text_mask[r3, 0]].tolist()) == [r3, r3 + 1]
    assert p.avail_text[r3, 1][p.text_mask[r3, 1]][0] == r3 and p.avail_text[r9, 0][p.text_mask[r9, 0]][0] == r9
    assert p.meta["text_placement"]["placed"] == 4 and p.meta["text_placement"]["dropped_out_of_panel"] == 1 and p.meta["text_placement"]["n_after_close"] == 1
    assert (p.avail_label == np.arange(120)[:, None] + 20).all()
    assert "Zihan1004/FNSPID" in p.meta["source"]


# ----------------------------------------------------------------------------------------------- D5 crypto pump-and-dump
def _klines(rng: np.random.Generator, t0: pd.Timestamp, n_before: int, n_after: int, p0: float, peak_offset: int, zero_vol_at: int):
    """Minute klines around t0 with a planted spike `peak_offset` minutes after t0 (format fixture)."""
    n = n_before + n_after + 1
    opens = (t0 - pd.Timedelta(minutes=n_before)) + pd.to_timedelta(np.arange(n), unit="m")
    close = p0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.002))
    high = close * (1 + np.abs(rng.standard_normal(n)) * 0.001)
    k = n_before + peak_offset
    high[k] = close[k] * 1.5
    close[k] = close[k] * 1.4
    low = close * (1 - np.abs(rng.standard_normal(n)) * 0.001)
    vol = rng.lognormal(3, 0.5, n)
    vol[n_before + zero_vol_at] = 0.0
    ms = (opens.values.astype("datetime64[ms]").astype(np.int64))
    return pd.DataFrame({"open_time": ms, "open": close, "high": high, "low": low, "close": close, "volume": vol, "close_time": ms + 59999,
                         "quote_volume": vol * close, "count": rng.integers(1, 50, n), "taker_buy_volume": vol / 2, "taker_buy_quote_volume": vol * close / 2, "ignore": 0})


def test_loaders_public_crypto_pd_format_fixture():
    root = _fixture_root("crypto_pd")
    rng = np.random.default_rng(5)
    d = os.path.join(root, "crypto_pd"); kd = os.path.join(d, "klines")
    os.makedirs(kd)
    pd.DataFrame([["YOYO", "Pump Channel A", "2018-11-14", "17:00", "Binance"], ["ABC", "Pump Channel B", "2018-11-20", "16:30", "Binance"],
                  ["XYZ", "Channel C", "2018-11-21", "18:00", "Yobit"]], columns=["symbol", "group", "date", "hour", "exchange"]
                 ).to_csv(os.path.join(d, "pump_telegram.csv"), index=False)
    t_yoyo, t_abc = pd.Timestamp("2018-11-14 17:00", tz="UTC"), pd.Timestamp("2018-11-20 16:30", tz="UTC")
    _klines(rng, t_yoyo, 90, 90, 1e-6, 7, -30).to_csv(os.path.join(kd, "YOYOBTC.csv"), header=False, index=False)       # Binance 12-col header-less
    _klines(rng, t_abc, 90, 90, 5e-5, 3, -10).to_csv(os.path.join(kd, "ABCBTC-1m-2018-11.csv"), index=False)            # headed monthly file
    pd.DataFrame([["Pump Channel A", "2018-11-14 16:55:00", "next pump in 5 minutes, get ready"],
                  ["Pump Channel A", "2018-11-14 17:00:05", "the coin is YOYO buy now on binance"],
                  ["Pump Channel B", "2018-11-20 16:30:30", "coin ABC go go go"]], columns=["channel", "timestamp", "text"]
                 ).to_csv(os.path.join(d, "messages.csv"), index=False)
    docs = pd.DataFrame(list(crypto_pd.iter_documents(root)))
    assert len(docs) == 3 and docs["asset"].tolist()[:2] == ["YOYOBTC@2018-11-14T17:00Z"] * 2
    build_doc_cache(cache_path(root, "crypto_pd"), docs, "crypto_pd", "2018-11-18", dim=4)
    cfg = {"root": root, "pre_minutes": 60, "post_minutes": 60, "lead": 3, "peak_window_minutes": 30, "end_minutes_after_peak": 20,
           "text_slots": 2, "horizons": [1, 2, 5, 10, 30]}
    p = _build("cg_crypto_pd", cfg)
    _common_checks(p, ("avail_feat", "avail_text", "avail_label", "universe"))
    T, pre = 121, 60
    assert p.T == T and p.N == 2 and p.assets == ["YOYOBTC@2018-11-14T17:00Z", "ABCBTC@2018-11-20T16:30Z"]
    assert p.meta["skipped"]["other_exchange"] == 1 and p.meta["peak_source"] == ["derived_argmax_high"] * 2
    assert p.meta["peak_row"].tolist() == [pre + 7, pre + 3] and p.meta["feat_names"] == ["log_ret", "log_volume", "hl_range", "log_trades"]
    # labels 1 on [announcement - lead, peak]; bars_to_peak exact and negative after the peak; label knowable at the event end
    assert p.labels[pre - 3 : pre + 8, 0].all() and not p.labels[: pre - 3, 0].any() and not p.labels[pre + 8 :, 0].any()
    assert p.bars_to_peak[pre, 0] == 7 and p.bars_to_peak[pre - 3, 0] == 10 and p.bars_to_peak[pre + 10, 0] == -3
    assert np.isnan(p.bars_to_peak[pre + 28, 0]) and p.bars_to_peak[pre + 27, 0] == -20
    assert (p.avail_label[pre - 3 : pre + 28, 0] == pre + 27).all() and p.avail_label[0, 0] == 30 and p.avail_label[pre + 28, 0] == pre + 28 + 30
    assert not p.tradable[pre - 30, 0] and not p.tradable[pre - 10, 1] and p.tradable[pre, 0]
    assert p.meta["bar_close_ns"][pre, 0] == t_yoyo.value + 60 * 10 ** 9 and p.meta["announce_ns"][0] == t_yoyo.value
    assert p.text_mask[pre, 0].sum() == 1 and p.text_mask[pre - 6, 0].sum() == 1 and p.text_mask[pre, 1].sum() == 1
    assert p.avail_text[pre, 0][p.text_mask[pre, 0]][0] == pre and p.avail_text[pre - 6, 0][p.text_mask[pre - 6, 0]][0] == pre - 6
    assert p.meta["split_axis"] == "assets" and "2105.00733" in p.meta["source"]
    # curated peak column (HH:MM on the announcement day) overrides the derived peak
    tab = pd.read_csv(os.path.join(d, "pump_telegram.csv"))
    tab["peak_time"] = ["17:12", "16:31", "18:05"]
    tab.to_csv(os.path.join(d, "pump_telegram.csv"), index=False)
    q = _build("cg_crypto_pd", cfg)
    assert q.meta["peak_source"] == ["curated"] * 2 and q.meta["peak_row"].tolist() == [pre + 11, pre]


# ----------------------------------------------------------------------------------------------- D8 Qlib
def _write_bin(path: str, start: int, values: np.ndarray):
    np.concatenate([[np.float32(start)], values.astype(np.float32)]).astype("<f4").tofile(path)


def test_loaders_public_qlib_format_fixture():
    root = _fixture_root("qlib")
    rng = np.random.default_rng(6)
    d = os.path.join(root, "qlib_us")
    os.makedirs(os.path.join(d, "calendars")); os.makedirs(os.path.join(d, "instruments"))
    dates = _bdays("2015-01-02", 120)
    with open(os.path.join(d, "calendars", "day.txt"), "w") as f:
        f.write("\n".join(dates.strftime("%Y-%m-%d")) + "\n")
    with open(os.path.join(d, "instruments", "all.txt"), "w") as f:
        f.write("AAA\t2015-01-02\t2015-12-31\nBBB\t2015-02-13\t2015-12-31\nCCC\t2015-01-02\t2015-12-31\n")
    with open(os.path.join(d, "instruments", "sp500.txt"), "w") as f:            # as-of membership ranges (two rows for CCC)
        f.write("AAA\t2015-01-02\t2015-12-31\nBBB\t2015-03-02\t2015-12-31\nCCC\t2015-01-02\t2015-02-27\nCCC\t2015-05-01\t2015-12-31\n")
    starts = {"AAA": 0, "BBB": 30, "CCC": 0}
    limit_day = 50
    for k, tk in enumerate(("AAA", "BBB", "CCC")):
        n = 120 - starts[tk]
        o, h, l, c, v = _ohlcv(rng, n, 100.0 + 50 * k)
        if tk == "AAA":                                                              # planted limit-up day (close == high, +10 %)
            c[limit_day] = c[limit_day - 1] * 1.10; h[limit_day] = c[limit_day]; o[limit_day] = c[limit_day - 1] * 1.02; l[limit_day] = o[limit_day]
            v[limit_day + 5] = 0                                                      # a zero-volume (suspended-like) day
        fdir = os.path.join(d, "features", tk.lower()); os.makedirs(fdir)
        chg = np.r_[np.nan, c[1:] / c[:-1] - 1]
        for name, arr in (("open", o), ("high", h), ("low", l), ("close", c), ("volume", v.astype(float)), ("factor", np.ones(n)), ("change", chg)):
            _write_bin(os.path.join(fdir, f"{name}.day.bin"), starts[tk], arr)
    p = _build("cg_qlib_alpha158", {"root": root, "region": "us", "labels": {"window": 30}})
    _common_checks(p, ("avail_feat", "avail_label", "universe"))
    assert p.T == 120 and p.assets == ["AAA", "CCC", "BBB"]                           # ordered by first membership start
    j = {a: i for i, a in enumerate(p.assets)}
    assert np.isnan(p.mid[:30, j["BBB"]]).all() and np.isfinite(p.mid[30:, j["BBB"]]).all()     # .bin start index honoured
    row = lambda day: int(np.searchsorted(dates.values, np.datetime64(day)))
    u = p.universe
    assert not u[: row("2015-03-02"), j["BBB"]].any() and u[row("2015-03-02") :, j["BBB"]].all()
    assert u[: row("2015-02-27") + 1, j["CCC"]].all() and not u[row("2015-03-02") : row("2015-05-01"), j["CCC"]].any() and u[row("2015-05-01") :, j["CCC"]].all()
    assert u[:, j["AAA"]].all()
    assert p.meta["feat_names"][:3] == ["log_ret", "log_volume", "hl_range"] and "KMID" in p.meta["feat_names"] and "ROC5" in p.meta["feat_names"]
    assert p.price_feat.shape[-1] == 3 + 4 + 7 * 4 and p.meta["alpha158"] == "lite" and "not applied" in p.meta["processors"]
    kmid = p.price_feat[:, j["AAA"], p.meta["feat_names"].index("KMID")]
    assert np.isfinite(kmid[5:]).all() and abs(kmid[limit_day] - (1.10 / 1.02 - 1)) < 1e-3
    assert p.tradable[limit_day, j["AAA"]] and not p.tradable[limit_day + 5, j["AAA"]]   # us: no limit rule; zero volume untradable
    assert p.meta["limit_rule"] == "none" and p.meta["n_limit_hit_bars"] == 0
    assert p.times[0] == pd.Timestamp("2015-01-02 16:00", tz="America/New_York").value and "sp500" in p.meta["universe_file"]
    # cn limit rule on the same fixture: the planted +10 % close-at-high day becomes untradable
    q = _build("cg_qlib_alpha158", {"root": root, "region": "cn", "subdir": "qlib_us", "universe": "sp500", "labels": {"window": 30}, "alpha158": "none"})
    assert q.meta["limit_rule"] == "cn_10pct" and q.meta["n_limit_hit_bars"] == 1 and not q.tradable[limit_day, j["AAA"]]
    assert q.price_feat.shape[-1] == 3 and q.times[0] == pd.Timestamp("2015-01-02 15:00", tz="Asia/Shanghai").value
    # explicit instrument subset and the missing-feature warning path
    r = _build("cg_qlib_alpha158", {"root": root, "instruments": ["AAA", "NOPE"], "labels": {"window": 30}, "alpha158": "none"})
    assert r.assets == ["AAA", "NOPE"] and np.isnan(r.mid[:, 1]).all() and any("NOPE" in w for w in r.meta["warnings"]) and not r.universe[:, 1].any()

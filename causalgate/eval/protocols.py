"""Protocol definitions and the runner for P-O (as published), P-W (purged walk-forward, offline inside the fold),
P-C (strictly causal), P-C/ADIA (detection-only, one observation at a time, TS-AUC) and the one-switch variants
(P-C with exactly one channel L1..L8 set to its offline convention; Zhang et al. 2026 design).

Switch -> where it acts
    normalisation   L1  baseline cfg norm_mode / published_norm / norm_rows            (model side)
    threshold       L2  baseline cfg thr_mode / published_threshold / gate              (model side)
    model_selection L3  realised ONLY as the threshold 'best_f1_test' (threshold chosen on test labels = selection
                        on test). The `model_selection` field itself selects nothing in the runner (no baseline has
                        a hyper-parameter search); it is kept for the hash / offline_channels bookkeeping.
    point_adjust    L4  metrics: PA-F1 (per asset, along time) reported as the headline F1  (metric side)
    label_join      L5  training labels with avail_label > train end masked when 'avail_time'; documents with
                        avail_text > t hidden (PITView semantics) when 'avail_time'     (data side)
    universe        L6  decision points (t, i) with universe[t, i] False removed (membership AT t only; prices
                        are never masked, so the decision set never depends on future membership)   (data side)
    fill            L7  decisions priced at the close of t ('same_bar') or t + latency  (pricing side)
    pretrained_check L8 audit-card flag only (baseline.cutoff_date vs the fold's test start). No baseline has a
                        switchable PIT / non-PIT checkpoint, so L8 is NOT a one-switch protocol (a causal+L8 run would
                        be bit-identical to causal); it is reported on the leakage audit card.
    detection_only      P-C/ADIA: scripts/cg_run.py runs `run_adia` on every fold whenever the protocol sets it.

Model-side channels (L1-L3) act only on baselines whose class lists them in `supported_switches` (the
ScoreToTradeBaseline scaffold: all three; CausalGate: none). A requested channel the class does not implement is
recorded in `not_applicable_channels` and the run is NOT stamped mode='published' for it.

P-O ('published') conventions, precedence (first hit wins), recorded per run in `published_conventions`:
    1. class attribute `published_protocol` = {'normalisation': global|train_fold|test_fold|streaming, 'threshold': ...}
    2. the baseline yaml's `published_norm` (train|global|test|rows) / `published_threshold`
    3. the class default (cfg.setdefault in __init__, or `published_norm_default` / `published_threshold_default`)
    4. the scaffold default (global / test_quantile)
The generic preset fields (normalisation='global', threshold='published') never override 1-3.

Every protocol has a content hash; `freeze()` writes configs/causalgate/PROTOCOL_HASH before any test-fold run (and
refuses to overwrite a differing freeze without force=True), and `verify_frozen()` refuses to run when a preset no
longer matches its frozen hash.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch

from warn2trade.data.base import EventDataset, build_loader
from warn2trade.data.schema import Panel
from warn2trade.data.splits import WalkForwardSplitter, Fold
from warn2trade.backtest.costs import CostModel
from warn2trade.backtest.engine import EventBacktester
from warn2trade.backtest import metrics as WM
from warn2trade.train.trainer import evaluate_outputs, to_device
from warn2trade.utils.registry import BASELINES
from warn2trade.utils.seed import set_seed
import copy
from ..data.pit import clone_panel, default_availability, get_avail, PITView, PIT_FIELDS
from ..models.gate import alarm_rate_error
from . import metrics as CM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HASH_FILE = os.path.join(ROOT, "configs", "causalgate", "PROTOCOL_HASH")


# --------------------------------------------------------------------------------------- protocol
@dataclass(frozen=True)
class Protocol:
    name: str
    normalisation: str = "streaming"      # global | test_fold | train_fold | streaming
    threshold: str = "delayed_aci"        # published | test_quantile | combined_quantile | best_f1_test | val_quantile | fixed | aci | delayed_aci | pot
    model_selection: str = "val"          # test | val
    point_adjust: bool = False
    label_join: str = "avail_time"        # event_time | avail_time
    universe: str = "as_of"               # today | as_of
    fill: str = "latency"                 # same_bar | latency
    pretrained_check: bool = True
    purge: bool = True
    embargo: bool = True
    detection_only: bool = False          # P-C/ADIA
    seeds: Tuple[int, ...] = (42, 2024, 2025, 2026)
    cost_grid_bps: Tuple[float, ...] = (1.0, 5.0, 10.0, 15.0, 30.0)
    train_cost_bps: float = 5.0
    min_latency: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        d = self.to_dict(); d.pop("name")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def offline_channels(self) -> List[str]:
        ch = []
        if self.normalisation in ("global", "test_fold"): ch.append("L1")
        if self.threshold in ("published", "test_quantile", "combined_quantile"): ch.append("L2")
        if self.model_selection == "test" or self.threshold == "best_f1_test": ch.append("L3")   # threshold chosen on test labels = selection on test
        if self.point_adjust: ch.append("L4")
        if self.label_join == "event_time": ch.append("L5")
        if self.universe == "today": ch.append("L6")
        if self.fill == "same_bar": ch.append("L7")
        if not self.pretrained_check: ch.append("L8")
        return ch


PRESETS: Dict[str, Protocol] = {
    # as published: the generic offline convention set; per-baseline overrides come from the class's `published_protocol`
    "published": Protocol("published", normalisation="global", threshold="published", model_selection="test", point_adjust=True,
                          label_join="event_time", universe="today", fill="latency", pretrained_check=False),
    # every offline convention switched on (upper bound of the leakage premium; used on the planted testbed)
    "offline_all": Protocol("offline_all", normalisation="global", threshold="test_quantile", model_selection="test", point_adjust=True,
                            label_join="event_time", universe="today", fill="same_bar", pretrained_check=False),
    # purged walk-forward with offline normalisation / thresholds inside the test fold (the common compromise)
    "walkforward": Protocol("walkforward", normalisation="test_fold", threshold="test_quantile", model_selection="val", point_adjust=False,
                            label_join="avail_time", universe="as_of", fill="latency", pretrained_check=True),
    "causal": Protocol("causal"),
    "causal_adia": Protocol("causal_adia", detection_only=True),
}

# L8 is deliberately absent: it changes no input of any baseline (audit-card flag only, see module docstring).
ONE_SWITCH: Dict[str, Dict[str, Any]] = {
    "L1": {"normalisation": "global"}, "L2": {"threshold": "test_quantile"}, "L3": {"model_selection": "test", "threshold": "best_f1_test"},
    "L4": {"point_adjust": True}, "L5": {"label_join": "event_time"}, "L6": {"universe": "today"}, "L7": {"fill": "same_bar"},
}
CARD_ONLY_CHANNELS = ("L8",)
MODEL_SIDE_CHANNELS = ("L1", "L2", "L3")
GATE_KINDS = {"fixed": "fixed", "val_quantile": "fixed", "aci": "aci", "delayed_aci": "delayed_aci", "pot": "pot"}
_NORM_FROM_YAML = {"train": "train_fold", "global": "global", "test": "test_fold", "test_fold": "test_fold", "train_fold": "train_fold",
                   "streaming": "streaming"}


def one_switch(channel: str) -> Protocol:
    return replace(PRESETS["causal"], name=f"causal+{channel}", **ONE_SWITCH[channel])


def all_protocols() -> Dict[str, Protocol]:
    d = dict(PRESETS)
    d.update({f"causal+{ch}": one_switch(ch) for ch in ONE_SWITCH})
    return d


def get_protocol(name: str) -> Protocol:
    return all_protocols()[name]


def freeze(path: str = HASH_FILE, force: bool = False) -> Dict[str, str]:
    """Write the protocol hashes. If `path` already holds a freeze: identical hashes -> left untouched (timestamp
    kept); differing hashes -> RuntimeError unless force=True, in which case the old file is appended to
    <path>.history before rewriting."""
    hashes = {k: v.hash() for k, v in all_protocols().items()}
    old = read_frozen(path)
    if old:
        if old == hashes:
            return hashes
        if not force:
            changed = sorted(k for k in set(old) | set(hashes) if old.get(k) != hashes.get(k))
            raise RuntimeError(f"{path} already holds a different freeze (changed: {changed}). Re-freezing after test-fold runs "
                               f"invalidates them; pass force=True / --force-refreeze deliberately and say so in the paper.")
        with open(path + ".history", "a") as fh:
            fh.write(f"# superseded {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + open(path).read() + "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# frozen {time.strftime('%Y-%m-%d %H:%M:%S')}; sha256[:16] of each protocol's canonical JSON (name excluded)\n")
        for k, v in hashes.items():
            fh.write(f"{k} {v}\n")
    return hashes


MIRROR_DIR = os.path.join(ROOT, "configs", "causalgate", "protocols")


def write_mirrors(out_dir: str = MIRROR_DIR) -> List[str]:
    """Regenerate configs/causalgate/protocols/*.yaml (read-only mirrors) from all_protocols(); stale mirrors of
    protocols that no longer exist are removed."""
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    want = set()
    for k, v in all_protocols().items():
        d = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.to_dict().items()}
        d.update({"hash": v.hash(), "offline_channels": v.offline_channels,
                  "_note": "mirror of causalgate/eval/protocols.py (source of truth); regenerate with "
                           "`python3 -c 'from causalgate.eval.protocols import write_mirrors; write_mirrors()'` and re-freeze"})
        fn = k.replace("+", "_") + ".yaml"; want.add(fn)
        with open(os.path.join(out_dir, fn), "w") as fh:
            yaml.safe_dump(d, fh, sort_keys=False)
    for fn in os.listdir(out_dir):
        if fn.endswith(".yaml") and fn not in want:
            os.remove(os.path.join(out_dir, fn))
    return sorted(want)


def read_frozen(path: str = HASH_FILE) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        k, v = line.split()
        out[k] = v
    return out


def verify_frozen(path: str = HASH_FILE, strict: bool = True) -> Dict[str, bool]:
    frozen = read_frozen(path)
    if not frozen:
        if strict:
            raise RuntimeError(f"No frozen protocol hashes at {path}; run `python3 scripts/cg_run.py --freeze` before any test-fold run")
        return {}
    cur = {k: v.hash() for k, v in all_protocols().items()}
    ok = {k: (frozen.get(k) == cur[k]) for k in cur}
    bad = [k for k, v in ok.items() if not v]
    if bad and strict:
        raise RuntimeError(f"Protocol definitions changed after freezing: {bad}. Re-freeze deliberately (and say so in the paper).")
    return ok


# --------------------------------------------------------------------------------------- data-side switches
def _times_to_date(times: np.ndarray, idx: int) -> Optional[str]:
    v = int(times[idx])
    if v > 10 ** 15:                                   # ns since epoch
        return str(pd.Timestamp(v, unit="ns").date())
    return None


def apply_data_switches(panel: Panel, protocol: Protocol, fold: Fold) -> Panel:
    """Return a panel view implementing L5 / L6 for this fold (the original panel is never modified)."""
    default_availability(panel)
    q = clone_panel(panel)
    q.meta = dict(q.meta)
    q.meta.pop("decision_universe", None)
    if protocol.universe == "as_of":
        uni = get_avail(panel, "universe")
        if uni is not None:
            # decision-point filter on membership AT t (F_t-measurable); prices are NOT masked, so paths into a
            # removal are kept and exits are priced on the real path (ProtocolRunner._loaders applies the filter)
            q.meta["decision_universe"] = np.asarray(uni, dtype=bool)
    if protocol.label_join == "avail_time":
        train_end = int(fold.val[-1]) if len(fold.val) else int(fold.train[-1])
        late = (q.avail_label > train_end)
        late[fold.test[0]:] = False                    # test labels are evaluation targets, never model inputs
        q.labels = q.labels.copy(); q.labels[late] = 0 # unknowable at training time -> treated as normal
        if q.text is not None:
            _replace_late_docs(q)                          # documents published after their slot bar move to their availability bar
    q.meta["protocol"] = protocol.name
    return q


def _replace_late_docs(q: Panel) -> None:
    """Loaders place a document at its content-date row with avail_text = publication bar (so that the event-time
    join is literally 'use the row'). Under the availability join the document is moved to its publication bar
    (age 0, first free slot; dropped with a count in meta if no slot is free or the bar is outside the panel)."""
    T, N, M = q.text_mask.shape
    rows = np.broadcast_to(np.arange(T)[:, None, None], q.text_mask.shape)
    late = q.text_mask & (q.avail_text > rows)
    if not late.any():
        return
    text = q.text.copy(); mask = q.text_mask.copy(); age = q.text_age.copy() if q.text_age is not None else np.zeros((T, N, M), np.float32)
    avail = q.avail_text.copy()
    dropped = 0
    for t, i, m in zip(*np.nonzero(late)):
        a = int(avail[t, i, m])
        mask[t, i, m] = False; text[t, i, m] = 0.0
        if a >= T:
            dropped += 1; continue
        free = np.nonzero(~mask[a, i])[0]
        if len(free) == 0:
            dropped += 1; continue
        s = int(free[0])
        text[a, i, s] = q.text[t, i, m]; mask[a, i, s] = True; age[a, i, s] = 0.0; avail[a, i, s] = a
    q.text, q.text_mask, q.text_age, q.avail_text = text, mask, age, avail
    q.meta = dict(q.meta); q.meta["late_docs_moved"] = int(late.sum()) - dropped; q.meta["late_docs_dropped"] = dropped


# --------------------------------------------------------------------------------------- pricing (L7)
def reprice_same_bar(trades: pd.DataFrame, panel: Panel, latency: int) -> pd.DataFrame:
    """Same-bar fill: the decision at t is filled at the close of t instead of t + latency; same exit horizon, same
    cost. Gross return = log mid[t + hold] - log mid[t]."""
    tr = trades.copy()
    lp = np.log(panel.mid)
    t = tr["t"].to_numpy().astype(int); i = tr["asset"].to_numpy().astype(int); h = tr["hold"].to_numpy().astype(int)
    T = lp.shape[0]
    t_exit = np.clip(t + h, 0, T - 1)
    gross = tr["w"].to_numpy() * (lp[t_exit, i] - lp[t, i])
    gross = np.where(tr["frozen"].to_numpy().astype(bool), 0.0, gross)
    tr["pnl"] = np.nan_to_num(gross) - tr["cost"].to_numpy()
    tr["exit_t"] = t_exit
    return tr


def truncate_panel(panel: Panel, t: int) -> Panel:
    """The panel as known at bar t: rows > t cut, data with availability > t masked (PITView semantics; labels not
    yet knowable -> 0). Used by the prequential replay."""
    v = PITView(panel, t)
    rows = lambda a: (a[: t + 1] if isinstance(a, np.ndarray) else a)
    lab = v.labels
    kw = dict(times=v.times, assets=list(panel.assets), mid=v.mid, price_feat=v.price_feat.astype(panel.price_feat.dtype), liq=v.liq.astype(panel.liq.dtype),
              tradable=v.tradable, labels=np.nan_to_num(lab, nan=0.0).astype(panel.labels.dtype), bars_to_peak=rows(panel.bars_to_peak),
              lob=rows(panel.lob), text=v.text, text_mask=v.text_mask, text_age=rows(panel.text_age), graph=rows(panel.graph),
              graph_mask=v.graph_mask if panel.graph is not None else None, meta=dict(panel.meta))
    q = Panel(**kw)
    for k in PIT_FIELDS:
        a = get_avail(panel, k)
        if a is not None:
            setattr(q, k, a[: t + 1])
    return q


# --------------------------------------------------------------------------------------- runner
class ProtocolRunner:
    def __init__(self, panel: Panel, horizons: Sequence[int], bars_per_year: int, data_cfg: Dict[str, Any], split_cfg: Dict[str, Any],
                 cost_cfg: Dict[str, Any], device: torch.device, out_dir: str, seed: int = 42, dataset_name: str = "dataset"):
        self.panel, self.horizons, self.bars_per_year = panel, list(horizons), int(bars_per_year)
        self.d, self.sp, self.c = dict(data_cfg), dict(split_cfg), dict(cost_cfg)
        self.device, self.out_dir, self.seed, self.dataset_name = device, out_dir, int(seed), dataset_name
        self.latency = max(int(self.d.get("latency", 1)), 1)
        self.exec_max = int(self.d.get("exec_max", 4))
        self.lookback = int(self.d.get("lookback", 64))
        default_availability(panel)

    def folds(self, protocol: Protocol, max_folds: Optional[int] = None) -> List[Fold]:
        hmax = max(self.horizons) + self.latency + self.exec_max
        purge = max(int(self.sp.get("purge", 0)), hmax) if protocol.purge else 0
        embargo = int(self.sp.get("embargo", 0)) if protocol.embargo else 0
        sp = WalkForwardSplitter(self.panel.T, self.sp["train"], self.sp["val"], self.sp["test"], self.sp.get("step"), purge, embargo,
                                 self.sp.get("anchored", False), max_folds if max_folds is not None else self.sp.get("max_folds"))
        return list(sp)

    def cost_model(self, bps: float) -> CostModel:
        return CostModel(bps, self.c.get("use_half_spread", True), self.c.get("impact_eta", 0.1), self.c.get("delay_zeta", 0.05), self.c.get("notional", 0.05))

    def _loaders(self, panel: Panel, fold: Fold, batch_size: int = 512):
        rng = np.random.default_rng(self.seed)
        uni = panel.meta.get("decision_universe")
        def mk(idx, neg):
            ds = EventDataset(panel, idx, self.lookback, self.horizons, self.latency, self.exec_max, int(self.d.get("vol_window", 60)), neg_subsample=neg, rng=rng)
            if uni is not None and len(ds.index):                       # L6 as-of: membership at the decision bar only
                ds.index = ds.index[uni[ds.index[:, 0], ds.index[:, 1]]]
            return ds
        ds = {"train": mk(fold.train, self.d.get("neg_subsample")), "val": mk(fold.val, None), "test": mk(fold.test, None)}
        return {k: build_loader(v, batch_size, shuffle=(k == "train"), seed=self.seed) for k, v in ds.items()}

    @staticmethod
    def resolve_published(base_cfg: Dict[str, Any], cls) -> Dict[str, Any]:
        """P-O conventions of one baseline (precedence in the module docstring). Returns {'normalisation', 'threshold',
        'norm_source', 'threshold_source'}; a None value means 'leave the key unset so the class default applies'."""
        pub = dict(getattr(cls, "published_protocol", {}) or {})
        if "normalisation" in pub:
            norm, ns = pub["normalisation"], "class.published_protocol"
        elif base_cfg.get("published_norm") is not None:
            norm, ns = _NORM_FROM_YAML.get(str(base_cfg["published_norm"]), "train_fold"), "baseline yaml"
        else:
            norm, ns = None, "class default"
        if "threshold" in pub:
            thr, ts = pub["threshold"], "class.published_protocol"
        elif base_cfg.get("published_threshold") is not None:
            thr, ts = str(base_cfg["published_threshold"]), "baseline yaml"
        else:
            thr, ts = None, "class default"
        return {"normalisation": norm, "threshold": thr, "norm_source": ns, "threshold_source": ts}

    def baseline_cfg(self, base_cfg: Dict[str, Any], protocol: Protocol, fold: Fold, panel: Panel, cls) -> Dict[str, Any]:
        cfg = dict(base_cfg)
        cfg.update({"panel": panel, "lookback": self.lookback, "latency": self.latency, "bars_per_year": self.bars_per_year,
                    "cost_bps": float(protocol.train_cost_bps), "ret_col": int(panel.meta.get("ret_col", 0))})
        supported = set(getattr(cls, "supported_switches", ()) or ())
        requested: List[str] = []
        if protocol.name == "published":
            r = self.resolve_published(base_cfg, cls)
            cfg["_published_conventions"] = dict(r)
            # normalisation: the baseline's own convention (default for the scaffold: offline on its published_norm)
            norm = r["normalisation"]
            if norm == "streaming":
                cfg["norm_mode"] = "streaming"
            else:
                cfg["norm_mode"] = "offline"; requested.append("L1")
                if norm is None:                                     # class default decides 'global' vs train rows
                    cfg.pop("published_norm", None); cfg["norm_rows"] = fold.train
                elif norm == "global":
                    cfg["published_norm"] = "global"; cfg["norm_rows"] = None
                else:
                    cfg["published_norm"] = "rows"; cfg["norm_rows"] = fold.test if norm == "test_fold" else fold.train
            thr = r["threshold"]
            if thr in GATE_KINDS:
                cfg["thr_mode"] = "gate"; g = dict(cfg.get("gate", {})); g["type"] = GATE_KINDS[thr]; cfg["gate"] = g
            else:
                cfg["thr_mode"] = "published"; requested.append("L2")
                if thr is None:
                    cfg.pop("published_threshold", None)
                else:
                    cfg["published_threshold"] = thr
                    if thr == "best_f1_test":
                        requested.append("L3")
        else:
            norm, thr = protocol.normalisation, protocol.threshold
            if norm == "streaming":
                cfg["norm_mode"] = "streaming"
            else:
                cfg["norm_mode"] = "offline"; requested.append("L1")
                cfg["published_norm"] = "global" if norm == "global" else "rows"
                cfg["norm_rows"] = None if norm == "global" else (fold.test if norm == "test_fold" else fold.train)
            if thr in GATE_KINDS:
                cfg["thr_mode"] = "gate"
                g = dict(cfg.get("gate", {})); g["type"] = GATE_KINDS[thr]; cfg["gate"] = g
            else:
                cfg["thr_mode"] = "published"
                cfg["published_threshold"] = cfg.get("published_threshold", "test_quantile") if thr == "published" else thr
                requested.append("L3" if thr == "best_f1_test" else "L2")
        applied = [c for c in requested if c in supported]
        cfg["_applied_channels"] = applied
        cfg["_not_applicable_channels"] = [c for c in requested if c not in supported]
        if "L1" not in applied:
            cfg["norm_mode"] = "streaming"
        if not ({"L2", "L3"} & set(applied)):
            cfg["thr_mode"] = "gate"
        cfg["mode"] = "published" if applied else "causal"
        return cfg

    # ---- one fold
    def _build(self, name: str, base_cfg: Dict[str, Any], protocol: Protocol, fold: Fold):
        set_seed(self.seed + fold.k)
        cls = BASELINES.get(name)
        panel = apply_data_switches(self.panel, protocol, fold)
        loaders = self._loaders(panel, fold)
        cfg = self.baseline_cfg(base_cfg, protocol, fold, panel, cls)
        bl = cls(cfg, panel.dims(), self.horizons, self.device)
        return cls, panel, loaders, cfg, bl

    def run_fold(self, name: str, base_cfg: Dict[str, Any], protocol: Protocol, fold: Fold, save: bool = True,
                 replay_check: bool = True, tag: Optional[str] = None) -> Dict[str, Any]:
        cls, panel, loaders, cfg, bl = self._build(name, base_cfg, protocol, fold)
        t0 = time.time()
        bl.fit(loaders["train"], loaders["val"])
        if hasattr(bl, "prepare_test"):
            bl.prepare_test(loaders["test"])
        fit_sec = time.time() - t0
        batches = [to_device(b, self.device) for b in loaders["test"]]
        outs = [bl.predict(b) for b in batches]
        applied = list(cfg.get("_applied_channels", []))
        if cfg["mode"] == "published" and getattr(bl, "thr_mode", None) == "published" and getattr(bl, "published_threshold", None) == "best_f1_test" and "L3" not in applied:
            applied.append("L3")                                     # class-default best_f1_test threshold (B6, B10)
        res: Dict[str, Any] = {"dataset": self.dataset_name, "baseline": name, "protocol": protocol.name, "protocol_hash": protocol.hash(),
                               "seed": self.seed, "fold": fold.k, "fit_sec": fit_sec, "mode": cfg["mode"], "offline_channels": protocol.offline_channels,
                               "applied_channels": applied, "not_applicable_channels": list(cfg.get("_not_applicable_channels", [])),
                               "test_start_date": _times_to_date(panel.times, int(fold.test[0])), "eval": {}}
        if "_published_conventions" in cfg:
            pc = dict(cfg["_published_conventions"])
            pc["normalisation_used"] = (getattr(bl, "published_norm", None) if getattr(bl, "norm_mode", "") == "offline" else "streaming")
            pc["threshold_used"] = (getattr(bl, "published_threshold", None) if getattr(bl, "thr_mode", "") == "published" else "gate")
            res["published_conventions"] = pc
        grid = sorted(set([float(b) for b in protocol.cost_grid_bps] + [float(protocol.train_cost_bps)]))
        bench = np.nanmean(np.log(self.panel.mid[fold.test]) - np.log(self.panel.mid[fold.test - 1]), axis=1)
        trades_main = None
        for bps in grid:
            bt = EventBacktester(self.horizons, self.latency, self.cost_model(bps))
            trades = pd.concat([bt.collect(o, b) for o, b in zip(outs, batches)], ignore_index=True)
            if protocol.fill == "same_bar":
                trades = reprice_same_bar(trades, panel, self.latency)
            m = evaluate_outputs(trades, self.bars_per_year, len(fold.test), int(fold.test[0]), bench, self.panel.N)
            m.update(self._extra_metrics(trades, outs, protocol))
            res["eval"][f"{bps:g}"] = m
            if bps == float(protocol.train_cost_bps):
                trades_main = trades
        if replay_check and cfg["mode"] == "causal":
            rp = self.prequential_replay(name, base_cfg, protocol, fold, outs, batches, bench)
            res["replay"] = {k: v for k, v in rp.items() if not k.startswith("_")}
            res["replay_max_abs_dw"] = rp["max_abs_dw"]
            res["deployability_gap_zero"] = bool(rp["max_abs_dw"] < 1e-9 and rp["max_abs_dabstain"] < 1e-9)
        res["diag"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in getattr(bl, "diag", {}).items()}
        res["cutoff_date"] = getattr(bl, "cutoff_date", None)
        res["status"] = getattr(bl, "status", "")
        if hasattr(bl, "guard"):
            try:
                bl.guard.assert_causal(); res["guard_violations"] = list(bl.guard.violations)
            except Exception as e:                 # strict guard raised
                res["guard_violations"] = [str(e)]
        if save:
            d = os.path.join(self.out_dir, self.dataset_name, name, tag or protocol.name, f"seed{self.seed}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"fold{fold.k}.json"), "w") as fh:
                json.dump(res, fh, indent=2, default=_json_default)
            if trades_main is not None:
                trades_main.to_csv(os.path.join(d, f"trades_fold{fold.k}.csv"), index=False)
        res["_trades"] = trades_main
        res["_baseline"] = bl
        res["_panel"] = panel
        return res

    def prequential_replay(self, name: str, base_cfg: Dict[str, Any], protocol: Protocol, fold: Fold, outs: List[Dict[str, torch.Tensor]],
                           batches: List[Dict[str, torch.Tensor]], bench: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Deployability-gap check (plan Sec. 2.4): a FRESH baseline is built and refitted on the same data (same seed,
        same loaders), then fed the test window bar by bar in time order; at bar t its panel is `truncate_panel(.., t)`
        (rows > t cut, late data masked) and any cached panel-derived state (`_Z`, the streaming normaliser) is rebuilt
        from that view. The replayed decisions are compared with the batch backtest decisions, and
        DG = M_backtest - M_replay is computed with metrics.deployability_gap. `data_cfg['replay_max_bars']` (default 250;
        None = every test bar) limits the replay to the first K test bars -- a valid prefix of the stream, recorded as
        n_bars. Cost is O(K * T) because the panel-derived state is rebuilt from scratch at every bar."""
        cls, panel, loaders, cfg, bl2 = self._build(name, base_cfg, protocol, fold)
        norm0 = copy.deepcopy(getattr(bl2, "norm", None)) if hasattr(bl2, "norm") else None
        if norm0 is not None and hasattr(norm0, "guard"):
            norm0.guard = None
        bl2.fit(loaders["train"], loaders["val"])
        if hasattr(bl2, "prepare_test"):
            bl2.prepare_test(loaders["test"])
        keys = list(batches[0].keys()) if batches else []
        allb = {k: torch.cat([b[k] for b in batches]) for k in keys if torch.is_tensor(batches[0][k])}
        n = len(allb["t"]) if allb else 0
        t_all = allb["t"].cpu().numpy() if n else np.zeros(0, int)
        bars = np.unique(t_all)
        kmax = self.d.get("replay_max_bars", 250)
        if kmax is not None:
            bars = bars[: int(kmax)]
        rep: Dict[str, np.ndarray] = {}
        uses_panel = getattr(bl2, "panel", None) is not None
        for tb in bars:
            rows = np.nonzero(t_all == tb)[0]
            if uses_panel:
                view = truncate_panel(panel, int(tb))
                bl2.panel = view
                if isinstance(getattr(bl2, "cfg", None), dict):
                    bl2.cfg["panel"] = view
                if hasattr(bl2, "_Z"):
                    bl2._Z = None
                if norm0 is not None:
                    bl2.norm = copy.deepcopy(norm0)
            sub = {k: v[torch.as_tensor(rows)] for k, v in allb.items()}
            o = bl2.predict(sub)
            for k, v in o.items():
                if torch.is_tensor(v):
                    rep.setdefault(k, np.zeros((n,) + tuple(v.shape[1:]), dtype=v.detach().cpu().numpy().dtype))
                    rep[k][rows] = v.detach().cpu().numpy()
        done = np.isin(t_all, bars)
        cat = lambda k: torch.cat([o[k] for o in outs]).detach().cpu().numpy()
        dw = float(np.abs(cat("w")[done] - rep["w"][done]).max()) if done.any() else 0.0
        dab = float(np.abs(cat("abstain")[done] - rep["abstain"][done]).max()) if done.any() else 0.0
        out: Dict[str, Any] = {"kind": "prequential_refit_bar_by_bar", "n_bars": int(len(bars)), "n_rows": int(done.sum()),
                               "uses_truncated_panel": bool(uses_panel), "max_abs_dw": dw, "max_abs_dabstain": dab}
        try:
            sel = torch.as_tensor(np.nonzero(done)[0])
            b_sel = {k: v[sel] for k, v in allb.items()}
            o_bt = {k: torch.cat([o[k] for o in outs])[sel] for k in outs[0] if torch.is_tensor(outs[0][k])}
            o_rp = {k: torch.as_tensor(v[done], device=self.device) for k, v in rep.items()}
            bt = EventBacktester(self.horizons, self.latency, self.cost_model(float(protocol.train_cost_bps)))
            nb = len(bars)
            m_bt = evaluate_outputs(bt.collect(o_bt, b_sel), self.bars_per_year, nb, int(fold.test[0]), None, self.panel.N)
            m_rp = evaluate_outputs(bt.collect(o_rp, b_sel), self.bars_per_year, nb, int(fold.test[0]), None, self.panel.N)
            out.update(CM.deployability_gap(m_bt, m_rp))
        except Exception as e:  # noqa: BLE001  (metrics are a report; the decision comparison above is the check)
            out["dg_error"] = f"{type(e).__name__}: {e}"
        return out

    def _extra_metrics(self, trades: pd.DataFrame, outs: List[Dict[str, torch.Tensor]], protocol: Protocol) -> Dict[str, float]:
        """Detection metrics. Rows of `trades` follow the concatenated `outs` (same batch order). Unscored rows (NaN raw
        score: warm-up / abstention) are excluded from every detection metric and from the alarm-rate denominator.
        PA and event F1 are computed per asset along time on the (T, N) pivot.
            PA_* / event_*   at the test-best (oracle) threshold -- diagnostics
            f1_@thr          F1 of the pipeline's own realised alarms (1 - abstain), no PA
            PA_f1_@thr       the same with point-adjust
            F1_headline      PA_f1_@thr under a PA protocol (L4), else f1_@thr"""
        y = trades["y"].to_numpy(); s = trades["score"].to_numpy().astype(np.float64); t = trades["t"].to_numpy().astype(int); i = trades["asset"].to_numpy().astype(int)
        cat = lambda k: np.concatenate([o[k].detach().cpu().numpy() for o in outs]) if outs and k in outs[0] else None
        sc = cat("scored")
        if sc is None:
            raw = cat("score")
            sc = (raw > -1.0) if raw is not None else np.ones(len(s), bool)
        sc = np.asarray(sc).astype(bool)
        abst = cat("abstain")
        alarm = (1.0 - abst) > 0.5 if abst is not None else np.zeros(len(s), bool)
        out: Dict[str, float] = {}
        tt = t - t.min(); T = int(tt.max()) + 1; N = int(i.max()) + 1
        Y = np.zeros((T, N), int); S = np.full((T, N), np.nan); M = np.zeros((T, N), bool); A = np.zeros((T, N), bool)
        Y[tt, i] = y; M[tt[sc], i[sc]] = True; S[tt[sc], i[sc]] = s[sc]; A[tt, i] = alarm & sc
        ys, ss = y[sc], s[sc]
        thr = WM.best_f1(ys, ss)[1] if len(ys) and ys.sum() > 0 else np.inf
        out.update(CM.pa_event_panel(Y, S, M, thr))
        own = CM.pa_event_panel(Y, S, M, thr, alarms=A)
        out["PA_f1_@thr"] = own["PA_f1"]
        out.update({f"{k}_@thr": v for k, v in CM.f1_with_optional_pa(ys, (alarm[sc]).astype(float), 0.5, pa=False).items()})
        out["F1_headline"] = out["PA_f1_@thr"] if protocol.point_adjust else out["f1_@thr"]
        out["TS_AUC"] = CM.ts_auc_from_panel(Y, S)
        out["VUS_PR"] = CM.vus_pr_panel(np.where(M, Y, 0), S)
        if abst is not None:
            out["alarm_rate"] = float(alarm[sc].mean()) if sc.any() else float("nan")
        out["scored_frac"] = float(sc.mean()) if len(sc) else float("nan")
        return out

    # ---- all folds
    def run(self, name: str, base_cfg: Dict[str, Any], protocol: Protocol, max_folds: Optional[int] = None, tag: Optional[str] = None,
            alpha: Optional[float] = None) -> Dict[str, Any]:
        folds = self.folds(protocol, max_folds)
        results = [self.run_fold(name, base_cfg, protocol, f, tag=tag) for f in folds]
        agg = aggregate_folds(results, f"{float(protocol.train_cost_bps):g}")
        if alpha is not None and "alarm_rate" in agg:
            agg["alarm_rate_error"] = abs(agg["alarm_rate"] - alpha)
        return {"folds": results, "agg": agg, "protocol": protocol.to_dict(), "protocol_hash": protocol.hash()}


def aggregate_folds(results: List[Dict[str, Any]], bps_key: str) -> Dict[str, float]:
    keys = set()
    for r in results:
        keys |= set(r["eval"].get(bps_key, {}).keys())
    agg = {}
    for k in sorted(keys):
        vals = [r["eval"][bps_key][k] for r in results if k in r["eval"].get(bps_key, {}) and isinstance(r["eval"][bps_key][k], (int, float))]
        vals = [v for v in vals if np.isfinite(v)]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["n_folds"] = len(results)
    return agg


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------------------- P-C/ADIA (detection only)
def _onset_truth(panel: Panel, T0: int, T1: int) -> np.ndarray:
    """(T1-T0+1, N) int: 1 from the first event ONSET (<= step) on. Onsets from panel.meta['events'] (onset_t, asset)
    when present; otherwise the first bar of each warning-label segment shifted forward by meta['lead'] (labels are
    1 on [onset - lead, ...]). Onsets before T0 count only for events whose label segment is still on at T0 -- an
    event that ended before the test window is not a break of this window."""
    N = panel.N
    first = np.full(N, np.iinfo(np.int64).max)
    ev = panel.meta.get("events")
    if ev:
        for e in ev:
            o, a = int(e["onset_t"]), int(e["asset"])
            if o > T1:
                continue
            if o < T0 and not bool(panel.labels[T0, a] == 1):
                continue
            first[a] = min(first[a], o)
    else:
        lead = int(panel.meta.get("lead", 0))
        for a in range(N):
            for u0, u1 in CM._segments(panel.labels[:, a]):
                o = u0 + lead
                if o <= T1 and (o >= T0 or u1 > T0):
                    first[a] = min(first[a], o); break
    steps = np.arange(T0, T1 + 1)[:, None]
    return (steps >= first[None, :]).astype(int)


def run_adia(bl, panel: Panel, fold: Fold, lookback: int, horizons: Sequence[int], latency: int = 1, exec_max: int = 4) -> Dict[str, float]:
    """One observation at a time per series: after each new bar of the test segment the fitted baseline emits a
    cumulative break score (running max of its causal anomaly probability); the ground truth at step s is whether an
    event ONSET has occurred in the series by step s (`_onset_truth`; not the lead-extended warning label). Metric:
    TS-AUC over steps (ADIA Structural Break Real-Time Edition convention). The baseline must already be fitted; its
    gate is reset per series. Pass the protocol-switched panel the baseline was fitted on (ProtocolRunner.run_fold
    returns it as res['_panel'])."""
    ds = EventDataset(panel, fold.test, lookback, horizons, latency, exec_max)
    uni = panel.meta.get("decision_universe")
    if uni is not None and len(ds.index):
        ds.index = ds.index[uni[ds.index[:, 0], ds.index[:, 1]]]
    rows = ds.index
    T0, T1 = int(rows[:, 0].min()), int(rows[:, 0].max())
    N = panel.N
    cum = np.full((T1 - T0 + 1, N), np.nan)
    truth = _onset_truth(panel, T0, T1)
    loader = build_loader(ds, 1024, shuffle=False)
    if hasattr(bl, "gate"):
        bl.gate.reset_stream()
    probs = {}
    for b in loader:
        o = bl.predict(b)
        for k in range(len(b["t"])):
            probs[(int(b["t"][k]), int(b["asset"][k]))] = float(o["a_prob"][k])
    for i in range(N):
        best = -np.inf
        for t in range(T0, T1 + 1):
            p = probs.get((t, i))
            if p is not None:
                best = max(best, p)
            cum[t - T0, i] = best if np.isfinite(best) else np.nan
    return {"TS_AUC_adia": CM.ts_auc_from_panel(truth, cum), "n_steps": int(T1 - T0 + 1), "truth": "event_onset"}

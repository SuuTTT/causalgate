"""Adapter that exposes a fitted event baseline as an FPIT-auditable batch pipeline over a panel.

`run(panel)` rebuilds the baseline on the (possibly perturbed) panel with the same seed, fits it on the fold's train /
validation rows, runs `prepare_test` and `predict` over the test rows and returns (T, N) arrays:
    "normaliser" (first feature of the normalised window at t), "scorer" (score), "threshold", "policy" (weight),
    plus "score", "thr", "alarm", "weight", "pnl" (gross log-return PnL of the decision taken at t under the
    protocol's fill convention: `fill='latency'` fills at t + latency, `fill='same_bar'` (L7) at the close of t, as
    ProtocolRunner's reprice_same_bar does; exit at fill + hold). `latency` stays the real latency so the FPIT fill-gap
    probe shifts prices from t + 1 on and fires on a same-bar fill.
"thr" is the per-decision threshold: the baseline's own 'thr' output when it returns one, the published threshold in
published mode, otherwise the gate's per-step threshold q (read by wrapping gate.step inside the adapter).
With refit=False the model is fitted once, a snapshot of the fitted baseline is kept, and every run (the base run
included) starts from a fresh deep copy of that snapshot, so streaming state (normaliser, gate) never carries over
between runs; only the test-time operators see the perturbed panel. The default refit=True is the complete test (the
fitted parameters are part of the pipeline). Determinism: `set_seed` is called before every fit / run, so a causal
pipeline reproduces bit-identical decisions on an unperturbed-past panel.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, Optional, Sequence
import numpy as np
import torch

from warn2trade.data.base import EventDataset, build_loader
from warn2trade.data.schema import Panel
from warn2trade.data.splits import Fold
from warn2trade.utils.registry import BASELINES
from warn2trade.utils.seed import set_seed
from warn2trade.train.trainer import to_device


class EventBaselinePipeline:
    operators = ("normaliser", "scorer", "threshold", "policy")

    def __init__(self, name: str, cfg: Dict[str, Any], fold: Fold, horizons: Sequence[int], lookback: int, latency: int = 1,
                 exec_max: int = 4, hold_index: int = 2, device: Optional[torch.device] = None, seed: int = 42, refit: bool = True,
                 batch_size: int = 1024, fill: str = "latency"):
        assert fill in ("latency", "same_bar"), fill
        self.fill = fill
        self.name, self.cfg, self.fold = name, dict(cfg), fold
        self.horizons, self.L, self.latency, self.exec_max = list(horizons), int(lookback), int(latency), int(exec_max)
        self.hold = int(self.horizons[hold_index])
        self.device = device or torch.device("cpu")
        self.seed, self.refit, self.bs = int(seed), bool(refit), int(batch_size)
        self._fitted = None

    def _loaders(self, panel: Panel):
        mk = lambda idx: EventDataset(panel, idx, self.L, self.horizons, self.latency, self.exec_max)
        return {k: build_loader(mk(getattr(self.fold, k)), self.bs, shuffle=(k == "train"), seed=self.seed) for k in ("train", "val", "test")}

    def _build(self, panel: Panel):
        cfg = dict(self.cfg); cfg["panel"] = panel
        return BASELINES.get(self.name)(cfg, panel.dims(), self.horizons, self.device)

    def run(self, panel: Panel) -> Dict[str, np.ndarray]:
        ld = self._loaders(panel)
        if self.refit:
            set_seed(self.seed)
            bl = self._build(panel)
            bl.fit(ld["train"], ld["val"])
        else:
            if self._fitted is None:
                set_seed(self.seed)
                fitted = self._build(panel)
                fitted.fit(ld["train"], ld["val"])
                self._fitted = copy.deepcopy(fitted, memo={id(panel): panel})
            set_seed(self.seed)
            old_panel = getattr(self._fitted, "panel", None)
            bl = copy.deepcopy(self._fitted, memo={id(old_panel): panel} if old_panel is not None else {})
            bl.panel = panel
            if hasattr(bl, "_Z"):
                bl._Z = None                                  # re-normalise on the perturbed panel (offline scalers refit here)
            if hasattr(bl, "gate") and hasattr(bl.gate, "reset_stream"):
                bl.gate.reset_stream()
            bl._last_t = -1
        q_log = []
        gate = getattr(bl, "gate", None)
        if gate is not None and hasattr(gate, "step") and getattr(bl, "thr_mode", "") != "published":
            _step = gate.step
            def _rec_step(*a, **k):
                r = _step(*a, **k)
                q_log.append(float(r[1]) if isinstance(r, tuple) and len(r) > 1 else np.nan)
                return r
            gate.step = _rec_step
        if hasattr(bl, "prepare_test"):
            bl.prepare_test(ld["test"])
        T, N = panel.mid.shape
        out = {k: np.full((T, N), np.nan) for k in ("normaliser", "scorer", "threshold", "policy", "alarm")}
        for b in ld["test"]:
            b = to_device(b, self.device)
            q_log.clear()
            o = bl.predict(b)
            t = b["t"].cpu().numpy(); i = b["asset"].cpu().numpy()
            if hasattr(bl, "_windows"):
                X = bl._windows(b); out["normaliser"][t, i] = X[:, -1, 0]
            out["scorer"][t, i] = o["score"].cpu().numpy() if "score" in o else o["a_prob"].cpu().numpy()
            if "thr" in o:
                out["threshold"][t, i] = o["thr"].cpu().numpy()
            elif getattr(bl, "thr_mode", "") == "published":
                out["threshold"][t, i] = getattr(bl, "_thr_published", np.nan)
            elif len(q_log) == len(t):
                # ScoreToTradeBaseline.predict steps the gate in lexsort((asset, t)) order and discards q: map it back
                out["threshold"][t[np.lexsort((i, t))], i[np.lexsort((i, t))]] = np.asarray(q_log)
            out["policy"][t, i] = o["w"].cpu().numpy()
            out["alarm"][t, i] = 1.0 - o["abstain"].cpu().numpy() if "abstain" in o else (o["w"].cpu().numpy() != 0).astype(float)
        test_rows = np.isfinite(out["scorer"])
        if test_rows.any() and np.isnan(out["threshold"][test_rows]).all():
            import warnings
            warnings.warn(f"{self.name}: no per-decision threshold could be read; FPIT cannot check the threshold operator")
        w = np.nan_to_num(out["policy"])
        lp = np.log(panel.mid)
        pnl = np.full((T, N), np.nan)
        lam = 0 if self.fill == "same_bar" else self.latency
        hi = T - lam - self.hold
        pnl[:hi] = w[:hi] * (lp[lam + self.hold : lam + self.hold + hi] - lp[lam : lam + hi])
        out.update({"score": out["scorer"], "thr": out["threshold"], "weight": out["policy"], "pnl": pnl})
        return out

"""B4 MemStream (Bhatia, Jain, Srivastava, Kawaguchi, Hooi, WWW 2022) as a score -> alarm -> trade baseline.

Official code: https://github.com/Stream-AD/MemStream  (code/memstream.py = the autoencoder-feature variant used in the
paper's main table; memstream-pca.py / memstream-ib.py / memstream-knn.py are the other feature extractors). The
official file is an argparse script (parses sys.argv at import, memstream.py ~L15-22) and cannot be imported as a
module, so the algorithm is re-implemented here *statement for statement* (status 'reimplemented'); THIRD_PARTY.md says
how the vendored copy is detected for the record.

Algorithm (memstream.py, line numbers ~ of the 2026-10-03 fetch):
    feature extractor   encoder Linear(in, 2 in) + Tanh, decoder Linear(2 in, in) (~L60-66), trained full-batch on the
                        initial memory records with Gaussian input noise 1e-3, Adam lr 1e-2, 5000 epochs (~L74-84)
    standardisation     (x - mem_data.mean(0)) / mem_data.std(0), zero where std == 0 - statistics of the records
                        currently IN MEMORY, recomputed on every memory update (~L75-77, ~L92, ~L106-107)
    memory              N = memlen records (default 2048, ~L20); initial memory = the first N records with label 0
                        (`init_data = numeric[labels == 0][:N]`, ~L130) - the official initialisation reads labels
    score               L1 distance to the nearest memory entry (k = 1): torch.norm(memory - enc(x), dim=1, p=1).min() (~L109)
    update              if score <= beta: FIFO replacement at count % N, memory + mem_data + mean/std updated (~L87-95)
    evaluation          ROC-AUC of the scores over the whole stream (~L141) - no threshold; beta is a per-dataset hyper
Here each record is the (flattened) last `record_bars` bars of the scaffold-normalised window (default 1 bar = one
multivariate observation, as in the paper's streaming setting); the memory statistics standardise on top of the
scaffold's L1 normalisation. Every `_score` call that carries (t, asset) bookkeeping is a prequential stream in (t, asset)
order (score with the memory built from earlier records, then admit the record if score <= beta): the scaffold's train and
validation passes in `fit` and the published-threshold pass in `prepare_test` all update the memory, as in the official
single pass. The replay snapshot is the memory after the train + val stream; a pass that starts strictly before the last
streamed bar (a replay, or the first predict pass after `prepare_test`) restores it, so repeated test passes are identical
(DG = 0). Only rows without (t, asset) bookkeeping are scored read-only.
"""
from __future__ import annotations
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import ScoreToTradeBaseline, register
from .b1_cpd_dmn import native_threshold_requested
from ..data.pit import AsOfStamped


THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party")


class _MemoryStamp(AsOfStamped):
    pass


class MemStreamCore:
    """Faithful re-implementation of class MemStream (memstream.py ~L46-110) with a vectorised read-only scorer."""

    def __init__(self, in_dim: int, memory_len: int, beta: Optional[float], lr: float, epochs: int, noise: float, seed: int, device: torch.device):
        torch.manual_seed(seed)
        self.in_dim, self.out_dim = int(in_dim), int(in_dim) * 2
        self.memory_len, self.beta, self.lr, self.epochs, self.noise = int(memory_len), beta, float(lr), int(epochs), float(noise)
        self.device = device
        self.encoder = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), nn.Tanh()).to(device)
        self.decoder = nn.Sequential(nn.Linear(self.out_dim, self.in_dim)).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim, device=device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim, device=device)
        self.mean = torch.zeros(self.in_dim, device=device); self.std = torch.ones(self.in_dim, device=device)
        self.count = 0
        self.n_updates = 0

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        return torch.nan_to_num(new, nan=0.0, posinf=0.0, neginf=0.0)

    def _stats(self) -> None:
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)

    def fit(self, init_data: torch.Tensor) -> None:
        """train_autoencoder (~L74-84) then initialize_memory (~L97-103) on the initial memory records."""
        init_data = init_data.to(self.device)
        n = init_data.shape[0]
        if n < self.memory_len:                                   # the official code assumes >= memlen normal records
            reps = int(np.ceil(self.memory_len / max(n, 1)))
            init_data = init_data.repeat(reps, 1)[: self.memory_len]
        self.mem_data = init_data[: self.memory_len].clone()
        self._stats()
        new = self._norm(self.mem_data)
        opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        loss_fn = nn.MSELoss()
        with torch.enable_grad():
            for _ in range(self.epochs):
                opt.zero_grad()
                out = self.decoder(self.encoder(new + self.noise * torch.randn_like(new)))
                loss = loss_fn(out, new)
                loss.backward()
                opt.step()
        with torch.no_grad():
            self.memory = self.encoder(self._norm(self.mem_data)).detach()
        self.count = 0

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Read-only: L1 distance to the nearest memory entry for every row (k = 1)."""
        e = self.encoder(self._norm(x.to(self.device)))
        return torch.cdist(e, self.memory, p=1).min(1).values

    @torch.no_grad()
    def stream(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential forward + update_memory (~L87-95, ~L105-110) over the rows of x in order."""
        out = torch.empty(x.shape[0], device=self.device)
        for r in range(x.shape[0]):
            xr = x[r : r + 1].to(self.device)
            e = self.encoder(self._norm(xr))
            d = torch.norm(self.memory - e, dim=1, p=1).min()
            out[r] = d
            if self.beta is not None and bool(d <= self.beta):
                pos = self.count % self.memory_len
                self.memory[pos] = e[0]
                self.mem_data[pos] = xr[0]
                self._stats()
                self.count += 1
                self.n_updates += 1
        return out

    def snapshot(self) -> Dict[str, Any]:
        return {"memory": self.memory.clone(), "mem_data": self.mem_data.clone(), "mean": self.mean.clone(), "std": self.std.clone(),
                "count": self.count, "n_updates": self.n_updates}

    def restore(self, s: Dict[str, Any]) -> None:
        self.memory, self.mem_data, self.mean, self.std = s["memory"].clone(), s["mem_data"].clone(), s["mean"].clone(), s["std"].clone()
        self.count, self.n_updates = s["count"], s["n_updates"]


# --------------------------------------------------------------------------------------- baseline
@register("cg_memstream")
class MemStreamBaseline(ScoreToTradeBaseline):
    name = "cg_memstream"
    paper = "Bhatia, Jain, Srivastava, Kawaguchi, Hooi. MemStream: Memory-Based Streaming Anomaly Detection. WWW 2022"
    official_repo = "https://github.com/Stream-AD/MemStream"
    status = "reimplemented"
    year = 2022
    cutoff_date = None
    conventions = {
        "L1": "official: records standardised with the mean / std of the records currently in memory (memstream.py ~L75-77 in "
              "train_autoencoder, ~L92 after every update, ~L106-107 in forward); initial statistics = the first N *normal* records. "
              "Causal apart from the label-based initial memory. Here: scaffold L1 first, then the same memory statistics.",
        "L2": "official: no alarm threshold - the paper reports ROC-AUC of the raw scores over the whole stream (~L141 "
              "metrics.roc_auc_score); the only decision threshold is beta, the memory-admission rule `output_loss <= max_thres` (~L88; "
              "default 0.1, ~L16, tuned per dataset in the README). published here: alarm <=> not admitted (score > beta) "
              "(published_rule 'beta'; 'quantile' switches to the scaffold's train_quantile); causal: delayed-ACI gate on the scores.",
        "L3": "official: initial memory `numeric[labels == 0][:N]` (~L130) uses the labels of the first N normal records; beta / memlen "
              "chosen per dataset with the test labels visible (README table). Here: published mode keeps the official label filter on "
              "the first N TRAIN-fold records (labels knowable by t_end), causal mode takes the first N training records regardless of "
              "label (cfg['model']['init_normal_only'] overrides); beta = quantile of the initial-memory leave-one-out distances unless given.",
        "L4": "official: none (ROC-AUC). Here: event-wise metrics.",
        "L5": "not applicable (records arrive in order; the official stream is the dataset file order).",
        "L6": "not applicable.",
        "L7": "official: score of record t is produced before the memory update with record t (forward ~L105-110) - prequential. Here: "
              "same; fill at t + latency by the protocol runner.",
        "L8": "none.",
    }
    unspecified = [
        "Record definition for a windowed panel: the last `record_bars` bars of the normalised window flattened (default 1).",
        "beta: official value 0.1 is on the raw-feature scale of their datasets; here cfg['model']['beta'] or, when null, the "
        "`beta_quantile` (default 0.95) quantile of the leave-one-out nearest-neighbour L1 distances inside the initial memory "
        "(the records' own scores are 0 because they sit in the memory).",
        "Memory size N (official 2048) and AE epochs (official 5000 full-batch) are reduced in the smoke config (256 / 200).",
        "Fewer than N training records: the initial data is tiled to length N (official code assumes >= N normal records).",
        "Streaming order inside a batch: (t, asset) lexicographic; the official stream is univariate in time.",
        "Scaffold train / val / published-threshold passes are prequential streams with memory updates (as the official single pass); "
        "predict replays from the post-calibration snapshot; only rows without (t, asset) bookkeeping are scored read-only.",
        "Label-filtered initial memory (official `labels == 0`) only when labels may be read (as-published row or an L3 threshold).",
    ]

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], horizons, device: torch.device):
        cfg = dict(cfg)
        cfg.setdefault("published_threshold", "train_quantile")
        cfg.setdefault("published_norm", "train")
        super().__init__(cfg, dims, horizons, device)
        m = dict(cfg.get("model", {}))
        self.record_bars = int(m.get("record_bars", 1))
        self.memory_len = int(m.get("memory_len", 2048))
        self.beta_cfg = m.get("beta", None)
        self.beta_quantile = float(m.get("beta_quantile", 0.95))
        self.lr = float(m.get("lr", 1e-2)); self.epochs = int(m.get("epochs", 5000)); self.noise = float(m.get("noise", 1e-3))
        self.seed = int(m.get("seed", 0))
        ino = m.get("init_normal_only", None)                       # None -> official label filter only when labels may be read
        self.init_normal_only = self._label_select_default() if ino is None else bool(ino)
        self.published_rule = str(m.get("published_rule", "beta")); assert self.published_rule in ("beta", "quantile"), self.published_rule
        self.core: Optional[MemStreamCore] = None
        self._snap: Optional[Dict[str, Any]] = None
        self._pending: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._pending_y: Optional[np.ndarray] = None
        self._stream_last_t = -1
        self.official_path = os.path.join(THIRD_PARTY, "MemStream", "code", "memstream.py")
        self.diag["official_import"] = ("script present (argparse at import: not importable, re-implementation used)"
                                        if os.path.exists(self.official_path) else "third_party/MemStream not found")
        self._stamp = _MemoryStamp()
        self.guard.register(self._stamp)

    def _label_select_default(self) -> bool:
        """The official label-filtered initial memory reads training labels (an L3-type choice), so it is keyed on the label /
        threshold switches, never on the combined `mode`: an explicit runner flag (`label_select` / `as_published`) when present,
        else on for the as-published row (offline normalisation + the native threshold) or a best_f1_test (L3) threshold."""
        if "_published_conventions" in self.cfg or "_applied_channels" in self.cfg:   # ProtocolRunner row
            return self.thr_mode == "published" and (self.published_threshold == "best_f1_test" or "_published_conventions" in self.cfg)
        for key in ("label_select", "as_published"):
            if key in self.cfg:
                return bool(self.cfg[key])
        if self.thr_mode != "published":
            return False
        return self.published_threshold == "best_f1_test" or (self.norm_mode == "offline" and native_threshold_requested(self))

    # ---- bookkeeping
    def _windows(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        self._pending = (batch["t"].cpu().numpy(), batch["asset"].cpu().numpy())
        return super()._windows(batch)

    def _collect(self, loader: DataLoader):
        Xs, ys, ts, iis = [], [], [], []
        for b in loader:
            Xs.append(self._windows(b)); ys.append(b["y"].cpu().numpy()); ts.append(b["t"].cpu().numpy()); iis.append(b["asset"].cpu().numpy())
        if not Xs:
            self._pending = None
            return np.zeros((0, self.L, self.dims["price"]), np.float32), np.zeros(0), np.zeros(0, dtype=int)
        t = np.concatenate(ts); i = np.concatenate(iis); y = np.concatenate(ys)
        self._pending = (t, i); self._pending_y = y
        return np.concatenate(Xs), y, t

    def _records(self, X: np.ndarray) -> torch.Tensor:
        Xn = np.asarray(X, dtype=np.float32)
        r = Xn[:, -self.record_bars :, :].reshape(Xn.shape[0], -1)
        return torch.as_tensor(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))

    # ---- model
    def _fit_model(self, X: np.ndarray, t_end: int) -> None:
        rec = self._records(X)
        order = np.arange(len(rec))
        if self._pending is not None and len(self._pending[0]) == len(rec):
            t, i = self._pending
            order = np.lexsort((i, t))                                      # time order: the FIRST N records form the memory
        ok = ~np.isnan(np.asarray(X, dtype=np.float32)[:, -1]).any(-1)
        if self.init_normal_only and self._pending_y is not None and len(self._pending_y) == len(rec):
            ok &= (self._pending_y == 0)
        order = order[ok[order]]
        init = rec[order[: self.memory_len]] if len(order) else rec[: self.memory_len]
        self.core = MemStreamCore(rec.shape[1], self.memory_len, None, self.lr, self.epochs, self.noise, self.seed, self.device)
        self.core.fit(init)
        if self.beta_cfg is not None:
            self.core.beta = float(self.beta_cfg)
        else:                                                                # leave-one-out NN distances inside the initial memory
            mem = self.core.memory
            d = torch.cdist(mem, mem, p=1)
            d.fill_diagonal_(float("inf"))
            s0 = d.min(1).values.cpu().numpy()
            s0 = s0[np.isfinite(s0)]
            self.core.beta = float(np.quantile(s0, self.beta_quantile)) if s0.size else None
        self._snap = self.core.snapshot()
        self._stamp.stamp(int(t_end))
        self.diag.update({"status": self.status, "beta": self.core.beta, "memory_len": self.memory_len, "record_dim": int(rec.shape[1])})

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Stream the rows in (t, asset) order (official: one pass over the whole record stream, memory updated after
        each score). A pass that starts strictly before the last streamed bar is a replay / second pricing pass and restarts
        from the snapshot taken after calibration, so repeated passes over the test window are identical (DG = 0).
        Rows without (t, asset) bookkeeping are scored read-only."""
        Xn = np.asarray(X, dtype=np.float32)
        if len(Xn) == 0:
            return np.zeros(0)
        rec = self._records(Xn)
        if self._pending is not None and len(self._pending[0]) == len(rec):
            t, i = self._pending
            order = np.lexsort((i, t))
            if int(t.min()) < self._stream_last_t and self._snap is not None:   # strictly earlier bar = replay (a bar may straddle batches)
                self.core.restore(self._snap)
            s = np.empty(len(rec))
            s[order] = self.core.stream(rec[order]).cpu().numpy()
            self._stream_last_t = int(t.max())                              # prequential: the state used at bar t holds bars < t only,
                                                                            # so the as-of stamp stays at the calibration end
        else:
            s = self.core.score(rec).cpu().numpy()
        s = s.astype(np.float64)
        s[np.isnan(Xn[:, -1]).any(-1)] = np.nan
        self.diag["memory_updates"] = self.core.n_updates
        return s

    def _published_threshold(self, s_train, s_val, s_test, y_test) -> float:
        """published_rule 'beta': alarm <=> the record is NOT admitted to memory (score > beta), the only decision threshold
        of the algorithm (memstream.py ~L88 `if output_loss <= self.max_thres`); 'quantile': the scaffold rule
        (`published_threshold`, default train_quantile of the streamed train scores). A protocol-requested kind (test_quantile
        / combined_quantile / best_f1_test from an L2 / L3 switch, P-W or offline_all) is always honoured."""
        if self.published_rule == "beta" and native_threshold_requested(self) and self.core is not None and self.core.beta is not None:
            self.diag["published_rule"] = "beta"
            return float(self.core.beta)
        self.diag["published_rule"] = self.published_threshold
        return super()._published_threshold(s_train, s_val, s_test, y_test)

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        self._stream_last_t = -1
        self._snap = None
        super().fit(train, val)                                              # streams train then val in time order
        self._snap = self.core.snapshot()                                    # replay point = memory state after calibration

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._stamp.check(int(batch["t"].min()))
        return super().predict(batch)

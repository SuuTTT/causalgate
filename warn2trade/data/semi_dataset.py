"""Label-budget protocol + the semi-supervised dataset consumed by SemiSupervisedTrainer.

Supervision available to a model trained for a fold [train_lo, as_of_t]:
    y_l = 1   on the warning windows of the K sampled anchors (nested per class, budget seed) that are available under the
              protocol at as_of_t AND whose conduct window intersects the training window
    y_l = 0   on confident normals: far (> neg_margin bars, same asset) from every *sampled* anchor and every LF positive.
              Unsampled anchors do not shape the negatives (an honest K-case budget).
    y_l = -1  everything else = the unlabelled pool.
A calibration set C0 (n_cal rows, or a fraction of the pool if n_cal < 1) is carved out of the unlabelled pool *uniformly at random at block level* (at most one row per
asset x cal_block bars, to weaken serial dependence). C0 rows are never labelled, never pseudo-labelled and never enter a
supervised or consistency loss; they share the SSL stages with the unlabelled pool so the gating score is a symmetric
function of C0 and D_u (needed for Proposition 2 in docs/sparse_label_research_plan.md).
Evaluation labels (y, bars_to_peak, event_id) are overridden from the ORACLE anchor raster when given, so that metrics
are computed against regulatory cases and not against the rule labels the weak supervision imitates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .base import EventDataset, build_loader
from .schema import Panel
from .anchors import available_anchors, sample_k_shot, anchor_label_arrays, confident_normal_pool, budget_report


@dataclass
class LabelBudget:
    k_per_class: Optional[int] = 10        # None = every anchor available under the protocol ("Full")
    n_neg_per_pos: int = 20
    min_neg: int = 50
    neg_margin: int = 20
    lead: int = 3
    seed: int = 7                          # BUDGET seed (which cases), decoupled from the training seed
    protocol: str = "release"


def make_labeled_arrays(panel: Panel, anchor_table: pd.DataFrame, as_of_t: int, budget: LabelBudget,
                        votes: Optional[np.ndarray] = None, train_lo: int = 0, neg_seed: int = 0) -> Dict[str, Any]:
    T, N = panel.mid.shape
    asset_index = {a: i for i, a in enumerate(panel.assets)}
    avail = available_anchors(anchor_table, as_of_t, budget.protocol)
    if budget.protocol != "oracle":
        avail = avail[(avail["end_t"] >= train_lo) & (avail["onset_t"] <= as_of_t)]   # usable inside this training window
    shot = sample_k_shot(avail, budget.k_per_class, budget.seed)
    classes = sorted(anchor_table["anomaly_class"].unique().tolist())
    arrs = anchor_label_arrays(shot, T, N, asset_index, budget.lead, class_names=classes)
    y_l = np.full((T, N), -1, dtype=np.int8)
    y_l[arrs["labels"] == 1] = 1
    y_l[: train_lo] = -1
    y_l[as_of_t + 1:] = -1
    excl = arrs["labels"].astype(bool)
    if votes is not None:
        excl |= (votes == 1).any(-1)
    valid = ~np.isnan(panel.mid)
    valid[: train_lo] = False
    valid[as_of_t + 1:] = False
    pool = confident_normal_pool(excl, budget.neg_margin, valid)
    n_pos = int((y_l == 1).sum())
    n_neg = max(budget.min_neg, budget.n_neg_per_pos * n_pos)
    cand = np.argwhere(pool)
    rng = np.random.default_rng(neg_seed)
    if len(cand):
        sel = cand[rng.choice(len(cand), size=min(n_neg, len(cand)), replace=False)]
        y_l[sel[:, 0], sel[:, 1]] = 0
    rep = budget_report(avail, shot, budget.k_per_class)
    return {"y_l": y_l, "anchors": shot, "available": avail, "n_pos": n_pos, "n_neg": int((y_l == 0).sum()),
            "n_cases": rep["k_eff_cases"], "budget": rep, "event_id": arrs["event_id"]}


def eval_label_arrays(panel: Panel, anchor_table: Optional[pd.DataFrame], mode: str = "anchors", lead: int = 3,
                      horizons: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Ground truth for EVALUATION. mode: 'anchors' (oracle anchor raster: every case, hindsight allowed for scoring),
    'rule' (Panel.labels = the builder's rule/proxy labels), 'union'. Event ids are anchor row ids for anchors and
    run-length ids for rule events (offset so they never collide)."""
    T, N = panel.mid.shape
    out_y = np.zeros((T, N), dtype=np.int8)
    out_btp = np.full((T, N), np.nan)
    out_eid = np.full((T, N), -1, dtype=np.int64)
    if mode in ("anchors", "union"):
        if anchor_table is None or len(anchor_table) == 0:
            raise ValueError("eval.labels='anchors' needs a non-empty anchor table")
        arr = anchor_label_arrays(anchor_table, T, N, {a: i for i, a in enumerate(panel.assets)}, lead)
        out_y |= arr["labels"].astype(np.int8)
        out_btp = np.where(arr["labels"] == 1, arr["bars_to_peak"], out_btp)
        out_eid = np.where(arr["labels"] == 1, arr["event_id"], out_eid)
    if mode in ("rule", "union"):
        rule = panel.labels.astype(bool)
        rid = _run_ids(rule) + 10_000_000
        take = rule & (out_y == 0)
        out_y[take] = 1
        out_btp = np.where(take, panel.bars_to_peak, out_btp)
        out_eid = np.where(take, rid, out_eid)
    if mode not in ("anchors", "rule", "union"):
        raise ValueError(mode)
    return {"y": out_y, "bars_to_peak": out_btp, "event_id": out_eid}


def _run_ids(mask: np.ndarray) -> np.ndarray:
    T, N = mask.shape
    ids = np.full((T, N), -1, dtype=np.int64)
    cur = 0
    for i in range(N):
        prev = False
        for t in range(T):
            if mask[t, i]:
                if not prev:
                    cur += 1
                ids[t, i] = cur
            prev = bool(mask[t, i])
    return ids


_PANEL_CACHE: Dict[Any, Dict[str, np.ndarray]] = {}


def panel_cache(panel: Panel, horizons: Sequence[int], vol_window: int) -> Dict[str, np.ndarray]:
    """impact (T,N,H), sigma (T,N), logret (T,N) and the row-validity mask of EventDataset(drop_nan=True), computed once."""
    from .labels import impact_curves, realized_vol
    key = (id(panel), tuple(horizons), int(vol_window))
    if key not in _PANEL_CACHE:
        impact = impact_curves(panel.mid, list(horizons))
        sigma = realized_vol(panel.mid, vol_window)
        logret = np.zeros_like(panel.mid)
        logret[1:] = np.log(panel.mid[1:]) - np.log(panel.mid[:-1])
        ok = ~np.isnan(panel.mid) & ~np.isnan(sigma) & ~np.isnan(impact).any(-1) & ~np.isnan(panel.price_feat).any(-1)
        _PANEL_CACHE.clear()                                              # one panel per process
        _PANEL_CACHE[key] = {"impact": impact, "sigma": sigma, "logret": logret, "ok": ok}
    return _PANEL_CACHE[key]


class SemiEventDataset(EventDataset):
    """EventDataset + votes, budgeted supervision y_l, a held-out calibration set, gate / pseudo-label state and
    optional evaluation-label overrides.

    Unlabelled sub-sampling is uniform (never conditioned on any label) so the pool keeps its natural anomaly rate.
    """

    def __init__(self, panel: Panel, time_idx: np.ndarray, lookback: int, horizons: Sequence[int], votes: np.ndarray,
                 y_l: np.ndarray, latency: int = 1, exec_max: int = 4, vol_window: int = 60,
                 unlabeled_subsample: Optional[float] = None, rng: Optional[np.random.Generator] = None,
                 lf_lookahead: Optional[np.ndarray] = None, n_cal: float = 0, cal_block: int = 20,
                 eval_labels: Optional[Dict[str, np.ndarray]] = None):
        self._init_cached(panel, time_idx, lookback, horizons, latency, exec_max, vol_window)
        assert votes.shape[:2] == panel.mid.shape and y_l.shape == panel.mid.shape
        self.votes, self.y_l = votes, y_l
        self.lf_lookahead = np.zeros(votes.shape[-1], dtype=int) if lf_lookahead is None else np.asarray(lf_lookahead, dtype=int)
        rng = rng or np.random.default_rng(0)
        T, N = panel.mid.shape
        # ---- calibration set: block-level uniform draw from the unlabelled rows of this window
        self.cal = np.zeros((T, N), dtype=bool)
        if n_cal > 0:
            unl = np.where(self.y_l[self.index[:, 0], self.index[:, 1]] < 0)[0]
            if 0 < float(n_cal) < 1:                                     # fraction of the unlabelled pool
                n_cal = int(round(float(n_cal) * len(unl)))
            if len(unl):
                blocks = self.index[unl, 1].astype(np.int64) * (T // max(cal_block, 1) + 1) + self.index[unl, 0] // max(cal_block, 1)
                ub = np.unique(blocks)
                pick_blocks = rng.permutation(ub)[: int(n_cal)]
                chosen = []
                for b in pick_blocks:
                    rows = unl[blocks == b]
                    chosen.append(rows[rng.integers(len(rows))])
                ch = np.asarray(chosen, dtype=np.int64)
                self.cal[self.index[ch, 0], self.index[ch, 1]] = True
        # ---- uniform unlabelled sub-sampling (calibration and labelled rows always kept)
        if unlabeled_subsample is not None and unlabeled_subsample < 1.0:
            keep_always = (self.y_l[self.index[:, 0], self.index[:, 1]] >= 0) | self.cal[self.index[:, 0], self.index[:, 1]]
            keep = keep_always | (rng.random(len(self.index)) < unlabeled_subsample)
            self.index = self.index[keep]
        self.pseudo = np.full((T, N), np.nan, dtype=np.float32)      # label-model posterior
        self.gate_ok = np.ones((T, N), dtype=bool)                    # BH-conformal selection (positives must be in it)
        self.omega = np.ones((T, N), dtype=np.float32)                # outcome weight, applied to teacher positives
        self.eval = eval_labels

    def _init_cached(self, panel: Panel, time_idx, lookback, horizons, latency, exec_max, vol_window) -> None:
        """Same fields and filtering as EventDataset.__init__ (drop_nan=True, no label-based sub-sampling), but the
        full-panel impact / volatility / log-return arrays are computed once per (panel, horizons, vol_window) and shared."""
        self.p, self.L, self.horizons = panel, lookback, list(horizons)
        self.hmax = max(self.horizons) + latency + exec_max
        self.latency, self.vol_window = latency, vol_window
        c = panel_cache(panel, self.horizons, vol_window)
        self._impact, self._sigma, self._logret = c["impact"], c["sigma"], c["logret"]
        T, N = panel.mid.shape
        lo, hi = max(int(np.min(time_idx)), lookback), min(int(np.max(time_idx)) + 1, T - self.hmax)
        if hi <= lo:
            self.index = np.zeros((0, 2), dtype=np.int64)
            return
        ok = c["ok"][lo:hi]
        tt, ii = np.nonzero(ok)
        self.index = np.stack([tt + lo, ii], 1).astype(np.int64)

    # ------------------------------------------------------------------ flat views aligned with self.index
    def flat(self, arr: np.ndarray) -> np.ndarray:
        return arr[self.index[:, 0], self.index[:, 1]]

    def labeled_indices(self) -> np.ndarray:
        return np.where((self.flat(self.y_l) >= 0) & ~self.flat(self.cal))[0]

    def unlabeled_indices(self) -> np.ndarray:
        return np.where((self.flat(self.y_l) < 0) & ~self.flat(self.cal))[0]

    def cal_indices(self) -> np.ndarray:
        return np.where(self.flat(self.cal))[0]

    def flat_impact(self) -> np.ndarray:
        return self._impact[self.index[:, 0], self.index[:, 1]]

    def flat_sigma(self) -> np.ndarray:
        return self._sigma[self.index[:, 0], self.index[:, 1]]

    def flat_eval(self, key: str = "y") -> np.ndarray:
        if self.eval is not None:
            return self.flat(self.eval[key])
        return self.flat(self.p.labels if key == "y" else (self.p.bars_to_peak if key == "bars_to_peak" else np.full(self.p.mid.shape, -1)))

    def causal_votes_flat(self) -> np.ndarray:
        """Votes with forward-looking LFs set to abstain: the only votes a *test-time* detector may use."""
        v = self.flat(self.votes).copy()
        v[:, self.lf_lookahead > 0] = 0
        return v

    def set_pseudo(self, prob: Optional[np.ndarray] = None, gate_ok: Optional[np.ndarray] = None, omega: Optional[np.ndarray] = None) -> None:
        tt, ii = self.index[:, 0], self.index[:, 1]
        if prob is not None:
            self.pseudo[tt, ii] = prob.astype(np.float32)
        if gate_ok is not None:
            self.gate_ok[tt, ii] = gate_ok.astype(bool)
        if omega is not None:
            self.omega[tt, ii] = omega.astype(np.float32)

    def supervision_summary(self) -> Dict[str, float]:
        yl, cal = self.flat(self.y_l), self.flat(self.cal)
        return {"n_items": int(len(self)), "n_pos_labeled": int(((yl == 1) & ~cal).sum()), "n_neg_labeled": int(((yl == 0) & ~cal).sum()),
                "n_unlabeled": int(((yl < 0) & ~cal).sum()), "n_cal": int(cal.sum()), "eval_pos_rate": float(self.flat_eval("y").mean()) if len(self) else 0.0}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        t, i = self.index[idx]
        if self.eval is not None:
            item["y"] = torch.tensor(float(self.eval["y"][t, i]), dtype=torch.float32)
            item["bars_to_peak"] = torch.tensor(float(self.eval["bars_to_peak"][t, i]), dtype=torch.float32)
            item["event_id"] = torch.tensor(int(self.eval["event_id"][t, i]))
        item["votes"] = torch.as_tensor(self.votes[t, i], dtype=torch.float32)
        item["y_l"] = torch.tensor(float(self.y_l[t, i]), dtype=torch.float32)
        item["is_cal"] = torch.tensor(bool(self.cal[t, i]))
        item["pseudo"] = torch.tensor(float(self.pseudo[t, i]), dtype=torch.float32)
        item["pseudo_ok"] = torch.tensor(bool(self.gate_ok[t, i]))
        item["omega"] = torch.tensor(float(self.omega[t, i]), dtype=torch.float32)
        return item


class TwoStreamLoader:
    """Iterates the unlabelled loader once per epoch while cycling the (much smaller) labelled loader."""

    def __init__(self, labeled: Optional[DataLoader], unlabeled: DataLoader):
        self.l, self.u = labeled, unlabeled

    def __len__(self) -> int:
        return len(self.u)

    def __iter__(self) -> Iterator[Tuple[Optional[Dict], Dict]]:
        it_l = iter(self.l) if self.l is not None else None
        for bu in self.u:
            bl = None
            if it_l is not None:
                try:
                    bl = next(it_l)
                except StopIteration:
                    it_l = iter(self.l)
                    bl = next(it_l)
            yield bl, bu


def make_two_stream_loaders(ds: SemiEventDataset, batch_size: int, seed: int, num_workers: int = 0,
                            labeled_batch_size: Optional[int] = None) -> Dict[str, Optional[DataLoader]]:
    """labeled | unlabeled (D_u) | unlabeled_cal (D_u + C0, symmetric SSL stages) | all (ordered, scoring)."""
    lab, unl, cal = ds.labeled_indices(), ds.unlabeled_indices(), ds.cal_indices()
    out: Dict[str, Optional[DataLoader]] = {"labeled": None}
    if len(lab):
        out["labeled"] = build_loader(Subset(ds, lab), min(labeled_batch_size or batch_size, len(lab)), True, num_workers, seed)
    out["unlabeled"] = build_loader(Subset(ds, unl if len(unl) else np.arange(len(ds))), batch_size, True, num_workers, seed + 1)
    uc = np.concatenate([unl, cal]) if len(cal) else unl
    out["unlabeled_cal"] = build_loader(Subset(ds, uc if len(uc) else np.arange(len(ds))), batch_size, True, num_workers, seed + 3)
    out["all"] = build_loader(ds, batch_size, False, num_workers, seed + 2)
    return out

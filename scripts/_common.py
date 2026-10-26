"""Shared CLI plumbing for scripts/*.py."""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from warn2trade.utils import load_config, deep_merge, set_seed, DATASETS, get_logger
from warn2trade.data import EventDataset, build_loader, WalkForwardSplitter
from warn2trade.backtest import CostModel
from warn2trade.train import resolve_device

log = get_logger("scripts")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/warn2trade/model.yaml"))
    p.add_argument("--dataset", default="synthetic", help="dataset name (configs/datasets/<name>.yaml)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=os.path.join(ROOT, "runs"))
    p.add_argument("--tag", default=None, help="run name (defaults to model/baseline name)")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides a.b=c")
    return p


def load_full_config(args) -> Dict[str, Any]:
    cfg = load_config(args.config, args.override)
    ds_cfg = load_config(os.path.join(ROOT, "configs/datasets", f"{args.dataset}.yaml"))
    cfg = deep_merge(cfg, {"dataset": ds_cfg})
    if "horizons" in ds_cfg:
        cfg["model"]["horizons"] = ds_cfg["horizons"]
    if "split" in ds_cfg:
        cfg["split"] = deep_merge(cfg["split"], ds_cfg["split"])
    cfg = load_config.__globals__["apply_overrides"](cfg, args.override)   # overrides win over dataset defaults
    cfg["seed"], cfg["device"] = args.seed, args.device
    if args.max_folds is not None:
        cfg["split"]["max_folds"] = args.max_folds
    return cfg


def build_panel_and_folds(cfg: Dict[str, Any]):
    builder = DATASETS.get(cfg["dataset"]["name"])(cfg["dataset"])
    panel = builder.build_panel()
    sp = cfg["split"]
    hmax = max(cfg["model"]["horizons"]) + cfg["data"]["latency"] + cfg["data"].get("exec_max", 4)
    purge = max(int(sp.get("purge", 0)), hmax)
    splitter = WalkForwardSplitter(panel.T, sp["train"], sp["val"], sp["test"], sp.get("step"), purge, sp.get("embargo", 0),
                                   sp.get("anchored", False), sp.get("max_folds"))
    return builder, panel, list(splitter)


def fold_loaders(cfg: Dict[str, Any], panel, fold, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    d = cfg["data"]
    rng = np.random.default_rng(seed)
    mk = lambda idx, neg: EventDataset(panel, idx, d["lookback"], cfg["model"]["horizons"], d["latency"], d.get("exec_max", 4), d.get("vol_window", 60), neg_subsample=neg, rng=rng)
    ds = {"train": mk(fold.train, d.get("neg_subsample")), "val": mk(fold.val, None), "test": mk(fold.test, None)}
    bs = int(cfg["train"]["batch_size"])
    loaders = {k: build_loader(v, bs, shuffle=(k == "train"), num_workers=int(d.get("num_workers", 0)), seed=seed) for k, v in ds.items()}
    return ds, loaders


def cost_model_from_cfg(cfg: Dict[str, Any], bps: float = None) -> CostModel:
    c = cfg["cost"]
    return CostModel(bps if bps is not None else c["commission_bps"], c.get("use_half_spread", True), c.get("impact_eta", 0.1), c.get("delay_zeta", 0.05), c.get("notional", 1.0))


def bench_returns(panel, fold) -> np.ndarray:
    """Equal-weight buy-and-hold of the universe over the test window (for the information ratio)."""
    lr = np.log(panel.mid[fold.test]) - np.log(panel.mid[fold.test - 1])
    return np.nanmean(lr, axis=1)


def bps_key(bps) -> str:
    return f"{float(bps):g}"


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

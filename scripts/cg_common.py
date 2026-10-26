"""Shared plumbing for scripts/cg_*.py (CausalGate track). Imports every causalgate loader / baseline module
tolerantly (a missing optional dependency disables one module, never the script)."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from warn2trade.utils import load_config, deep_merge, apply_overrides, DATASETS, BASELINES, get_logger
from warn2trade.train import resolve_device
from causalgate.eval.protocols import ProtocolRunner, all_protocols, get_protocol, freeze, verify_frozen, HASH_FILE

log = get_logger("cg")
CFG_DIR = os.path.join(ROOT, "configs", "causalgate")
RESULTS = os.path.join(ROOT, "results", "causalgate")

LOADER_MODULES = ("causalgate.data.planted", "causalgate.data.loaders.stocknet", "causalgate.data.loaders.fi2010", "causalgate.data.loaders.fnspid",
                  "causalgate.data.loaders.crypto_pd", "causalgate.data.loaders.qlib_alpha158", "causalgate.data.loaders.edgar",
                  "causalgate.data.loaders.csi300_limit", "causalgate.data.loaders.sp500_tick", "causalgate.data.loaders.wsb_squeeze",
                  "causalgate.data.loaders.fin_graph")
BASELINE_MODULES = ("causalgate.models.causalgate", "causalgate.baselines.b1_cpd_dmn", "causalgate.baselines.b2_anomaly_transformer",
                    "causalgate.baselines.b3_damp", "causalgate.baselines.b4_memstream", "causalgate.baselines.b5_dcdetector",
                    "causalgate.baselines.b6_moment", "causalgate.baselines.b7_catch", "causalgate.baselines.b8_stage",
                    "causalgate.baselines.b9_alridhawi", "causalgate.baselines.b10_konczal_evt")


def import_all(verbose: bool = False) -> Dict[str, str]:
    status = {}
    for m in LOADER_MODULES + BASELINE_MODULES:
        try:
            importlib.import_module(m); status[m] = "ok"
        except Exception as e:  # noqa: BLE001
            status[m] = f"unavailable: {type(e).__name__}: {e}"
            if verbose:
                log.warning(f"{m}: {status[m]}")
    return status


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--dataset", default="cg_planted", help="configs/causalgate/datasets/<name>.yaml")
    p.add_argument("--baseline", default="causalgate", help="registry key (causalgate, cg_damp, ...)")
    p.add_argument("--protocol", default="causal", help="published | walkforward | causal | causal_adia | offline_all | causal+L1 .. causal+L8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--out", default=RESULTS)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true", help="merge the baseline yaml's smoke: block (tiny sizes)")
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides a.b=c (applied to the merged config)")
    p.add_argument("--freeze", action="store_true", help="write configs/causalgate/PROTOCOL_HASH and exit (refuses to overwrite a differing freeze)")
    p.add_argument("--force-refreeze", action="store_true", help="with --freeze: overwrite a differing freeze (old one appended to PROTOCOL_HASH.history)")
    p.add_argument("--no-freeze-check", action="store_true", help="skip the frozen-hash check (smoke tests only)")
    return p


def load_cfgs(args) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = load_config(os.path.join(CFG_DIR, "default.yaml"))
    # registry-style names (cg_stocknet) resolve to datasets/stocknet.yaml; args.dataset stays the result-dir key
    cand = [args.dataset] + ([args.dataset[3:]] if args.dataset.startswith("cg_") else [])
    ds_path = next((p for p in (os.path.join(CFG_DIR, "datasets", f"{c}.yaml") for c in cand) if os.path.exists(p)), None)
    if ds_path is None:
        raise FileNotFoundError(f"no dataset config for {args.dataset} (tried {cand} under {os.path.join(CFG_DIR, 'datasets')})")
    ds_cfg = load_config(ds_path)
    for k in ("lookback", "latency", "exec_max", "vol_window", "neg_subsample"):
        if k in ds_cfg:
            cfg["data"][k] = ds_cfg[k]
    if "split" in ds_cfg:
        cfg["split"] = deep_merge(cfg["split"], ds_cfg["split"])
    if "horizons" in ds_cfg:
        cfg["horizons"] = list(ds_cfg["horizons"])
    key = args.baseline[3:] if args.baseline.startswith("cg_") else args.baseline
    bl_path = os.path.join(CFG_DIR, "baselines", f"{key}.yaml")
    bcfg = load_config(bl_path) if os.path.exists(bl_path) else {}
    if args.smoke and "smoke" in bcfg:
        bcfg = deep_merge(bcfg, bcfg["smoke"])
    bcfg.pop("smoke", None)
    bcfg = deep_merge({"n_exec": cfg["policy"]["n_exec"], "w_max": cfg["policy"]["w_max"], "hold_index": cfg["policy"]["hold_index"]}, bcfg)
    cfg = apply_overrides(cfg, [o for o in args.override if not o.startswith("baseline.")])
    bcfg = apply_overrides(bcfg, [o[len("baseline."):] for o in args.override if o.startswith("baseline.")])
    if args.max_folds is not None:
        cfg["split"]["max_folds"] = args.max_folds
    return cfg, ds_cfg, bcfg


def build_panel(ds_cfg: Dict[str, Any], cache: bool = True):
    """Build (or load) the dataset panel. The built Panel, PIT fields included, is pickled under
    <root>/cache/panels/<name>_<sha1(config + loader source)>.pkl so that grid jobs do not re-parse raw files; any change to
    the dataset yaml or to the loader module invalidates the cache. Set CG_PANEL_CACHE=0 to disable."""
    import hashlib, inspect, pickle
    builder = DATASETS.get(ds_cfg["name"])(ds_cfg)
    if not cache or os.environ.get("CG_PANEL_CACHE", "1") == "0" or ds_cfg["name"] == "cg_planted":
        return builder, builder.build_panel()
    try:
        src = inspect.getsource(inspect.getmodule(type(builder)))
        from causalgate.data.loaders import common as _lc
        src += inspect.getsource(_lc)
    except Exception:  # noqa: BLE001
        src = ""
    key = hashlib.sha1((json.dumps(ds_cfg, sort_keys=True, default=str) + src).encode()).hexdigest()[:16]
    cdir = os.environ.get("CG_PANEL_CACHE_DIR") or os.path.join(builder.root if os.path.isabs(builder.root) else os.path.join(ROOT, builder.root), "cache", "panels")
    path = os.path.join(cdir, f"{ds_cfg['name']}_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return builder, pickle.load(fh)
    panel = builder.build_panel()
    os.makedirs(cdir, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(panel, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return builder, panel


def make_runner(cfg: Dict[str, Any], ds_cfg: Dict[str, Any], builder, panel, args) -> ProtocolRunner:
    horizons = cfg.get("horizons") or list(panel.meta.get("horizons", builder.horizons))
    bpy = int(ds_cfg.get("bars_per_year", panel.meta.get("bars_per_year", builder.bars_per_year)))
    return ProtocolRunner(panel, horizons, bpy, cfg["data"], cfg["split"], cfg["cost"], resolve_device(args.device), args.out, args.seed, args.dataset)


def protocol_from_args(args):
    if args.freeze:
        h = freeze(HASH_FILE, force=bool(getattr(args, "force_refreeze", False)))
        log.info(f"froze {len(h)} protocols to {HASH_FILE}")
        sys.exit(0)
    if not args.no_freeze_check:
        verify_frozen(HASH_FILE, strict=True)
    return get_protocol(args.protocol)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))

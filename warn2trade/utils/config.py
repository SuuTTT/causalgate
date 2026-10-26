"""YAML/JSON config loading with `_base_` inheritance and dotted CLI overrides (a.b.c=1)."""
from __future__ import annotations
import copy
import json
import os
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def _read(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        txt = f.read()
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            raise ImportError("pip install pyyaml, or use .json configs")
        return yaml.safe_load(txt) or {}
    return json.loads(txt)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(v: str) -> Any:
    try:
        return json.loads(v)
    except Exception:
        return v


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like a.b=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def load_config(path: str, overrides: Optional[List[str]] = None, base_key: str = "_base_") -> Dict[str, Any]:
    cfg = _read(path)
    bases = cfg.pop(base_key, [])
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        b_path = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        merged = deep_merge(merged, load_config(b_path))
    merged = deep_merge(merged, cfg)
    return apply_overrides(merged, overrides)

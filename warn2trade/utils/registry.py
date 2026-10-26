from __future__ import annotations
from typing import Callable, Dict, Optional


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable] = {}

    def register(self, key: Optional[str] = None):
        def deco(obj):
            k = key or obj.__name__
            if k in self._items:
                raise KeyError(f"{k} already registered in {self.name}")
            self._items[k] = obj
            return obj
        return deco

    def get(self, key: str):
        if key not in self._items:
            raise KeyError(f"{key!r} not in registry {self.name}. Available: {sorted(self._items)}")
        return self._items[key]

    def keys(self):
        return sorted(self._items)


DATASETS = Registry("datasets")
MODELS = Registry("models")
BASELINES = Registry("baselines")

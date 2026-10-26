"""Purged, embargoed walk-forward (rolling out-of-sample) validation over the time axis.

train [......)  purge  val [....)  purge  test [....)  embargo
Purge >= max(horizons) + latency guarantees no forward-looking label/impact of a train sample overlaps a test bar.
Embargo additionally skips bars right after the test window before the next fold's train may start (López de Prado, 2018).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np


@dataclass
class Fold:
    k: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __repr__(self) -> str:
        return (f"Fold(k={self.k}, train=[{self.train[0]},{self.train[-1]}], "
                f"val=[{self.val[0]},{self.val[-1]}], test=[{self.test[0]},{self.test[-1]}])")


class WalkForwardSplitter:
    def __init__(self, n_times: int, train: int, val: int, test: int, step: Optional[int] = None,
                 purge: int = 0, embargo: int = 0, anchored: bool = False, max_folds: Optional[int] = None):
        self.n, self.train, self.val, self.test = n_times, train, val, test
        self.step = step or test
        self.purge, self.embargo, self.anchored, self.max_folds = purge, embargo, anchored, max_folds

    def __iter__(self) -> Iterator[Fold]:
        k = 0
        start = 0
        while True:
            tr_lo = 0 if self.anchored else start
            tr_hi = start + self.train
            va_lo = tr_hi + self.purge
            va_hi = va_lo + self.val
            te_lo = va_hi + self.purge
            te_hi = te_lo + self.test
            if te_hi > self.n:
                break
            yield Fold(k, np.arange(tr_lo, tr_hi), np.arange(va_lo, va_hi), np.arange(te_lo, te_hi))
            k += 1
            if self.max_folds is not None and k >= self.max_folds:
                break
            start += self.step + (self.embargo if not self.anchored else 0)

    def __len__(self) -> int:
        return sum(1 for _ in self)

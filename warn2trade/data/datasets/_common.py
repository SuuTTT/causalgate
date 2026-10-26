from __future__ import annotations
import os
from typing import Iterable


def require_files(root: str, files: Iterable[str], hint: str) -> None:
    missing = [f for f in files if not os.path.exists(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under {root}.\n{hint}")

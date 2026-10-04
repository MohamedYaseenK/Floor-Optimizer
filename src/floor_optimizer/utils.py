"""Small shared helpers."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from floor_optimizer.logger import get_logger

_log = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs (models receive the seed explicitly as well)."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    """Write JSON atomically (temp file + rename) so readers never see a partial file."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)
    return path


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Log how long a block takes."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _log.info("%s took %.1fs", label, time.perf_counter() - start)

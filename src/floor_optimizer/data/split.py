"""Chronological train/val/test loading from the parquet cache."""

from __future__ import annotations

import pandas as pd

from floor_optimizer.config import AppConfig
from floor_optimizer.data.ingestion import processed_path
from floor_optimizer.data.schema import LABEL_COL
from floor_optimizer.logger import get_logger

log = get_logger(__name__)


def _load_days(cfg: AppConfig, days: list[int]) -> pd.DataFrame:
    frames = []
    for day in sorted(days):  # sorted -> rows stay in chronological order
        path = processed_path(cfg, day)
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed data for day {day} not found at {path}. "
                "Run the ingestion step first."
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_split(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    """Return {'train','val','test'} frames. Order is guaranteed by SplitConfig."""
    split = {
        "train": _load_days(cfg, cfg.split.train_days),
        "val": _load_days(cfg, cfg.split.val_days),
        "test": _load_days(cfg, cfg.split.test_days),
    }
    for name, df in split.items():
        log.info("%s: %d rows, click rate %.4f", name, len(df), df[LABEL_COL].mean())
    return split

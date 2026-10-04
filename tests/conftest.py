"""Shared fixtures: a tiny synthetic Criteo-style dataset and a matching config."""

from __future__ import annotations

import pytest

from floor_optimizer.config import AppConfig
from floor_optimizer.data.synthetic import write_synthetic_days

N_ROWS = 4000
DAYS = [0, 1, 2, 3, 4]


@pytest.fixture()
def n_rows() -> int:
    return N_ROWS


@pytest.fixture()
def synthetic_cfg(tmp_path) -> AppConfig:
    raw_dir = tmp_path / "raw"
    write_synthetic_days(raw_dir, DAYS, n_rows=N_ROWS, seed=1)
    return AppConfig(
        seed=7,
        paths={
            "raw_dir": raw_dir,
            "processed_dir": tmp_path / "processed",
            "artifacts_dir": tmp_path / "artifacts",
        },
        data={"max_rows_per_day": None, "chunksize": 1500},
        split={"train_days": [0, 1, 2], "val_days": [3], "test_days": [4]},
        features={"n_hash_features": 65536, "min_category_count": 2},
        models={
            "logreg": {"C": 0.1, "max_iter": 100},
            "lightgbm": {
                "n_estimators": 60,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "min_child_samples": 20,
                "early_stopping_rounds": 10,
            },
        },
    )

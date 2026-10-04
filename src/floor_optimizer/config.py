"""Typed, validated configuration loaded from YAML.

Validation happens at load time so a bad config fails immediately, not an hour
into training. Most importantly, the split config enforces a strict
chronological order (train < val < test) to prevent temporal leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from floor_optimizer.exception import ConfigError


class _Strict(BaseModel):
    """Reject unknown keys so typos in YAML are caught instead of silently ignored."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")


class DataConfig(_Strict):
    file_pattern: str = "day_{day}.gz"
    max_rows_per_day: int | None = Field(default=1_000_000, gt=0)
    chunksize: int = Field(default=500_000, gt=0)

    @model_validator(mode="after")
    def _pattern_has_placeholder(self) -> DataConfig:
        if "{day}" not in self.file_pattern:
            raise ValueError("file_pattern must contain the '{day}' placeholder")
        return self


class SplitConfig(_Strict):
    train_days: list[int] = Field(default_factory=lambda: [0, 1, 2])
    val_days: list[int] = Field(default_factory=lambda: [3])
    test_days: list[int] = Field(default_factory=lambda: [4])

    @model_validator(mode="after")
    def _chronological(self) -> SplitConfig:
        for name in ("train_days", "val_days", "test_days"):
            days = getattr(self, name)
            if not days:
                raise ValueError(f"{name} must not be empty")
            if min(days) < 0:
                raise ValueError(f"{name} must contain non-negative day indices")
        if not (
            max(self.train_days) < min(self.val_days)
            and max(self.val_days) < min(self.test_days)
        ):
            raise ValueError(
                "Splits must be strictly chronological (train < val < test, no overlap) "
                "to avoid temporal leakage"
            )
        return self

    @property
    def all_days(self) -> list[int]:
        return sorted(set(self.train_days) | set(self.val_days) | set(self.test_days))


class FeatureConfig(_Strict):
    n_hash_features: int = Field(default=2**20, gt=0, le=2**31 - 1)
    min_category_count: int = Field(default=5, ge=1)


class LogRegConfig(_Strict):
    C: float = Field(default=0.1, gt=0)  # inverse L2 strength (smaller = stronger)
    max_iter: int = Field(default=100, gt=0)


class LightGBMConfig(_Strict):
    n_estimators: int = Field(default=500, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=63, gt=1)
    min_child_samples: int = Field(default=100, gt=0)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    early_stopping_rounds: int = Field(default=30, gt=0)


class ModelsConfig(_Strict):
    logreg: LogRegConfig = LogRegConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()


class EvaluationConfig(_Strict):
    n_calibration_bins: int = Field(default=15, gt=1)


class AppConfig(_Strict):
    seed: int = 42
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    split: SplitConfig = SplitConfig()
    features: FeatureConfig = FeatureConfig()
    models: ModelsConfig = ModelsConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc

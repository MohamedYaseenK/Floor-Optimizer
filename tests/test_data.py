"""Config validation, ingestion and chronological split."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from floor_optimizer.config import AppConfig, SplitConfig, load_config
from floor_optimizer.data.ingestion import ingest, processed_path, read_day
from floor_optimizer.data.preprocessing import hash_categorical, validate_frame
from floor_optimizer.data.schema import CAT_COLS, INT_COLS, LABEL_COL
from floor_optimizer.data.split import load_split
from floor_optimizer.exception import ConfigError, DataValidationError


def test_split_config_rejects_overlap_and_disorder():
    with pytest.raises(ValidationError):
        SplitConfig(train_days=[0, 1], val_days=[1], test_days=[2])  # overlap
    with pytest.raises(ValidationError):
        SplitConfig(train_days=[0, 3], val_days=[2], test_days=[4])  # train after val starts
    with pytest.raises(ValidationError):
        SplitConfig(train_days=[], val_days=[1], test_days=[2])  # empty


def test_config_rejects_unknown_keys_and_bad_pattern():
    with pytest.raises(ValidationError):
        AppConfig(unknown_key=1)
    with pytest.raises(ValidationError):
        AppConfig(data={"file_pattern": "day.gz"})


def test_load_config_errors(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("split:\n  train_days: [3]\n  val_days: [2]\n  test_days: [4]\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_shipped_configs_are_valid():
    for name in ("config.yaml", "config_synthetic.yaml"):
        load_config(f"configs/{name}")


def test_hash_categorical_is_deterministic_and_missing_is_distinct():
    s = pd.Series(["abc", None, "abc", "xyz"])
    h = hash_categorical(s)
    assert h.dtype.name == "uint64"
    assert h[0] == h[2]
    assert len({h[0], h[1], h[3]}) == 3
    assert (hash_categorical(s) == h).all()


def test_read_day_parses_schema_and_respects_row_cap(synthetic_cfg, n_rows):
    from floor_optimizer.data.ingestion import raw_path

    path = raw_path(synthetic_cfg, 0)
    full = read_day(path, None, 1000)
    assert len(full) == n_rows
    assert full[LABEL_COL].dtype.name == "int8"
    assert all(full[c].dtype.name == "float32" for c in INT_COLS)
    assert all(full[c].dtype.name == "uint64" for c in CAT_COLS)
    assert full[list(INT_COLS)].isna().any().any()  # missing numerics preserved as NaN

    capped = read_day(path, 1234, 500)
    assert len(capped) == 1234
    pd.testing.assert_frame_equal(capped, full.iloc[:1234].reset_index(drop=True))


def test_read_day_missing_file_gives_helpful_error(synthetic_cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="Raw file not found"):
        read_day(tmp_path / "nope.gz", None, 1000)


def test_ingest_caches_and_split_is_chronological(synthetic_cfg, n_rows):
    paths = ingest(synthetic_cfg)
    assert len(paths) == 5 and all(p.is_file() for p in paths)
    mtime = paths[0].stat().st_mtime_ns
    ingest(synthetic_cfg)  # second run is a cache hit
    assert paths[0].stat().st_mtime_ns == mtime

    split = load_split(synthetic_cfg)
    assert {k: len(v) for k, v in split.items()} == {
        "train": 3 * n_rows,
        "val": n_rows,
        "test": n_rows,
    }
    # Train rows are exactly days 0..2 in order: first block equals the day-0 cache.
    day0 = pd.read_parquet(processed_path(synthetic_cfg, 0))
    pd.testing.assert_frame_equal(split["train"].iloc[:n_rows].reset_index(drop=True), day0)


def test_load_split_without_ingestion_fails_clearly(synthetic_cfg):
    with pytest.raises(FileNotFoundError, match="ingestion"):
        load_split(synthetic_cfg)


def test_validate_frame_catches_bad_labels():
    df = pd.DataFrame({c: [0, 1] for c in [LABEL_COL, *INT_COLS, *CAT_COLS]})
    validate_frame(df)
    df[LABEL_COL] = [0, 2]
    with pytest.raises(DataValidationError):
        validate_frame(df)
    df[LABEL_COL] = [1, 1]
    with pytest.raises(DataValidationError):
        validate_frame(df)

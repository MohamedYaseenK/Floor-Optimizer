"""Stage 1: read raw Criteo day files once, parse, and cache as parquet.

Usage:
    python -m floor_optimizer.data.ingestion --config configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from floor_optimizer.config import AppConfig, load_config
from floor_optimizer.data.preprocessing import parse_chunk, validate_frame
from floor_optimizer.data.schema import CAT_COLS, RAW_COLUMNS
from floor_optimizer.exception import DataValidationError
from floor_optimizer.logger import get_logger
from floor_optimizer.utils import ensure_dir, timed

log = get_logger(__name__)


def raw_path(cfg: AppConfig, day: int) -> Path:
    return cfg.paths.raw_dir / cfg.data.file_pattern.format(day=day)


def processed_path(cfg: AppConfig, day: int) -> Path:
    """Cache path. The row cap is part of the name so changing it never reuses stale data."""
    rows = cfg.data.max_rows_per_day
    tag = "all" if rows is None else str(rows)
    return cfg.paths.processed_dir / f"day_{day}_rows_{tag}.parquet"


def read_day(path: Path, max_rows: int | None, chunksize: int) -> pd.DataFrame:
    """Read one raw day file in chunks and return the parsed frame.

    Note: with a row cap this reads the *head* of the file, i.e. the earliest
    hours of that day. Train/val/test all share this bias, so comparisons stay
    fair, but it is not a uniform sample of the day.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw file not found: {path}. Download it (scripts/download_data.sh) "
            "or generate synthetic data (scripts/make_synthetic_data.py)."
        )
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(RAW_COLUMNS),
        dtype={c: str for c in CAT_COLS},
        nrows=max_rows,
        chunksize=chunksize,
        keep_default_na=False,
        na_values=[""],
    )
    parts = [parse_chunk(chunk) for chunk in reader]
    if not parts:
        raise DataValidationError(f"No rows read from {path}")
    df = pd.concat(parts, ignore_index=True)
    validate_frame(df)
    return df


def ingest_day(cfg: AppConfig, day: int, force: bool = False) -> Path:
    """Parse one day and cache it. Skips work if the cache already exists."""
    out = processed_path(cfg, day)
    if out.is_file() and not force:
        log.info("day %d: cache hit (%s)", day, out)
        return out
    with timed(f"ingest day {day}"):
        df = read_day(raw_path(cfg, day), cfg.data.max_rows_per_day, cfg.data.chunksize)
    ensure_dir(out.parent)
    tmp = out.with_name(out.name + ".tmp")  # atomic write: never leave a half-written cache
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    log.info("day %d: %d rows, click rate %.4f -> %s", day, len(df), df["label"].mean(), out)
    return out


def ingest(cfg: AppConfig, force: bool = False) -> list[Path]:
    """Ingest every day referenced by the split config."""
    return [ingest_day(cfg, day, force=force) for day in cfg.split.all_days]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="re-ingest even if cached")
    args = parser.parse_args()
    ingest(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()

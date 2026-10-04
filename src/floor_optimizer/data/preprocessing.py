"""Raw-chunk parsing and validation.

Design note: categorical strings are converted to stable 64-bit hashes
immediately. This keeps memory low (8 bytes/cell instead of a Python string) and,
crucially, the *same* function can be reused at serving time so training and
online features cannot drift apart (training/serving parity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floor_optimizer.data.schema import (
    CAT_COLS,
    INT_COLS,
    LABEL_COL,
    MISSING_CAT_TOKEN,
    RAW_COLUMNS,
)
from floor_optimizer.exception import DataValidationError


def hash_categorical(values: pd.Series) -> np.ndarray:
    """Map a string column to deterministic uint64 hashes (missing -> own token).

    pandas' hash_array uses a fixed key, so results are identical across runs,
    machines and processes (unlike Python's built-in hash()).
    """
    filled = values.fillna(MISSING_CAT_TOKEN).astype(str)
    return pd.util.hash_array(filled.to_numpy(dtype=object))


def parse_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw chunk into the compact modelling schema.

    Output dtypes: label int8, I* float32 (NaN kept for missing), C* uint64 hashes.
    """
    missing = [c for c in RAW_COLUMNS if c not in chunk.columns]
    if missing:
        raise DataValidationError(f"Raw chunk is missing columns: {missing}")
    if chunk[LABEL_COL].isna().any():
        raise DataValidationError("Found missing labels in raw data")

    data: dict[str, np.ndarray] = {LABEL_COL: chunk[LABEL_COL].to_numpy().astype(np.int8)}
    for col in INT_COLS:
        data[col] = chunk[col].to_numpy(dtype=np.float32, na_value=np.nan)
    for col in CAT_COLS:
        data[col] = hash_categorical(chunk[col])
    return pd.DataFrame(data)


def validate_frame(df: pd.DataFrame) -> None:
    """Basic data contract checks on a parsed frame. Raises DataValidationError."""
    if df.empty:
        raise DataValidationError("Parsed frame is empty")
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Frame is missing columns: {missing}")
    if not df[LABEL_COL].isin([0, 1]).all():
        raise DataValidationError("Label column must contain only 0/1")
    if df[LABEL_COL].nunique() < 2:
        raise DataValidationError("Label column has a single class; cannot train or evaluate")

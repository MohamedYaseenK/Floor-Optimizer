"""Stateless feature hashing for the linear CTR model.

Every feature (13 bucketised numerics + 26 categoricals) becomes one active
index in a fixed-size sparse vector. There is nothing to fit, so:
  * no leakage from validation/test data is possible, and
  * the exact same function can run online at serving time (training/serving parity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from floor_optimizer.data.schema import CAT_COLS, INT_COLS

_MISSING_NUMERIC_BUCKET = 10_000  # far outside any real log2 bucket (|bucket| < 64)
_GOLDEN = 0x9E3779B97F4A7C15
_MASK64 = (1 << 64) - 1


def log_bucketize(x: np.ndarray) -> np.ndarray:
    """Sign-preserving log2 bucket of a numeric array; NaN -> a dedicated bucket.

    Count-like features are heavy-tailed, so log-scale buckets are the standard
    way to turn them into categorical tokens for a linear model.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, _MISSING_NUMERIC_BUCKET, dtype=np.int64)
    valid = ~np.isnan(x)
    v = x[valid]
    out[valid] = (np.sign(v) * np.floor(np.log2(np.abs(v) + 1.0))).astype(np.int64)
    return out


def _column_salt(col_idx: int) -> np.uint64:
    """Per-column salt so equal raw values in different columns land in different buckets."""
    return np.uint64(((col_idx + 1) * _GOLDEN) & _MASK64)


def hash_to_sparse(df: pd.DataFrame, n_features: int) -> sparse.csr_matrix:
    """Hash a parsed frame into a CSR matrix with <= 39 active entries per row.

    (Hash collisions between two columns of the same row are summed, so a row
    can occasionally have fewer than 39 distinct entries.)
    """
    if not 0 < n_features <= 2**31 - 1:
        raise ValueError("n_features must be in (0, 2**31 - 1]")

    columns = (*INT_COLS, *CAT_COLS)
    n_rows, n_cols = len(df), len(columns)
    n_buckets = np.uint64(n_features)
    indices = np.empty((n_rows, n_cols), dtype=np.int32)

    for j, col in enumerate(columns):
        if col in INT_COLS:
            raw_hash = pd.util.hash_array(log_bucketize(df[col].to_numpy()))
        else:
            raw_hash = df[col].to_numpy(dtype=np.uint64)  # already hashed at ingestion
        # uint64 array arithmetic wraps around silently, which is what we want for hashing
        indices[:, j] = ((raw_hash + _column_salt(j)) % n_buckets).astype(np.int32)

    indptr = np.arange(0, (n_rows + 1) * n_cols, n_cols, dtype=np.int64)
    data = np.ones(n_rows * n_cols, dtype=np.float32)
    matrix = sparse.csr_matrix((data, indices.ravel(), indptr), shape=(n_rows, n_features))
    matrix.sum_duplicates()
    return matrix

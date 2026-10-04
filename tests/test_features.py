"""Hashing and frequency-encoding behaviour (determinism, no leakage, edge cases)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floor_optimizer.data.ingestion import raw_path, read_day
from floor_optimizer.data.schema import CAT_COLS, INT_COLS
from floor_optimizer.features.build_features import FrequencyEncoder
from floor_optimizer.features.hashing import hash_to_sparse, log_bucketize

N_FEATURES = 2**16


@pytest.fixture()
def frame(synthetic_cfg) -> pd.DataFrame:
    return read_day(raw_path(synthetic_cfg, 0), 1000, 500)


def test_log_bucketize_handles_signs_and_missing():
    out = log_bucketize(np.array([np.nan, 0.0, 1.0, 3.0, 7.0, -1.0, -7.0]))
    assert out.tolist() == [10_000, 0, 1, 2, 3, -1, -3]


def test_hash_to_sparse_shape_and_nnz(frame):
    X = hash_to_sparse(frame, N_FEATURES)
    assert X.shape == (len(frame), N_FEATURES)
    nnz_per_row = np.diff(X.indptr)
    n_cols = len(INT_COLS) + len(CAT_COLS)
    assert nnz_per_row.max() <= n_cols
    assert nnz_per_row.min() >= n_cols - 6  # only rare same-row collisions
    assert X.data.min() >= 1.0
    assert X.indices.max() < N_FEATURES


def test_hash_to_sparse_is_deterministic_and_row_independent(frame):
    full = hash_to_sparse(frame, N_FEATURES)
    again = hash_to_sparse(frame, N_FEATURES)
    assert (full != again).nnz == 0
    # Serving parity: scoring a single row gives the same vector as inside a batch.
    single = hash_to_sparse(frame.iloc[[17]], N_FEATURES)
    assert (single != full[17]).nnz == 0


def test_hash_to_sparse_same_value_in_different_columns_differs():
    row = {c: 0.0 for c in INT_COLS}
    row.update({c: np.uint64(42) for c in CAT_COLS})
    X = hash_to_sparse(pd.DataFrame([row]).astype({c: "uint64" for c in CAT_COLS}), N_FEATURES)
    assert X.nnz >= len(CAT_COLS)  # 26 identical raw values -> not collapsed to one bucket


def test_hash_to_sparse_empty_frame_and_bad_args(frame):
    empty = hash_to_sparse(frame.iloc[:0], N_FEATURES)
    assert empty.shape == (0, N_FEATURES)
    with pytest.raises(ValueError):
        hash_to_sparse(frame, 0)


def test_frequency_encoder_fit_on_train_only(frame):
    train, other = frame.iloc[:600], frame.iloc[600:]
    enc = FrequencyEncoder(min_count=2).fit(train)
    out = enc.transform(other)
    assert out.shape == (len(other), len(INT_COLS) + len(CAT_COLS))
    assert out.dtypes.map(lambda d: d == np.float32).all()
    assert out[list(CAT_COLS)].min().min() >= 0.0
    assert out[list(CAT_COLS)].max().max() <= 1.0

    # An id never seen in training encodes to 0.0, no matter what it is in `other`.
    unseen = other.iloc[:1].copy()
    unseen["C3"] = np.uint64(2**63 + 5)
    assert enc.transform(unseen)["C3"].iloc[0] == 0.0


def test_frequency_encoder_min_count_and_errors(frame):
    enc = FrequencyEncoder(min_count=10**9).fit(frame)  # nothing survives the threshold
    assert (enc.transform(frame)[list(CAT_COLS)] == 0.0).all().all()
    with pytest.raises(RuntimeError):
        FrequencyEncoder().transform(frame)
    with pytest.raises(ValueError):
        FrequencyEncoder().fit(frame.iloc[:0])
    with pytest.raises(ValueError):
        FrequencyEncoder(min_count=0)

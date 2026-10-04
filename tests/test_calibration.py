"""Metric correctness (AUC, log-loss, NE, ECE). Calibrators will be tested here later."""

from __future__ import annotations

import numpy as np
import pytest

from floor_optimizer.evaluation.metrics import (
    evaluate_ctr,
    expected_calibration_error,
    log_loss,
    normalized_entropy,
)
from floor_optimizer.exception import DataValidationError


def test_log_loss_known_value():
    y = np.array([1, 0])
    p = np.array([0.9, 0.2])
    expected = -(np.log(0.9) + np.log(0.8)) / 2
    assert log_loss(y, p) == pytest.approx(expected)


def test_log_loss_is_finite_for_extreme_probabilities():
    assert np.isfinite(log_loss(np.array([1, 0]), np.array([0.0, 1.0])))


def test_normalized_entropy_is_one_for_base_rate_predictor():
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.2).astype(int)
    ne = normalized_entropy(y, np.full(y.shape, y.mean()), base_ctr=float(y.mean()))
    assert ne == pytest.approx(1.0, abs=1e-9)


def test_ece_small_when_calibrated_large_when_not():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.6, 200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 2.0, 0, 1)) > 0.1


def test_ece_handles_boundary_probabilities():
    y = np.array([0, 1, 1, 0])
    assert expected_calibration_error(y, np.array([0.0, 1.0, 1.0, 0.0])) == 0.0


def test_evaluate_ctr_returns_plain_floats_and_validates():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.5, 5_000)
    y = (rng.random(p.size) < p).astype(int)
    out = evaluate_ctr(y, p, base_ctr=float(y.mean()))
    assert all(isinstance(v, float) for v in out.values())
    assert out["auc"] > 0.6 and out["normalized_entropy"] < 1.0

    with pytest.raises(DataValidationError):
        evaluate_ctr(np.zeros(10), np.full(10, 0.3), base_ctr=0.3)  # one class only
    with pytest.raises(DataValidationError):
        evaluate_ctr(y, p + 1.0, base_ctr=0.3)  # probabilities out of range
    with pytest.raises(DataValidationError):
        evaluate_ctr(y[:-1], p, base_ctr=0.3)  # length mismatch

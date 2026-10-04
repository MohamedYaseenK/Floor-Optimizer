"""CTR evaluation metrics.

Beyond AUC (ranking), ad-tech cares about *probability quality*: bids and floors
are computed from predicted probabilities, so we report log-loss, Normalized
Entropy (NE) and Expected Calibration Error (ECE) as well.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from floor_optimizer.exception import DataValidationError

_EPS = 1e-15


def _check_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise DataValidationError("y_true and y_prob must be 1-D arrays of equal length")
    if y_true.size == 0:
        raise DataValidationError("Cannot evaluate on empty arrays")
    if not np.isfinite(y_prob).all() or y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise DataValidationError("Predicted probabilities must be finite and within [0, 1]")
    if not np.isin(y_true, (0, 1)).all():
        raise DataValidationError("y_true must contain only 0/1")
    return y_true.astype(np.float64), y_prob


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true, y_prob = _check_inputs(y_true, y_prob)
    p = np.clip(y_prob, _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def normalized_entropy(y_true: np.ndarray, y_prob: np.ndarray, base_ctr: float) -> float:
    """Log-loss divided by the entropy of a constant predictor at `base_ctr`.

    NE < 1 means the model beats always predicting the training click rate.
    """
    if not 0.0 < base_ctr < 1.0:
        raise DataValidationError("base_ctr must be strictly between 0 and 1")
    background = -(base_ctr * np.log(base_ctr) + (1.0 - base_ctr) * np.log(1.0 - base_ctr))
    return float(log_loss(y_true, y_prob) / background)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Weighted mean |predicted - observed click rate| over equal-width probability bins."""
    y_true, y_prob = _check_inputs(y_true, y_prob)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    bin_idx = np.digitize(y_prob, edges)  # 0 .. n_bins-1
    counts = np.bincount(bin_idx, minlength=n_bins)
    sum_prob = np.bincount(bin_idx, weights=y_prob, minlength=n_bins)
    sum_true = np.bincount(bin_idx, weights=y_true, minlength=n_bins)
    occupied = counts > 0
    return float(np.abs(sum_prob[occupied] - sum_true[occupied]).sum() / y_true.size)


def evaluate_ctr(
    y_true: np.ndarray, y_prob: np.ndarray, base_ctr: float, n_bins: int = 15
) -> dict[str, float]:
    """All CTR metrics in one dict of plain Python floats (JSON friendly)."""
    y_true_f, y_prob_f = _check_inputs(y_true, y_prob)
    if np.unique(y_true_f).size < 2:
        raise DataValidationError("AUC needs both classes present in y_true")
    return {
        "auc": float(roc_auc_score(y_true_f, y_prob_f)),
        "logloss": log_loss(y_true_f, y_prob_f),
        "normalized_entropy": normalized_entropy(y_true_f, y_prob_f, base_ctr),
        "ece": expected_calibration_error(y_true_f, y_prob_f, n_bins),
        "mean_predicted_ctr": float(y_prob_f.mean()),
        "observed_ctr": float(y_true_f.mean()),
        "n_samples": float(y_true_f.size),
    }

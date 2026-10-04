"""Baseline CTR models: hashed logistic regression (L-BFGS) and LightGBM.

Both expose probabilities of the positive class. Hyper-parameters come from the
validated config; nothing is hard-coded here.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from floor_optimizer.config import LightGBMConfig, LogRegConfig
from floor_optimizer.logger import get_logger

log = get_logger(__name__)


def train_logreg(
    X_train: sparse.csr_matrix, y_train: np.ndarray, cfg: LogRegConfig, seed: int
) -> LogisticRegression:
    """L2-regularised logistic regression (L-BFGS) on sparse hashed features.

    L-BFGS is used instead of SGD on purpose: in our tests plain SGD reached a
    similar AUC but produced badly over-confident probabilities (NE >> 1), which
    is unacceptable when probabilities drive prices. L-BFGS is deterministic and
    well calibrated out of the box.
    """
    model = LogisticRegression(C=cfg.C, solver="lbfgs", max_iter=cfg.max_iter, random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: LightGBMConfig,
    seed: int,
) -> lgb.LGBMClassifier:
    """Gradient-boosted trees with early stopping on validation log-loss."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    with warnings.catch_warnings():
        # newer LightGBM prefers eval_X/eval_y; eval_set works on every supported version
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
    log.info("lightgbm best_iteration=%s", model.best_iteration_)
    return model


def predict_proba(model: LogisticRegression | lgb.LGBMClassifier, X) -> np.ndarray:
    """Probability of a click, as a 1-D float64 array."""
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)

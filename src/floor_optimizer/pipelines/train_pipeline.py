"""Stage 2: train baseline CTR models and evaluate on chronological val/test.

Usage:
    python -m floor_optimizer.pipelines.train_pipeline --config configs/config.yaml

Outputs (under paths.artifacts_dir):
    models/<name>.joblib            fitted models (+ gbdt_encoder.joblib)
    predictions/<name>_<split>.npy  predicted CTRs, kept for the calibration phase
    reports/metrics.json            metrics, data stats and the config used
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np

from floor_optimizer.config import AppConfig, load_config
from floor_optimizer.data.schema import LABEL_COL
from floor_optimizer.data.split import load_split
from floor_optimizer.evaluation.metrics import evaluate_ctr
from floor_optimizer.features.build_features import FrequencyEncoder
from floor_optimizer.features.hashing import hash_to_sparse
from floor_optimizer.logger import get_logger
from floor_optimizer.models.ctr_baselines import predict_proba, train_lightgbm, train_logreg
from floor_optimizer.utils import ensure_dir, save_json, set_seed, timed

log = get_logger(__name__)

MODEL_CHOICES = ("logreg", "lightgbm")


def _evaluate(
    name: str,
    model: Any,
    features: dict[str, Any],
    labels: dict[str, np.ndarray],
    base_ctr: float,
    cfg: AppConfig,
    preds_dir,
) -> dict[str, dict[str, float]]:
    """Score val and test, persist predictions, return metrics per split."""
    results: dict[str, dict[str, float]] = {}
    for split in ("val", "test"):
        probs = predict_proba(model, features[split])
        np.save(preds_dir / f"{name}_{split}.npy", probs)
        results[split] = evaluate_ctr(
            labels[split], probs, base_ctr, cfg.evaluation.n_calibration_bins
        )
        log.info(
            "%s/%s | AUC %.4f | logloss %.4f | NE %.4f | ECE %.4f",
            name,
            split,
            results[split]["auc"],
            results[split]["logloss"],
            results[split]["normalized_entropy"],
            results[split]["ece"],
        )
    return results


def run(cfg: AppConfig, models: tuple[str, ...] = MODEL_CHOICES) -> dict[str, Any]:
    """Train the requested models and write all artifacts. Returns the report dict."""
    unknown = set(models) - set(MODEL_CHOICES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}; choose from {MODEL_CHOICES}")

    set_seed(cfg.seed)
    artifacts = cfg.paths.artifacts_dir
    models_dir = ensure_dir(artifacts / "models")
    preds_dir = ensure_dir(artifacts / "predictions")
    reports_dir = ensure_dir(artifacts / "reports")

    splits = load_split(cfg)
    labels = {k: v[LABEL_COL].to_numpy() for k, v in splits.items()}
    base_ctr = float(labels["train"].mean())  # NE reference: train click rate only

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "data": {
            split: {"rows": int(len(df)), "ctr": float(labels[split].mean())}
            for split, df in splits.items()
        },
        "models": {},
        "config": cfg.model_dump(mode="json"),
    }

    if "logreg" in models:
        with timed("logreg: hash features"):
            X = {k: hash_to_sparse(df, cfg.features.n_hash_features) for k, df in splits.items()}
        with timed("logreg: train"):
            model = train_logreg(X["train"], labels["train"], cfg.models.logreg, cfg.seed)
        report["models"]["logreg"] = _evaluate("logreg", model, X, labels, base_ctr, cfg, preds_dir)
        joblib.dump(model, models_dir / "logreg.joblib")
        del X

    if "lightgbm" in models:
        with timed("lightgbm: encode features"):
            encoder = FrequencyEncoder(cfg.features.min_category_count).fit(splits["train"])
            X = {k: encoder.transform(df) for k, df in splits.items()}
        with timed("lightgbm: train"):
            model = train_lightgbm(
                X["train"], labels["train"], X["val"], labels["val"], cfg.models.lightgbm, cfg.seed
            )
        report["models"]["lightgbm"] = _evaluate(
            "lightgbm", model, X, labels, base_ctr, cfg, preds_dir
        )
        joblib.dump(model, models_dir / "lightgbm.joblib")
        joblib.dump(encoder, models_dir / "gbdt_encoder.joblib")
        del X

    save_json(report, reports_dir / "metrics.json")
    log.info("report written to %s", reports_dir / "metrics.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CHOICES), choices=MODEL_CHOICES)
    args = parser.parse_args()
    run(load_config(args.config), tuple(args.models))


if __name__ == "__main__":
    main()

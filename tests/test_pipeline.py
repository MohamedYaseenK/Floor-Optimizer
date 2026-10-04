"""End-to-end: synthetic raw files -> ingestion -> training -> metrics/artifacts."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from floor_optimizer.data.ingestion import ingest
from floor_optimizer.features.build_features import FrequencyEncoder
from floor_optimizer.pipelines.train_pipeline import run


def test_train_pipeline_end_to_end(synthetic_cfg, n_rows):
    ingest(synthetic_cfg)
    report = run(synthetic_cfg)

    # Both models learn real signal out-of-time and beat the constant-CTR predictor.
    for name in ("logreg", "lightgbm"):
        test = report["models"][name]["test"]
        assert test["auc"] > 0.62, (name, test)
        assert test["normalized_entropy"] < 1.0, (name, test)

    artifacts = synthetic_cfg.paths.artifacts_dir
    metrics = json.loads((artifacts / "reports" / "metrics.json").read_text())
    assert metrics["data"]["train"]["rows"] == 3 * n_rows
    assert metrics["config"]["split"]["test_days"] == [4]

    for name in ("logreg", "lightgbm"):
        for split in ("val", "test"):
            preds = np.load(artifacts / "predictions" / f"{name}_{split}.npy")
            assert preds.shape == (n_rows,) and ((preds >= 0) & (preds <= 1)).all()
        assert (artifacts / "models" / f"{name}.joblib").is_file()
    assert isinstance(joblib.load(artifacts / "models" / "gbdt_encoder.joblib"), FrequencyEncoder)


def test_run_single_model_and_rejects_unknown(synthetic_cfg):
    ingest(synthetic_cfg)
    report = run(synthetic_cfg, models=("logreg",))
    assert set(report["models"]) == {"logreg"}
    with pytest.raises(ValueError, match="Unknown models"):
        run(synthetic_cfg, models=("xgboost",))

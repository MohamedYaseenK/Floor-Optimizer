#!/usr/bin/env bash
# Scaffolds the empty file structure for the SSP floor-price optimizer project.
# Usage: bash template.sh   (run inside the folder where you want the project)

set -euo pipefail

PROJECT="pubmatic-floor-optimizer"
PKG="floor_optimizer"

list_of_files=(
  # ---------- root ----------
  "README.md"
  "requirements.txt"
  "Dockerfile"
  "docker-compose.yml"
  "Makefile"
  ".gitignore"
  ".env.example"

  # ---------- configs ----------
  "configs/config.yaml"
  "configs/features.yaml"

  # ---------- data (kept out of git, folders only) ----------
  "data/raw/.gitkeep"
  "data/interim/.gitkeep"
  "data/processed/.gitkeep"

  # ---------- artifacts ----------
  "artifacts/models/.gitkeep"
  "artifacts/reports/.gitkeep"

  # ---------- notebooks ----------
  "notebooks/.gitkeep"

  # ---------- package ----------
  "src/${PKG}/__init__.py"
  "src/${PKG}/config.py"
  "src/${PKG}/logger.py"
  "src/${PKG}/exception.py"
  "src/${PKG}/utils.py"

  # data layer
  "src/${PKG}/data/__init__.py"
  "src/${PKG}/data/ingestion.py"
  "src/${PKG}/data/preprocessing.py"
  "src/${PKG}/data/split.py"

  # feature layer
  "src/${PKG}/features/__init__.py"
  "src/${PKG}/features/build_features.py"
  "src/${PKG}/features/hashing.py"
  "src/${PKG}/features/online_store.py"

  # models
  "src/${PKG}/models/__init__.py"
  "src/${PKG}/models/ctr_baselines.py"
  "src/${PKG}/models/deepfm.py"
  "src/${PKG}/models/calibration.py"
  "src/${PKG}/models/win_price.py"
  "src/${PKG}/models/fraud.py"

  # floor optimization + offline replay
  "src/${PKG}/optimization/__init__.py"
  "src/${PKG}/optimization/floor_optimizer.py"
  "src/${PKG}/optimization/replay_simulator.py"

  # evaluation
  "src/${PKG}/evaluation/__init__.py"
  "src/${PKG}/evaluation/metrics.py"

  # pipelines (entry points)
  "src/${PKG}/pipelines/__init__.py"
  "src/${PKG}/pipelines/train_pipeline.py"
  "src/${PKG}/pipelines/evaluate_pipeline.py"
  "src/${PKG}/pipelines/export_onnx.py"

  # serving
  "src/${PKG}/serving/__init__.py"
  "src/${PKG}/serving/app.py"
  "src/${PKG}/serving/schemas.py"
  "src/${PKG}/serving/predictor.py"

  # streaming
  "src/${PKG}/streaming/__init__.py"
  "src/${PKG}/streaming/producer.py"
  "src/${PKG}/streaming/spark_features.py"

  # monitoring
  "src/${PKG}/monitoring/__init__.py"
  "src/${PKG}/monitoring/drift.py"
  "src/${PKG}/monitoring/metrics_exporter.py"

  # ---------- scripts ----------
  "scripts/download_data.sh"
  "scripts/locustfile.py"

  # ---------- monitoring configs ----------
  "monitoring/prometheus.yml"
  "monitoring/grafana/dashboards/.gitkeep"

  # ---------- tests ----------
  "tests/__init__.py"
  "tests/test_features.py"
  "tests/test_calibration.py"
  "tests/test_floor_optimizer.py"
  "tests/test_api.py"
)

mkdir -p "${PROJECT}"
cd "${PROJECT}"

for filepath in "${list_of_files[@]}"; do
  filedir="$(dirname "${filepath}")"
  if [ "${filedir}" != "." ]; then
    mkdir -p "${filedir}"
  fi
  if [ ! -f "${filepath}" ]; then
    touch "${filepath}"
    echo "Created: ${PROJECT}/${filepath}"
  else
    echo "Exists : ${PROJECT}/${filepath}"
  fi
done

echo ""
echo "Done. Next: cd ${PROJECT} && python -m venv .venv && pip install -r requirements.txt"

#!/usr/bin/env bash
# Download selected Criteo day files from Hugging Face into data/raw.
# Usage: bash scripts/download_data.sh [DAY ...]      (default: 0 1 2 3 4)
#        OUT_DIR=/big/disk/raw bash scripts/download_data.sh 0 1 2
# Each day file is ~1.6 GB compressed. Requires: pip install huggingface_hub
set -euo pipefail

DAYS="${*:-0 1 2 3 4}"
OUT_DIR="${OUT_DIR:-data/raw}"
mkdir -p "${OUT_DIR}"

for day in ${DAYS}; do
  echo "Downloading day_${day}.gz ..."
  python - <<PY
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="criteo/CriteoClickLogs",
    filename="day_${day}.gz",
    repo_type="dataset",
    local_dir="${OUT_DIR}",
)
PY
done
echo "Done. Files are in ${OUT_DIR}"

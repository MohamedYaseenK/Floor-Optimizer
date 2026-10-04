.PHONY: install test smoke ingest train

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest

# End-to-end run on synthetic data (no download needed)
smoke:
	python scripts/make_synthetic_data.py
	python -m floor_optimizer.data.ingestion --config configs/config_synthetic.yaml
	python -m floor_optimizer.pipelines.train_pipeline --config configs/config_synthetic.yaml

# Real data (download first: bash scripts/download_data.sh)
ingest:
	python -m floor_optimizer.data.ingestion --config configs/config.yaml

train:
	python -m floor_optimizer.pipelines.train_pipeline --config configs/config.yaml

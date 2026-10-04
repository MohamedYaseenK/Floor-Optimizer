# Real-time Floor-Price Optimization (SSP) — CTR modelling foundation

Predict click probability for ad impressions on the Criteo click logs, as the first
building block of a floor-price optimizer for a sell-side platform. Probabilities (not
just rankings) feed pricing decisions, so the project tracks **calibration** and
**Normalized Entropy** alongside AUC.

**Status — Phase 1 (this repo state):** data ingestion, chronological split, features,
two baseline CTR models, metrics, tests.
**Next:** calibration -> DeepFM -> win-price model -> floor optimizer + replay ->
ONNX/FastAPI serving with latency benchmarks -> Redis/Kafka -> monitoring.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install            # pip install -r requirements.txt && pip install -e .
make test               # 25 tests
make smoke              # full pipeline on synthetic data, ~1 minute, no download

# real data (each day file is ~1.6 GB compressed)
bash scripts/download_data.sh 0 1 2 3 4
make ingest             # parse + cache as parquet (one-off per day)
make train              # train + evaluate, writes artifacts/
```

Tune `configs/config.yaml` (days, row cap per day, hyper-parameters). Invalid configs fail
at load time, including any split that is not strictly chronological.

## Pipeline

```
data/raw/day_N.gz --ingestion--> data/processed/day_N_rows_<cap>.parquet
        (chunked read, categoricals -> deterministic uint64 hashes, validation, atomic write)
                         |
              load_split (train < val < test by day)
                 /                      \
   hash_to_sparse (stateless)     FrequencyEncoder (fit on train only)
   LogisticRegression (L-BFGS)    LightGBM (early stopping on val)
                 \                      /
        evaluate: AUC, log-loss, NE, ECE  -> artifacts/{models,predictions,reports}
```

## Design decisions (and why)

- **Time-based split by day.** Random splits leak the future and inflate metrics. Val/test
  are strictly later days than train; `SplitConfig` enforces it.
- **Stateless hashing for the linear model.** Nothing is fit, so no leakage and the same
  function can run online -> training/serving parity (critical for latency-sensitive serving).
- **Frequency encoding fit on train only** for GBDT; unseen/rare ids map to 0.
- **L-BFGS logistic regression, not SGD.** In testing SGD reached similar AUC but gave
  badly over-confident probabilities (NE 6-9). Probabilities drive prices, so we use the
  solver that is well calibrated.
- **Deterministic hashes** (`pd.util.hash_array`) rather than Python's `hash()`, which
  changes between processes.
- **Atomic writes** for caches and reports; processed cache name includes the row cap so a
  config change never reuses stale data.

## Known limitations (read before quoting numbers)

- With `max_rows_per_day` set, the **head** of each day is read (earliest hours), not a
  uniform sample. All splits share the bias, so comparisons stay fair.
- Criteo **subsamples clicks and non-clicks at different rates**, so the base CTR in this
  data is not real traffic CTR. Treat absolute probabilities accordingly; the relative model
  comparison is what matters here.
- Public data, advertiser-side/proxy, **not SSP logs**. Any revenue figures later will be
  *simulated offline estimates*.
- Metrics from `make smoke` use synthetic data and are only a pipeline sanity check.
- The download script and real-data run were not executed in the build environment (no
  access to Hugging Face there); the synthetic path was tested end to end.

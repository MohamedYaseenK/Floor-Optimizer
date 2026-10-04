"""Generate Criteo-formatted synthetic click logs for smoke tests and unit tests.

The files match the real layout exactly (tab separated, gzip, no header, empty
string for missing values) and contain a learnable signal, so the full pipeline
can be exercised without downloading gigabytes. Not for reporting results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from floor_optimizer.data.schema import CAT_COLS, INT_COLS, LABEL_COL, RAW_COLUMNS

_VOCAB = 200


def _hex_ids(ids: np.ndarray) -> np.ndarray:
    """Integer ids -> 8-char hex strings, like Criteo's hashed categoricals."""
    return np.char.mod("%08x", (ids.astype(np.int64) * 2654435761) % (2**32))


def generate_day(n_rows: int, day: int, seed: int = 0) -> pd.DataFrame:
    """One synthetic day in the raw Criteo layout (string categoricals, NaN for missing)."""
    rng = np.random.default_rng(seed * 10_007 + day)
    effects_rng = np.random.default_rng(12345)  # same category effects every day
    effect_c1 = effects_rng.normal(0.0, 0.8, _VOCAB)
    effect_c2 = effects_rng.normal(0.0, 0.6, _VOCAB)

    cols: dict[str, np.ndarray] = {}
    ints: dict[str, np.ndarray] = {}
    for i, name in enumerate(INT_COLS):
        values = rng.poisson(3 + 2 * i, n_rows).astype(float)
        values[rng.random(n_rows) < 0.15] = np.nan
        ints[name] = values

    # Zipf-like popularity so frequency encoding can tell categories apart.
    popularity = 1.0 / (np.arange(_VOCAB) + 1.0)
    popularity /= popularity.sum()
    cat_ids: dict[str, np.ndarray] = {}
    for j, name in enumerate(CAT_COLS):
        cat_ids[name] = rng.choice(_VOCAB, size=n_rows, p=popularity) + j * _VOCAB

    logit = (
        -1.4
        + 0.45 * np.log1p(np.nan_to_num(ints["I1"], nan=0.0))
        - 0.25 * np.log1p(np.nan_to_num(ints["I2"], nan=0.0))
        + effect_c1[cat_ids["C1"] % _VOCAB]
        + effect_c2[cat_ids["C2"] % _VOCAB]
        + rng.normal(0.0, 0.3, n_rows)
    )
    label = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int8)

    cols[LABEL_COL] = label
    for name in INT_COLS:
        cols[name] = ints[name]
    for name in CAT_COLS:
        strings = _hex_ids(cat_ids[name]).astype(object)
        strings[rng.random(n_rows) < 0.05] = None  # missing categoricals
        cols[name] = strings

    df = pd.DataFrame(cols)[list(RAW_COLUMNS)]
    for name in INT_COLS:
        df[name] = df[name].astype("Int64")  # written as "5", not "5.0"; NA -> empty
    return df


def write_synthetic_days(
    out_dir: str | Path,
    days: list[int],
    n_rows: int,
    seed: int = 0,
    file_pattern: str = "day_{day}.gz",
) -> list[Path]:
    """Write one gzip TSV per day and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for day in days:
        path = out_dir / file_pattern.format(day=day)
        generate_day(n_rows, day, seed).to_csv(
            path, sep="\t", header=False, index=False, compression="gzip"
        )
        paths.append(path)
    return paths

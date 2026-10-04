"""Dense features for the gradient-boosted model.

Categoricals are replaced by their training-set frequency (count encoding), a
robust choice for ~10^6-cardinality hashed ids. The encoder is fit on TRAIN only
and then applied unchanged to val/test, so there is no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floor_optimizer.data.schema import CAT_COLS, INT_COLS


class FrequencyEncoder:
    """Replace each categorical id by its relative frequency in the training data.

    Ids seen fewer than `min_count` times in training, and ids never seen at all,
    both map to 0.0 (a shared "rare/unseen" value). Numeric columns pass through
    untouched; LightGBM handles NaN natively.
    """

    def __init__(self, min_count: int = 5) -> None:
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        self.min_count = min_count
        self.n_train_: int | None = None
        self.maps_: dict[str, pd.Series] = {}

    def fit(self, df: pd.DataFrame) -> FrequencyEncoder:
        n = len(df)
        if n == 0:
            raise ValueError("Cannot fit on an empty frame")
        self.n_train_ = n
        self.maps_ = {}
        for col in CAT_COLS:
            counts = df[col].value_counts()
            counts = counts[counts >= self.min_count]
            self.maps_[col] = (counts / n).astype(np.float32)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.n_train_ is None:
            raise RuntimeError("FrequencyEncoder must be fit before transform")
        data: dict[str, np.ndarray] = {
            col: df[col].to_numpy(dtype=np.float32) for col in INT_COLS
        }
        for col in CAT_COLS:
            mapped = df[col].map(self.maps_[col])
            data[col] = mapped.fillna(0.0).to_numpy(dtype=np.float32)
        return pd.DataFrame(data, index=df.index)

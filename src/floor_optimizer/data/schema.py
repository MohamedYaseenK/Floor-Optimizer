"""Criteo click-log schema and shared constants.

Raw layout (tab separated, no header):
    <label> <I1..I13: integer features> <C1..C26: hashed categorical features>
"""

from __future__ import annotations

LABEL_COL = "label"
INT_COLS: tuple[str, ...] = tuple(f"I{i}" for i in range(1, 14))
CAT_COLS: tuple[str, ...] = tuple(f"C{i}" for i in range(1, 27))
RAW_COLUMNS: tuple[str, ...] = (LABEL_COL, *INT_COLS, *CAT_COLS)
FEATURE_COLS: tuple[str, ...] = (*INT_COLS, *CAT_COLS)

# Explicit token for missing categoricals, so "missing" is its own category.
MISSING_CAT_TOKEN = "__missing__"

"""Generate synthetic Criteo-style day files for smoke-testing the pipeline.

Usage:
    python scripts/make_synthetic_data.py --out-dir data/synthetic/raw --rows 100000
"""

from __future__ import annotations

import argparse

from floor_optimizer.data.synthetic import write_synthetic_days


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/synthetic/raw")
    parser.add_argument("--days", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rows", type=int, default=100_000, help="rows per day")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = write_synthetic_days(args.out_dir, args.days, args.rows, args.seed)
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

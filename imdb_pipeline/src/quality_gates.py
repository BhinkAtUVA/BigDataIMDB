"""
quality_gates.py
────────────────
Each gate is a function that takes a DataFrame and raises a
RuntimeError with a clear message if something critical is wrong.
Add new gates freely — build() will call run_all_gates().
"""

import pandas as pd
from config import REQUIRED_COLS, LABEL_COL, YEAR_MIN, YEAR_MAX


def gate_required_columns(df: pd.DataFrame, split: str) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise RuntimeError(
            f"[{split}] Missing required columns: {missing}\n"
            f"  Available: {set(df.columns)}"
        )


def gate_tconst_not_null(df: pd.DataFrame, split: str) -> None:
    n = df["tconst"].isna().sum()
    if n > 0:
        raise RuntimeError(f"[{split}] {n} rows with NULL tconst — cannot proceed.")


def gate_tconst_unique(df: pd.DataFrame, split: str) -> None:
    dups = df["tconst"].duplicated().sum()
    if dups > 0:
        raise RuntimeError(
            f"[{split}] {dups} duplicate tconst values after deduplication — "
            "dedup logic may have failed."
        )


def gate_label_values(df: pd.DataFrame, split: str) -> None:
    if LABEL_COL not in df.columns:
        return  # test set has no label
    invalid = ~df[LABEL_COL].isin([0, 1, True, False])
    n = invalid.sum()
    if n > 0:
        raise RuntimeError(
            f"[{split}] {n} rows with invalid label values: "
            f"{df.loc[invalid, LABEL_COL].unique()}"
        )


def gate_year_range(df: pd.DataFrame, split: str) -> None:
    if "startYear" not in df.columns:
        return
    numeric = pd.to_numeric(df["startYear"], errors="coerce").dropna()
    out = ((numeric < YEAR_MIN) | (numeric > YEAR_MAX)).sum()
    if out > 0:
        raise RuntimeError(
            f"[{split}] {out} rows with startYear outside [{YEAR_MIN},{YEAR_MAX}]. "
            "Check year-swap cleaning."
        )


def run_all_gates(df: pd.DataFrame, split: str) -> None:
    """Run all gates in order. Stops at first failure."""
    gate_required_columns(df, split)
    gate_tconst_not_null(df, split)
    gate_tconst_unique(df, split)
    gate_label_values(df, split)
    gate_year_range(df, split)
    print(f"  ✓ All quality gates passed for [{split}]")

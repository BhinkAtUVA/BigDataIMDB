"""
cleaning_hooks.py
─────────────────
This is the EDA team's file.

The pipeline calls apply_cleaning_hooks(df, split) at a fixed point
in the Silver stage. The EDA team registers their cleaning functions
here — the core pipeline (build_features.py) never needs to change.

HOW TO ADD A CLEANING STEP
───────────────────────────
1. Write a function with this signature:
       def your_function(df: pd.DataFrame) -> pd.DataFrame

2. Register it at the bottom of this file:
       register(your_function)

That's it. Functions run in registration order.
Each function must return the full DataFrame (modified or not).

RULES
─────
- Never hardcode column values. Use config.py constants for thresholds.
- If a column you expect is missing, skip gracefully (don't crash).
- Log what you changed with print() so it shows in the pipeline output.
- Do NOT filter rows based on label (data leakage).
"""

import pandas as pd
import numpy as np
from typing import Callable

# ── Registry ──────────────────────────────────────────────────────────────────

_hooks: list[Callable[[pd.DataFrame], pd.DataFrame]] = []


def register(fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Register a cleaning function to be applied in the Silver stage."""
    _hooks.append(fn)


def apply_cleaning_hooks(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Called by build_features.py. Runs all registered hooks in order."""
    if not _hooks:
        print(f"  [hooks] No cleaning hooks registered for [{split}]")
        return df
    for fn in _hooks:
        before = len(df)
        df = fn(df)
        after = len(df)
        dropped = before - after
        tag = f" (dropped {dropped} rows)" if dropped else ""
        print(f"  [hook] {fn.__name__}{tag} → [{split}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# EDA TEAM: ADD YOUR CLEANING FUNCTIONS BELOW
# ══════════════════════════════════════════════════════════════════════════════

# ── Example: impute missing numVotes with median ───────────────────────────
def impute_numVotes_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numVotes with the median of the available values.
    Safe: skips silently if numVotes column is absent.
    """
    if "numVotes" not in df.columns:
        return df
    median = df["numVotes"].median()
    n_filled = df["numVotes"].isna().sum()
    df["numVotes"] = df["numVotes"].fillna(median)
    if n_filled:
        print(f"    imputed {n_filled} missing numVotes with median={median:.0f}")
    return df


# ── Example: drop endYear (100% null, not useful for movies) ──────────────
def drop_endYear(df: pd.DataFrame) -> pd.DataFrame:
    """Drop endYear — it is always NULL for movies (only relevant for TV series)."""
    if "endYear" in df.columns:
        df = df.drop(columns=["endYear"])
    return df


# ── Example: clip extreme runtimeMinutes ──────────────────────────────────
def clip_runtime_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag runtimeMinutes > 300 as a separate binary feature rather than
    clipping, so the information is preserved for the model.
    EDA team: adjust threshold in config.py (RUNTIME_MAX) if needed.
    """
    if "runtimeMinutes" not in df.columns:
        return df
    from config import RUNTIME_MAX
    df["is_long_film"] = (df["runtimeMinutes"].astype(float) > RUNTIME_MAX).astype("Int8")
    return df


# ── Placeholder: EDA team fills this in ───────────────────────────────────
# def fix_XYZ(df: pd.DataFrame) -> pd.DataFrame:
#     """Describe what you found and how you fix it."""
#     ...
#     return df


# ══════════════════════════════════════════════════════════════════════════════
# REGISTER YOUR FUNCTIONS HERE (order matters)
# ══════════════════════════════════════════════════════════════════════════════

register(drop_endYear)
register(impute_numVotes_median)
register(clip_runtime_outliers)

# register(fix_XYZ)   ← uncomment when ready

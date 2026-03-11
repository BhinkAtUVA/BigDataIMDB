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

import re
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
# EDA CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Drop endYear (90% null for movies, already used in year-swap) ──────
def drop_endYear(df: pd.DataFrame) -> pd.DataFrame:
    """Drop endYear — it is 90%+ NULL for movies (only relevant for TV series).
    The useful information was already extracted by the year-swap fix in Silver."""
    if "endYear" in df.columns:
        df = df.drop(columns=["endYear"])
    return df


# ── 2. Impute missing numVotes with median ────────────────────────────────
def impute_numVotes_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numVotes with the median of the available values.
    EDA finding: ~10% of Bronze rows have null numVotes. After Silver
    cast_numeric this is resolved, but a safety net is needed.
    """
    if "numVotes" not in df.columns:
        return df
    median = df["numVotes"].median()
    n_filled = df["numVotes"].isna().sum()
    if n_filled:
        df["numVotes"] = df["numVotes"].fillna(median)
        print(f"    imputed {n_filled} missing numVotes with median={median:.0f}")
    return df


# ── 3. Impute missing runtimeMinutes with median ─────────────────────────
def impute_runtimeMinutes_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA finding: 13 rows (0.16%) still have null runtimeMinutes in Gold.
    The sklearn Imputer handles this, but imputing here is cleaner —
    it lets quality gates verify completeness, and the Gold SQL
    computes derived features (is_long_film) without NaN propagation.
    """
    if "runtimeMinutes" not in df.columns:
        return df
    median = df["runtimeMinutes"].median()
    n_filled = df["runtimeMinutes"].isna().sum()
    if n_filled:
        df["runtimeMinutes"] = df["runtimeMinutes"].fillna(median)
        print(f"    imputed {n_filled} missing runtimeMinutes with median={median:.0f}")
    return df


# ── 4. Cap runtimeMinutes outliers ────────────────────────────────────────
def cap_runtime_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA finding: 8 films have runtime > 300 min (up to 551 min).
    These are documentaries / anthologies that distort the distribution.
    Cap at RUNTIME_CAP and preserve the original signal via is_long_film.
    """
    if "runtimeMinutes" not in df.columns:
        return df
    from config import RUNTIME_CAP
    extreme = df["runtimeMinutes"] > RUNTIME_CAP
    n_capped = extreme.sum()
    # Create is_long_film flag BEFORE capping so the info is preserved
    df["is_long_film"] = extreme.astype("Int8")
    if n_capped:
        df.loc[extreme, "runtimeMinutes"] = RUNTIME_CAP
        print(f"    capped {n_capped} runtimeMinutes > {RUNTIME_CAP} → {RUNTIME_CAP}")
    return df


# ── 5. Impute missing startYear with median ──────────────────────────────
def impute_startYear_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA finding: year-swap fix resolves most missing startYear, but a few
    edge-case rows (val/test) may still have nulls. Median imputation
    preserves temporal signal better than dropping rows.
    """
    if "startYear" not in df.columns:
        return df
    n_miss = df["startYear"].isna().sum()
    if n_miss:
        median = df["startYear"].median()
        df["startYear"] = df["startYear"].fillna(median)
        print(f"    imputed {n_miss} missing startYear with median={median:.0f}")
    return df


# ── 6. Create is_foreign_title flag ──────────────────────────────────────
def create_is_foreign_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA finding: 37% of non-null originalTitle values differ from
    primaryTitle → these are foreign-language films with translated
    English titles. Foreign films have different rating dynamics.
    """
    if "originalTitle" not in df.columns or "primaryTitle" not in df.columns:
        return df
    has_orig = df["originalTitle"].notna()
    differs  = df["originalTitle"] != df["primaryTitle"]
    df["is_foreign_title"] = (has_orig & differs).astype("Int8")
    n_foreign = df["is_foreign_title"].sum()
    print(f"    is_foreign_title: {n_foreign} foreign-language films flagged")
    return df


# ── 7. Create title_has_number flag ──────────────────────────────────────
def create_title_has_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA finding: Titles containing numbers (sequels, year-named films)
    have different rating distributions. '2', 'III', '1984', etc.
    """
    if "primaryTitle" not in df.columns:
        return df
    df["title_has_number"] = df["primaryTitle"].str.contains(
        r'\d', na=False, regex=True
    ).astype("Int8")
    n = df["title_has_number"].sum()
    print(f"    title_has_number: {n} titles contain digits")
    return df


# ── 8. Create title_word_count feature ───────────────────────────────────
def create_title_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA finding: Number of words in a title correlates weakly with genre
    and rating. Single-word titles behave differently from long ones.
    """
    if "primaryTitle" not in df.columns:
        return df
    df["title_word_count"] = (
        df["primaryTitle"]
        .fillna("")
        .str.split()
        .str.len()
        .astype("Int16")
    )
    return df


# ── 9. Drop originalTitle (50% null, signal extracted into is_foreign) ───
def drop_originalTitle(df: pd.DataFrame) -> pd.DataFrame:
    """
    After extracting is_foreign_title, the raw originalTitle column has no
    further use — it's 50% null and free text. Drop to keep the feature
    set clean for the model.
    """
    if "originalTitle" in df.columns:
        df = df.drop(columns=["originalTitle"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# REGISTER FUNCTIONS HERE (order matters!)
#
# Execution order rationale:
#   1. drop_endYear          — remove useless column first
#   2. impute_numVotes       — fill nulls before any derived features
#   3. impute_runtimeMinutes — fill nulls before capping
#   4. cap_runtime_outliers  — cap + create is_long_film (needs imputed runtime)
#   5. impute_startYear      — safety net after year-swap
#   6. create_is_foreign     — needs originalTitle + primaryTitle
#   7. create_title_has_num  — needs primaryTitle
#   8. create_title_word_cnt — needs primaryTitle
#   9. drop_originalTitle    — must come AFTER is_foreign extraction
# ══════════════════════════════════════════════════════════════════════════════

register(drop_endYear)
register(impute_numVotes_median)
register(impute_runtimeMinutes_median)
register(cap_runtime_outliers)
register(impute_startYear_median)
register(create_is_foreign_title)
register(create_title_has_number)
register(create_title_word_count)
register(drop_originalTitle)

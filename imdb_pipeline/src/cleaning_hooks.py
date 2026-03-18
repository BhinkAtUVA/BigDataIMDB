import pandas as pd
import numpy as np
from typing import Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import RUNTIME_MAX, YEAR_MIN



# Registry

_hooks: list[Callable[[pd.DataFrame], pd.DataFrame]] = []


def register(fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    # a cleaning function to be applied after Silver cleaning.
    _hooks.append(fn)


def apply_cleaning_hooks(df: pd.DataFrame, split: str) -> pd.DataFrame:

    if not _hooks:
        print(f"  [hooks] No hooks registered for [{split}]")
        return df
    for fn in _hooks:
        before_rows = len(df)
        before_cols = set(df.columns)
        df = fn(df)
        after_rows  = len(df)
        new_cols    = set(df.columns) - before_cols
        dropped     = before_rows - after_rows
        tag_rows    = f" (dropped {dropped} rows)" if dropped else ""
        tag_cols    = f" (added cols: {sorted(new_cols)})" if new_cols else ""
        print(f"  [hook] {fn.__name__}{tag_rows}{tag_cols} -> [{split}]")
    return df



# v1 HOOKS — original set

def drop_endYear(df: pd.DataFrame) -> pd.DataFrame:

    if "endYear" in df.columns:
        df = df.drop(columns=["endYear"])
    return df


def impute_numVotes_median(df: pd.DataFrame) -> pd.DataFrame:

    if "numVotes" not in df.columns:
        return df
    n_missing = int(df["numVotes"].isna().sum())
    if n_missing == 0:
        return df
    median = df["numVotes"].median()
    df = df.copy()
    df["numVotes"] = df["numVotes"].fillna(median)
    print(f"    imputed {n_missing} missing numVotes with median={median:.0f}")
    return df


def clip_runtime_outliers(df: pd.DataFrame) -> pd.DataFrame:

    if "runtimeMinutes" not in df.columns:
        return df
    df = df.copy()
    df["is_long_film"] = (
        pd.to_numeric(df["runtimeMinutes"], errors="coerce") > RUNTIME_MAX
    ).astype("Int8")
    return df



# v2 HOOKS — EDA-driven additions

def impute_runtime_median(df: pd.DataFrame) -> pd.DataFrame:

    if "runtimeMinutes" not in df.columns:
        return df
    n_missing = int(
        pd.to_numeric(df["runtimeMinutes"], errors="coerce").isna().sum()
    )
    if n_missing == 0:
        return df
    df = df.copy()
    rt = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    median = rt.median()
    df["runtimeMinutes"] = rt.fillna(median)
    print(f"    imputed {n_missing} missing runtimeMinutes with median={median:.0f}")
    return df


def flag_old_film(df: pd.DataFrame) -> pd.DataFrame:

    if "startYear" not in df.columns:
        return df
    df = df.copy()
    df["is_old_film"] = (
        pd.to_numeric(df["startYear"], errors="coerce") < 1930
    ).astype("Int8")
    n = int(df["is_old_film"].sum())
    print(f"    flagged {n} pre-1930 films as is_old_film=1")
    return df


def flag_high_votes(df: pd.DataFrame) -> pd.DataFrame:

    if "numVotes" not in df.columns:
        return df
    df = df.copy()
    df["is_blockbuster"] = (
        pd.to_numeric(df["numVotes"], errors="coerce") > 1_000_000
    ).astype("Int8")
    n = int(df["is_blockbuster"].sum())
    print(f"    flagged {n} films as is_blockbuster=1 (numVotes > 1M)")
    return df


def flag_foreign_title(df: pd.DataFrame) -> pd.DataFrame:

    if "primaryTitle" not in df.columns or "originalTitle" not in df.columns:
        return df
    df = df.copy()
    df["is_foreign_title"] = (
        df["originalTitle"].notna() &
        (df["primaryTitle"].str.strip() != df["originalTitle"].str.strip())
    ).astype("Int8")
    n = int(df["is_foreign_title"].sum())
    print(f"    flagged {n} films as is_foreign_title=1")
    return df


def add_title_features(df: pd.DataFrame) -> pd.DataFrame:

    if "primaryTitle" not in df.columns:
        return df
    df = df.copy()
    df["title_word_count"] = df["primaryTitle"].str.split().str.len().astype("Int16")
    df["title_has_number"] = (
        df["primaryTitle"].str.contains(r"\d", regex=True, na=False)
    ).astype("Int8")
    return df


# v1 — structural fixes
register(drop_endYear)
register(impute_numVotes_median)
register(clip_runtime_outliers)      # creates is_long_film BEFORE imputation

# v2 — EDA-driven additions
register(impute_runtime_median)      # imputes AFTER is_long_film is created
register(flag_old_film)
register(flag_high_votes)
register(flag_foreign_title)
register(add_title_features)
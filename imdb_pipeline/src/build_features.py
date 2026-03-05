"""
build_features.py
─────────────────
Main pipeline: Bronze → Silver → Gold → Parquet outputs

  Bronze : raw ingestion, union by name, sentinel→NULL
  Silver : type casting, accent normalisation, year-swap fix, range clipping
  Gold   : join relations, compute features, export

Run:
  cd imdb_pipeline
  python src/build_features.py

Outputs:
  data/processed/train_features.csv
  data/processed/val_features.csv
  data/processed/test_features.csv
  outputs/profiles/profile_*.csv

NOTE — DuckDB swap points are marked with:
  # [DUCKDB] replace this block with the DuckDB equivalent shown in the comment
"""

import glob
import json
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAW, PROCESSED, PROFILES,
    TRAIN_GLOB, VAL_FILE, TEST_FILE,
    DIRECTING_FILE, WRITING_FILE,
    REQUIRED_COLS, LABEL_COL,
    NULL_SENTINELS,
    RUNTIME_MIN, RUNTIME_MAX, NUMVOTES_MIN, YEAR_MIN, YEAR_MAX,
)
from json_to_relations import load_directing, load_writing
from quality_gates import run_all_gates
from cleaning_hooks import apply_cleaning_hooks


# ══════════════════════════════════════════════════════════════════════════════
# BRONZE — Raw ingestion
# ══════════════════════════════════════════════════════════════════════════════

def load_csv_robust(path: Path) -> pd.DataFrame:
    """
    Load a single CSV tolerantly:
      - strips an accidental integer index column if present
      - replaces all null sentinels with pd.NA
      - strips whitespace from string columns
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Drop accidental index column (unnamed, sequential integers)
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Replace null sentinels
    df = df.replace(list(NULL_SENTINELS), pd.NA)

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    return df


def ingest_train(raw_dir: Path) -> pd.DataFrame:
    """
    [DUCKDB] Equivalent:
        CREATE OR REPLACE VIEW bronze_train AS
        SELECT * FROM read_csv_auto('data/raw/train-*.csv',
                                    union_by_name=true, ignore_errors=true);
    """
    files = sorted(glob.glob(str(raw_dir / TRAIN_GLOB)))
    if not files:
        raise FileNotFoundError(f"No train files found matching {raw_dir / TRAIN_GLOB}")

    parts = []
    for f in files:
        df = load_csv_robust(Path(f))
        df["_source_file"] = Path(f).name
        parts.append(df)
        print(f"  Loaded {Path(f).name}: {len(df)} rows")

    # union_by_name: align on column name, fill missing cols with NA
    combined = pd.concat(parts, ignore_index=True, sort=False)
    print(f"  → Combined train: {len(combined)} rows, {combined['tconst'].nunique()} unique tconst")
    return combined


def ingest_single(raw_dir: Path, filename: str) -> pd.DataFrame:
    """Load a single CSV (val or test)."""
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = load_csv_robust(path)
    print(f"  Loaded {filename}: {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SILVER — Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def normalize_accents(s: str) -> str:
    """
    Remove synthetically injected accent marks from titles.
    Strategy: NFD decompose → strip all combining diacritical marks → recompose.
    e.g. 'Báttling Bútlér' → 'Battling Butler'
         'Thé Grápés ớf Wráth' → 'The Grapes of Wrath'
    This is safe for truly non-English titles too: stripping diacritics from
    German/French originals only applies to primaryTitle (which should be the
    English version); originalTitle is left untouched.
    """
    if not isinstance(s, str):
        return s
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped


def fix_year_swap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detected issue: ~10% of rows have startYear='\\N' but endYear holds
    the actual release year (the two columns are swapped).
    Fix: when startYear is NULL and endYear is a plausible year → swap them.
    This is data-driven, not hardcoded: we check for plausible year range.
    """
    df = df.copy()
    sy = pd.to_numeric(df["startYear"], errors="coerce")
    ey = pd.to_numeric(df["endYear"],   errors="coerce")

    swap_mask = sy.isna() & ey.between(YEAR_MIN, YEAR_MAX)
    n_swapped = swap_mask.sum()
    if n_swapped > 0:
        print(f"    year-swap fix: {n_swapped} rows (startYear↔endYear swapped)")
        df.loc[swap_mask, "startYear"] = df.loc[swap_mask, "endYear"]
        df.loc[swap_mask, "endYear"]   = pd.NA

    return df


def cast_numeric(df: pd.DataFrame, col: str, dtype=float,
                 lo=None, hi=None) -> pd.DataFrame:
    """
    TRY_CAST equivalent: coerce to numeric, clip to range, keep NULLs.
    [DUCKDB] Equivalent: TRY_CAST({col} AS DOUBLE)
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    if lo is not None:
        df.loc[df[col] < lo, col] = pd.NA
    if hi is not None:
        df.loc[df[col] > hi, col] = pd.NA
    return df


def silver_clean(df: pd.DataFrame, has_label: bool) -> pd.DataFrame:
    """
    [DUCKDB] Equivalent silver layer:
        CREATE OR REPLACE VIEW silver AS
        SELECT
            CAST(tconst AS VARCHAR)                         AS tconst,
            normalize_accents(primaryTitle)                 AS primaryTitle,
            originalTitle,
            TRY_CAST(startYear AS INTEGER)                  AS startYear,
            TRY_CAST(endYear   AS INTEGER)                  AS endYear,
            TRY_CAST(runtimeMinutes AS DOUBLE)              AS runtimeMinutes,
            TRY_CAST(numVotes AS DOUBLE)                    AS numVotes
        FROM bronze;
    """
    df = df.copy()

    # ── 1. Normalize accents on primaryTitle (English title only)
    if "primaryTitle" in df.columns:
        df["primaryTitle"] = df["primaryTitle"].apply(normalize_accents)

    # ── 2. Fix swapped year columns
    df = fix_year_swap(df)

    # ── 3. Cast numerics with range validation
    df = cast_numeric(df, "startYear",      lo=YEAR_MIN,    hi=YEAR_MAX)
    df = cast_numeric(df, "endYear",        lo=YEAR_MIN,    hi=YEAR_MAX)
    df = cast_numeric(df, "runtimeMinutes", lo=RUNTIME_MIN, hi=RUNTIME_MAX)
    df = cast_numeric(df, "numVotes",       lo=NUMVOTES_MIN)

    # ── 4. Cast label to int (0/1)
    if has_label and LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].map(
            {"True": 1, "False": 0, "true": 1, "false": 0,
             True: 1, False: 0, "1": 1, "0": 0}
        ).astype("Int8")

    # ── 5. Deduplicate on tconst (keep first occurrence)
    #  [DUCKDB] ROW_NUMBER() OVER (PARTITION BY tconst ORDER BY tconst) = 1
    n_before = len(df)
    df = df.drop_duplicates(subset=["tconst"], keep="first").reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"    dedup: removed {n_before - n_after} duplicate tconst rows")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# GOLD — Feature engineering + joins
# ══════════════════════════════════════════════════════════════════════════════

def build_relation_aggregates(directing_df: pd.DataFrame,
                               writing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate directors and writers per movie.
    [DUCKDB] Equivalent:
        CREATE VIEW agg_directors AS
        SELECT tconst, COUNT(DISTINCT nconst) AS n_directors FROM rel_directing GROUP BY tconst;
        CREATE VIEW agg_writers AS
        SELECT tconst, COUNT(DISTINCT nconst) AS n_writers FROM rel_writing GROUP BY tconst;
    """
    agg_dir = (
        directing_df.groupby("tconst")["nconst"]
        .nunique()
        .reset_index()
        .rename(columns={"nconst": "n_directors"})
    )
    agg_wri = (
        writing_df.groupby("tconst")["nconst"]
        .nunique()
        .reset_index()
        .rename(columns={"nconst": "n_writers"})
    )
    agg = agg_dir.merge(agg_wri, on="tconst", how="outer")
    return agg


def gold_features(df: pd.DataFrame, agg: pd.DataFrame,
                  has_label: bool) -> pd.DataFrame:
    """
    Join aggregates and compute derived features.
    [DUCKDB] Equivalent:
        CREATE TABLE gold AS
        SELECT
            m.*,
            COALESCE(d.n_directors, 0)            AS n_directors,
            COALESCE(w.n_writers, 0)              AS n_writers,
            LN(numVotes + 1)                      AS log_numVotes,
            LENGTH(primaryTitle)                  AS len_primaryTitle,
            CASE WHEN endYear IS NULL THEN 0 ELSE 1 END AS has_endYear,
            ...
        FROM silver m
        LEFT JOIN agg_directors d USING (tconst)
        LEFT JOIN agg_writers   w USING (tconst);
    """
    df = df.merge(agg, on="tconst", how="left")
    df["n_directors"] = df["n_directors"].fillna(0).astype(int)
    df["n_writers"]   = df["n_writers"].fillna(0).astype(int)

    # Derived features
    df["log_numVotes"]     = np.log1p(df["numVotes"].astype(float))
    df["len_primaryTitle"] = df["primaryTitle"].str.len().fillna(0).astype(int)
    df["has_endYear"]      = df["endYear"].notna().astype(int)
    df["decade"]           = (df["startYear"].astype(float) // 10 * 10).astype("Int64")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def profile_dataframe(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    """
    Write a missingness + basic stats profile CSV for the given DataFrame.
    Useful for presenting evidence of data quality on the poster.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        miss   = series.isna().sum()
        n_unique = series.nunique(dropna=True)
        rows.append({
            "split":       name,
            "column":      col,
            "n_rows":      n,
            "n_missing":   int(miss),
            "pct_missing": round(float(miss) / max(n, 1) * 100, 2),
            "n_unique":    int(n_unique),
            "dtype":       str(series.dtype),
        })
    profile = pd.DataFrame(rows)
    out_path = out_dir / f"profile_{name}.csv"
    profile.to_csv(out_path, index=False)
    print(f"  Profile saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build(raw_dir: Path = RAW,
          processed_dir: Path = PROCESSED,
          profiles_dir: Path = PROFILES) -> None:

    processed_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    print("\n── BRONZE: Ingestion ─────────────────────────────────────────")
    bronze_train = ingest_train(raw_dir)
    bronze_val   = ingest_single(raw_dir, VAL_FILE)
    bronze_test  = ingest_single(raw_dir, TEST_FILE)

    print("\n── JSON Relations ────────────────────────────────────────────")
    directing_df = load_directing(raw_dir / DIRECTING_FILE)
    writing_df   = load_writing(raw_dir   / WRITING_FILE)
    print(f"  directing: {len(directing_df)} rows, {directing_df['tconst'].nunique()} movies")
    print(f"  writing:   {len(writing_df)} rows, {writing_df['tconst'].nunique()} movies")

    agg = build_relation_aggregates(directing_df, writing_df)
    print(f"  aggregates: {len(agg)} movies with director/writer counts")

    print("\n── SILVER: Cleaning ──────────────────────────────────────────")
    silver_train = silver_clean(bronze_train, has_label=True)
    silver_val   = silver_clean(bronze_val,   has_label=True)
    silver_test  = silver_clean(bronze_test,  has_label=False)

    print("\n── Cleaning Hooks (EDA injections) ───────────────────────────")
    silver_train = apply_cleaning_hooks(silver_train, "train")
    silver_val   = apply_cleaning_hooks(silver_val,   "val")
    silver_test  = apply_cleaning_hooks(silver_test,  "test")

    print("\n── Quality Gates (Silver) ────────────────────────────────────")

    print("\n── Quality Gates (Silver) ────────────────────────────────────")
    run_all_gates(silver_train, "train")
    run_all_gates(silver_val,   "val")
    run_all_gates(silver_test,  "test")

    print("\n── GOLD: Features + Joins ────────────────────────────────────")
    gold_train = gold_features(silver_train, agg, has_label=True)
    gold_val   = gold_features(silver_val,   agg, has_label=True)
    gold_test  = gold_features(silver_test,  agg, has_label=False)

    print("\n── Profiles ──────────────────────────────────────────────────")
    for df, name in [(gold_train, "gold_train"),
                     (gold_val,   "gold_val"),
                     (gold_test,  "gold_test")]:
        profile_dataframe(df, name, profiles_dir)

    print("\n── Export Parquet ────────────────────────────────────────────")
    # [DUCKDB] COPY gold_train TO 'data/processed/train_features.csv' (FORMAT PARQUET);
    gold_train.to_csv(processed_dir / "train_features.csv", index=False)
    gold_val.to_csv(  processed_dir / "val_features.csv",   index=False)
    gold_test.to_csv( processed_dir / "test_features.csv",  index=False)

    print(f"\n✓ Pipeline complete.")
    print(f"  train : {len(gold_train)} rows  |  {gold_train[LABEL_COL].sum()} positive")
    print(f"  val   : {len(gold_val)} rows  |  {gold_val[LABEL_COL].sum() if LABEL_COL in gold_val.columns else "no label"} positive")
    print(f"  test  : {len(gold_test)} rows")
    print(f"  features: {[c for c in gold_train.columns if c not in ('tconst','_source_file',LABEL_COL)]}")


if __name__ == "__main__":
    build()

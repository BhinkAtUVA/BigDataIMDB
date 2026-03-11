"""
build_features.py
─────────────────
Main pipeline: Bronze → Silver → Gold → CSV outputs

  Bronze : DuckDB — raw ingestion, union by name, sentinel→NULL
  Silver : pandas — type casting, accent normalisation, year-swap fix
  Hooks  : pandas — EDA team cleaning functions
  Gates  : pandas — validation checks
  Gold   : DuckDB — joins, aggregations, feature computation

WHY THIS SPLIT:
  DuckDB handles what SQL is built for:
    - ingesting multiple files with union_by_name (schema-drift tolerant)
    - set-based joins and GROUP BY aggregations (faster, less code)
  pandas handles what SQL is bad at:
    - row-level string manipulation (accent normalisation)
    - conditional column swaps (year-swap fix)
    - arbitrary Python functions (cleaning hooks)
  sklearn handles what neither SQL nor pandas does:
    - model training and prediction

Run:
    cd imdb_pipeline
    python src/build_features.py

Outputs:
    data/processed/train_features.csv
    data/processed/val_features.csv
    data/processed/test_features.csv
    outputs/profiles/profile_*.csv
"""

import unicodedata
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLEANED, MERGED, RAW, PROCESSED, PROFILES,
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
# BRONZE — DuckDB ingestion
# ══════════════════════════════════════════════════════════════════════════════

def bronze_ingest(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """
    Load all raw CSV files into DuckDB bronze views.

    Key DuckDB features used:
      - read_csv_auto: infers types, handles inconsistent schemas
      - union_by_name=true: aligns files on column name, not position
        → train-6.csv has runtimeMinutes as int64, others as string: handled automatically
      - ignore_errors=false: malformed rows should lead to a crash to attract attention

    Equivalent pandas approach would require manual dtype reconciliation
    across 8 files with schema drift — DuckDB does this in one line.
    """
    train_glob = str(raw_dir / TRAIN_GLOB)
    val_path   = str(raw_dir / VAL_FILE)
    test_path  = str(raw_dir / TEST_FILE)

    sentinel_list = ", ".join(f"'{s}'" for s in NULL_SENTINELS if s.strip())

    def sentinel_col(col: str) -> str:
        return (
            f"CASE WHEN TRIM(CAST({col} AS VARCHAR)) IN ({sentinel_list}) "
            f"THEN NULL ELSE TRIM(CAST({col} AS VARCHAR)) END AS {col}"
        )

    cols = ["tconst", "primaryTitle", "originalTitle",
            "startYear", "endYear", "runtimeMinutes", "numVotes"]

    col_sql = ",\n            ".join(sentinel_col(c) for c in cols)

    # Train: glob all train-*.csv with union_by_name
    con.execute(f"""
        CREATE OR REPLACE VIEW bronze_train AS
        SELECT
            {col_sql},
            label
        FROM read_csv_auto(
            '{train_glob}',
            union_by_name = true,
            ignore_errors = false
        )
    """)
    n_train = con.execute("SELECT COUNT(*) FROM bronze_train").fetchone()[0]
    print(f"  bronze_train: {n_train} rows")

    con.execute(f"""
        CREATE OR REPLACE VIEW bronze_val AS
        SELECT {col_sql}
        FROM read_csv_auto('{val_path}', ignore_errors = true)
    """)
    n_val = con.execute("SELECT COUNT(*) FROM bronze_val").fetchone()[0]
    print(f"  bronze_val:   {n_val} rows")

    con.execute(f"""
        CREATE OR REPLACE VIEW bronze_test AS
        SELECT {col_sql}
        FROM read_csv_auto('{test_path}', ignore_errors = true)
    """)
    n_test = con.execute("SELECT COUNT(*) FROM bronze_test").fetchone()[0]
    print(f"  bronze_test:  {n_test} rows")


def bronze_to_pandas(con: duckdb.DuckDBPyConnection,
                     view: str) -> pd.DataFrame:
    """Materialise a bronze DuckDB view into pandas for Silver."""
    return con.execute(f"SELECT * FROM {view}").df()


# ══════════════════════════════════════════════════════════════════════════════
# SILVER — pandas cleaning
# (row-level transformations: DuckDB has no advantage here)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_accents(s: str) -> str:
    """
    Remove synthetically injected accent marks from titles.
    NFD decompose → strip combining diacritical marks.
    'Báttling Bútlér' → 'Battling Butler'
    Works for any accent on any letter — no lookup table needed.
    """
    if not isinstance(s, str):
        return s
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def fix_year_swap(df: pd.DataFrame) -> pd.DataFrame:
    """
    ~10% of rows have startYear=NULL but endYear holds the release year.
    Detection is data-driven: startYear NULL AND endYear in plausible range.
    """
    df = df.copy()
    sy = pd.to_numeric(df["startYear"], errors="coerce")
    ey = pd.to_numeric(df["endYear"],   errors="coerce")
    swap_mask = sy.isna() & ey.between(YEAR_MIN, YEAR_MAX)
    n = swap_mask.sum()
    if n > 0:
        print(f"    year-swap fix: {n} rows")
        df.loc[swap_mask, "startYear"] = df.loc[swap_mask, "endYear"]
        df.loc[swap_mask, "endYear"]   = pd.NA
    return df


def cast_numeric(df: pd.DataFrame, col: str,
                 lo=None, hi=None) -> pd.DataFrame:
    """TRY_CAST equivalent: invalid → NULL, out-of-range → NULL."""
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    if lo is not None:
        df.loc[df[col] < lo, col] = pd.NA
    if hi is not None:
        df.loc[df[col] > hi, col] = pd.NA
    return df


def silver_clean(df: pd.DataFrame, has_label: bool) -> pd.DataFrame:
    df = df.copy()

    if "primaryTitle" in df.columns:
        df["primaryTitle"] = df["primaryTitle"].apply(normalize_accents)

    df = fix_year_swap(df)

    df = cast_numeric(df, "startYear",      lo=YEAR_MIN,    hi=YEAR_MAX)
    df = cast_numeric(df, "endYear",        lo=YEAR_MIN,    hi=YEAR_MAX)
    df = cast_numeric(df, "runtimeMinutes", lo=RUNTIME_MIN, hi=RUNTIME_MAX)
    df = cast_numeric(df, "numVotes",       lo=NUMVOTES_MIN)

    if has_label and LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].map(
            {"True": 1, "False": 0, "true": 1, "false": 0,
             True: 1, False: 0, "1": 1, "0": 0}
        ).astype("Int8")

    n_before = len(df)
    df = df.drop_duplicates(subset=["tconst"], keep="first").reset_index(drop=True)
    if len(df) < n_before:
        print(f"    dedup: removed {n_before - len(df)} duplicate rows")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# GOLD — DuckDB joins + aggregations
# ══════════════════════════════════════════════════════════════════════════════

def gold_build(con: duckdb.DuckDBPyConnection,
               silver_train: pd.DataFrame,
               silver_val:   pd.DataFrame,
               silver_test:  pd.DataFrame,
               directing_df: pd.DataFrame,
               writing_df:   pd.DataFrame) -> tuple:
    """
    Use DuckDB for Gold stage joins and aggregations.

    Why DuckDB here instead of pandas merge+groupby:
      - Aggregation + join in a single SQL expression is more readable
      - Hash-join optimiser is faster than pandas merge at scale
      - Adding new aggregations is one SQL line
      - Logic is transparent and auditable for the poster diagram
    """
    con.register("silver_train_df", silver_train)
    con.register("silver_val_df",   silver_val)
    con.register("silver_test_df",  silver_test)
    con.register("directing_df",    directing_df)
    con.register("writing_df",      writing_df)

    con.execute("""
        CREATE OR REPLACE VIEW agg_directors AS
        SELECT tconst, COUNT(DISTINCT nconst) AS n_directors
        FROM directing_df
        GROUP BY tconst
    """)
    con.execute("""
        CREATE OR REPLACE VIEW agg_writers AS
        SELECT tconst, COUNT(DISTINCT nconst) AS n_writers
        FROM writing_df
        GROUP BY tconst
    """)

    print(f"  agg_directors: {con.execute('SELECT COUNT(*) FROM agg_directors').fetchone()[0]} movies")
    print(f"  agg_writers:   {con.execute('SELECT COUNT(*) FROM agg_writers').fetchone()[0]} movies")

    def gold_sql(source: str, include_label: bool) -> str:
        label_col = f", CAST(m.{LABEL_COL} AS INTEGER) AS {LABEL_COL}" if include_label else ""
        return f"""
            SELECT
                m.tconst,
                m.primaryTitle,
                m.originalTitle,
                CAST(m.startYear AS DOUBLE)             AS startYear,
                CAST(m.runtimeMinutes AS DOUBLE)        AS runtimeMinutes,
                CAST(m.numVotes AS DOUBLE)              AS numVotes
                {label_col},
                COALESCE(d.n_directors, 0)              AS n_directors,
                COALESCE(w.n_writers,   0)              AS n_writers,
                LN(CAST(m.numVotes AS DOUBLE) + 1)      AS log_numVotes,
                LENGTH(m.primaryTitle)                  AS len_primaryTitle,
                0                                       AS has_endYear,
                FLOOR(CAST(m.startYear AS DOUBLE) / 10) * 10 AS decade
            FROM {source} m
            LEFT JOIN agg_directors d USING (tconst)
            LEFT JOIN agg_writers   w USING (tconst)
        """

    gold_train = con.execute(gold_sql("silver_train_df", include_label=True)).df()
    gold_val   = con.execute(gold_sql("silver_val_df",   include_label=False)).df()
    gold_test  = con.execute(gold_sql("silver_test_df",  include_label=False)).df()

    return gold_train, gold_val, gold_test


# ══════════════════════════════════════════════════════════════════════════════
# PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def profile_dataframe(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        miss   = series.isna().sum()
        rows.append({
            "split":       name,
            "column":      col,
            "n_rows":      n,
            "n_missing":   int(miss),
            "pct_missing": round(float(miss) / max(n, 1) * 100, 2),
            "n_unique":    int(series.nunique(dropna=True)),
            "dtype":       str(series.dtype),
        })
    pd.DataFrame(rows).to_csv(out_dir / f"profile_{name}.csv", index=False)
    print(f"  Profile saved: profile_{name}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build(raw_dir: Path = RAW,
          merged_dir: Path = MERGED,
          cleaned_dir: Path = CLEANED,
          processed_dir: Path = PROCESSED,
          profiles_dir: Path = PROFILES) -> None:

    merged_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()  # in-memory DuckDB instance

    print("\n── BRONZE: Ingestion (DuckDB) ────────────────────────────────")
    bronze_ingest(con, raw_dir)
    bronze_train_df = bronze_to_pandas(con, "bronze_train")
    bronze_val_df   = bronze_to_pandas(con, "bronze_val")
    bronze_test_df  = bronze_to_pandas(con, "bronze_test")

    bronze_train_df.to_csv(merged_dir / "train_features.csv", index=False)
    bronze_val_df.to_csv(  merged_dir / "val_features.csv",   index=False)
    bronze_test_df.to_csv( merged_dir / "test_features.csv",  index=False)

    print("\n── JSON Relations ────────────────────────────────────────────")
    directing_df = load_directing(raw_dir / DIRECTING_FILE)
    writing_df   = load_writing(raw_dir   / WRITING_FILE)
    print(f"  directing: {len(directing_df)} rows, {directing_df['tconst'].nunique()} movies")
    print(f"  writing:   {len(writing_df)} rows, {writing_df['tconst'].nunique()} movies")

    print("\n── SILVER: Cleaning (pandas) ─────────────────────────────────")
    silver_train = silver_clean(bronze_train_df, has_label=True)
    silver_val   = silver_clean(bronze_val_df,   has_label=True)
    silver_test  = silver_clean(bronze_test_df,  has_label=False)

    print("\n── Cleaning Hooks (EDA injections) ───────────────────────────")
    silver_train = apply_cleaning_hooks(silver_train, "train")
    silver_val   = apply_cleaning_hooks(silver_val,   "val")
    silver_test  = apply_cleaning_hooks(silver_test,  "test")

    print("\n── Quality Gates ─────────────────────────────────────────────")
    run_all_gates(silver_train, "train")
    run_all_gates(silver_val,   "val")
    run_all_gates(silver_test,  "test")

    silver_train.to_csv(cleaned_dir / "train_features.csv", index=False)
    silver_val.to_csv(  cleaned_dir / "val_features.csv",   index=False)
    silver_test.to_csv( cleaned_dir / "test_features.csv",  index=False)

    print("\n── GOLD: Features + Joins (DuckDB) ───────────────────────────")
    gold_train, gold_val, gold_test = gold_build(
        con, silver_train, silver_val, silver_test,
        directing_df, writing_df
    )

    print("\n── Profiles ──────────────────────────────────────────────────")
    for df, name in [(bronze_train_df, "bronze_train"),
                     (bronze_val_df,   "bronze_val"),
                     (bronze_test_df,  "bronze_test"),
                     (silver_train, "silver_train"),
                     (silver_val,   "silver_val"),
                     (silver_test,  "silver_test"),
                     (gold_train, "gold_train"),
                     (gold_val,   "gold_val"),
                     (gold_test,  "gold_test")]:
        profile_dataframe(df, name, profiles_dir)

    print("\n── Export ────────────────────────────────────────────────────")
    gold_train.to_csv(processed_dir / "train_features.csv", index=False)
    gold_val.to_csv(  processed_dir / "val_features.csv",   index=False)
    gold_test.to_csv( processed_dir / "test_features.csv",  index=False)

    con.close()

    print(f"\n✓ Pipeline complete.")
    print(f"  train : {len(gold_train)} rows  |  {gold_train[LABEL_COL].sum()} positive")
    val_pos = gold_val[LABEL_COL].sum() if LABEL_COL in gold_val.columns else "N/A"
    print(f"  val   : {len(gold_val)} rows  |  {val_pos} positive")
    print(f"  test  : {len(gold_test)} rows")
    print(f"  features: {[c for c in gold_train.columns if c not in ('tconst', LABEL_COL)]}")


if __name__ == "__main__":
    build()
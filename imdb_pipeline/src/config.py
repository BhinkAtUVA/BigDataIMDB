from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW         = ROOT / "data" / "raw"
MERGED      = ROOT / "data" / "merged"
CLEANED     = ROOT / "data" / "cleaned"
PROCESSED   = ROOT / "data" / "processed"
PROFILES    = ROOT / "outputs" / "profiles"
PLOTS       = ROOT / "outputs" / "plots"

TRAIN_GLOB       = "train-*.csv"
VAL_FILE         = "validation_hidden.csv"
TEST_FILE        = "test_hidden.csv"
DIRECTING_FILE   = "directing.json"
WRITING_FILE     = "writing.json"

# ── Schema ────────────────────────────────────────────────────────────────────
# Columns we REQUIRE to exist; pipeline raises early if any are absent
REQUIRED_COLS = {"tconst", "primaryTitle", "startYear", "runtimeMinutes", "numVotes"}
LABEL_COL     = "label"

# ── Cleaning thresholds (change here, not in code) ───────────────────────────
YEAR_MIN            = 1880   # earliest plausible film year
YEAR_MAX            = 2030
RUNTIME_MIN         = 1      # minutes
RUNTIME_MAX         = 600
NUMVOTES_MIN        = 0

# ── NULL sentinel values found in the data ───────────────────────────────────
NULL_SENTINELS = {"\\N", "\\\\N", "NA", "N/A", "None", "none", "null", "NULL", "", " "}

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW         = ROOT / "data" / "raw"
EXTERNAL    = ROOT / "data" / "external"
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

# ── External IMDB public datasets (datasets.imdbws.com) ──────────────────────
BASICS_FILE      = "title.basics.tsv"      # genres, titleType, isAdult
RATINGS_FILE     = "title.ratings.tsv"     # averageRating, numVotes (public)

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

# ── Outlier-capping thresholds (EDA-driven) ──────────────────────────────────
RUNTIME_CAP         = 300    # cap runtimeMinutes at 300 (99th-pctl ≈ 200)
N_DIRECTORS_CAP     = 5      # anthology films can have 35; cap at 5
N_WRITERS_CAP       = 10     # cap extreme writer counts

# ── NULL sentinel values found in the data ───────────────────────────────────
NULL_SENTINELS = {"\\N", "\\\\N", "NA", "N/A", "None", "none", "null", "NULL", "", " "}

# ── Top IMDB genres to one-hot encode (covers >95% of movies) ────────────────
TOP_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
    "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport",
    "Thriller", "War", "Western",
]

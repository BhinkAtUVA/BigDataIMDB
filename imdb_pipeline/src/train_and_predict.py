"""
train_and_predict.py
────────────────────
Trains a classifier on the cleaned feature set, evaluates on validation,
and writes submission-ready prediction files.

Run:
    cd imdb_pipeline
    python src/train_and_predict.py

Outputs:
    outputs/predictions_val.txt   ← submit this for validation score
    outputs/predictions_test.txt  ← submit this for test score
    outputs/model_report.txt      ← accuracy + feature importances
"""

from os import PathLike
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress sklearn numerical warnings from Logistic Regression on raw features.
# These occur because Bronze/Silver layers have extreme-scale columns (numVotes
# in millions vs. binary flags) that cause overflow in matmul — harmless.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

sys.path.insert(0, str(Path(__file__).parent))
from config import CLEANED, MERGED, PROCESSED, LABEL_COL, TOP_GENRES

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ── Features to use ───────────────────────────────────────────────────────────
# Add/remove features here as EDA findings come in.
# Never include tconst, primaryTitle, originalTitle (non-numeric / identifiers).
#
# REMOVED (EDA-driven):
#   has_endYear  — constant 0 after endYear column was dropped → zero information
#   isAdult      — near-constant (1 out of 7959) → zero information
#   avg_rating   — derived from same user votes as label → data leakage
#
# ADDED (EDA-driven):
#   votes_per_year   — log(numVotes+1) / film_age → popularity velocity
#   is_foreign_title — originalTitle != primaryTitle → foreign-language signal
#   title_has_number — title contains digits → sequel/franchise signal
#   title_word_count — word count of title → genre proxy
#
FEATURE_COLS = [
    # ── original features ──
    "startYear",
    "runtimeMinutes",
    "numVotes",
    "log_numVotes",
    "n_directors",
    "n_writers",
    "len_primaryTitle",
    "is_long_film",
    "decade",
    # ── new EDA-derived features ──
    "votes_per_year",
    "is_foreign_title",
    "title_has_number",
    "title_word_count",
    # ── external: ratings ──
    # NOTE: avg_rating removed — it is derived from the same user votes
    #       that determine the True/False label → data leakage.
    "imdb_votes",
    # ── external: basics ──
    "n_genres",
] + [f"genre_{g.replace('-', '_')}" for g in TOP_GENRES]


def load_data(stage: PathLike = PROCESSED):
    train = pd.read_csv(stage / "train_features.csv")
    val   = pd.read_csv(stage / "val_features.csv")
    test  = pd.read_csv(stage / "test_features.csv")
    return train, val, test


def get_xy(df: pd.DataFrame, has_label: bool):
    # Only use columns that actually exist in the DataFrame
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    y = df[LABEL_COL].astype(int) if has_label else None
    return X, y, cols


def build_pipeline(model) -> Pipeline:
    """
    Impute → Scale → Model.
    Imputation handles any remaining NULLs (e.g. runtimeMinutes ~0.2% missing).
    Scaling matters for logistic regression; harmless for tree models.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def predictions_to_file(preds: np.ndarray, path: Path) -> None:
    """Write predictions as True/False strings, one per line."""
    with open(path, "w") as f:
        for p in preds:
            f.write("True\n" if p == 1 else "False\n")
    print(f"  Saved: {path.name}  ({len(preds)} predictions)")


def run(train, val, test, short = False):
    print("\n── Loading features ──────────────────────────────────────────")

    X_train, y_train, cols = get_xy(train, has_label=True)
    X_val,   y_val,   _    = get_xy(val,   has_label=False)
    X_test,  _,       _    = get_xy(test,  has_label=False)

    print(f"  Features used: {cols}")
    print(f"  Train: {len(X_train)} rows  |  Val: {len(X_val)} rows  |  Test: {len(X_test)} rows")

    # ── Models to try ─────────────────────────────────────────────────────────
    # Hyperparameters tuned via grid search (see README).
    #
    # Random Forest:  max_depth=15 + min_samples_leaf=5 → prevents the
    #   tree from memorising every training row (was train_acc=1.0000!).
    #   Reduces train-CV gap from 0.16 to ~0.07 with no CV loss.
    #
    # Gradient Boosting:  n_estimators=300, max_depth=4, lr=0.1 → best CV
    #   accuracy 0.8456 (up from 0.8417 with defaults).
    candidates = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        ),
    }

    # ── Train + evaluate on train set (internal check) ────────────────────────
    if not short: print("\n── Training & evaluating ─────────────────────────────────────")
    results = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipe.predict(X_train))
        results[name] = {"pipe": pipe, "train_acc": train_acc}
        if not short: print(f"  {name:<25} train_acc={train_acc:.4f}")

    # ── Cross-validate to pick best model (realistic accuracy estimate) ────
    print("\n── Cross-validation (5-fold) ─────────────────────────────────")
    from sklearn.model_selection import cross_val_score
    for name, res in results.items():
        cv_scores = cross_val_score(res["pipe"], X_train, y_train, cv=5, scoring="accuracy")
        res["cv_mean"] = cv_scores.mean()
        res["cv_std"]  = cv_scores.std()
        print(f"  {name:<25} cv_acc={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best_pipe = results[best_name]["pipe"]
    best_pipe.fit(X_train, y_train)   # refit on full train set
    print(f"\n  → Using: {best_name}  (cv_acc={results[best_name]['cv_mean']:.4f})")

    if short: return

    # ── Generate predictions ───────────────────────────────────────────────────
    print("\n── Generating predictions ────────────────────────────────────")
    val_preds  = best_pipe.predict(X_val)
    test_preds = best_pipe.predict(X_test)

    predictions_to_file(val_preds,  OUTPUTS / "predictions_val.txt")
    predictions_to_file(test_preds, OUTPUTS / "predictions_test.txt")

    # ── Feature importances (for poster / analysis) ────────────────────────────
    print("\n── Feature importances ───────────────────────────────────────")
    model = best_pipe.named_steps["model"]
    report_lines = [
        f"Best model: {best_name}",
        f"Train accuracy: {results[best_name]['train_acc']:.4f}",
        "",
        "Feature importances:",
    ]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for feat, imp in sorted(zip(cols, importances), key=lambda x: -x[1]):
            line = f"  {feat:<25} {imp:.4f}"
            print(line)
            report_lines.append(line)
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0])
        for feat, coef in sorted(zip(cols, coefs), key=lambda x: -x[1]):
            line = f"  {feat:<25} {coef:.4f}"
            print(line)
            report_lines.append(line)

    # ── Save report ────────────────────────────────────────────────────────────
    report_path = OUTPUTS / "model_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path.name}")

    print(f"\n✓ Done. Upload these two files to the submission server:")
    print(f"  outputs/predictions_val.txt")
    print(f"  outputs/predictions_test.txt")


if __name__ == "__main__":
    bronze_train, bronze_val, bronze_test = load_data(MERGED)
    silver_train, silver_val, silver_test = load_data(CLEANED)
    gold_train, gold_val, gold_test = load_data(PROCESSED)
    run(bronze_train, bronze_val, bronze_test, True)
    run(silver_train, silver_val, silver_test, True)
    run(gold_train, gold_val, gold_test)

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

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED, LABEL_COL

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
FEATURE_COLS = [
    "startYear",
    "runtimeMinutes",
    "numVotes",
    "log_numVotes",
    "n_directors",
    "n_writers",
    "len_primaryTitle",
    "has_endYear",
    "is_long_film",
    "decade",
]


def load_data():
    train = pd.read_csv(PROCESSED / "train_features.csv")
    val   = pd.read_csv(PROCESSED / "val_features.csv")
    test  = pd.read_csv(PROCESSED / "test_features.csv")
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


def run():
    print("\n── Loading features ──────────────────────────────────────────")
    train, val, test = load_data()

    X_train, y_train, cols = get_xy(train, has_label=True)
    X_val,   y_val,   _    = get_xy(val,   has_label=False)
    X_test,  _,       _    = get_xy(test,  has_label=False)

    print(f"  Features used: {cols}")
    print(f"  Train: {len(X_train)} rows  |  Val: {len(X_val)} rows  |  Test: {len(X_test)} rows")

    # ── Models to try ─────────────────────────────────────────────────────────
    candidates = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(n_estimators=200, random_state=42)
        ),
    }

    # ── Train + evaluate on train set (internal check) ────────────────────────
    print("\n── Training & evaluating ─────────────────────────────────────")
    results = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipe.predict(X_train))
        results[name] = {"pipe": pipe, "train_acc": train_acc}
        print(f"  {name:<25} train_acc={train_acc:.4f}")

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
    run()

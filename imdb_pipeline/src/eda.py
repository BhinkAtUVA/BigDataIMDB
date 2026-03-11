"""
eda.py
──────
Exploratory Data Analysis for the IMDB pipeline.

Generates a full suite of visualisations and a summary report across
all three pipeline layers (Bronze → Silver → Gold).

Run:
    cd imdb_pipeline
    python src/eda.py

Outputs:
    outputs/plots/eda_*.png          ← individual EDA charts
    outputs/plots/eda_report.md      ← markdown summary of findings
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless / CI

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MERGED, CLEANED, PROCESSED, PLOTS,
    LABEL_COL, TOP_GENRES, YEAR_MIN, YEAR_MAX,
)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook", palette="muted")
FIGSIZE_WIDE   = (14, 5)
FIGSIZE_SQUARE = (7, 5)
FIGSIZE_TALL   = (10, 8)
DPI = 150


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    """Save figure and close it to free memory."""
    path = PLOTS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def _label_str(val: int) -> str:
    """Map 0/1 label to readable string."""
    return "Highly Rated" if val == 1 else "Not Highly Rated"


# ══════════════════════════════════════════════════════════════════════════════
# 1. MISSING VALUES — heatmap across all layers
# ══════════════════════════════════════════════════════════════════════════════

def plot_missing_heatmap(bronze: pd.DataFrame,
                         silver: pd.DataFrame,
                         gold:   pd.DataFrame) -> None:
    """
    Side-by-side heatmaps showing % missing per column at each layer.
    Purpose: demonstrate that each pipeline stage reduces missingness.
    """
    def pct_missing(df, name):
        return (df.isna().sum() / len(df) * 100).rename(name)

    b = pct_missing(bronze, "Bronze")
    s = pct_missing(silver, "Silver")
    g = pct_missing(gold,   "Gold")

    # Combine — only keep columns that appear in at least one layer
    combined = pd.concat([b, s, g], axis=1).fillna(0)
    # Drop columns that are 0% missing everywhere (not informative)
    combined = combined.loc[combined.max(axis=1) > 0]

    if combined.empty:
        print("  ⊘ No missing values to plot — skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(combined))))
    sns.heatmap(combined, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% Missing"})
    ax.set_title("Missing Values Across Pipeline Layers (% of rows)")
    ax.set_xlabel("Pipeline Layer")
    ax.set_ylabel("")
    _save(fig, "eda_missing_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LABEL DISTRIBUTION (train only)
# ══════════════════════════════════════════════════════════════════════════════

def plot_label_distribution(train: pd.DataFrame) -> None:
    """Bar chart of class balance (True / False)."""
    if LABEL_COL not in train.columns:
        return

    counts = train[LABEL_COL].value_counts().sort_index()
    labels = [_label_str(v) for v in counts.index]
    colours = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    bars = ax.bar(labels, counts.values, color=colours, edgecolor="black", width=0.5)
    ax.bar_label(bars, fmt="%d", fontsize=11)
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v / 2, f"{v / total:.1%}", ha="center", va="center",
                fontweight="bold", fontsize=12, color="white")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Class Distribution in Training Set")
    _save(fig, "eda_label_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. NUMERIC FEATURE DISTRIBUTIONS (Gold layer)
# ══════════════════════════════════════════════════════════════════════════════

def plot_numeric_distributions(gold_train: pd.DataFrame) -> None:
    """
    Histogram + KDE for key numeric features, split by label.
    Shows how feature distributions differ between classes.
    """
    features = ["startYear", "runtimeMinutes", "numVotes", "log_numVotes",
                "n_directors", "n_writers", "len_primaryTitle", "avg_rating",
                "n_genres"]
    features = [f for f in features if f in gold_train.columns]

    if LABEL_COL not in gold_train.columns or not features:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for lab, color in [(0, "#e74c3c"), (1, "#2ecc71")]:
            subset = gold_train.loc[gold_train[LABEL_COL] == lab, feat].dropna()
            ax.hist(subset, bins=40, alpha=0.5, color=color,
                    label=_label_str(lab), density=True, edgecolor="none")
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class (Gold Layer — Train)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "eda_numeric_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION MATRIX (Gold layer)
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(gold_train: pd.DataFrame) -> None:
    """Pearson correlation heatmap for all numeric features + label."""
    numeric = gold_train.select_dtypes(include=[np.number])
    # Drop identifier-like or constant columns
    drop = {"tconst"}
    numeric = numeric.drop(columns=[c for c in drop if c in numeric.columns],
                           errors="ignore")
    if numeric.shape[1] < 2:
        return

    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(10, corr.shape[0] * 0.55),
                                    max(8, corr.shape[0] * 0.45)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax,
                annot_kws={"size": 7}, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix (Gold Layer — Train)", fontsize=13)
    _save(fig, "eda_correlation_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BOX PLOTS — numeric features by label
# ══════════════════════════════════════════════════════════════════════════════

def plot_boxplots_by_label(gold_train: pd.DataFrame) -> None:
    """Box plots of key features grouped by label."""
    features = ["startYear", "runtimeMinutes", "log_numVotes",
                "n_directors", "n_writers", "len_primaryTitle",
                "avg_rating", "n_genres"]
    features = [f for f in features if f in gold_train.columns]

    if LABEL_COL not in gold_train.columns or not features:
        return

    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = gold_train[[feat, LABEL_COL]].dropna()
        data["class"] = data[LABEL_COL].map(
            {0: "Not Highly\nRated", 1: "Highly\nRated"})
        sns.boxplot(data=data, x="class", y=feat, hue="class", ax=ax,
                    palette={"Not Highly\nRated": "#e74c3c",
                             "Highly\nRated": "#2ecc71"},
                    width=0.5, legend=False)
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Box Plots by Class (Gold Layer — Train)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "eda_boxplots_by_label.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. YEAR DISTRIBUTION — release year over time, by label
# ══════════════════════════════════════════════════════════════════════════════

def plot_year_distribution(gold_train: pd.DataFrame) -> None:
    """Stacked histogram of startYear by class + line for highly-rated ratio."""
    if LABEL_COL not in gold_train.columns or "startYear" not in gold_train.columns:
        return

    data = gold_train[["startYear", LABEL_COL]].dropna()

    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)

    bins = np.arange(data["startYear"].min() // 10 * 10,
                     data["startYear"].max() + 10, 5)

    for lab, color, lbl in [(0, "#e74c3c", "Not Highly Rated"),
                            (1, "#2ecc71", "Highly Rated")]:
        ax1.hist(data.loc[data[LABEL_COL] == lab, "startYear"],
                 bins=bins, alpha=0.6, color=color, label=lbl, edgecolor="white")

    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper left")
    ax1.set_title("Movies Over Time by Class (Gold Layer — Train)")

    # Overlay: % highly rated per decade
    data["decade"] = (data["startYear"] // 10 * 10).astype(int)
    decade_stats = data.groupby("decade")[LABEL_COL].mean() * 100
    ax2 = ax1.twinx()
    ax2.plot(decade_stats.index + 5, decade_stats.values,
             color="navy", marker="o", linewidth=2, label="% Highly Rated")
    ax2.set_ylabel("% Highly Rated")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(loc="upper right")

    _save(fig, "eda_year_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. RUNTIME vs. VOTES scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_runtime_vs_votes(gold_train: pd.DataFrame) -> None:
    """Scatter: runtimeMinutes vs log_numVotes, coloured by label."""
    needed = {"runtimeMinutes", "log_numVotes", LABEL_COL}
    if not needed.issubset(gold_train.columns):
        return

    data = gold_train[list(needed)].dropna()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for lab, color, marker, lbl in [(0, "#e74c3c", "x", "Not Highly Rated"),
                                    (1, "#2ecc71", "o", "Highly Rated")]:
        sub = data[data[LABEL_COL] == lab]
        ax.scatter(sub["runtimeMinutes"], sub["log_numVotes"],
                   alpha=0.35, c=color, marker=marker, s=15, label=lbl)

    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("log(numVotes + 1)")
    ax.set_title("Runtime vs. Vote Popularity by Class")
    ax.legend()
    _save(fig, "eda_runtime_vs_votes.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. GENRE ANALYSIS — prevalence & class-conditional rates
# ══════════════════════════════════════════════════════════════════════════════

def plot_genre_analysis(gold_train: pd.DataFrame) -> None:
    """
    Two-panel chart:
      Left  — genre prevalence (% of movies)
      Right — % highly rated per genre
    """
    genre_cols = [c for c in gold_train.columns if c.startswith("genre_")]
    if not genre_cols or LABEL_COL not in gold_train.columns:
        return

    stats = []
    for col in genre_cols:
        genre_name = col.replace("genre_", "").replace("_", "-")
        n_movies = gold_train[col].sum()
        if n_movies == 0:
            continue
        pct_movies = n_movies / len(gold_train) * 100
        pct_high = gold_train.loc[gold_train[col] == 1, LABEL_COL].mean() * 100
        stats.append({"genre": genre_name, "pct_movies": pct_movies,
                       "pct_highly_rated": pct_high})

    if not stats:
        return

    df_stats = pd.DataFrame(stats).sort_values("pct_movies", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(df_stats))))

    # Left: prevalence
    ax1.barh(df_stats["genre"], df_stats["pct_movies"], color="steelblue")
    ax1.set_xlabel("% of Movies")
    ax1.set_title("Genre Prevalence")

    # Right: highly-rated rate
    colors = ["#2ecc71" if v >= 50 else "#e74c3c" for v in df_stats["pct_highly_rated"]]
    ax2.barh(df_stats["genre"], df_stats["pct_highly_rated"], color=colors)
    ax2.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("% Highly Rated")
    ax2.set_title("Highly-Rated Rate per Genre")

    fig.suptitle("Genre Analysis (Gold Layer — Train)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "eda_genre_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. DIRECTORS / WRITERS — count distributions
# ══════════════════════════════════════════════════════════════════════════════

def plot_crew_distributions(gold_train: pd.DataFrame) -> None:
    """Bar chart: n_directors and n_writers counts, coloured by label."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ax, col, title in zip(
        axes,
        ["n_directors", "n_writers"],
        ["Number of Directors", "Number of Writers"],
    ):
        if col not in gold_train.columns or LABEL_COL not in gold_train.columns:
            ax.set_visible(False)
            continue

        data = gold_train[[col, LABEL_COL]].dropna()
        # Bin high counts
        data["count_bin"] = data[col].clip(upper=5).astype(int)
        data["count_bin"] = data["count_bin"].replace(
            {5: "5+"}).astype(str)

        ct = pd.crosstab(data["count_bin"], data[LABEL_COL].map(_label_str),
                         normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=["#e74c3c", "#2ecc71"], edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("% of Movies")
        ax.legend(title="", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    fig.suptitle("Crew Size vs. Rating (Gold Layer — Train)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "eda_crew_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. LAYER COMPARISON — accuracy lift from Bronze to Silver to Gold
# ══════════════════════════════════════════════════════════════════════════════

def plot_layer_comparison(bronze: pd.DataFrame,
                          silver: pd.DataFrame,
                          gold:   pd.DataFrame) -> None:
    """
    Compare summary statistics across pipeline layers for key columns.
    Stacked bar: shows how each layer refines the data.
    """
    common_cols = ["startYear", "runtimeMinutes", "numVotes"]

    records = []
    for name, df in [("Bronze", bronze), ("Silver", silver), ("Gold", gold)]:
        for col in common_cols:
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            records.append({
                "layer": name,
                "column": col,
                "pct_non_null": numeric.notna().mean() * 100,
                "mean": numeric.mean(),
                "median": numeric.median(),
                "std": numeric.std(),
            })

    if not records:
        return

    df_stats = pd.DataFrame(records)

    fig, axes = plt.subplots(1, len(common_cols), figsize=(5 * len(common_cols), 5))
    if len(common_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, common_cols):
        subset = df_stats[df_stats["column"] == col]
        x = np.arange(len(subset))
        ax.bar(x, subset["pct_non_null"], width=0.5,
               color=["#3498db", "#f39c12", "#2ecc71"], edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(subset["layer"])
        ax.set_ylabel("% Non-Null")
        ax.set_title(f"{col}")
        ax.set_ylim(0, 110)
        for xi, val in zip(x, subset["pct_non_null"]):
            ax.text(xi, val + 1, f"{val:.1f}%", ha="center", fontsize=10)

    fig.suptitle("Data Completeness Across Pipeline Layers (Train)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "eda_layer_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. PAIR PLOT — top features
# ══════════════════════════════════════════════════════════════════════════════

def plot_pairplot(gold_train: pd.DataFrame) -> None:
    """Seaborn pairplot for the top-5 most informative features."""
    candidates = ["log_numVotes", "startYear", "runtimeMinutes",
                  "n_directors", "avg_rating"]
    cols = [c for c in candidates if c in gold_train.columns]
    if LABEL_COL not in gold_train.columns or len(cols) < 2:
        return

    data = gold_train[cols + [LABEL_COL]].dropna().sample(
        n=min(2000, len(gold_train)), random_state=42)
    data["class"] = data[LABEL_COL].map(_label_str)

    g = sns.pairplot(data, hue="class", vars=cols,
                     palette={"Not Highly Rated": "#e74c3c",
                              "Highly Rated": "#2ecc71"},
                     diag_kind="kde", plot_kws={"alpha": 0.4, "s": 12})
    g.figure.suptitle("Pair Plot of Top Features (Gold Layer — Train)",
                      y=1.01, fontsize=14)
    _save(g.figure, "eda_pairplot.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12. VOTES DISTRIBUTION — log-scale deep-dive
# ══════════════════════════════════════════════════════════════════════════════

def plot_votes_analysis(gold_train: pd.DataFrame) -> None:
    """numVotes on raw and log scale, showing the heavy right-skew."""
    if "numVotes" not in gold_train.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    votes = gold_train["numVotes"].dropna()

    # Raw
    ax1.hist(votes, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("numVotes")
    ax1.set_ylabel("Count")
    ax1.set_title("numVotes (Raw Scale)")
    ax1.axvline(votes.median(), color="red", linestyle="--", label=f"Median={votes.median():.0f}")
    ax1.axvline(votes.mean(), color="orange", linestyle="--", label=f"Mean={votes.mean():.0f}")
    ax1.legend()

    # Log
    log_votes = np.log1p(votes)
    ax2.hist(log_votes, bins=60, color="teal", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("log(numVotes + 1)")
    ax2.set_ylabel("Count")
    ax2.set_title("numVotes (Log Scale)")
    ax2.axvline(log_votes.median(), color="red", linestyle="--",
                label=f"Median={log_votes.median():.2f}")
    ax2.legend()

    fig.suptitle("Vote Count Distribution — Heavy Right-Skew Motivates Log Transform",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "eda_votes_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13. SPLIT COMPARISON — train vs. val vs. test
# ══════════════════════════════════════════════════════════════════════════════

def plot_split_comparison(gold_train: pd.DataFrame,
                          gold_val:   pd.DataFrame,
                          gold_test:  pd.DataFrame) -> None:
    """
    Overlay KDE plots for key features across train / val / test
    to check for distribution shift.
    """
    features = ["startYear", "runtimeMinutes", "log_numVotes", "n_genres"]
    features = [f for f in features if f in gold_train.columns]

    if not features:
        return

    fig, axes = plt.subplots(1, len(features),
                             figsize=(4.5 * len(features), 4))
    if len(features) == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        for df, name, color in [(gold_train, "Train", "#3498db"),
                                (gold_val,   "Val",   "#e67e22"),
                                (gold_test,  "Test",  "#9b59b6")]:
            vals = pd.to_numeric(df[feat], errors="coerce").dropna()
            if len(vals) > 5:
                vals.plot.kde(ax=ax, label=name, color=color, linewidth=1.5)
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Feature Distribution Across Splits — Checking for Shift",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "eda_split_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 14. DECADE ANALYSIS — highly-rated rate per decade
# ══════════════════════════════════════════════════════════════════════════════

def plot_decade_analysis(gold_train: pd.DataFrame) -> None:
    """Bar chart: count of movies per decade + line for % highly rated."""
    if "decade" not in gold_train.columns or LABEL_COL not in gold_train.columns:
        return

    data = gold_train[["decade", LABEL_COL]].dropna()
    data["decade"] = data["decade"].astype(int)
    decade_counts = data.groupby("decade").size()
    decade_rate   = data.groupby("decade")[LABEL_COL].mean() * 100

    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)
    ax1.bar(decade_counts.index, decade_counts.values,
            width=8, color="steelblue", alpha=0.7, label="# Movies")
    ax1.set_xlabel("Decade")
    ax1.set_ylabel("Number of Movies")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(decade_rate.index, decade_rate.values,
             color="crimson", marker="s", linewidth=2, label="% Highly Rated")
    ax2.set_ylabel("% Highly Rated")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(loc="upper right")

    ax1.set_title("Movies per Decade & Highly-Rated Rate (Gold — Train)",
                  fontsize=13)
    fig.tight_layout()
    _save(fig, "eda_decade_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 15. TOP CORRELATED FEATURES WITH LABEL
# ══════════════════════════════════════════════════════════════════════════════

def plot_label_correlations(gold_train: pd.DataFrame) -> None:
    """Horizontal bar chart of Pearson correlation of each feature with label."""
    if LABEL_COL not in gold_train.columns:
        return

    numeric = gold_train.select_dtypes(include=[np.number])
    drop_cols = {"tconst"}
    numeric = numeric.drop(columns=[c for c in drop_cols if c in numeric.columns],
                           errors="ignore")

    corrs = numeric.corr()[LABEL_COL].drop(LABEL_COL, errors="ignore").sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(corrs))))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in corrs.values]
    ax.barh(corrs.index, corrs.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Label")
    ax.set_title("Feature Correlation with Highly-Rated Label (Gold — Train)")
    _save(fig, "eda_label_correlations.png")


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(bronze_train: pd.DataFrame,
                    silver_train: pd.DataFrame,
                    gold_train:   pd.DataFrame,
                    gold_val:     pd.DataFrame,
                    gold_test:    pd.DataFrame) -> None:
    """Write a Markdown summary of key EDA findings."""
    lines = [
        "# EDA Report — IMDB Pipeline",
        "",
        f"**Generated automatically by `src/eda.py`**",
        "",
        "---",
        "",
        "## Dataset Overview",
        "",
        f"| Split | Rows (Bronze) | Rows (Silver) | Rows (Gold) |",
        f"|-------|--------------|--------------|------------|",
    ]

    # We only have train labels, but show row counts for context
    rows_b = len(bronze_train)
    rows_s = len(silver_train)
    rows_g = len(gold_train)
    lines.append(f"| Train | {rows_b:,} | {rows_s:,} | {rows_g:,} |")
    lines.append(f"| Val   | — | — | {len(gold_val):,} |")
    lines.append(f"| Test  | — | — | {len(gold_test):,} |")
    lines.append("")

    # Class balance
    if LABEL_COL in gold_train.columns:
        vc = gold_train[LABEL_COL].value_counts()
        pos = vc.get(1, 0)
        neg = vc.get(0, 0)
        lines += [
            "## Class Balance (Train)",
            "",
            f"| Class | Count | Percentage |",
            f"|-------|------:|----------:|",
            f"| Highly Rated (1) | {pos:,} | {pos / len(gold_train):.1%} |",
            f"| Not Highly Rated (0) | {neg:,} | {neg / len(gold_train):.1%} |",
            "",
        ]

    # Numeric summary
    numeric = gold_train.select_dtypes(include=[np.number])
    drop_cols = {"tconst"}
    numeric = numeric.drop(columns=[c for c in drop_cols if c in numeric.columns],
                           errors="ignore")
    lines += ["## Numeric Feature Summary (Gold — Train)", ""]
    desc = numeric.describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["Mean", "Std", "Min", "Median", "Max"]
    lines.append(desc.to_markdown())
    lines.append("")

    # Missing values at each layer
    lines += ["## Missing Values (% per column, Train)", ""]
    layers = [("Bronze", bronze_train), ("Silver", silver_train), ("Gold", gold_train)]
    miss_records = []
    for name, df in layers:
        for col in df.columns:
            pct = df[col].isna().sum() / len(df) * 100
            if pct > 0:
                miss_records.append({"Layer": name, "Column": col,
                                     "% Missing": f"{pct:.2f}"})
    if miss_records:
        lines.append(pd.DataFrame(miss_records).to_markdown(index=False))
    else:
        lines.append("No missing values detected.")
    lines.append("")

    # Top correlations with label
    if LABEL_COL in gold_train.columns:
        corrs = numeric.corr()[LABEL_COL].drop(
            LABEL_COL, errors="ignore").abs().sort_values(ascending=False).head(10)
        lines += [
            "## Top 10 Features Correlated with Label (absolute Pearson r)",
            "",
        ]
        lines.append(corrs.to_frame("| abs(r) |").to_markdown())
        lines.append("")

    # Plots index
    lines += [
        "## Generated Plots",
        "",
        "| # | File | Description |",
        "|---|------|-------------|",
        "| 1 | `eda_missing_heatmap.png` | Missing values across pipeline layers |",
        "| 2 | `eda_label_distribution.png` | Class balance bar chart |",
        "| 3 | `eda_numeric_distributions.png` | Histograms by class |",
        "| 4 | `eda_correlation_matrix.png` | Feature correlation heatmap |",
        "| 5 | `eda_boxplots_by_label.png` | Box plots by class |",
        "| 6 | `eda_year_distribution.png` | Release year timeline by class |",
        "| 7 | `eda_runtime_vs_votes.png` | Runtime vs. votes scatter |",
        "| 8 | `eda_genre_analysis.png` | Genre prevalence & rating rates |",
        "| 9 | `eda_crew_distributions.png` | Director/writer count analysis |",
        "| 10 | `eda_layer_comparison.png` | Data completeness across layers |",
        "| 11 | `eda_pairplot.png` | Pair plot of top features |",
        "| 12 | `eda_votes_analysis.png` | Vote count distribution (raw + log) |",
        "| 13 | `eda_split_comparison.png` | Train / val / test distribution shift |",
        "| 14 | `eda_decade_analysis.png` | Movies per decade + rating rate |",
        "| 15 | `eda_label_correlations.png` | Feature–label correlation bars |",
        "",
    ]

    report_path = PLOTS / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ eda_report.md")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("── Loading data ──────────────────────────────────────────────")
    bronze_train = pd.read_csv(MERGED    / "train_features.csv")
    silver_train = pd.read_csv(CLEANED   / "train_features.csv")
    gold_train   = pd.read_csv(PROCESSED / "train_features.csv")
    gold_val     = pd.read_csv(PROCESSED / "val_features.csv")
    gold_test    = pd.read_csv(PROCESSED / "test_features.csv")
    print(f"  Bronze train: {len(bronze_train):,} rows")
    print(f"  Silver train: {len(silver_train):,} rows")
    print(f"  Gold   train: {len(gold_train):,} rows  |  val: {len(gold_val):,}  |  test: {len(gold_test):,}")

    print("\n── Generating EDA plots ──────────────────────────────────────")
    plot_missing_heatmap(bronze_train, silver_train, gold_train)
    plot_label_distribution(gold_train)
    plot_numeric_distributions(gold_train)
    plot_correlation_matrix(gold_train)
    plot_boxplots_by_label(gold_train)
    plot_year_distribution(gold_train)
    plot_runtime_vs_votes(gold_train)
    plot_genre_analysis(gold_train)
    plot_crew_distributions(gold_train)
    plot_layer_comparison(bronze_train, silver_train, gold_train)
    plot_pairplot(gold_train)
    plot_votes_analysis(gold_train)
    plot_split_comparison(gold_train, gold_val, gold_test)
    plot_decade_analysis(gold_train)
    plot_label_correlations(gold_train)

    print("\n── Generating report ─────────────────────────────────────────")
    generate_report(bronze_train, silver_train, gold_train, gold_val, gold_test)

    print(f"\n✓ EDA complete. All outputs in: {PLOTS}")


if __name__ == "__main__":
    main()

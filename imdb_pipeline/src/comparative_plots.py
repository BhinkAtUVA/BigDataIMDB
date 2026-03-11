import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import CLEANED, MERGED, PLOTS, PROCESSED
from build_features import normalize_accents

if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    bronze_train = pd.read_csv(MERGED / "train_features.csv")
    silver_train = pd.read_csv(CLEANED / "train_features.csv")
    gold_train = pd.read_csv(PROCESSED / "train_features.csv")

    titles = bronze_train["primaryTitle"]
    sus_titles = titles[titles.map(lambda t: t != normalize_accents(t))].sort_values()
    selection = sus_titles.tail(10).copy()
    pd.DataFrame({
        "dirty": selection,
        "clean": selection.map(lambda t: normalize_accents(t))
    }).to_markdown(PLOTS / "dirty_titles.md")

    start_pct_available = round(bronze_train["startYear"].notna().sum() / bronze_train.shape[0] * 100, 2)
    end_pct_available = round(bronze_train["endYear"].notna().sum() / bronze_train.shape[0] * 100, 2)
    votes_pct_available = round(bronze_train["numVotes"].notna().sum() / bronze_train.shape[0] * 100, 2)

    silver_year_pct_available = silver_train["startYear"].notna().sum() / silver_train.shape[0] * 100
    silver_votes_pct_available = silver_train["numVotes"].notna().sum() / silver_train.shape[0] * 100

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    start_bar = ax[0].bar(["Start Year"], [start_pct_available])
    ax[0].bar_label(start_bar, label_type="edge")
    end_bar = ax[0].bar(["End Year"], [end_pct_available])
    ax[0].bar_label(end_bar, label_type="edge")
    votes_bar = ax[0].bar(["Votes"], [votes_pct_available])
    ax[0].bar_label(votes_bar, label_type="edge")
    ax[0].set_ylim((0, 110))
    ax[0].set_title("Before")
    fixed_bar = ax[1].bar(["Start Year"], [silver_year_pct_available])
    ax[1].bar_label(fixed_bar, label_type="edge")
    ax[1].bar(["Start Year"], [0])
    imputed_bar = ax[1].bar(["Votes"], [silver_votes_pct_available])
    ax[1].bar_label(imputed_bar, label_type="edge")
    ax[1].set_ylim((0, 110))
    ax[1].set_title("After")
    fig.suptitle("Amount of non-missing entries for selected columns before and after data cleaning (%)")
    fig.savefig(PLOTS / "release_missing.svg")


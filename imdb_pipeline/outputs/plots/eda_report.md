# EDA Report — IMDB Pipeline

**Generated automatically by `src/eda.py`**

---

## Dataset Overview

| Split | Rows (Bronze) | Rows (Silver) | Rows (Gold) |
|-------|--------------|--------------|------------|
| Train | 7,959 | 7,959 | 7,959 |
| Val   | — | — | 955 |
| Test  | — | — | 1,086 |

## Class Balance (Train)

| Class | Count | Percentage |
|-------|------:|----------:|
| Highly Rated (1) | 3,990 | 50.1% |
| Not Highly Rated (0) | 3,969 | 49.9% |

## Numeric Feature Summary (Gold — Train)

|                   |            Mean |            Std |        Min |     Median |            Max |
|:------------------|----------------:|---------------:|-----------:|-----------:|---------------:|
| startYear         |  1998.07        |     21.9854    | 1918       | 2006       | 2021           |
| runtimeMinutes    |   105.687       |     25.3963    |   45       |  100       |  551           |
| numVotes          | 26943.6         | 108898         | 1001       | 3559       |    2.50364e+06 |
| label             |     0.501319    |      0.50003   |    0       |    1       |    1           |
| n_directors       |     1.121       |      0.744122  |    0       |    1       |   35           |
| n_writers         |     2.22201     |      1.57356   |    0       |    2       |   29           |
| log_numVotes      |     8.56828     |      1.41877   |    6.90975 |    8.17752 |   14.7333      |
| len_primaryTitle  |    15.6516      |      8.80338   |    1       |   14       |   86           |
| has_endYear       |     0           |      0         |    0       |    0       |    0           |
| decade            |  1993.38        |     22.0393    | 1910       | 2000       | 2020           |
| avg_rating        |     6.1875      |      1.46953   |    0       |    6       |    9.3         |
| imdb_votes        | 36098.8         | 142029         |    0       | 4421       |    3.16603e+06 |
| isAdult           |     0.000125644 |      0.0112091 |    0       |    0       |    1           |
| n_genres          |     2.34364     |      0.75972   |    0       |    3       |    3           |
| genre_Action      |     0.18055     |      0.384669  |    0       |    0       |    1           |
| genre_Adventure   |     0.109938    |      0.312833  |    0       |    0       |    1           |
| genre_Animation   |     0.031034    |      0.173421  |    0       |    0       |    1           |
| genre_Biography   |     0.0542782   |      0.22658   |    0       |    0       |    1           |
| genre_Comedy      |     0.334087    |      0.4717    |    0       |    0       |    1           |
| genre_Crime       |     0.158688    |      0.365408  |    0       |    0       |    1           |
| genre_Documentary |     0.064204    |      0.245131  |    0       |    0       |    1           |
| genre_Drama       |     0.526071    |      0.499351  |    0       |    1       |    1           |
| genre_Family      |     0.0474934   |      0.212705  |    0       |    0       |    1           |
| genre_Fantasy     |     0.0601834   |      0.237841  |    0       |    0       |    1           |
| genre_History     |     0.0417138   |      0.199947  |    0       |    0       |    1           |
| genre_Horror      |     0.154291    |      0.36125   |    0       |    0       |    1           |
| genre_Music       |     0.0530217   |      0.224091  |    0       |    0       |    1           |
| genre_Musical     |     0.0202287   |      0.14079   |    0       |    0       |    1           |
| genre_Mystery     |     0.0843071   |      0.277865  |    0       |    0       |    1           |
| genre_Romance     |     0.160699    |      0.367276  |    0       |    0       |    1           |
| genre_Sci_Fi      |     0.0594296   |      0.236442  |    0       |    0       |    1           |
| genre_Sport       |     0.0219877   |      0.146652  |    0       |    0       |    1           |
| genre_Thriller    |     0.149642    |      0.356742  |    0       |    0       |    1           |
| genre_War         |     0.0282699   |      0.165753  |    0       |    0       |    1           |
| genre_Western     |     0.0123131   |      0.110286  |    0       |    0       |    1           |

## Missing Values (% per column, Train)

| Layer   | Column         |   % Missing |
|:--------|:---------------|------------:|
| Bronze  | originalTitle  |       50.11 |
| Bronze  | startYear      |        9.88 |
| Bronze  | endYear        |       90.12 |
| Bronze  | runtimeMinutes |        0.16 |
| Bronze  | numVotes       |        9.93 |
| Silver  | originalTitle  |       50.11 |
| Silver  | runtimeMinutes |        0.16 |
| Gold    | originalTitle  |       50.11 |
| Gold    | runtimeMinutes |        0.16 |

## Top 10 Features Correlated with Label (absolute Pearson r)

|                   |   | abs(r) | |
|:------------------|-------------:|
| avg_rating        |     0.864148 |
| genre_Horror      |     0.350345 |
| runtimeMinutes    |     0.302308 |
| genre_Drama       |     0.297416 |
| startYear         |     0.263745 |
| decade            |     0.260896 |
| log_numVotes      |     0.234181 |
| genre_Documentary |     0.232538 |
| genre_Biography   |     0.181264 |
| genre_Thriller    |     0.174047 |

## Generated Plots

| # | File | Description |
|---|------|-------------|
| 1 | `eda_missing_heatmap.png` | Missing values across pipeline layers |
| 2 | `eda_label_distribution.png` | Class balance bar chart |
| 3 | `eda_numeric_distributions.png` | Histograms by class |
| 4 | `eda_correlation_matrix.png` | Feature correlation heatmap |
| 5 | `eda_boxplots_by_label.png` | Box plots by class |
| 6 | `eda_year_distribution.png` | Release year timeline by class |
| 7 | `eda_runtime_vs_votes.png` | Runtime vs. votes scatter |
| 8 | `eda_genre_analysis.png` | Genre prevalence & rating rates |
| 9 | `eda_crew_distributions.png` | Director/writer count analysis |
| 10 | `eda_layer_comparison.png` | Data completeness across layers |
| 11 | `eda_pairplot.png` | Pair plot of top features |
| 12 | `eda_votes_analysis.png` | Vote count distribution (raw + log) |
| 13 | `eda_split_comparison.png` | Train / val / test distribution shift |
| 14 | `eda_decade_analysis.png` | Movies per decade + rating rate |
| 15 | `eda_label_correlations.png` | Feature–label correlation bars |

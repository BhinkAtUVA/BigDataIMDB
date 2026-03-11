from __future__ import annotations

"""
json_to_relations.py
────────────────────
Parses directing.json and writing.json into clean 2-column DataFrames
(tconst, nconst).  Robust to both known shapes and future shape changes.
"""

import json
import pandas as pd
from pathlib import Path
from config import NULL_SENTINELS


def _clean_relation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shared post-processing for any relation table."""
    df = df.copy()
    df.columns = ["tconst", "nconst"]
    df = df.astype(str)
    # Replace all null sentinels
    df = df.replace(list(NULL_SENTINELS), pd.NA)
    df = df.dropna(subset=["tconst", "nconst"])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def load_directing(path: str | Path) -> pd.DataFrame:
    """
    Handles two known shapes:
      • dict-of-columns: {"movie": {idx: tconst}, "director": {idx: nconst}}
      • list-of-records:  [{"movie": tconst, "director": nconst}, ...]
    Raises ValueError with a clear message for any unrecognised shape.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        # dict-of-columns (pandas orient="dict")
        col_map = {"movie": None, "director": None}
        for key in obj:
            k = key.lower().strip()
            if k in ("movie", "tconst"):
                col_map["movie"] = key
            elif k in ("director", "director_id", "nconst"):
                col_map["director"] = key
        if None in col_map.values():
            raise ValueError(
                f"directing.json dict shape unrecognised. Keys found: {list(obj.keys())}"
            )
        movie   = pd.Series(obj[col_map["movie"]],   name="tconst")
        nconst  = pd.Series(obj[col_map["director"]], name="nconst")
        df = pd.concat([movie, nconst], axis=1).reset_index(drop=True)

    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
        # Flexible column rename
        rename = {}
        for col in df.columns:
            c = col.lower().strip()
            if c in ("movie", "tconst"):
                rename[col] = "tconst"
            elif c in ("director", "director_id", "nconst"):
                rename[col] = "nconst"
        df = df.rename(columns=rename)

    else:
        raise ValueError(f"directing.json: unexpected top-level type {type(obj)}")

    return _clean_relation_df(df[["tconst", "nconst"]])


def load_writing(path: str | Path) -> pd.DataFrame:
    """
    Handles two known shapes:
      • list-of-records:  [{"movie": tconst, "writer": nconst}, ...]
      • dict-of-columns:  {"movie": {idx: tconst}, "writer": {idx: nconst}}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        df = pd.DataFrame(obj)
        rename = {}
        for col in df.columns:
            c = col.lower().strip()
            if c in ("movie", "tconst"):
                rename[col] = "tconst"
            elif c in ("writer", "writer_id", "nconst"):
                rename[col] = "nconst"
        df = df.rename(columns=rename)

    elif isinstance(obj, dict):
        col_map = {"movie": None, "writer": None}
        for key in obj:
            k = key.lower().strip()
            if k in ("movie", "tconst"):
                col_map["movie"] = key
            elif k in ("writer", "writer_id", "nconst"):
                col_map["writer"] = key
        if None in col_map.values():
            raise ValueError(
                f"writing.json dict shape unrecognised. Keys found: {list(obj.keys())}"
            )
        movie  = pd.Series(obj[col_map["movie"]],  name="tconst")
        nconst = pd.Series(obj[col_map["writer"]], name="nconst")
        df = pd.concat([movie, nconst], axis=1).reset_index(drop=True)

    else:
        raise ValueError(f"writing.json: unexpected top-level type {type(obj)}")

    return _clean_relation_df(df[["tconst", "nconst"]])

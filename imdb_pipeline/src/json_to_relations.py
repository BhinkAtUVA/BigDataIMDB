import json
import pandas as pd
from pathlib import Path
from config import NULL_SENTINELS


def _clean_relation_df(df: pd.DataFrame) -> pd.DataFrame:
    # Shared post-processing for any relation table. 
    df = df.copy()
    df.columns = ["tconst", "nconst"]
    df = df.astype(str)
    df = df.replace(list(NULL_SENTINELS), pd.NA)
    df = df.dropna(subset=["tconst", "nconst"])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def load_directing(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
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

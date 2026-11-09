"""Load the official UCI splits and apply the same cleaning rules as the course notebook."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

LABELS = ["Negative", "Neutral", "Positive"]
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
RESULTS = ROOT / "results"

_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]+")
_SPACE = re.compile(r"\s+")


def clean_text(value) -> str:
    text = html.unescape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))
    return text.strip().strip('"').strip()


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["review"] = frame["review"].map(clean_text)
    frame["condition"] = frame["condition"].map(
        lambda value: clean_text(value) if isinstance(value, str) or pd.notna(value) else ""
    )
    frame.loc[frame["condition"].str.len() == 0, "condition"] = pd.NA
    frame = frame[frame["review"].notna() & (frame["review"].str.len() >= 10)]
    frame = frame[frame["condition"].notna() & ~frame["condition"].str.contains(_TAG, na=False)]
    frame = frame.drop_duplicates("review").drop_duplicates("uniqueID")
    frame = frame[frame["rating"].between(1, 10)]
    frame["review_norm"] = (
        frame["review"].str.lower().str.replace(_NON_ALPHA, " ", regex=True).str.replace(_SPACE, " ", regex=True).str.strip()
    )
    frame = frame[frame["review_norm"].str.len() >= 10]
    frame["sentiment"] = pd.cut(
        frame["rating"],
        bins=[0, 4, 6, 10],
        labels=LABELS,
    )
    frame["label"] = frame["sentiment"].map({name: index for index, name in enumerate(LABELS)}).astype(int)
    frame["review_len"] = frame["review"].str.len().astype(float)
    frame["usefulCount"] = frame["usefulCount"].fillna(0).astype(float)
    return frame.reset_index(drop=True)


def load_official() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_split(DATASET / "drugsComTrain_raw.csv")
    test = load_split(DATASET / "drugsComTest_raw.csv")
    return train, test

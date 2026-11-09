"""Sentence-level effectiveness vs side-effect cues, scored with VADER.

The UCI file has no aspect labels. This does not invent supervised aspect tags.
It splits each review into sentences, tags a sentence when it mentions benefit or
a side effect, and records the polarity of those sentences as extra features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_EFFECT = re.compile(
    r"\b(works?|worked|working|effective|effectiveness|relief|relieved|cures?|cured|helps?|helped|helping|improves?|improved|improvement)\b",
    re.I,
)
_SIDE = re.compile(
    r"\b(side effects?|nausea|nauseous|nauseated|dizz(?:y|iness)|headaches?|migraines?|insomnia|rashes?|fatigue|drowsy|drowsiness|vomit(?:ing)?|weight gain|weight loss|bleeding|bleed|spotting|diarrhea|constipation)\b",
    re.I,
)
_SENT = re.compile(r"[.!?]+")

ASPECT_COLUMNS = [
    "vader",
    "vader_effect",
    "vader_side",
    "vader_other",
    "n_effect",
    "n_side",
    "n_sents",
    "aspect_gap",
]


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENT.split(text) if part and len(part.strip()) > 2]
    return parts or [text]


def aspect_features(reviews: pd.Series) -> pd.DataFrame:
    scorer = SentimentIntensityAnalyzer()
    rows: list[dict[str, float]] = []
    total = len(reviews)
    for index, text in enumerate(reviews.fillna(""), start=1):
        sentences = _sentences(text)
        effect: list[str] = []
        side: list[str] = []
        other: list[str] = []
        for sentence in sentences:
            if _SIDE.search(sentence):
                side.append(sentence)
            elif _EFFECT.search(sentence):
                effect.append(sentence)
            else:
                other.append(sentence)

        def mean_compound(group: list[str]) -> float:
            if not group:
                return 0.0
            return float(np.mean([scorer.polarity_scores(sentence)["compound"] for sentence in group]))

        effect_score = mean_compound(effect)
        side_score = mean_compound(side)
        rows.append(
            {
                "vader": scorer.polarity_scores(text)["compound"] if text else 0.0,
                "vader_effect": effect_score,
                "vader_side": side_score,
                "vader_other": mean_compound(other),
                "n_effect": float(len(effect)),
                "n_side": float(len(side)),
                "n_sents": float(len(sentences)),
                "aspect_gap": effect_score - side_score,
            }
        )
        if index % 20000 == 0 or index == total:
            print(f"  aspect features {index}/{total}", flush=True)
    return pd.DataFrame(rows, columns=ASPECT_COLUMNS)

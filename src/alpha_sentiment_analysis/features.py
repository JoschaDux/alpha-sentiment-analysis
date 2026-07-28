from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from alpha_sentiment_analysis.text_processing import tokenize


NEGATIVE_WORDS = {
    "adverse",
    "decline",
    "decrease",
    "default",
    "deteriorate",
    "difficult",
    "failure",
    "loss",
    "losses",
    "negative",
    "risk",
    "risks",
    "uncertain",
    "weakness",
}

POSITIVE_WORDS = {
    "benefit",
    "efficient",
    "favorable",
    "gain",
    "gains",
    "growth",
    "improve",
    "improved",
    "opportunity",
    "positive",
    "profit",
    "strong",
}

UNCERTAINTY_WORDS = {
    "approximately",
    "contingent",
    "depend",
    "depends",
    "fluctuate",
    "may",
    "might",
    "possible",
    "uncertain",
    "uncertainty",
    "variable",
    "whether",
}

LITIGIOUS_WORDS = {
    "claim",
    "claims",
    "complaint",
    "court",
    "legal",
    "litigation",
    "plaintiff",
    "regulatory",
    "settlement",
    "sue",
}


def _share(counter: Counter[str], vocabulary: set[str], total: int) -> float:
    if total == 0:
        return 0.0
    return sum(counter[word] for word in vocabulary) / total


def text_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    counter = Counter(tokens)
    total = len(tokens)
    unique = len(counter)

    sentence_count = max(text.count(".") + text.count("?") + text.count("!"), 1)
    avg_word_length = sum(len(token) for token in tokens) / total if total else 0.0

    return {
        "word_count": float(total),
        "unique_word_count": float(unique),
        "type_token_ratio": unique / total if total else 0.0,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": total / sentence_count if total else 0.0,
        "log_word_count": math.log1p(total),
        "negative_share": _share(counter, NEGATIVE_WORDS, total),
        "positive_share": _share(counter, POSITIVE_WORDS, total),
        "uncertainty_share": _share(counter, UNCERTAINTY_WORDS, total),
        "litigious_share": _share(counter, LITIGIOUS_WORDS, total),
    }


def add_change_features(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["ticker", "filing_date"]).copy()
    feature_cols = [
        "negative_share",
        "positive_share",
        "uncertainty_share",
        "litigious_share",
        "log_word_count",
    ]
    for column in feature_cols:
        ordered[f"{column}_yoy_change"] = ordered.groupby("ticker")[column].diff()
    return ordered

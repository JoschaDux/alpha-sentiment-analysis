from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "word_count",
    "unique_word_count",
    "type_token_ratio",
    "avg_word_length",
    "avg_sentence_length",
    "log_word_count",
    "negative_share",
    "positive_share",
    "uncertainty_share",
    "litigious_share",
    "negative_share_yoy_change",
    "positive_share_yoy_change",
    "uncertainty_share_yoy_change",
    "litigious_share_yoy_change",
    "log_word_count_yoy_change",
]


def time_split(frame: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["filing_year"] = pd.to_datetime(data["filing_date"]).dt.year
    train = data.loc[data["filing_year"] < test_year]
    test = data.loc[data["filing_year"] >= test_year]
    return train, test


def train_baseline_classifier(frame: pd.DataFrame, test_year: int = 2022) -> dict[str, object]:
    data = frame.dropna(subset=["outperformed"]).copy()
    train, test = time_split(data, test_year=test_year)
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")

    features = [column for column in FEATURE_COLUMNS if column in data.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    model.fit(train[features], train["outperformed"].astype(int))
    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": model,
        "features": features,
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy_score(test["outperformed"].astype(int), predictions),
        "roc_auc": roc_auc_score(test["outperformed"].astype(int), probabilities),
    }

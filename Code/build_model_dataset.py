from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MERGE_KEYS = ["ticker", "accession_number", "filing_date"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge 10-K text features with return labels into a modeling dataset."
    )
    parser.add_argument(
        "--text-features",
        default="data/processed/text_features.csv",
        help="CSV produced by Code/build_text_features.py.",
    )
    parser.add_argument(
        "--return-labels",
        default="data/processed/return_labels.csv",
        help="CSV produced by Code/build_return_labels.py.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/model_dataset.csv",
        help="Output CSV for the final modeling dataset.",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep rows with failed text-feature extraction or failed return labels.",
    )
    return parser.parse_args()


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper()
    data["accession_number"] = data["accession_number"].astype(str)
    data["filing_date"] = pd.to_datetime(data["filing_date"]).dt.date.astype(str)
    return data


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    text_path = Path(args.text_features)
    labels_path = Path(args.return_labels)

    if not text_path.exists():
        raise FileNotFoundError(
            f"Missing text features file: {text_path}. "
            "Run Code/build_text_features.py first."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing return labels file: {labels_path}. "
            "Run Code/build_return_labels.py first."
        )

    text_features = normalize_keys(pd.read_csv(text_path))
    return_labels = normalize_keys(pd.read_csv(labels_path))

    require_columns(text_features, MERGE_KEYS, "text features")
    require_columns(return_labels, MERGE_KEYS, "return labels")

    if not args.keep_failed:
        if "feature_status" in text_features.columns:
            text_features = text_features.loc[text_features["feature_status"] == "ok"]
        if "label_status" in return_labels.columns:
            return_labels = return_labels.loc[return_labels["label_status"] == "ok"]

    dataset = text_features.merge(
        return_labels,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_label"),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Text-feature rows: {len(text_features)}")
    print(f"Return-label rows: {len(return_labels)}")
    print(f"Final model rows: {len(dataset)}")
    print(f"Saved model dataset to {output_path}")


if __name__ == "__main__":
    main()

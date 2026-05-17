from __future__ import annotations

import argparse

import pandas as pd

from alpha_sentiment_analysis.modeling import train_baseline_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline return-direction classifier.")
    parser.add_argument("--dataset", default="data/processed/model_dataset.csv")
    parser.add_argument("--test-year", type=int, default=2022)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.dataset)
    result = train_baseline_classifier(dataset, test_year=args.test_year)
    print("Baseline logistic regression")
    print(f"Features: {len(result['features'])}")
    print(f"Train rows: {result['train_rows']}")
    print(f"Test rows: {result['test_rows']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"ROC-AUC: {result['roc_auc']:.3f}")


if __name__ == "__main__":
    main()

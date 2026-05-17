from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
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
    "word_count_yoy_change",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models on the 10-K modeling dataset.")
    parser.add_argument("--dataset", default="data/processed/model_dataset.csv")
    parser.add_argument("--output-dir", default="reports/model_results")
    parser.add_argument("--test-year", type=int, default=2023)
    parser.add_argument("--horizon", type=int, default=252)
    return parser.parse_args()


def prepare_data(dataset_path: str, horizon: int) -> tuple[pd.DataFrame, dict[str, str]]:
    data = pd.read_csv(dataset_path)
    data["filing_year"] = pd.to_datetime(data["filing_date"]).dt.year

    targets = {
        "outperformance": f"outperformed_{horizon}d",
        "excess_return": f"excess_return_{horizon}d",
        "realized_volatility": f"realized_volatility_{horizon}d",
    }
    missing_targets = [target for target in targets.values() if target not in data.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    available_features = [feature for feature in FEATURE_COLUMNS if feature in data.columns]
    if not available_features:
        raise ValueError("No expected text feature columns found.")

    return data, targets


def time_split(data: pd.DataFrame, target: str, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = data.dropna(subset=[target]).copy()
    train = usable.loc[usable["filing_year"] < test_year].copy()
    test = usable.loc[usable["filing_year"] >= test_year].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty train/test split for target {target}.")
    return train, test


def linear_preprocessor_model(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def tree_model(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def train_classification(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_year: int,
) -> tuple[list[dict], pd.DataFrame]:
    train, test = time_split(data, target, test_year)
    x_train, y_train = train[features], train[target].astype(int)
    x_test, y_test = test[features], test[target].astype(int)

    models = {
        "logistic_regression": linear_preprocessor_model(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "random_forest_classifier": tree_model(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        ),
    }

    majority_prediction = int(y_train.mean() >= 0.5)
    baseline_accuracy = accuracy_score(y_test, np.repeat(majority_prediction, len(y_test)))

    results = []
    predictions = test[["ticker", "accession_number", "filing_date", target]].copy()
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predicted = (probabilities >= 0.5).astype(int)

        results.append(
            {
                "task": "classification",
                "target": target,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_year": test_year,
                "metric_primary": "roc_auc",
                "roc_auc": roc_auc_score(y_test, probabilities),
                "accuracy": accuracy_score(y_test, predicted),
                "baseline_accuracy": baseline_accuracy,
            }
        )
        predictions[f"{model_name}_probability"] = probabilities
        predictions[f"{model_name}_prediction"] = predicted

    return results, predictions


def spearman_corr(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def train_regression(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_year: int,
    task_name: str,
) -> tuple[list[dict], pd.DataFrame]:
    train, test = time_split(data, target, test_year)
    x_train, y_train = train[features], train[target].astype(float)
    x_test, y_test = test[features], test[target].astype(float)

    models = {
        "ridge_regression": linear_preprocessor_model(Ridge(alpha=1.0)),
        "random_forest_regressor": tree_model(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    baseline_prediction = np.repeat(float(y_train.mean()), len(y_test))
    baseline_mae = mean_absolute_error(y_test, baseline_prediction)

    results = []
    predictions = test[["ticker", "accession_number", "filing_date", target]].copy()
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rmse = mean_squared_error(y_test, predicted) ** 0.5

        results.append(
            {
                "task": task_name,
                "target": target,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_year": test_year,
                "metric_primary": "mae",
                "mae": mean_absolute_error(y_test, predicted),
                "rmse": rmse,
                "r2": r2_score(y_test, predicted),
                "spearman": spearman_corr(y_test, predicted),
                "baseline_mae": baseline_mae,
            }
        )
        predictions[f"{model_name}_prediction"] = predicted

    return results, predictions


def save_feature_correlations(data: pd.DataFrame, features: list[str], targets: dict[str, str], output_dir: Path) -> None:
    rows = []
    for target_name, target in targets.items():
        for feature in features:
            sample = data[[feature, target]].dropna()
            if len(sample) < 10:
                continue
            rows.append(
                {
                    "target_name": target_name,
                    "target": target,
                    "feature": feature,
                    "pearson": sample[feature].corr(sample[target], method="pearson"),
                    "spearman": sample[feature].corr(sample[target], method="spearman"),
                    "rows": len(sample),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "feature_target_correlations.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, targets = prepare_data(args.dataset, args.horizon)
    features = [feature for feature in FEATURE_COLUMNS if feature in data.columns]

    results = []

    classification_results, classification_predictions = train_classification(
        data=data,
        features=features,
        target=targets["outperformance"],
        test_year=args.test_year,
    )
    results.extend(classification_results)
    classification_predictions.to_csv(output_dir / "classification_predictions.csv", index=False)

    excess_results, excess_predictions = train_regression(
        data=data,
        features=features,
        target=targets["excess_return"],
        test_year=args.test_year,
        task_name="excess_return_regression",
    )
    results.extend(excess_results)
    excess_predictions.to_csv(output_dir / "excess_return_predictions.csv", index=False)

    volatility_results, volatility_predictions = train_regression(
        data=data,
        features=features,
        target=targets["realized_volatility"],
        test_year=args.test_year,
        task_name="volatility_regression",
    )
    results.extend(volatility_results)
    volatility_predictions.to_csv(output_dir / "volatility_predictions.csv", index=False)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_dir / "baseline_model_results.csv", index=False)
    save_feature_correlations(data, features, targets, output_dir)

    print(results_frame.to_string(index=False))
    print(f"\nSaved model results to {output_dir}")


if __name__ == "__main__":
    main()

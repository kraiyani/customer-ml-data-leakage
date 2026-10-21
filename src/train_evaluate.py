"""Fixed-model comparisons with explicit holdout and provenance audits."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import build_features, validate_feature_times
from src.generate_data import ExperimentConfig, generate_data
from src.validation_strategies import (
    chronological_split,
    customer_grouped_split,
    random_row_split,
)

BASE_COLUMNS = ["account_age_months", "activity_score", "payment_load"]
HISTORY_COLUMNS = ["historical_missed_rate", "history_count"]


def make_model(numeric_columns: Sequence[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(numeric_columns),
            ),
            ("customer", OneHotEncoder(handle_unknown="ignore"), ["customer_id"]),
        ]
    )
    return make_pipeline(
        preprocess, LogisticRegression(C=10.0, max_iter=2000, tol=1e-8)
    )


def run_experiment(
    features: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit five predeclared cases; never tune against these holdouts."""
    random_train, random_test = random_row_split(features, config.seed)
    grouped_train, grouped_test = customer_grouped_split(features, config.seed)
    time_train, time_test = chronological_split(features)
    cases = [
        ("random_rows", random_train, random_test, BASE_COLUMNS),
        ("customer_grouped", grouped_train, grouped_test, BASE_COLUMNS),
        ("chronological", time_train, time_test, BASE_COLUMNS),
        (
            "future_feature",
            time_train.assign(future_missed_payment=time_train["missed_payment"]),
            time_test.assign(future_missed_payment=time_test["missed_payment"]),
            [*BASE_COLUMNS, "future_missed_payment"],
        ),
        ("point_in_time", time_train, time_test, [*BASE_COLUMNS, *HISTORY_COLUMNS]),
    ]
    metrics = []
    predictions = []
    for name, train, test, numeric_columns in cases:
        validate_feature_times(train)
        validate_feature_times(test)
        future_feature_row_count = 0
        if name == "future_feature":
            negative_control = test.assign(
                feature_source_max_time=test["label_available_time"],
                feature_available_time=test["label_available_time"],
            )
            try:
                validate_feature_times(negative_control)
            except ValueError:
                future_feature_row_count = int(
                    (test["label_available_time"] > test["snapshot_time"]).sum()
                )
            else:
                raise ValueError("The intentionally leaked control escaped the audit")
        model = make_model(numeric_columns)
        model.fit(train, train["missed_payment"])
        probabilities = model.predict_proba(test)[:, 1]
        cutoff_time = test["snapshot_time"].min()
        metrics.append(
            {
                "experiment": name,
                "roc_auc": float(roc_auc_score(test["missed_payment"], probabilities)),
                "average_precision": float(
                    average_precision_score(test["missed_payment"], probabilities)
                ),
                "accuracy": float(
                    accuracy_score(test["missed_payment"], probabilities >= 0.5)
                ),
                "train_row_count": len(train),
                "test_row_count": len(test),
                "customer_overlap_count": len(
                    set(train["customer_id"]) & set(test["customer_id"])
                ),
                "future_feature_row_count": future_feature_row_count,
                "training_labels_after_test_start_count": int(
                    (train["label_available_time"] > cutoff_time).sum()
                ),
                "test_positive_rate": float(test["missed_payment"].mean()),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "experiment": name,
                    "row_id": test["row_id"].to_numpy(),
                    "customer_id": test["customer_id"].to_numpy(),
                    "snapshot_time": test["snapshot_time"].to_numpy(),
                    "target": test["missed_payment"].to_numpy(),
                    "probability": probabilities,
                }
            )
        )
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def main() -> None:
    from src.reporting import save_artifacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=300)
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        customer_count=args.customers, period_count=args.periods, seed=args.seed
    )
    snapshots, events = generate_data(config)
    features = build_features(snapshots, events)
    metrics, predictions = run_experiment(features, config)
    save_artifacts(
        args.output, config, snapshots, events, features, metrics, predictions
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()

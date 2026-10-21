import pandas as pd

from src.feature_engineering import build_features
from src.generate_data import ExperimentConfig, generate_data
from src.train_evaluate import run_experiment


def test_paired_leaked_and_corrected_cases_use_identical_holdout_rows():
    config = ExperimentConfig(customer_count=100, period_count=12)
    snapshots, events = generate_data(config)
    features = build_features(snapshots, events)
    metrics, predictions = run_experiment(features, config)
    assert len(metrics) == 5
    leaked = predictions.loc[predictions["experiment"] == "future_feature"]
    corrected = predictions.loc[predictions["experiment"] == "point_in_time"]
    pd.testing.assert_frame_equal(
        leaked[["row_id", "target"]].reset_index(drop=True),
        corrected[["row_id", "target"]].reset_index(drop=True),
    )
    assert (
        metrics.loc[
            metrics["experiment"] == "customer_grouped", "customer_overlap_count"
        ].item()
        == 0
    )
    assert (
        metrics.loc[
            metrics["experiment"] == "future_feature", "future_feature_row_count"
        ].item()
        > 0
    )
    assert (
        metrics.loc[
            metrics["experiment"] == "point_in_time", "future_feature_row_count"
        ].item()
        == 0
    )
    assert metrics["roc_auc"].between(0, 1).all()

import pandas as pd
import pytest

from src.feature_engineering import build_features, validate_feature_times
from src.generate_data import ExperimentConfig, generate_data
from src.validation_strategies import chronological_split, validate_label_maturity


def test_late_arrivals_and_future_events_do_not_enter_history():
    cutoff_time = pd.Timestamp("2024-03-01", tz="UTC")
    snapshots = pd.DataFrame(
        {
            "customer_id": ["A"],
            "snapshot_time": [cutoff_time],
            "baseline_available_time": [cutoff_time - pd.Timedelta(days=1)],
        }
    )
    events = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A"],
            "event_time": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-04-01"], utc=True
            ),
            "available_time": pd.to_datetime(
                ["2024-01-02", "2024-03-02", "2024-04-02"], utc=True
            ),
            "missed_payment": [0, 1, 1],
        }
    )
    features = build_features(snapshots, events)
    assert features.loc[0, "historical_missed_rate"] == 0
    assert features.loc[0, "history_count"] == 1
    validate_feature_times(features)


def test_information_available_exactly_at_cutoff_is_included():
    cutoff_time = pd.Timestamp("2024-03-01", tz="UTC")
    snapshots = pd.DataFrame(
        {
            "customer_id": ["A"],
            "snapshot_time": [cutoff_time],
            "baseline_available_time": [cutoff_time],
        }
    )
    events = pd.DataFrame(
        {
            "customer_id": ["A"],
            "event_time": [cutoff_time],
            "available_time": [cutoff_time],
            "missed_payment": [1],
        }
    )
    features = build_features(snapshots, events)
    assert features.loc[0, "historical_missed_rate"] == 1


def test_future_feature_provenance_is_rejected():
    cutoff_time = pd.Timestamp("2024-03-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "snapshot_time": [cutoff_time],
            "feature_source_max_time": [cutoff_time],
            "feature_available_time": [cutoff_time + pd.Timedelta(days=1)],
        }
    )
    with pytest.raises(ValueError, match="available"):
        validate_feature_times(frame)


def test_missing_prediction_time_cannot_pass_the_temporal_audit():
    known_time = pd.Timestamp("2024-03-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "snapshot_time": [pd.NaT],
            "feature_source_max_time": [known_time],
            "feature_available_time": [known_time],
        }
    )
    with pytest.raises(ValueError, match="prediction"):
        validate_feature_times(frame)


def test_valid_history_does_not_mask_missing_baseline_provenance():
    cutoff_time = pd.Timestamp("2024-03-01", tz="UTC")
    snapshots = pd.DataFrame(
        {
            "customer_id": ["A"],
            "snapshot_time": [cutoff_time],
            "baseline_available_time": pd.Series([pd.NaT], dtype="datetime64[ns, UTC]"),
        }
    )
    events = pd.DataFrame(
        {
            "customer_id": ["A"],
            "event_time": [cutoff_time - pd.Timedelta(days=2)],
            "available_time": [cutoff_time - pd.Timedelta(days=1)],
            "missed_payment": [1],
        }
    )
    with pytest.raises(ValueError, match="baseline"):
        build_features(snapshots, events)


def test_chronological_training_labels_are_mature_before_test_starts():
    snapshots, events = generate_data(
        ExperimentConfig(customer_count=80, period_count=12)
    )
    features = build_features(snapshots, events)
    train, test = chronological_split(features)
    cutoff_time = test["snapshot_time"].min()
    assert train["snapshot_time"].max() < cutoff_time
    assert train["label_available_time"].max() <= cutoff_time
    validate_label_maturity(train, cutoff_time)
    invalid = train.copy()
    invalid.loc[invalid.index[0], "label_available_time"] = cutoff_time + pd.Timedelta(
        days=1
    )
    with pytest.raises(ValueError, match="label"):
        validate_label_maturity(invalid, cutoff_time)

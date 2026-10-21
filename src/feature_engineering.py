"""Point-in-time historical features use arrival time as well as event time."""

import numpy as np
import pandas as pd


def validate_feature_times(frame: pd.DataFrame) -> None:
    """Reject unavailable features, including missing provenance."""
    if frame["snapshot_time"].isna().any():
        raise ValueError("Every prediction needs a valid cutoff time")
    if frame[["feature_source_max_time", "feature_available_time"]].isna().any().any():
        raise ValueError("Feature provenance must include source and available times")
    if (frame["feature_available_time"] > frame["snapshot_time"]).any():
        raise ValueError("A feature was not available at the prediction cutoff")
    if (frame["feature_source_max_time"] > frame["snapshot_time"]).any():
        raise ValueError("A feature uses a future source event")


def build_features(snapshots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Use a backward availability-time join to cumulative customer history."""
    if snapshots["baseline_available_time"].isna().any():
        raise ValueError("Every baseline feature needs an available time")
    if events[["event_time", "available_time"]].isna().any().any():
        raise ValueError("Events require source and available times")
    if (events["event_time"] > events["available_time"]).any():
        raise ValueError("An event cannot be available before its source event time")
    history = events.sort_values(["available_time", "customer_id"]).copy()
    grouped = history.groupby("customer_id", sort=False)
    history["history_count"] = grouped.cumcount() + 1
    history["historical_missed_rate"] = (
        grouped["missed_payment"].cumsum() / history["history_count"]
    )
    history["feature_source_max_time"] = grouped["event_time"].cummax()
    history = history.rename(columns={"available_time": "history_available_time"})
    left = snapshots.assign(_row_order=np.arange(len(snapshots)))
    features = pd.merge_asof(
        left.sort_values(["snapshot_time", "customer_id"]),
        history[
            [
                "customer_id",
                "history_available_time",
                "feature_source_max_time",
                "history_count",
                "historical_missed_rate",
            ]
        ].sort_values(["history_available_time", "customer_id"]),
        by="customer_id",
        left_on="snapshot_time",
        right_on="history_available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    features["history_count"] = features["history_count"].fillna(0).astype(int)
    features["feature_source_max_time"] = features[
        ["feature_source_max_time", "baseline_available_time"]
    ].max(axis=1)
    features["feature_available_time"] = features[
        ["history_available_time", "baseline_available_time"]
    ].max(axis=1)
    features = features.sort_values("_row_order").drop(columns="_row_order")
    validate_feature_times(features)
    return features.reset_index(drop=True)

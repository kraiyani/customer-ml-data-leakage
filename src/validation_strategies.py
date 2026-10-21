"""Each holdout answers an explicitly different deployment question."""

from datetime import datetime

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def random_row_split(
    frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=0.25,
        random_state=seed,
        stratify=frame["missed_payment"],
    )
    return train.copy(), test.copy()


def validate_customer_isolation(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if set(train["customer_id"]) & set(test["customer_id"]):
        raise ValueError("Unseen-customer holdout contains overlapping customer IDs")


def customer_grouped_split(
    frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_indices, test_indices = next(
        splitter.split(frame, groups=frame["customer_id"])
    )
    train, test = frame.iloc[train_indices].copy(), frame.iloc[test_indices].copy()
    validate_customer_isolation(train, test)
    return train, test


def validate_label_maturity(train: pd.DataFrame, cutoff_time: datetime) -> None:
    if pd.isna(cutoff_time):
        raise ValueError("Label maturity needs a valid model fitting time")
    if train["label_available_time"].isna().any():
        raise ValueError("Training labels need an available time")
    if (train["label_available_time"] > cutoff_time).any():
        raise ValueError("A training label was unavailable at model fitting time")


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot_times = sorted(frame["snapshot_time"].unique())
    cutoff_time = datetime.fromisoformat(
        str(snapshot_times[int(len(snapshot_times) * 0.75)])
    )
    train = frame.loc[
        (frame["snapshot_time"] < cutoff_time)
        & (frame["label_available_time"] <= cutoff_time)
    ].copy()
    test = frame.loc[frame["snapshot_time"] >= cutoff_time].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological evaluation requires training and test rows")
    validate_label_maturity(train, cutoff_time)
    return train, test

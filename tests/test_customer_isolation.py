import pandas as pd
import pytest

from src.generate_data import ExperimentConfig, generate_data
from src.validation_strategies import (
    customer_grouped_split,
    validate_customer_isolation,
)


def test_generation_repeats_with_the_same_seed():
    config = ExperimentConfig(customer_count=80, period_count=12)
    first, first_events = generate_data(config)
    second, second_events = generate_data(config)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_events, second_events)
    assert first.groupby("customer_id").size().min() == 12
    assert first["missed_payment"].nunique() == 2


def test_grouped_split_holds_out_every_row_for_test_customers():
    snapshots, _ = generate_data(ExperimentConfig(customer_count=80, period_count=12))
    train, test = customer_grouped_split(snapshots, seed=42)
    validate_customer_isolation(train, test)
    assert len(train) + len(test) == len(snapshots)
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])


def test_customer_overlap_is_rejected_when_isolation_is_required():
    frame = pd.DataFrame({"customer_id": ["A", "B"]})
    with pytest.raises(ValueError, match="customer"):
        validate_customer_isolation(frame, frame)

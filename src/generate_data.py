"""Generate recurrent synthetic payment outcomes, not real customer records."""

from datetime import timedelta
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, NonNegativeInt


class ExperimentConfig(BaseModel):
    customer_count: Annotated[int, Field(ge=40)] = 300
    ingestion_lag_duration: Annotated[timedelta, Field(ge=timedelta())] = timedelta(
        days=2
    )
    label_horizon_duration: Annotated[timedelta, Field(gt=timedelta())] = timedelta(
        days=30
    )
    period_count: Annotated[int, Field(ge=12)] = 24
    seed: NonNegativeInt = 42


def generate_data(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return snapshots and outcome events with explicit availability times."""
    rng = np.random.default_rng(config.seed)
    start_time = pd.Timestamp("2023-01-01", tz="UTC")
    snapshots = []
    events = []
    for customer_index in range(config.customer_count):
        customer_id = f"C{customer_index:05d}"
        latent_risk = float(rng.normal(0, 1.6))
        initial_tenure = int(rng.integers(1, 72))
        for period_index in range(config.period_count):
            snapshot_time = start_time + period_index * config.label_horizon_duration
            event_time = snapshot_time + config.label_horizon_duration
            available_time = event_time + config.ingestion_lag_duration
            activity_score = float(-0.35 * latent_risk + rng.normal(0, 1.2))
            payment_load = float(rng.lognormal(0, 0.4))
            logit = (
                -1.0
                + 0.9 * latent_risk
                - 0.5 * activity_score
                + 0.2 * payment_load
                + 0.35 * period_index / config.period_count
            )
            probability = 1 / (1 + np.exp(-logit))
            missed_payment = int(rng.random() < probability)
            snapshots.append(
                (
                    len(snapshots),
                    customer_id,
                    snapshot_time,
                    snapshot_time - pd.Timedelta(days=1),
                    available_time,
                    initial_tenure + period_index,
                    activity_score,
                    payment_load,
                    missed_payment,
                )
            )
            events.append((customer_id, event_time, available_time, missed_payment))
    snapshot_frame = pd.DataFrame(
        snapshots,
        columns=np.array(
            [
                "row_id",
                "customer_id",
                "snapshot_time",
                "baseline_available_time",
                "label_available_time",
                "account_age_months",
                "activity_score",
                "payment_load",
                "missed_payment",
            ]
        ),
    )
    event_frame = pd.DataFrame(
        events,
        columns=np.array(
            ["customer_id", "event_time", "available_time", "missed_payment"]
        ),
    )
    return snapshot_frame, event_frame

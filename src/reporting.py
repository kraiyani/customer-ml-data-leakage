"""Reproducible CSVs, a measured-results chart and versioned provenance."""

import hashlib
import importlib.metadata
import platform
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from src.generate_data import ExperimentConfig


class RunMetadata(BaseModel):
    configuration: ExperimentConfig
    data_sha256: dict[str, str]
    model: str
    package_versions: dict[str, str]
    python_version: str
    source_sha256: dict[str, str]


def plot_comparison(metrics: pd.DataFrame, filename: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    comparisons = [
        (
            ["random_rows", "customer_grouped", "chronological"],
            [
                "Random rows\nMixed customers/time",
                "Customer holdout\nUnseen IDs",
                "Chronological\nFuture periods",
            ],
            "Different evaluation questions",
            ["#4568b0", "#8071ae", "#268577"],
        ),
        (
            ["chronological", "future_feature", "point_in_time"],
            [
                "Baseline",
                "Future target copy\nINVALID control",
                "Historical features\nPoint-in-time",
            ],
            "Same chronological holdout",
            ["#268577", "#ba5552", "#477db5"],
        ),
    ]
    indexed = metrics.set_index("experiment")
    for axis, (names, labels, title, colors) in zip(axes, comparisons, strict=True):
        values = indexed.loc[names, "roc_auc"].to_numpy()
        bars = axis.bar(labels, values, color=colors, width=0.65)
        axis.set_ylim(0, 1.12)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis.set_ylabel("ROC-AUC")
        axis.set_title(title, fontsize=12, pad=18)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=9)
    figure.suptitle("Synthetic customer validation: measured results", fontsize=15)
    figure.supxlabel(
        "Single fixed seed; different holdouts are not a ranking of split quality.\n"
        "The future-target control deliberately violates feature availability.",
        fontsize=9,
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        filename, dpi=180, facecolor="white", metadata={"Software": "Matplotlib"}
    )
    with (
        plt.rc_context({"svg.hashsalt": "customer-ml-data-leakage"}),
        StringIO() as svg,
    ):
        figure.savefig(svg, format="svg", metadata={"Date": None})
        svg_source = "\n".join(line.rstrip() for line in svg.getvalue().splitlines())
    filename.with_suffix(".svg").write_text(
        svg_source + "\n", encoding="utf-8", newline="\n"
    )
    plt.close(figure)


def save_artifacts(
    root: Path,
    config: ExperimentConfig,
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    data_directory = root / "data"
    results_directory = root / "results"
    data_directory.mkdir(parents=True, exist_ok=True)
    results_directory.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("synthetic_customers.csv", snapshots),
        ("payment_events.csv", events),
        ("point_in_time_features.csv", features),
    ]:
        frame.to_csv(
            data_directory / name, index=False, float_format="%.8f", lineterminator="\n"
        )
    metrics.to_csv(
        results_directory / "metrics.csv",
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )
    predictions.to_csv(
        results_directory / "predictions.csv",
        index=False,
        float_format="%.8f",
        lineterminator="\n",
    )
    metadata = RunMetadata(
        configuration=config,
        data_sha256={
            filename.name: hashlib.sha256(filename.read_bytes()).hexdigest()
            for filename in sorted(data_directory.glob("*.csv"))
        },
        model=(
            "Training-only preprocessing + one-hot customer ID + "
            "LogisticRegression(C=10, max_iter=2000, tol=1e-8)"
        ),
        package_versions={
            name: importlib.metadata.version(name)
            for name in ["matplotlib", "numpy", "pandas", "pydantic", "scikit-learn"]
        },
        python_version=platform.python_version(),
        source_sha256={
            filename.name: hashlib.sha256(filename.read_bytes()).hexdigest()
            for filename in sorted((Path(__file__).parent).glob("*.py"))
        },
    )
    (results_directory / "run_metadata.json").write_text(
        metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(metrics, results_directory / "validation_comparison.png")
    lines = [
        "# Measured synthetic results",
        "",
        f"Seed {config.seed}; {len(snapshots):,} snapshots; "
        f"{config.customer_count} customers.",
        "",
        "| Experiment | ROC-AUC | Average precision | Accuracy | Train rows | "
        "Test rows | Customer overlap | Future-feature rows |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['roc_auc']:.6f} | "
            f"{row['average_precision']:.6f} | {row['accuracy']:.6f} | "
            f"{row['train_row_count']} | {row['test_row_count']} | "
            f"{row['customer_overlap_count']} | {row['future_feature_row_count']} |"
        )
    lines.extend(
        [
            "",
            "The first three cases answer different generalization questions. "
            "Only the last three share an identical chronological holdout.",
            "The future-feature case copies the target and is deliberately invalid; "
            "it is a negative control, not a deployable result.",
            "Historical outcomes enter features only after arrival; "
            "the fitted model stays fixed throughout the holdout.",
            "These are single-seed synthetic results, not expected real-world "
            "scores or a statistical significance claim.",
        ]
    )
    (results_directory / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

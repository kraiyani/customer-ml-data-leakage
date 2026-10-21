"""Export measured results into the companion website's article template."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.reporting import RunMetadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--website", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    website = args.website.resolve()
    template = website / "templates/leakage-article.md"
    if not template.is_file():
        raise ValueError("Website must contain templates/leakage-article.md")
    metrics = pd.read_csv(root / "results/metrics.csv").set_index("experiment")
    metadata = RunMetadata.model_validate_json(
        (root / "results/run_metadata.json").read_text(encoding="utf-8")
    )
    config = metadata.configuration
    names = {
        "random_rows": "Random row split",
        "customer_grouped": "Customer-grouped split",
        "chronological": "Chronological baseline",
        "future_feature": "Future target copy (invalid control)",
        "point_in_time": "Corrected historical features",
    }
    lines = [
        "| Experiment | ROC-AUC | Average precision | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for key, label in names.items():
        row = metrics.loc[key]
        lines.append(
            f"| {label} | {row['roc_auc']:.6f} | "
            f"{row['average_precision']:.6f} | {row['accuracy']:.6f} |"
        )
    source = template.read_text(encoding="utf-8")
    if (
        source.count("{{RESULTS_TABLE}}") != 1
        or source.count("{{DATASET_DESCRIPTION}}") != 1
    ):
        raise ValueError("Article template needs results and dataset slots")
    destination = website / "content/posts/preventing-data-leakage-customer-level-ml.md"
    destination.write_text(
        source.replace("{{RESULTS_TABLE}}", "\n".join(lines)).replace(
            "{{DATASET_DESCRIPTION}}",
            f"This run uses {config.customer_count} synthetic customers, "
            f"{config.period_count} fixed "
            f"{config.label_horizon_duration.total_seconds() / 86400:g}-day periods "
            f"({config.customer_count * config.period_count:,} snapshots), "
            f"and random seed {config.seed}. Outcome records arrive "
            f"{config.ingestion_lag_duration.total_seconds() / 86400:g} days "
            "after the outcome event.",
        ),
        encoding="utf-8",
    )
    assets = website / "assets/customer-ml-data-leakage"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ["metrics.csv", "run_metadata.json", "validation_comparison.png"]:
        shutil.copyfile(root / "results" / name, assets / name)
    print("Exported article source, measured metrics, provenance and chart")


if __name__ == "__main__":
    main()

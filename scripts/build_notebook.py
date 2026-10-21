"""Build the demonstration notebook from source and optionally execute it."""

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Customer-level ML data leakage\n\n"
            "A reproducible, entirely synthetic experiment. The target is a "
            "missed payment in the next 30-day period, which can recur. "
            "No real customers or employer data are used. The target-copy "
            "control is deliberately invalid, not a deployable model."
        ),
        nbformat.v4.new_code_cell(
            dedent("""\
            from pathlib import Path
            import sys

            project_root = Path.cwd()
            if not (project_root / "src").exists():
                project_root = project_root.parent
            sys.path.insert(0, str(project_root))

            from src.generate_data import ExperimentConfig, generate_data
            from src.feature_engineering import build_features
            from src.train_evaluate import run_experiment
            from src.reporting import RunMetadata, plot_comparison
            """)
        ),
        nbformat.v4.new_code_cell(
            'metadata_path = project_root / "results/run_metadata.json"\n'
            "config = (\n"
            "    RunMetadata.model_validate_json(\n"
            '        metadata_path.read_text(encoding="utf-8")\n'
            "    ).configuration\n"
            "    if metadata_path.exists() else ExperimentConfig()\n"
            ")\n"
            "snapshots, events = generate_data(config)\n"
            "features = build_features(snapshots, events)\n"
            "snapshots.head()"
        ),
        nbformat.v4.new_markdown_cell(
            "## Five declared cases\n\n"
            "The first three use identical baseline feature definitions with "
            "different split questions. The last three use exactly the same "
            "chronological training and test rows. Labels must have matured "
            "by the training cutoff. No holdout is used to select parameters."
        ),
        nbformat.v4.new_code_cell(
            "metrics, predictions = run_experiment(features, config)\nmetrics"
        ),
        nbformat.v4.new_code_cell(
            dedent("""\
            from IPython.display import Image, display

            chart_path = project_root / ".cache/notebook-validation.png"
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            plot_comparison(metrics, chart_path)
            display(Image(filename=str(chart_path)))
            """)
        ),
        nbformat.v4.new_code_cell(
            dedent("""\
            from src.validation_strategies import chronological_split

            train, test = chronological_split(features)
            assert train.label_available_time.max() <= test.snapshot_time.min()
            grouped = metrics.set_index("experiment")
            assert grouped.loc["customer_grouped", "customer_overlap_count"] == 0
            assert grouped.loc["point_in_time", "future_feature_row_count"] == 0
            assert grouped.loc["future_feature", "future_feature_row_count"] > 0
            print("Customer isolation, label maturity and provenance checks passed.")
            """)
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation\n\n"
            "Customer ID is one-hot encoded, so repeated known customers let "
            "the baseline learn stable latent propensities. Unknown IDs use "
            "the encoder's ignore behaviour. Grouped validation therefore "
            "answers a different question from row validation. Corrected "
            "historical features need not beat the chronological baseline. "
            "The feature store can incorporate newly arrived historical "
            "outcomes during the holdout; the fitted model does not change. "
            "These single-seed results are a constructed demonstration, not "
            "real-world performance estimates or confidence intervals."
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{index:02d}"
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    if args.execute:
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    destination = root / "notebooks/leakage_experiment.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, destination)
    print(f"Wrote {destination.name}; executed={args.execute}")


if __name__ == "__main__":
    main()

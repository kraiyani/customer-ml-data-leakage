import hashlib
import runpy
import shutil
import sys
from pathlib import Path

import nbformat
import pandas as pd

from src.feature_engineering import build_features
from src.generate_data import ExperimentConfig, generate_data
from src.reporting import save_artifacts
from src.train_evaluate import run_experiment


def test_notebook_respects_saved_config_and_preserves_canonical_results(
    tmp_path, monkeypatch
):
    config = ExperimentConfig(customer_count=80, period_count=12, seed=7)
    snapshots, events = generate_data(config)
    features = build_features(snapshots, events)
    metrics, predictions = run_experiment(features, config)
    save_artifacts(tmp_path, config, snapshots, events, features, metrics, predictions)
    before = {
        filename.name: hashlib.sha256(filename.read_bytes()).hexdigest()
        for filename in (tmp_path / "results").iterdir()
    }
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    script = tmp_path / "scripts/build_notebook.py"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/build_notebook.py", script
    )
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    notebook = nbformat.read(
        tmp_path / "notebooks/leakage_experiment.ipynb", as_version=4
    )
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "notebook-cell", "exec"), namespace)
    assert namespace["config"] == config
    pd.testing.assert_frame_equal(namespace["metrics"].round(6), metrics.round(6))
    after = {
        filename.name: hashlib.sha256(filename.read_bytes()).hexdigest()
        for filename in (tmp_path / "results").iterdir()
    }
    assert before == after

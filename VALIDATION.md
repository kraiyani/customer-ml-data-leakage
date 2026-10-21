# Validation record

Validated locally on 21 October 2026 with Python 3.11.7 on Windows.
Runtime package versions and the exact model/data configuration are recorded
in `results/run_metadata.json`.

## Executed checks

- `python -m src.train_evaluate`: regenerated all five declared experiments,
  synthetic CSVs, measured metrics, predictions, charts and provenance.
- `./scripts/commit-gate.ps1`: passed Ruff formatting, Ruff linting, ty type
  checking, and all 11 pytest tests. No tests were skipped or weakened.
- `python -m scripts.build_notebook --execute`: executed all five code cells
  in a Python kernel, including the embedded results chart and audit checks.
- Independently checked every recorded source/data SHA-256 hash, notebook
  schema, execution counts, absence of error outputs, and exact agreement of
  the metrics, chart and metadata exported to the website.

Ruff and ty were supplied through a local tools directory. The existing
Python environment supplied the pinned runtime packages. Unrelated ambient
pytest plugins were excluded using `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`;
project tests all ran. Jupyter emitted environment/deprecation warnings,
without cell errors. This was not a fresh installation test.

## Regression evidence

Both new regressions failed before their corresponding fixes and passed
afterward:

- Notebook execution now uses a saved nondefault configuration and leaves
  all canonical result-file hashes unchanged.
- Valid historical records cannot mask a missing baseline availability time.

Other tests cover reproducible generation, customer isolation, late arrivals,
future-event exclusion, exact-cutoff availability, missing prediction times,
training-label maturity, and identical chronological evaluation rows.

## Interpretation limits

The published example is one declared seed, one model and synthetic data.
It does not establish confidence intervals or expected production scores.
Different split populations answer different questions. The perfect-score
future-target case is an intentionally invalid control, and valid historical
features did not improve this run's chronological baseline.

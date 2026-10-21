# Measured synthetic results

Seed 42; 7,200 snapshots; 300 customers.

| Experiment | ROC-AUC | Average precision | Accuracy | Train rows | Test rows | Customer overlap | Future-feature rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_rows | 0.836026 | 0.789650 | 0.767778 | 5400 | 1800 | 299 | 0 |
| customer_grouped | 0.713793 | 0.649145 | 0.657222 | 5400 | 1800 | 0 | 0 |
| chronological | 0.826301 | 0.781191 | 0.760000 | 5100 | 1800 | 300 | 0 |
| future_feature | 1.000000 | 1.000000 | 1.000000 | 5100 | 1800 | 300 | 1800 |
| point_in_time | 0.825284 | 0.781239 | 0.758889 | 5100 | 1800 | 300 | 0 |

The first three cases answer different generalization questions. Only the last three share an identical chronological holdout.
The future-feature case copies the target and is deliberately invalid; it is a negative control, not a deployable result.
Historical outcomes enter features only after arrival; the fitted model stays fixed throughout the holdout.
These are single-seed synthetic results, not expected real-world scores or a statistical significance claim.

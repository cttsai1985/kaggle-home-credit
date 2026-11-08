# From a configurable modeling framework to an inspectable credit-risk experiment

The original repository orchestrated feature caching, Bayesian hyperparameter searches and stacked models. Its readable standalone selected-feature LightGBM path provides an attributable starting point for a current, reproducible experiment. The modern package separates data acquisition, applicant-level feature construction, model fitting, evaluation and explicit tracking.

## Features and attribution

Application ratios and historical aggregates derive from the implementation credited to [Joao Aguiar](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features) and [Olivier Grellier](https://www.kaggle.com/ogrellier/lighgbm-with-selected-features). The original source and framework remain under `legacy/`; the baseline revision is [a378d5f](https://github.com/cttsai1985/kaggle-home-credit-solution-mlops/tree/a378d5f).

The feature table is one row per `SK_ID_CURR`. Bureau balance aggregates by bureau account before applicant aggregation; all other child histories aggregate to applicant level before joining. Every join validates its many-to-one relationship and unchanged application row count. Missing history stays missing. Target values are never aggregated into features.

The recipe retains its four-row `XNA` gender exclusion, binary/category encodings and fixed selected-feature list. Organization-income medians and category vocabularies are transductive summaries of the combined labeled/unlabeled application population. They contain no target values, but they are not a fold-fitted inductive preprocessing pipeline. The resulting validation answers the original competition setting; it does not validate a production lending system on future applicants.

## Current implementation choices

- Pandas `concat` replaces removed `append`; explicit numerical reductions handle mixed string/numeric application flags.
- String categories work with current Pandas string dtypes, not only old `object` columns.
- Numerical input floats are stored as float32 to reduce memory. Infinite ratios become missing values. These choices can change exact thresholds and predictions relative to the old stack.
- LightGBM uses its current callback API for early stopping and logging. CPU threads and deterministic histogram settings are explicit.
- Stable model column identifiers avoid LightGBM's JSON-character restrictions; `features.json` preserves original names and ordering for inspection and reload.
- The reference experiment uses five shuffled applicant folds, seed 1001 and the original selected-model hyperparameters. Every labeled applicant has one held-out prediction. Averaged test predictions come from those same fold models.
- Plain model text, Parquet features and CSV/JSON evidence replace framework-specific HDF5 caches and hidden module paths. Training and submission are separate commands.

The current package does not reimplement old library internals or assert numerical equivalence. Historical HPO and stacked-model artifacts were not committed and cannot be reconstructed from their filename scores. Stronger current recipes should be labeled as improved models and evaluated against the historical benchmark, rather than described as the same trained model.

## Benchmark comparison

The account's historical Kaggle records identify these experiments:

| Experiment | Public ROC AUC | Private ROC AUC |
|---|---:|---:|
| Selected-feature LightGBM, 2018-07-15 | 0.79571 | 0.79075 |
| Final XGBoost stack, 2018-08-30 | 0.79682 | 0.79237 |
| Historical blend, 2018-08-29 | 0.79764 | 0.79158 |

The original README's PB/LB labels were ambiguous. Current reporting names public and private metrics explicitly. An OOF AUC is internal validation and must not be substituted for either leaderboard score. See [results](results.md) for measured current runs.

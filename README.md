# Home Credit: Default Risk

Assessing repayment risk is especially difficult when an application has a limited credit history. [Home Credit's challenge](https://www.kaggle.com/c/home-credit-default-risk) asks how application and repayment information can improve that assessment. This solution aggregates relational histories into one row per application and trains a five-fold LightGBM ensemble.

Credit histories earn their place in the model: at the same 100-round budget, adding selected relational features raises mean OOF AUC from **0.746335 to 0.755201**, improving all ten matched seed/fold comparisons. The separate full-budget stratified run reaches **0.79257 OOF AUC** across **307,507 eligible applications**.

For a lender, the next question is whether better risk differentiation improves the approval/loss tradeoff on future applications. The benchmark establishes model discrimination; the [decision and value discussion](docs/problem-and-value.md) explains what a lending pilot would still need to measure.

[Feature study](docs/modeling-study.md) · [Results](docs/results.md) · [Run the solution](#run-the-experiment)

<!-- portfolio-results:start -->
## Results

**Public leaderboard — ROC AUC ↑.**

| Model / submission | Public score | Public reference rank / teams | Top % | Reference ties |
|---|---:|---:|---:|---:|
| Historical selected-feature LightGBM | 0.79571 | 2,699 / 7,180 | **37.59%** | 1 |
| Historical final XGBoost stack | 0.79682 | 2,272–2,275 / 7,180 | **31.64%** | 4 |
| Recorded current KFold reference | 0.79641 | 2,396 / 7,180 | **33.37%** | 1 |
| Historical blend | 0.79764 | 2,126–2,127 / 7,180 | **29.61%** | 2 |

Reference: **2026-11-09T00:02:26Z**, **7,180 unique scored teams**. Ranks are score-equivalent comparisons with the dated public export, not awarded final ranks. Lower Top % is better. A zero tie count denotes an insertion position; ties occupy the shown interval.

**OOF/holdout rank and Top %: not applicable. Private leaderboard rank and Top %: unavailable without a matching private reference population.** Internal validation and private scores remain separately reported in [detailed results](docs/results.md). [Reference data and calculation](docs/leaderboard-context.json).
<!-- portfolio-results:end -->

## Where risk ranking breaks down

On all 307,507 retained OOF applications, the hardest 10% within each class accounts for about 34.5% of that class's pairwise-ranking errors. The two views overlap and must not be added. External-score patterns motivate a controlled history-feature comparison within predefined strata, rather than removing difficult applicants or changing the reported evaluation. [Hard-case analysis and next experiment](docs/error-analysis.md).

## Relational features and LightGBM

The feature set combines:

- **Application ratios:** credit-to-annuity, income, employment and external-score summaries.
- **Bureau history:** credit amounts, active/closed accounts and monthly status summaries.
- **Previous applications:** approved/refused amounts and credit-decision patterns.
- **Repayment behavior:** delinquency, installment timing and differences between due and paid amounts.

The selected-feature recipe builds on the original repository's credited Aguiar/Grellier implementation. It excludes four labeled `CODE_GENDER=XNA` rows and retains **307,507 training applicants**. No applicant identifiers or target values enter the model inputs.

Five fixed-seed stratified folds produce one held-out prediction per training applicant while preserving the target proportions. Early stopping uses that fold's validation AUC; the final test probability averages the five fold models. The default profile fixes StratifiedKFold seed 1001 and the model settings before evaluation. A historical KFold reference and one capacity comparison remain separate, recorded experiments.

Some unlabeled category vocabularies and organization-income summaries use the combined application population, as in the historical competition design. This transductive experiment does not establish performance on future production applicants. [Method notes](docs/migration.md) explain the assumptions, attribution and changes.

## Testing the contribution of credit history

The bounded feature comparison gives a reason to retain the history-table complexity: it improves all ten matched seed/fold comparisons at the same 100-round budget. The result is useful for validating the recipe; it does not establish a novel credit model or prospective lending performance. [Controlled study](docs/modeling-study.md) · [Executed notebook](notebooks/02-controlled-ablations.ipynb).

Artifact delivery has its own failure boundary. A successful local experiment remains valid if a tracker transfer fails; recovery should reuse its identity and verify remote bytes, rather than retrain or silently create a replacement run. Fault-injection tests exercise that client contract. The shared platform's real saved-model batch drill uses Iceberg; it is not an execution claim for this credit model.

This solution uses shared tracking and storage services. Its responsibilities are data/model contracts, reproducible evaluation and verified batch outputs. [Consumer contract](docs/consumer-contract.md) · [Engineering boundaries](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/mlops-assessment.md) · [Original walkthrough](notebooks/01-credit-risk.ipynb).

## Use the saved model

The [model packaging and serving guide](docs/serving.md) explains the actual input/output contract, export from saved artifacts and the shared HTTP parity check. A canary release retains the tested source, dependency lock and aggregate verification evidence; private model weights and request rows are not release assets.

[Real-model canary verification](docs/canary-validation.json) records the source, package/runtime identity, native-to-HTTP parity and rejected invalid request.

## Joining the data can change what the model learns

The prediction unit is an applicant, while bureau and previous-application tables contain multiple records per applicant. On this input snapshot, directly left-joining those two histories would expand **307,507 applicants into 8,091,067 rows**. The same applicants would then receive unequal weights: positive prevalence changes from **8.07% per applicant to 8.87% per joined row** without adding a single applicant.

That is why aggregation and join-grain validation belong before training. They preserve the intended learning unit, rather than merely saving memory. The maintained pipeline aggregates first and keeps one row per applicant; this audit quantifies the counterfactual join it prevents. It is not a discovered production incident or a measured model-score effect. [Calculation and source identities](docs/data-insight.json) · [Consumer requirements](docs/consumer-contract.md).

## Application data and credit histories

The competition provides **307,511 labeled applications** and **48,744 test applications**, plus six historical tables. `SK_ID_CURR` identifies the applicant; `TARGET=1` indicates payment difficulty under the competition definition.

A single applicant can have many bureau accounts, monthly balances and payment records. Joining these tables directly would multiply application rows. The pipeline aggregates each history to applicant level before joining, validates the join grain, and preserves test applicant identity and ordering.

## Stratified validation and diagnostics

`home-credit analyze` independently checks the configured assignments, applicant identity, complete OOF coverage and per-fold target balance. It recomputes pooled AUC, reports fold mean/std/min/max and best iterations, and summarizes covariates for weaker-fold and train/test comparisons. Reports link model/data hashes without rewriting the trained run.

The full stratified run reaches **0.79257 OOF AUC** over 307,507 applicants. The eligible dataset has 24,825 positive targets; stratification assigns 4,965 to each validation fold. Fold AUC ranges from **0.78784 to 0.79653**, with sample standard deviation **0.00338**. Balanced targets do not guarantee equal case difficulty or lower AUC variance. Fold spread is descriptive, not a confidence interval. Early stopping uses the reported validation fold for checkpoint selection, so its AUC can be optimistic. See [cross-validation diagnostics](docs/cross-validation.md).

## Benchmark results

The competition metric is **ROC AUC**, where higher is better. Historical account records identify the selected-feature model and the final stack separately:

| Historical experiment | Public AUC | Private AUC |
|---|---:|---:|
| Selected-feature LightGBM | 0.79571 | 0.79075 |
| Final XGBoost stack | 0.79682 | 0.79237 |
| Blend | 0.79764 | 0.79158 |

The recorded full KFold reference reaches **0.79281 OOF AUC**, **0.79641 public AUC** and **0.79065 private AUC**. It improves the historical selected model's public score by 0.00070 and is 0.00010 lower on the private score; the historical final stack remains stronger. The default stratified run is evaluated separately. A higher-capacity KFold comparison was weaker on OOF validation and was not selected for submission.

Exact experiment settings and file identities are recorded in [the benchmark record](docs/results.md). Out-of-fold validation and leaderboard scores are reported separately; a local AUC is not a reproduced leaderboard score.

The historical final stack's public AUC **0.79682** corresponds to approximately **top 31.6%** of the downloaded public leaderboard; the modern reference's **0.79641** corresponds to **top 33.4%**. [Percentile method and snapshot](docs/results.md#leaderboard-percentile-context). These are score-equivalent positions, separate from official late-entry ranks. AUC measures ranking, not classification accuracy.

## Run the experiment

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --locked --group dev --group lab --group data --group tracking
export HOME_CREDIT_DATA_DIR=/absolute/path/outside/git/home-credit
.venv/bin/kaggle auth login
.venv/bin/home-credit download --data-dir "$HOME_CREDIT_DATA_DIR"
.venv/bin/home-credit validate --data-dir "$HOME_CREDIT_DATA_DIR"
.venv/bin/home-credit features --data-dir "$HOME_CREDIT_DATA_DIR" --output .local/cache
.venv/bin/home-credit train --cache .local/cache --config configs/baseline.json \
  --run-dir .local/stratified-baseline
.venv/bin/home-credit evaluate --cache .local/cache --run-dir .local/stratified-baseline
.venv/bin/home-credit analyze --cache .local/cache --run-dir .local/stratified-baseline \
  --output .local/analyses/stratified.json
```

Download requires accepted Kaggle competition rules. Use a new directory for every feature preparation and training invocation. The local runtime uses CPU/Linux/Python 3.12; the package also supports the hosted Python 3.13 runtime; the default configuration caps training at six threads.

Open [the executed walkthrough](notebooks/01-credit-risk.ipynb) to inspect class imbalance, relational aggregation, fold performance, ROC curves, feature contributions and checkpoint reconstruction. The [runbook](docs/reproduction.md) covers prediction, artifacts and resource requirements.

## JupyterLab and experiment tracking

```bash
.venv/bin/python scripts/create_lab_token.py
docker compose up --build -d lab
```

Open http://127.0.0.1:8891/lab using the token in ignored `.local/jupyter_token`. Docker provides an isolated environment with read-only competition data and writable experiment outputs. A VS Code dev container uses the same image.

Each run retains configuration, data and source fingerprints, fold assignments, portable model files, validation metrics and artifact checksums. The full real-data benchmark was imported into shared MLflow and its model artifact retrieved with a matching checksum; repeating the import reused the same run. The client requires the existing `kaggle/home-credit` experiment declared by this repository and records project, logical-run and source identities. Shared storage and service deployment remain external; an optional override exposes an existing SeaweedFS client mount. See [tracking and storage](docs/reproduction.md#existing-tracking-and-storage-services).

## Kaggle execution

The [hosted notebook](notebooks/kaggle/rerun.ipynb) runs the same package from an attached code wheel and retains a portable run bundle. The full stratified hosted run completed with the same input and fold-assignment identities as native execution; its model and CV report passed shared-store import and checksum read-back. [Kaggle delivery instructions](docs/kaggle.md) cover packaging, hosted-runtime requirements and scoring verified predictions.

## Development

```bash
uv sync --locked --group dev --group serving --group tracking --group tracking-test
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/pytest -q
```

The maintained implementation lives in `src/home_credit`, training recipes in `configs`, notebooks in `notebooks`, and container configuration in `docker`. Data, feature caches, weights and credentials are excluded from Git.

## License

Source code is licensed under the [Apache License 2.0](LICENSE). Competition data follows its own terms.

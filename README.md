# Home Credit Default Risk

**Credit histories help rank applicants, but a useful model also needs trustworthy joins and validation.** This project revisits a 2018 Kaggle solution with modern LightGBM, applicant-level relational features and a reproducible batch workflow.

| What matters | What the evidence says |
|---|---|
| Competition context | The modern reference public AUC **0.79641** corresponds to approximately **top 33.4%** of the downloaded public leaderboard. The historical final stack's 0.79682 corresponds to **top 31.6%**. These are score-equivalent positions, not new official competition ranks. |
| Controlled ablation | Adding selected history features raises mean OOF AUC **0.74634 → 0.75520** at 100 fixed rounds. The paired gain is **+0.00864 to +0.00910** across both model seeds on identical folds. |
| Validation finding | Five stratified folds contain exactly **4,965 positive cases each**. OOF AUC is **0.79257**; fold AUC still spans **0.78784–0.79653**. Equal label balance does not make case difficulty equal. |

AUC measures how well the model ranks payment-difficulty cases above other applicants; **0.79641 is not 79.6% accuracy**. Local out-of-fold predictions and hidden-test leaderboard scores answer different questions. The modern reference is close to the historical selected-feature model; it does not surpass the final historical stack. [Leaderboard percentile method and snapshot](docs/results.md#leaderboard-percentile-context) · [Executed walkthrough](notebooks/01-credit-risk.ipynb).

The [controlled study](docs/modeling-study.md) separates feature contribution from training-seed variability. Its fixed budget is deliberately separate from the full early-stopped benchmark. The [executed ablation notebook](notebooks/02-controlled-ablations.ipynb) shows both seeds and every fold.

## From applied science to an operated experiment

[Consumer contract](docs/consumer-contract.md) · [Controlled modeling study](docs/modeling-study.md)

The pipeline validates join grain, retains applicant identity, saves every fold model and independently reconstructs predictions. Local execution and Kaggle use the same package; artifact imports retain their source/data identity in the shared tracker. The [shared operations case study](https://github.com/cttsai1985/mlops-platform/blob/main/docs/batch-operations.md) demonstrates orchestration and recovery on Iceberg; it is not evidence that this credit model has been deployed to lending operations.

This solution is a **consumer of MLOps capabilities** for reproducible batch research. Some historical feature construction uses the unlabeled combined population, and early stopping uses each reported validation fold. This is not a prospective lending validation or a deployed credit-decision service. [MLOps capability assessment](https://github.com/cttsai1985/mlops-platform/blob/main/docs/mlops-assessment.md).

## Application data and credit histories

The competition provides **307,511 labeled applications** and **48,744 test applications**, plus six historical tables. `SK_ID_CURR` identifies the applicant; `TARGET=1` indicates payment difficulty under the competition definition.

A single applicant can have many bureau accounts, monthly balances and payment records. Joining these tables directly would multiply application rows. The pipeline aggregates each history to applicant level before joining, validates the join grain, and preserves test applicant identity and ordering.

## Relational features and LightGBM

The feature set combines:

- **Application ratios:** credit-to-annuity, income, employment and external-score summaries.
- **Bureau history:** credit amounts, active/closed accounts and monthly status summaries.
- **Previous applications:** approved/refused amounts and credit-decision patterns.
- **Repayment behavior:** delinquency, installment timing and differences between due and paid amounts.

The selected-feature recipe builds on the original repository's credited Aguiar/Grellier implementation. It excludes four labeled `CODE_GENDER=XNA` rows and retains **307,507 training applicants**. No applicant identifiers or target values enter the model inputs.

Five fixed-seed stratified folds produce one held-out prediction per training applicant while preserving the target proportions. Early stopping uses that fold's validation AUC; the final test probability averages the five fold models. The default profile fixes StratifiedKFold seed 1001 and the model settings before evaluation. A historical KFold reference and one capacity comparison remain separate, recorded experiments.

Some unlabeled category vocabularies and organization-income summaries use the combined application population, as in the historical competition design. This transductive experiment does not establish performance on future production applicants. [Method notes](docs/migration.md) explain the assumptions, attribution and changes.

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

Each run retains configuration, data and source fingerprints, fold assignments, portable model files, validation metrics and artifact checksums. The full real-data benchmark was imported into shared MLflow and its model artifact retrieved with a matching checksum; repeating the import reused the same run. The client requires the platform-provisioned `kaggle/home-credit` experiment and records project, logical-run and source identities. Shared storage and service deployment remain external; an optional override exposes an existing SeaweedFS client mount. See [tracking and storage](docs/reproduction.md#existing-tracking-and-storage-services).

## Kaggle execution

The [hosted notebook](notebooks/kaggle/rerun.ipynb) runs the same package from an attached code wheel and retains a portable run bundle. The full stratified hosted run completed with the same input and fold-assignment identities as native execution; its model and CV report passed shared-store import and checksum read-back. [Kaggle delivery instructions](docs/kaggle.md) cover packaging, hosted-runtime requirements and scoring verified predictions.

## Development

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/pytest -q
```

The maintained implementation lives in `src/home_credit`, training recipes in `configs`, notebooks in `notebooks`, and container configuration in `docker`. Data, feature caches, weights and credentials are excluded from Git.

## License

Source code is licensed under the [Apache License 2.0](LICENSE). Competition data follows its own terms.

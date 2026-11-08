# Home Credit Default Risk

Rank loan applicants by the likelihood of payment difficulty using application data and credit histories. The model combines applicant-level financial ratios with bureau, prior-loan and payment aggregates, then trains five cross-validated LightGBM models.

This repository develops Chia-Ta Tsai's 2018 Kaggle solution into a current, reproducible experiment. The [notebook](notebooks/01-credit-risk.ipynb) examines the data, feature construction, out-of-fold predictions and saved-model checks. The original modeling framework remains in [legacy/](legacy/README.md).

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

Five shuffled folds produce one held-out prediction per training applicant. Early stopping uses that fold's validation AUC; the final test probability averages the five fold models. The reference profile retains the original selected-model settings. A separate capacity profile tests more leaves and lower minimum leaf Hessian while keeping the feature table and fold assignment fixed.

Some unlabeled category vocabularies and organization-income summaries use the combined application population, as in the historical competition design. This transductive experiment does not establish performance on future production applicants. [Method notes](docs/migration.md) explain the assumptions, attribution and changes.

## Benchmark results

The competition metric is **ROC AUC**, where higher is better. Historical account records identify the selected-feature model and the final stack separately:

| Historical experiment | Public AUC | Private AUC |
|---|---:|---:|
| Selected-feature LightGBM | 0.79571 | 0.79075 |
| Final XGBoost stack | 0.79682 | 0.79237 |
| Blend | 0.79764 | 0.79158 |

The current full five-fold baseline reaches **0.79281 OOF AUC**, **0.79641 public AUC** and **0.79065 private AUC**. It improves the historical selected model's public score by 0.00070 and is 0.00010 lower on the private score; the historical final stack remains stronger. A higher-capacity comparison was weaker on OOF validation and was not selected for submission.

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
.venv/bin/home-credit train --cache .local/cache --config configs/reference.json \
  --run-dir .local/benchmark
.venv/bin/home-credit evaluate --cache .local/cache --run-dir .local/benchmark
```

Download requires accepted Kaggle competition rules. Use a new directory for every feature preparation and training invocation. The local runtime uses CPU/Linux/Python 3.12; the package also supports the hosted Python 3.13 runtime; the reference configuration caps training at six threads.

Open [the executed walkthrough](notebooks/01-credit-risk.ipynb) to inspect class imbalance, relational aggregation, fold performance, ROC curves, feature contributions and checkpoint reconstruction. The [runbook](docs/reproduction.md) covers prediction, artifacts and resource requirements.

## JupyterLab and experiment tracking

```bash
.venv/bin/python scripts/create_lab_token.py
docker compose up --build -d lab
```

Open http://127.0.0.1:8891/lab using the token in ignored `.local/jupyter_token`. Docker provides an isolated environment with read-only competition data and writable experiment outputs. A VS Code dev container uses the same image.

Each run retains configuration, data and source fingerprints, fold assignments, portable model files, validation metrics and artifact checksums. An optional client imports completed runs into an existing MLflow server. Shared storage and service deployment remain external; an optional override exposes an existing SeaweedFS client mount. See [tracking and storage](docs/reproduction.md#existing-tracking-and-storage-services).

## Kaggle execution

The [hosted notebook](notebooks/kaggle/rerun.ipynb) runs the same package from an attached code wheel and retains a portable run bundle. [Kaggle delivery instructions](docs/kaggle.md) cover packaging, hosted-runtime requirements and scoring verified predictions.

## Development

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/pytest -q
```

The maintained implementation lives in `src/home_credit`, training recipes in `configs`, notebooks in `notebooks`, and container configuration in `docker`. Data, feature caches, weights and credentials are excluded from Git. Code retains the [Apache 2.0 license](LICENSE); competition data follows its own terms.

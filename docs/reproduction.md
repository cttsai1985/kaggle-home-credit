# Reproduce the five-fold experiment

The project runs on CPU/Linux with Python 3.12. `uv.lock` fixes package versions; `uv sync` creates a project-local `.venv`. Competition data, feature caches and model outputs stay outside Git.

## Download and validate official inputs

```bash
uv sync --locked --group dev --group lab --group data --group tracking
export HOME_CREDIT_DATA_DIR=/absolute/path/outside/git/home-credit
.venv/bin/kaggle auth login
.venv/bin/home-credit download --data-dir "$HOME_CREDIT_DATA_DIR"
.venv/bin/home-credit validate --data-dir "$HOME_CREDIT_DATA_DIR"
```

Accept the competition rules in your Kaggle account. The downloader retains archive and payload hashes, refuses unsafe archive paths and retains existing payload files. Existing files are a cache: do not edit them and continue claiming official-file identity.

## Build the applicant-level feature table

```bash
.venv/bin/home-credit features --data-dir "$HOME_CREDIT_DATA_DIR" --output .local/cache
```

This reads all application and history tables, aggregates child rows before joining, and writes `selected.parquet` plus a manifest. It keeps missing numerical values for LightGBM, converts undefined infinite ratios to missing and uses the fixed historical feature-exclusion list. `--all-features` retains the preselection features. `--row-limit` is a software diagnostic and does not reproduce a competition score.

The full selected cache has 307,507 labeled applicants after excluding four historical `CODE_GENDER=XNA` rows, and all 48,744 test applicants. Neither IDs nor targets are model features. Some category vocabularies and organization-income medians use the combined application population without labels; this is the original transductive competition feature design. See [method changes](migration.md).

## Train, reload and generate test predictions

```bash
.venv/bin/home-credit train --cache .local/cache --config configs/baseline.json \
  --run-dir .local/stratified-baseline
.venv/bin/home-credit evaluate --cache .local/cache --run-dir .local/stratified-baseline
.venv/bin/home-credit predict --cache .local/cache --run-dir .local/stratified-baseline \
  --output .local/reloaded-submission.csv
```

Use a fresh run directory for each training invocation. The default configuration trains five models with shuffled StratifiedKFold seed 1001, up to 10,000 boosting rounds and validation-only early stopping after 200 rounds. Threads are capped at six. Out-of-fold predictions cover each training applicant once; test predictions average the five saved models. Evaluation reloads each model, reconstructs its validation predictions and verifies artifact hashes.

```bash
.venv/bin/home-credit analyze --cache .local/cache --run-dir .local/stratified-baseline \
  --output .local/analyses/stratified.json
```

The analyzer checks exact seeded assignments, coverage and label balance, then reports fold dispersion and covariate context. The historical KFold recipe remains `configs/reference.json`; use it with a separate new run directory to reproduce the recorded leaderboard benchmark. See [cross-validation](cross-validation.md).

| Artifact | Purpose |
|---|---|
| `manifest.json` | Source, environment, data/feature identities, recipe, status and checksums |
| `config.json` / `features.json` | Resolved training settings and ordered readable feature names |
| `fold-*.txt` | Portable LightGBM model files |
| `fold-metrics.json` | Fold AUC, best iteration and training/validation counts |
| `oof.csv` | Applicant ID, target, held-out probability and fold assignment |
| `submission.csv` | Complete test IDs in official order and averaged probability |
| `importance.csv` / `metrics.json` | Gain importance and aggregated validation metrics |

Keep failed run manifests for diagnosis. Do not import an incomplete run as a successful experiment. A run's hashes verify stored bytes; they do not establish score parity by themselves.

## JupyterLab and development containers

```bash
.venv/bin/python scripts/create_lab_token.py
docker compose up --build -d lab
```

Set `HOME_CREDIT_DATA_DIR` first. Open http://127.0.0.1:8891/lab using the token in ignored `.local/jupyter_token`. The repository is writable at `/workspace`, official data is read-only at `/data`, and `.local` is writable at `/outputs`. The image installs its own environment at `/opt/venv`; it does not use the host `.venv`. Lab is limited to six CPUs and 24 GiB RAM.

The supplied user has UID 1000; adjust ownership/user IDs on other hosts if needed. The dev container uses the same Lab image with a read-only data mount. Opening an interactive VS Code session is a separate editor check.

For isolated runtime evaluation:

```bash
docker build -f docker/Dockerfile --target runtime -t home-credit-runtime:0.1 .
docker run --rm --network none --cpus 6 --memory 24g \
  -v "$PWD:/workspace:ro" home-credit-runtime:0.1 \
  evaluate --cache /workspace/.local/cache --run-dir /workspace/.local/stratified-baseline
```

## Existing tracking and storage services

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5500
.venv/bin/home-credit track --run-dir .local/stratified-baseline
```

The client passes an isolated SQLite-backed MLflow test covering artifact upload/retrieval, repeated imports and conflict rejection. Run it with `uv sync --locked --group dev --group tracking --group tracking-test` and `.venv/bin/pytest -q`. The real benchmark was also imported into the shared experiment, reimported without duplication, and its first model retrieved with a matching SHA-256. Exact evidence is in [validation.json](validation.json).

The current host route is port 5500; configure it for your deployment. A host-loopback binding cannot be reached simply through Docker's host gateway. For Lab, attach the platform client network as shown below and use `http://mlflow:5000`. Use runtime authentication, not credentials in the URI. The client requires a preprovisioned `kaggle/home-credit` experiment and fails with a bootstrap instruction when it is absent. It checks artifact hashes and logical run identity before import, rejects other project identities and refuses unexpected artifact names. The shared MLflow server owns its artifact-store route; SeaweedFS can remain behind that route. An optional client-side filesystem mount is separate:

```bash
export SEAWEED_MOUNT=/absolute/path/to/existing/client-mount
docker compose -f compose.yaml -f docker/compose.seaweed.yaml up -d lab
```

That exposes the existing mount read-only at `/shared`; it neither deploys SeaweedFS nor creates a FUSE mount. No shared services are started, upgraded or reconfigured by this repository. Prefect and W&B are not prerequisites for this experiment and are not implemented integrations.

### Connect Lab to the existing client network

```bash
# Existing deployment adoption; do not create or reconfigure its networks.
export MLOPS_DOCKER_NETWORK=kaggle-quant-collections_ui
docker compose -f compose.yaml -f docker/compose.platform.yaml up -d lab
```

The override adds the existing client network alongside Lab's own network and sets `MLFLOW_TRACKING_URI=http://mlflow:5000`. Its default network name is `mlops-platform_clients` for the shared platform. It does not attach database or object-store administration networks. Use an actual reachable endpoint and verify connectivity before importing runs.

### Shared platform identities

| Resource | Identity |
|---|---|
| Project/run tag | `home-credit` |
| MLflow experiment | `kaggle/home-credit` |
| Reserved registered model | `kaggle-home-credit` |
| Prefect flow/deployment contract | `kaggle-home-credit/local` |
| Platform artifact proxy prefix | `mlflow-artifacts:/projects/home-credit` |

The [shared MLOps platform](https://github.com/cttsai1985/mlops-platform) owns service provisioning, registry mappings and artifact-store configuration. The prefix above is a platform bootstrap contract; this project does not migrate an existing experiment or object store. Tracking uses runtime endpoint configuration and tags each import with project, logical run and available source identities. Older recorded benchmark manifests are accepted through the project's default identity without rewriting their source claims. Reserved registry/deployment names do not imply a promoted model or deployed Prefect flow.

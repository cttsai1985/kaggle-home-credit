# Run the package in Kaggle

The educational notebook is `notebooks/01-credit-risk.ipynb`. The thin hosted entry at `notebooks/kaggle/rerun.ipynb` imports the same package, constructs full applicant features, trains five folds and writes `submission.csv` plus `run-bundle.zip` under `/kaggle/working`.

## Prepare code-only assets

```bash
.venv/bin/python scripts/export_kaggle.py
```

The ignored `kaggle-bundle/code-dataset` contains a code wheel, reference configuration, license and checksums. No competition data, credentials or trained model is included. The kernel folder contains the hosted notebook and metadata. Adjust owner/slug metadata for another account.

```bash
# Account-writing commands; inspect staged files first.
.venv/bin/kaggle datasets create -p kaggle-bundle/code-dataset
.venv/bin/kaggle datasets status cttsai/home-credit-risk-code
# Wait for READY before pushing; unresolved attachments may be silently dropped.
.venv/bin/kaggle kernels push -p kaggle-bundle/kernel
.venv/bin/kaggle kernels pull cttsai/home-credit-modern-benchmark \
  -p .local/kaggle-readback -m
# Verify read-back metadata retains the intended dataset_sources and is_private=true.
.venv/bin/kaggle kernels status cttsai/home-credit-modern-benchmark
.venv/bin/kaggle kernels output cttsai/home-credit-modern-benchmark \
  -p .local/kaggle-output
```

Metadata defaults to private, CPU and internet disabled with the official competition and code dataset attached. The observed hosted image uses Python 3.13.18, NumPy 2.1.3, Pandas 2.3.3, LightGBM 4.6.0, scikit-learn 1.6.1 and PyArrow 23.0.1. The package supports Python 3.12/3.13. The hosted image must already contain compatible scientific packages; the notebook installs only the project wheel with `--no-deps`. The notebook checks the wheel checksum and installed source-file hashes against the attached code receipt, then records wheel identity in the portable run manifest. This runtime is independently validated from the local locked Docker image.

Retrieve exact notebook/output versions before comparing results. Safely extract a retrieved bundle into a new directory, retain its manifest, verify hashes and then explicitly import it into the existing tracker. The offline notebook never needs private-service credentials. Retrieval and import are separate from leaderboard submission.

A competition submission must contain `SK_ID_CURR,TARGET` in official test order. Score only verified complete predictions; keep public/private scores and the exact file hash together in the benchmark evidence. Hosted notebook publication, execution and scoring statuses are recorded separately in [results](results.md).

Metadata follows the [official Kaggle dataset schema](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md), including the `apache-2.0` license identifier.

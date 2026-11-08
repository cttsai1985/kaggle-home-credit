# Stratified validation and fold diagnostics

The default experiment uses **five StratifiedKFold partitions with seed 1001**. One feature row represents one applicant, so an applicant cannot cross the training/validation boundary within a fold. All labeled applicants receive exactly one held-out prediction. The seed is fixed before examining scores; no split-seed search is performed.

The eligible population has 307,507 applicants and 24,825 positive targets, approximately 8.07%. Stratification assigns 4,965 positives to each validation fold; fold sizes differ by at most one applicant. The historical KFold benchmark is retained separately, so changing validation policy does not silently relabel an old result.

## Recompute the diagnostics

```bash
.venv/bin/home-credit analyze --cache .local/cache --run-dir .local/stratified-baseline \
  --output .local/analyses/stratified.json
```

The analyzer reads the completed run and recorded feature cache, verifies relevant artifact checksums, and independently checks:

- Applicant identity/order, unique IDs and complete OOF coverage.
- Targets against the feature cache and predictions for finite values in [0, 1].
- Deterministic assignments against the declared splitter, fold count and seed.
- No training/validation ID overlap, class counts/rates and best iterations per fold.
- Pooled OOF AUC, mean fold AUC, sample standard deviation, minimum/maximum and the weakest fold.
- Missingness and numerical distributions for selected covariates in training, each holdout and test data.

The report links the exact model-manifest, OOF and cache hashes. It is a separate analysis artifact; running it does not rewrite completed model artifacts. Duplicate IDs, missing coverage, incorrect target identities, changed assignments and checksum mismatches fail explicitly.

## Interpret the results

Pooled OOF AUC is calculated over the entire held-out prediction vector. Mean fold AUC averages five separate ranking evaluations. AUC is nonlinear, so these values need not be identical.

Early stopping selects the checkpoint using the same fold whose AUC is reported. Its labels are excluded from fitting tree gradients but participate in model selection, so this OOF result is a validation estimate with possible selection optimism, not an untouched nested outer-test estimate. A stricter design would use an inner stopping partition and keep the outer fold independent. The fixed feature list and settings also come from prior work on this competition.

Fold standard deviation uses `ddof=1` and describes variation among these five evaluations. Because the training sets overlap, the folds are not independent experiments and their spread is not a confidence interval. A fixed seed makes assignments repeatable; it does not prove generalization to future populations.

Stratification controls target proportions. It does **not** guarantee identical feature distributions, equally difficult cases or a smaller AUC spread. Retain weaker folds and inspect their class counts and covariate context. Median, tail and missingness summaries can identify questions about distribution shift; they do not establish a cause or replace temporal/cohort evaluation.

The feature recipe still includes its declared unlabeled transductive summaries. Fold-level supervised fitting is isolated, but a strict inductive production pipeline would also fit learned preprocessing inside training folds and enforce an observation-time boundary.

See [the executed notebook](../notebooks/01-credit-risk.ipynb) for the class-balance and score-dispersion plots, and [results](results.md) for the recorded experiments.

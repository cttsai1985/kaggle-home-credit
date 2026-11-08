# Stratified cross-validation and benchmark evidence

The current baseline uses all **307,507 eligible labeled applicants**, all **48,744 test applicants**, **528 selected features** and five disjoint applicant folds. The official competition metric is ROC AUC, where higher is better.

| Experiment | OOF AUC | Public AUC | Private AUC |
|---|---:|---:|---:|
| Current stratified baseline | 0.79257 | Not submitted | Not submitted |
| Historical selected-feature LightGBM | Not retained | 0.79571 | 0.79075 |
| Historical final XGBoost stack | Not retained | 0.79682 | 0.79237 |
| Recorded KFold reference | 0.79281 | **0.79641** | **0.79065** |
| Current higher-capacity comparison | 0.79255 | Not submitted | Not submitted |

The recorded KFold reference improves the historical selected model's public score by **0.00070**, while its private score is **0.00010 lower**. This is close benchmark performance from a current implementation; it is not numerical equivalence or evidence of a general improvement. The historical final stack remains stronger on both leaderboards.

The default now uses a fixed-seed stratified split rather than the historical KFold policy. It assigns exactly **4,965 positives per fold**. Its pooled OOF AUC is **0.7925749500**, fold mean **0.7925757387**, sample standard deviation **0.0033772991**, and range **0.7878401227–0.7965282163**. The standard deviation is higher than the KFold reference's **0.0020550848**: balanced targets do not guarantee less AUC dispersion. No seed was selected to obtain a more favorable spread. See the [stratified report](cv-stratified.json) and [CV methodology](cross-validation.md).

The capacity comparison changes only the maximum leaves (20→31), minimum leaf Hessian (60→30) and runtime thread budget. Its OOF AUC is **-0.00026** below the KFold reference. It was not selected for submission. No additional hyperparameter search or leaderboard-driven tuning was performed.

## Measurement and provenance

The verified submission is Kaggle submission **58137603**, `home-credit-reference-9192529d5c79.csv`, scored on 2026-11-08. Its SHA-256 is:

```text
9192529d5c79647bf12c0f32b57803c2cb3feffe67ed890ba8cfdc0ef8ca0f43
```

Historical values come from [the retained Kaggle score records](historical-benchmarks.json). Current source/data/artifact identities and checks are retained in [the validation receipt](validation.json). Raw records, model files and prediction rows remain outside Git.

The KFold reference's complete OOF AUC is **0.7928089864**; the simple mean of fold AUCs is **0.7928202007**. These are distinct aggregations. Internal OOF validation and public/private leaderboard evaluation use different applicants and must not be substituted for one another.

Feature construction took about **30 seconds** and reference five-fold training/evaluation took **17.4 minutes**; the full stratified run took **15.9 minutes** on the observed six-thread CPU environment. These timings are environment-specific.

## Hosted execution and shared artifacts

The full reference also completed on Kaggle: Python 3.13.18, Pandas 2.3.3 and LightGBM 4.6.0 produced OOF AUC **0.7929941861**. Its raw inputs match the local fingerprints, while the execution environment differs from the local lock. All 48,744 test IDs/order and every retrieved model-bundle artifact were checked. This hosted result is not asserted numerically identical to the local run.

The offline bundle was imported into shared MLflow experiment 6, run `3ab213afcc0d41a8afe3274fa301ff45`; repeat import was idempotent and a retrieved model matched its hash. The native stratified model and content-addressed CV report are tracked in run `d056184ce0a74f14a3433c7efd5c2d44`, with verified model and report read-back. See [hosted evidence](hosted-reference.json) and [validation](validation.json).

## Verification

- Every eligible training applicant has exactly one held-out prediction and fold assignment.
- Independent checkpoint evaluation reconstructed all 307,507 OOF probabilities from the five model files and matched saved predictions.
- All 48,744 test IDs and their order match the official sample; output probabilities are finite and bounded.
- The teaching notebook executes against this real experiment and retains aggregate figures and model diagnostics.
- The real benchmark was tracked in shared MLflow experiment `kaggle/home-credit` (ID 6), run `cfc881e7401642638d17c2a754469568`. Reimport was idempotent; a model artifact retrieved through the shared proxy matched its checksum.
- Network-isolated Docker evaluation reconstructed the same complete OOF result, and Lab rejected anonymous content access while accepting its token.
- Tests cover current string categories, payment aggregation, full OOF coverage, model reload, output identity, corrupted artifacts, installed-wheel provenance and isolated MLflow artifact import.

The historical feature recipe includes unlabeled transductive summaries; it is not validated as a train-only preprocessing pipeline for future lending applications. Independent seeds, strict inductive preprocessing and point-in-time evaluation would answer additional generalization questions.

Hosted notebook execution and container checks are reported separately in the validation receipt; a successful local score does not establish the hosted runtime's behavior.

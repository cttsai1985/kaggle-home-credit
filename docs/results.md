# Five-fold LightGBM recovers the selected-model benchmark

The current baseline uses all **307,507 eligible labeled applicants**, all **48,744 test applicants**, **528 selected features** and five disjoint applicant folds. The official competition metric is ROC AUC, where higher is better.

| Experiment | OOF AUC | Public AUC | Private AUC |
|---|---:|---:|---:|
| Historical selected-feature LightGBM | Not retained | 0.79571 | 0.79075 |
| Historical final XGBoost stack | Not retained | 0.79682 | 0.79237 |
| Current LightGBM baseline | 0.79281 | **0.79641** | **0.79065** |
| Current higher-capacity comparison | 0.79255 | Not submitted | Not submitted |

The baseline improves the historical selected model's public score by **0.00070**, while its private score is **0.00010 lower**. This is close benchmark performance from a current implementation; it is not numerical equivalence or evidence of a general improvement. The historical final stack remains stronger on both leaderboards.

The capacity comparison changes only the maximum leaves (20→31), minimum leaf Hessian (60→30) and runtime thread budget. Its OOF AUC is **-0.00026** below the baseline. It was not selected for submission. No additional hyperparameter search or leaderboard-driven tuning was performed.

## Measurement and provenance

The verified submission is Kaggle submission **58137603**, `home-credit-reference-9192529d5c79.csv`, scored on 2026-11-08. Its SHA-256 is:

```text
9192529d5c79647bf12c0f32b57803c2cb3feffe67ed890ba8cfdc0ef8ca0f43
```

Historical values come from [the retained Kaggle score records](historical-benchmarks.json). Current source/data/artifact identities and checks are retained in [the validation receipt](validation.json). Raw records, model files and prediction rows remain outside Git.

The baseline's complete OOF AUC is **0.7928089864**; the simple mean of fold AUCs is **0.7928202007**. These are distinct aggregations. Internal OOF validation and public/private leaderboard evaluation use different applicants and must not be substituted for one another.

Feature construction took about **30 seconds** and reference five-fold training/evaluation took **17.4 minutes** on the observed six-thread CPU environment. These timings are environment-specific.

## Verification

- Every eligible training applicant has exactly one held-out prediction and fold assignment.
- Independent checkpoint evaluation reconstructed all 307,507 OOF probabilities from the five model files and matched saved predictions.
- All 48,744 test IDs and their order match the official sample; output probabilities are finite and bounded.
- The teaching notebook executes against this real experiment and retains aggregate figures and model diagnostics.
- Tests cover current string categories, payment aggregation, full OOF coverage, model reload, output identity, corrupted artifacts, installed-wheel provenance and isolated MLflow artifact import.

The historical feature recipe includes unlabeled transductive summaries; it is not validated as a train-only preprocessing pipeline for future lending applications. Independent seeds, strict inductive preprocessing and point-in-time evaluation would answer additional generalization questions.

Hosted notebook execution and container checks are reported separately in the validation receipt; a successful local score does not establish the hosted runtime's behavior.

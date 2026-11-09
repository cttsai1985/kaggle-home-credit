# Credit history earns its place in this fixed-budget model

This study uses a conventional competition model to examine evaluation and reproducibility decisions. Its role in the portfolio is engineering validation; the component comparison is not presented as a novel modeling contribution.

**Removing the history tables sacrifices ranking quality even when applicants, folds and training budget stay fixed.** Adding selected credit-history aggregates raises mean out-of-fold AUC from **0.746335 to 0.755201** at 100 boosting rounds. This gives a concrete reason to retain the history features in the current recipe: the simpler application-only version loses predictive information under the tested conditions.

The modeling question is whether the extra relational data contribute anything beyond the current application. Comparing the full historical pipeline with a separately tuned application model would leave several explanations open. Here the two variants use every eligible applicant, the same folds and parameters, and exactly 100 rounds without early stopping. The controlled change is inclusion of selected history features. Their paired AUC gains are **+0.009098** and **+0.008635**, larger than the observed variation between model seeds within either feature set.

The retained fold results make the conclusion more specific: **all ten matched seed/fold comparisons improve**, with gains from **+0.007815 to +0.010286**. The pooled improvement therefore is not carried by one favorable held-out partition. These are arithmetic differences from the `runs[].folds` entries in the [study receipt](modeling-study.json), not additional experiments or ten independent replications.

The bounded interpretation is that selected applicant histories contain useful ranking information for this recipe and budget. The study does not identify which history table or behavior supplies the gain: all selected relational groups enter together. Nor does it establish that the larger feature set remains preferable after full-budget tuning. The feature selection was inherited, and the cache includes unlabeled category and organization-income summaries drawn partly from test rows. Those limits prevent translating the gain into a prospective lending claim.

The decision is to retain history information in this benchmark rather than simplify away the joins on the assumption that application attributes suffice. It is not a claim that every history aggregate is necessary or that its operational cost has been optimized. A useful next question is which history groups preserve the gain; a deployment-oriented study would first need point-in-time, train-only feature construction. Neither follow-up has been performed here.

## Retained comparison

| Feature set | Features | Mean OOF AUC across 2 model seeds | Model-seed range | Sample SD across seeds |
|---|---:|---:|---:|---:|
| Selected application features | 129 | 0.746335 | 0.746278–0.746391 | 0.000080 |
| Application + selected relational features | 528 | 0.755201 | 0.755025–0.755376 | 0.000248 |

| Matched model seed | Add relational features: Δ OOF AUC |
|---|---:|
| 0 | +0.009098 |
| 1 | +0.008635 |

## Hypothesis and controlled comparison

The application contains current borrower attributes; historical tables describe prior credit and payment behavior. The hypothesis is that applicant-level history aggregates add ranking information beyond the selected application features.

The application-only set is the intersection of the existing selected-feature cache and the columns produced by `application_train_test`. The second variant adds the selected bureau, previous-application, POS, installment and credit-card aggregates already present in that cache. Thus this is **selected application features versus selected application-plus-history features**, not a claim to compare all possible raw application features. Neither feature selection nor hyperparameters were retuned for this experiment.

All **307,507 eligible labeled applicants** use identical five-fold stratified assignments (split seed 1001). Every fold has **4,965 positives**. LightGBM parameters, rows, preprocessing cache and **100 boosting rounds** are matched. There is **no early stopping** and no test prediction or submission. Each variant is evaluated at that fixed final budget, avoiding outer-fold epoch/round selection in the study.

Model seeds **0 and 1** are predeclared, including bagging, feature-sampling and data-random seeds. For each seed, only feature inclusion changes. Both models keep the same row and column sampling fractions; because the number of available features differs, the sampled subsets differ as an expected consequence of the ablation. Both seeds and all folds remain visible in the receipt.

## Variability and interpretation

The seed range and sample SD describe two stochastic training realizations on the same partitions. They are not confidence intervals or strong evidence of universal stability. Fold metrics show partition heterogeneity; overlapping fold training sets make fold SD unsuitable as an ordinary independent-observation standard error.

This **bounded fixed-100-round study is separate from the full early-stopped benchmark**. Its absolute AUC should not be substituted for that benchmark, and its relative effect may change at a larger budget. The cache retains the historical recipe’s unlabeled category vocabularies and organization-income summaries, including test rows. This controlled comparison isolates feature inclusion within that recipe; it does not establish prospective, fully inductive credit-risk validation.

## Verification and budget

The study fitted **20 fold models**, using four CPU threads in **264.0 seconds**. All four OOF files were independently reloaded and rescored; all 20 saved models were reloaded against their held-out applicants. Maximum model-versus-saved-probability delta was **0**, below the declared 1e-12 tolerance. Input, feature-group, fold, output and model hashes were checked. All raw applicants, predictions and trained weights remain private.

```bash
uv sync --locked --group dev --group lab
.venv/bin/python scripts/ablation_study.py \
  --cache .local/cache \
  --data-dir "$HOME_CREDIT_DATA_DIR" \
  --output .local/my-ablation --threads 4
```

Run from the repository root and choose a new output directory. The cache must first be built through the documented [feature workflow](reproduction.md). [The aggregate study receipt](modeling-study.json) retains actual model parameters, per-fold metrics, model/assignment fingerprints and source/runtime identities. [The executed companion notebook](../notebooks/02-controlled-ablations.ipynb) shows matched effects across seeds.

The study tests a modeling hypothesis, not a production lending policy or a leaderboard-tuning strategy. A deployment-oriented follow-up would require point-in-time features, train-only category/aggregate fitting and an appropriate temporal population; none is inferred from these competition rows.

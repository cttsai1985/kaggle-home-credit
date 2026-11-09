# Where the application ranking fails

The retained model's ranking errors are concentrated, but a hard example is not synonymous with a probability above 0.5. In the lowest-ranked **10% of payment-difficulty applications**, the median model probability is only **0.02474**. In the highest-ranked **10% of other applications**, it is **0.23484**. These are the two ends of a class-conditional ranking problem; selecting a lending cutoff is a separate decision.

This audit keeps all **307,507 eligible applications**, including **24,825 payment-difficulty labels** and **282,682 other labels**. Its independent pair accounting reproduces pooled OOF AUC **0.7925749500** exactly, and reproduces all five retained fold AUCs. No application is dropped and no score or model is changed. [Aggregate evidence](error-analysis.json) · [Full result context](results.md).

## Count the ranking errors directly

AUC compares every positive application with every negative application. A correctly ordered pair contributes one, an exact tie contributes one half, and a reversed pair contributes zero. The analyzer sorts each class's probabilities and uses binary search to count comparisons; it does not allocate a positive-by-negative pair matrix.

For a **positive** application, its burden is the fraction of negative applications scored above it, with half weight for ties. For a **negative** application, it is the fraction of positive applications scored below it, again with half weight for ties. The mean burden **within either class** equals `1 - AUC`, here **0.20742505**.

The cohort rule is fixed at the highest **10% burden within each class**, rounded up. For this binary ranking, that is the lowest-scored positives and highest-scored negatives. Exact burden ties use retained OOF row order. Class-conditional selection exposes failures on both sides without letting the majority class dominate the count.

| Retrospective cohort | Applications | Mean opposite-class misranking fraction | Share of its class view's pairwise errors |
|---|---:|---:|---:|
| Hard payment-difficulty applications | 2,483 | 0.715073 | 34.48% |
| Hard other applications | 28,269 | 0.718436 | 34.64% |

**Do not add the two shares.** They account for the same set of positive/negative pairs from two perspectives, and a pair can involve both a hard positive and a hard negative. Neither percentage is a share of financial loss. The cohorts require observed outcomes, so they cannot directly select unseen applications for review.

## The difficult cases are not confined to one fold

| Fold | OOF AUC | Hard positives | Hard negatives |
|---|---:|---:|---:|
| 1 | 0.787840 | 510 | 5,829 |
| 2 | 0.792971 | 510 | 5,437 |
| 3 | 0.796528 | 491 | 5,493 |
| 4 | 0.794686 | 469 | 5,781 |
| 5 | 0.790853 | 503 | 5,729 |

Every fold has exactly **4,965 positive applications**, and every fold contributes to both cohorts. The weakest fold has the lowest within-fold AUC, but the error profile does not support discarding it or attributing all disagreement to that partition. The global burdens and within-fold AUCs compare different pair populations and are retained separately in the receipt.

## Look at the feature conditions around disagreement

The optional profile reads only seven retained input features plus the application ID from the verified cache. It does not rematerialize the full feature table. The supplied external-score fields show a consistent association:

| Retained feature median | Hard positives | Other positives | Hard negatives | Other negatives |
|---|---:|---:|---:|---:|
| `EXT_SOURCE_1` | 0.6050 | 0.3364 | 0.2818 | 0.5403 |
| `EXT_SOURCE_2` | 0.6213 | 0.4087 | 0.3413 | 0.5888 |
| `EXT_SOURCE_3` | 0.6093 | 0.3491 | 0.2822 | 0.5656 |

The lowest-scored positives have higher values in these supplied fields than other positives; the highest-scored negatives have lower values than other negatives. This is a feature association **after selection on the model's own scores and observed class**. It is not an independent feature-importance experiment, evidence of a causal effect, or proof that an external data provider was wrong. Field definitions also do not justify inventing a customer-specific interpretation of the anonymized sources.

Missingness in the retained `DAYS_EMPLOYED` representation differs as well: **23.96%** in hard positives versus **10.72%** in remaining positives, and **7.24%** in hard negatives versus **19.79%** in remaining negatives. Missingness here does not establish employment status. The receipt includes valid-value support and missingness for every profiled feature so medians are not mistaken for full-population summaries.

## From pattern to a controlled next experiment

**Observation:** both error directions include applications whose supplied external-score features differ strongly from the remainder of their observed class. **Hypothesis:** the incremental history information that helps the overall model may contribute differently within those feature strata. The existing [fixed-budget feature study](modeling-study.md) establishes an overall history-feature gain; this audit does not establish where that gain comes from.

A useful next experiment would define external-feature strata on fit data, retain missingness as an explicit state, and compare application-only versus application-plus-history features on exactly the same predeclared folds, model seeds and boosting budget. Report the full AUC and per-stratum support/performance, plus calibration if probabilities will support a decision policy. Keep all folds, including weak and sparse strata. Do not optimize a new model against these already-selected hard cohorts and describe the result as untouched validation.

A prospective lending assessment additionally needs point-in-time, train-only feature construction and later application cohorts. The current competition cache includes inherited transductive processing; the retrospective ranking audit does not establish an approval policy, future default rate or commercial benefit. [Problem and value](problem-and-value.md) explains the institutional validation needed beyond the competition objective.

## Reproduce the audit

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv/bin/python scripts/analyze_hard_cases.py \
  --run-dir .local/stratified-baseline \
  --applications "$HOME_CREDIT_DATA_DIR/application_train.csv" \
  --cache .local/features-reference \
  --output .local/home-credit-hard-cases.json
```

Use a new output path. The script verifies the retained OOF/metric files, official application source and optional cache hashes. It checks unique IDs, the declared eligibility rule, exact source labels and fold coverage before recomputing the headline and fold metrics. Tests cover independently known AUC examples, half-weighted ties, perfect/reversed rankings, deterministic class-cohort selection and identity failures.

Only aggregate diagnostics are published. Application IDs, request features, individual predictions and the full feature cache remain private. [Analyzer](../scripts/analyze_hard_cases.py) · [Tests](../tests/test_hard_cases.py).

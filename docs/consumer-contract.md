# Home Credit: the MLOps consumer contract

The solution owns the modeling question, validation method and acceptance criteria. It consumes shared execution, tracking and storage services. Its scientific claims do not become stronger merely because a run appears in MLflow.

## Why the applicant is a data contract

A row is not automatically a training observation. This snapshot has 307,507 eligible applicants, but 227,415 have multiple bureau records and 238,520 have multiple previous applications. Directly left-joining both histories would create **8,091,067 rows**, averaging **26.31 rows per applicant**. Some applicants would receive much more implicit weight than others.

The effect is visible without fitting a model: positive prevalence is **8.07% when each applicant counts once**, but **8.87% when each hypothetical joined row counts once**. The labels have not changed; their weights have. This explains why aggregation and row-grain assertions protect the learning objective, rather than merely making a join faster.

The maintained pipeline aggregates histories before joining and preserves one applicant per feature row. At delivery, the same identity contract requires every expected test applicant exactly once in the correct order. A finite probability array alone cannot establish a correct handoff. The audit describes a counterfactual naive join, not a production incident or a measured effect on model AUC. [Aggregate calculation and exact source hashes](data-insight.json).

The calculation avoids materializing the expanded join. For each eligible applicant, let `b` be its bureau-row count and `p` its previous-application count. A pair of left joins creates `max(b, 1) × max(p, 1)` rows. Sum this multiplicity to obtain the joined row count; weight each applicant's target by it to obtain the joined-row positive fraction. Eligibility excludes the same four `CODE_GENDER=XNA` rows as the selected-feature recipe. The source tables and formula make this descriptive check reproducible.

## From scientific evidence to an executable job

| Boundary | This solution supplies | What it expects from shared services |
|---|---|---|
| Inputs | Authorized data, schema/ID checks and exact input hashes | Read-only input access and explicit identity on execution |
| Experiment | Fixed configuration, folds, seeds, preprocessing and budget | An immutable environment/image reference and recorded task state |
| Evaluation | OOF/holdout analyzer, retained weak folds, variance and ablation evidence | Durable results and logs for both favorable and unfavorable experiments |
| Delivery | Verified checkpoint and ordered batch outputs | Artifact transfer with checksummed readback; explicit failure/recovery |
| Troubleshooting | A portable local run bundle and project-specific validation errors | Stage/flow correlation, owner-visible status and safe retry semantics |

The canonical tracking experiment is `kaggle/home-credit`. The platform provisions it; this client does not silently create a different namespace. The run retains its own logical identity and source/input/model fingerprints, including when it originates in an offline Kaggle notebook.

## Failure behavior matters to research

A transient tracker outage must not require retraining an expensive completed experiment. `track --resume` recovers the same logical/remote run only after the operator confirms the earlier importer has stopped. A conflicting manifest or duplicate logical identity fails closed. Artifact delivery is marked finished only after byte-hash readback. [Project-specific commands and recovery details](reproduction.md).

A poor model remains an experiment worth recording. A delivery gate may reject a candidate without hiding its metrics. Acceptance thresholds must be declared for the intended use and evaluated on the right population; a convenient competition holdout threshold is not production approval.

## Evidence and limits

The [modeling study](modeling-study.md) tests component contributions and descriptive variability. The [shared batch case study](https://github.com/cttsai1985/mlops-platform/blob/main/docs/batch-operations.md) demonstrates an Iceberg saved-model workflow with real inputs, task logs, delivery gating, failure and recovery. It does not imply every historical experiment or every solution ran under that scheduler.

Current CI and recovery tests exercise the client contracts. The [capability assessment](https://github.com/cttsai1985/mlops-platform/blob/main/docs/mlops-assessment.md) distinguishes these controls from live service monitoring, automatic retraining, model promotion and disaster recovery. This repository makes no claim to operate those production capabilities.

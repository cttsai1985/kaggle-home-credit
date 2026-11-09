# Home Credit requirements audit

Review date: **2026-11-09** · Acceptance standard: [Portfolio standard 1.0](portfolio-standard.md).

This audit covers the current `fix/portfolio-evidence` correction, the recorded
canary and retained modeling/evaluation evidence. **Evidenced** identifies a scoped
implementation or completed check; **release gate** identifies final CI/merge/
publication work for this correction. **Open P1** items remain accepted work.
Nothing in this matrix declares a deployed production lending system.

## The regression and its correction

The editorial change retained scientific result files but weakened result visibility:

| Surface | Before the editorial change | Prior canary presentation | Current correction |
|---|---|---|---|
| README ranking paragraph | Line **5** at `e601bc7` | Line **72** at `0f4ecb0`, after model/validation detail | A primary **Results** section immediately after the brief opening and before model/reproduction/infrastructure detail |
| GitHub description | Score/ranking context was a portfolio requirement | Replaced by a problem-only summary without score, reference position/N or Top % | Live description pairs historical and current public results with reference counts and Top % |
| Secondary historical blend | Its score was already reported in the historical benchmark material | The public-score percentile receipt did not include its comparison | The existing **0.79764** blend now has an explicit **2,126–2,127/7,180, Top 29.61%** comparison |

README line movement is verified from Git blobs. GitHub descriptions are external
metadata; Git alone does not reconstruct their editing history. The restored live
value is captured in [public-profile.json](public-profile.json). It displays the
historical blend **0.79764, 2,126–2,127/7,180, Top 29.61%** and current KFold
reference **0.79641, 2,396/7,180, Top 33.37%**. The primary README table also keeps
the selected-feature model **0.79571, 2,699/7,180, Top 37.59%** and final stack
**0.79682, 2,272–2,275/7,180, Top 31.64%**. The common public snapshot is
**2026-11-09T00:02:26Z**. [Reference receipt](leaderboard-context.json).

These are public **score-equivalent reference positions**, not awarded final ranks.
The blend row is a new comparison of an already recorded score, not a new submission
or a new fitted model. Internal OOF/holdout ranks are not applicable, and private
rank is unavailable without a matched private reference population. The current
stratified default's OOF AUC is not assigned the KFold submission's public rank.

## Requirement-by-requirement record

| ID | Priority | Status / applicability | Evidence and remaining acceptance |
|---|---|---|---|
| R01 | P0 | Evidenced; final portfolio review remains a release responsibility | Every R01–R26 requirement in [standard 1.0](portfolio-standard.md) is mapped here. Naming, serving or business prose cannot silently replace evaluation, ranking or production requirements. |
| R02 | P0 | Evidenced — complete solution | [README](../README.md) connects repayment-risk assessment with application-level aggregation, five-fold LightGBM, controlled feature evidence, validation and runnable prediction/export paths. |
| R03 | P0 | Restored; source-backed guard added | The first Results section visibly pairs all four public scores with reference rank/N, Top % and ties. [Public profile](public-profile.json) restores historical/current description entries. [Portfolio check](../scripts/check_portfolio.py) protects this display against the reference receipt. The regression is recorded above. |
| R04 | P0 | Evidenced — metric and population scope | [Results](results.md) distinguishes current stratified OOF, KFold reference, capacity comparison, hosted runs and historical public/private AUCs. Public comparisons carry metric direction, matched population, dated snapshot and tie intervals in [leaderboard context](leaderboard-context.json). |
| R05 | P0 | Evidenced — no transferred rank | README/results explicitly distinguish score-equivalent positions from awarded rank, public/private populations, authenticated historical scores from source-reproduced models, and OOF from leaderboard evaluation. No private or OOF percentile is invented. |
| R06 | P0 | Evidenced — controlled feature insight | [Modeling study](modeling-study.md) holds eligible applications, folds, seeds and 100-round budget fixed: selected application/history features raise mean OOF AUC **0.746335 → 0.755201**, with all ten paired seed/fold comparisons retained. It explains the decision and untested table-specific/longitudinal follow-ups. |
| R07 | P0 | Evidenced within retained OOF; mechanisms remain hypotheses | [Error analysis](error-analysis.md) links engineered feature conditions, five-fold ranking and observed failures using exact pairwise burden. It profiles supported external-score/missingness associations and proposes a fit-defined stratified history comparison. It does not claim causal feature explanations. |
| R08 | P0 | Evidenced with important exceptions | [CV diagnostics](cross-validation.md), [stratified receipt](cv-stratified.json), [validation](validation.json) and [results](results.md) preserve exact IDs/folds, all-row OOF, balanced positives, weak folds and independent reload/metric checks. Early stopping uses those validation folds; the inherited cache includes transductive processing. Neither exception is hidden as prospective train-only validation. |
| R09 | P0 | Evidenced — ranking audit only | [Hard-case analyzer](../scripts/analyze_hard_cases.py) keeps all **307,507** eligible applications and all five folds, reconciles unchanged AUC and fixes the top 10% rule within each observed class. It does not tune a cutoff, discard hard applications or call descriptive feature differences causal. |
| R10 | P0 | Evidenced — evaluated saved ensemble | [Serving guide](serving.md), [adapter](../src/home_credit/serving.py) and [canary receipt](canary-validation.json) bind all five native boosters, 528 ordered features, producer/payload hashes, prediction policy and model environment. No refit is needed to reload/predict. |
| R11 | P0 | Evidenced for the recorded real-model canary | Native ensemble → exported package → fresh-workspace HTTP inference was checked on **8 real-input rows**, maximum absolute error **0**, required-field rejection **HTTP 400**, package integrity and cleanup. [Receipt](canary-validation.json). This is an interface/parity sample, not a new full-data quality or serving-load result. |
| R12 | P0 | Evidenced — engineered-feature boundary | The endpoint accepts **528 engineered numeric application features**, with JSON null for missing values, and averages saved fold predictions. [Feature assembly](serving.md#feature-assembly-remains-part-of-the-solution) remains an explicit upstream workflow; the endpoint is not a raw relational-data or approved lending-policy service. |
| R13 | P0 | Evidenced — maintained execution paths | [Reproduction](reproduction.md), feature/train/analyze/predict CLI, Docker/Jupyter/dev-container files, executed notebooks and offline Kaggle entry remain available. [Hosted receipts](hosted-stratified.json) distinguish actual runtime results and input/fold identity from native numerical equivalence. |
| R14 | P0 | Evidenced — one-way client integration | [Consumer contract](consumer-contract.md) leaves experiment/model/flow/artifact identities with this repo and consumes generic [ML infrastructure](https://github.com/cttsai1985/ml-infrastructure) interfaces. Adding the solution does not require a platform-owned project catalog. |
| R15 | P0 | Delegated to shared infrastructure; client handoff reviewed | Protected deployment state, lightweight lifecycle CLI and operator status/logs belong to shared infrastructure. This solution supplies its model runtime/export path explicitly and does not need another deployment manager or tracking-server model dependencies. This audit does not rerun platform state/volume tests. |
| R16 | P0 | Evidenced for the correction/canary scope | Client identity `kaggle/home-credit` and recorded artifact routes are preserved. The real-model canary reports no modification of shared services; current changes are presentation, diagnostics and metadata, not an adoption of running volumes/ports/credentials. No new live-service inventory is claimed here. |
| R17 | P0 | Evidenced for public outputs; release inspection gate | [Error receipt](error-analysis.json) publishes aggregate class/fold/feature summaries and hashes, without IDs or row-level features/predictions. Canary private assets remain unpublished. Model bundles, request rows, caches and logs stay outside Git; release assets still need final inspection. |
| R18 | P0 | Scoped prior evidence retained; current-source release gate | [Canary receipt](canary-validation.json) binds its clean adapter/controller commits, lock, package and runtime hashes. New analyzer hashes match current source and retained OOF/input files. The correction package is **0.2.0.dev2**; final CI/source/lock/checksum evidence must distinguish it from the older HTTP check instead of claiming an unperformed rerun. |
| R19 | P0 | Recorded prerelease exists; correction publication is a release gate | [v0.2.0-canary.1](https://github.com/cttsai1985/kaggle-home-credit-solution/releases/tag/v0.2.0-canary.1) is a prerelease targeting `0f4ecb0`. The correction canary must identify its own source/lock/new diagnostics. Canary does not mean a live traffic split; other solution/infrastructure release records have separate audits. |
| R20 | P0 | Evidenced — solution-first naming and attribution | Canonical repo is `kaggle-home-credit-solution`; README opening leads with the problem, model and measured history-feature decision. Original Aguiar/Grellier attribution, [legacy source](../legacy/README.md) and [Apache-2.0 license](../LICENSE) remain. |
| R21 | P0 | Evidenced — primary-source problem/value explanation | [Problem and value](problem-and-value.md) uses the official competition/data description and sponsor context. It separates observed discrimination from a proposed institutional approval/loss tradeoff, and identifies point-in-time, calibration/policy and future-cohort evidence still needed. No customer ROI is claimed. |
| R22 | P1 | **Open P1 — production target retained** | Real-model package/HTTP usability is implemented, while production research/batch and online operation remain accepted goals under [shared requirements](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/requirements.md). The target is not reduced to demo-only to close open work. |
| R23 | P1 | **Open P1 — platform and model operations** | Hosting/ownership, service and model objectives, measured capacity, TLS/authentication, monitoring/alert delivery, off-host backup/restore and production rollout/rollback require deployment evidence. The canary explicitly excludes production ingress, serving-image and zero-downtime validation. |
| R24 | P1 | Client recovery evidenced; production batch operation **open P1** | [Consumer contract](consumer-contract.md), tracker tests in [test_pipeline.py](../tests/test_pipeline.py) and shared-store receipts cover idempotent import/conflict handling and verified artifacts. The platform's saved-model fault drill used Iceberg; it is not a scheduled Home Credit production-batch claim. |
| R25 | P1 | Current mining implemented; hypothesis validation **open P1** | [Error analysis](error-analysis.md) exactly reconciles full OOF/fold AUC, half-weighted ties and both class views, then reports declared hard cohorts and supported feature associations. The fit-defined feature-strata/history comparison and prospective point-in-time validation have not been run. |
| R26 | P0 | Canonical URLs reviewed; final release/branch hygiene gate | [Public profile](public-profile.json) records the live `*-solution` URL. Current package/export/migration links are updated without rewriting historical receipts. Keep `fix/portfolio-evidence` until merge/checks; delete only verified merged branches and preserve unmerged work/active worktrees. |

## Scientific preservation: exact comparison scope

The inspected comparisons are:

1. **Before the editorial/serving canary:**
   `e601bc7c7af0854a144cab148deb07121c3459b4`.
2. **Published prior-canary merge:**
   `0f4ecb0ba2099e243032db9389b3b354c932c3d4`.
3. **Current correction:** the working tree based on that prior-canary merge,
   reviewed before its final correction commit/release.

`git diff --name-status` is empty for this path scope in both
before→canary and canary→current comparisons:

- `configs/`, `notebooks/`, `legacy/` and `LICENSE`;
- `src/home_credit/features.py`, `pipeline.py`, `cli.py`, `tracking.py`,
  `import_support.py` and `feature_exclusions.json`;
- `docs/modeling-study.md`, `docs/modeling-study.json`, `docs/data-insight.json`,
  `docs/validation.json`, `docs/cv-stratified.json`, `docs/cv-reference.json`,
  `docs/cv-capacity.json`, `docs/historical-benchmarks.json`,
  `docs/hosted-reference.json` and `docs/hosted-stratified.json`.

The canary added `src/home_credit/serving.py` and changed package identity metadata
in `src/home_credit/__init__.py`; those changes are outside the preserved training
scope. The current correction leaves the serving adapter unchanged.
`docs/canary-validation.json` remains byte-identical to `0f4ecb0`.

There are intentional evidence-presentation additions:

- `docs/leaderboard-context.json` appends the existing historical blend's public
  comparison; earlier comparison entries and the dated source population remain.
- `docs/results.md` adds the same blend comparison and links to error analysis/audit.
- New error-analysis files retain their own script/source/OOF/cache hashes and
  exactly reconciled headline/fold metrics. They do not replace the old feature
  study, source scoring records or canary.

README, consumer-contract wording, CI/guard, canonical URL and package/lock version
changes are reviewed separately from the protected scientific scope. This is not a
blanket statement that nothing else changed or that every past revision was audited.

## Explicit open work and release gates

- **Production operation — R22/R23/R24:** select accountable owners and the actual
  service/workload targets, then verify access/TLS, capacity, observability/alerts,
  off-host backup/restore and controlled rollout/recovery. Model parity on a local
  HTTP sample does not close these accepted obligations.
- **Scientific follow-up — R25:** test history-feature contributions within strata
  defined on fit data, with matched folds/seeds/budget and retained weak strata.
  A prospective evaluation needs point-in-time/train-only features and later
  cohorts. Current post-hoc associations do not establish a lending rule or cause.
- **Correction publication — R18/R19/R26:** pass complete current-source CI and the
  portfolio guard, verify live metadata and the full public Results comparison,
  publish the correction canary/source/lock/checksums, then clean only verified
  merged branches. Keep historical evidence tied to its actual producer/source.

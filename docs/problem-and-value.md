# Using credit history to assess repayment risk

Home Credit's 2018 competition connected risk prediction with its stated aim of serving borrowers who have limited credit histories. [Official competition brief](https://www.kaggle.com/c/home-credit-default-risk). Its contemporary corporate description also emphasizes this segment. [2018 company statement, page 2](https://www.homecredit.net/wp-content/uploads/2022/11/181008-hcgbv-press-release-airbank-hc-moneta-english-v2.pdf).

A lender's business problem is a tradeoff: expanding approvals to people able to repay while controlling the losses and customer harm from unsuitable lending. A model can contribute a ranking signal, but a lending decision also needs the institution's policy, calibrated risk, product economics and required review. The sponsor's ambition is not an outcome demonstrated by this repository.

## Historical information has measurable predictive value

The fixed-budget comparison holds eligible applications, folds and LightGBM parameters constant. Selected application features achieve mean OOF AUC **0.746335** across two model seeds; adding the selected relational features raises it to **0.755201**. All ten matched seed/fold comparisons improve, with gains from **0.007815 to 0.010286**. [Controlled study](modeling-study.md).

This supports retaining those history features at the tested 100-round budget. It does not isolate the value of each table or establish how much credit loss an institution would avoid. Full-budget early-stopped results are reported separately: the current stratified baseline reaches **0.7925749500 pooled OOF AUC**. [Results](results.md).

## Preserve one application per decision

The official files describe a main loan-application row plus histories at credit, application, month and installment/payment grains. [Data description](https://www.kaggle.com/c/home-credit-default-risk/data?select=application_train.csv).

The local audit calculates that directly joining bureau and previous-application rows would expand **307,507 eligible applications to 8,091,067 rows**. The same underlying applications would receive unequal training weights; target prevalence would move from **8.07% per application to 8.87% per joined row**. The maintained pipeline aggregates first. This is a measured counterfactual data-grain audit, not an observed lending incident or a fitted causal score effect. [Audit receipt](data-insight.json).

The practical insight is that data integration changes what a model learns. Preserving the unit on which the lender actually decides is part of modeling, not just a database optimization.

## AUC is the model metric, not the business result

The competition evaluates ROC AUC. [Official evaluation](https://www.kaggle.com/c/home-credit-default-risk). AUC assesses discrimination across the labeled population. It does not select an approval cutoff, calibrate a probability of default, measure expected loss or prove equitable outcomes.

For an institutional evaluation, useful decision metrics include approval rate at an agreed observed loss/arrears budget, calibrated risk by relevant segment, and stability over later cohorts. These are proposed validation measures; no live lending data or approval-policy experiment is part of this repository.

A later Home Credit competition explicitly made temporal stability part of its objective. [2024 official brief](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/tracks-and-awards). That reinforces why lifecycle evaluation matters, but the later data and metric are separate from the 2018 benchmark used here.

## What must change before a real lending pilot

The retained competition recipe includes transductive vocabulary and organization-income summaries using unlabeled test rows. A prospective workflow needs point-in-time histories, transformations fitted only on available training data, and evaluation on later applications. Data available after the decision must not enter that request.

The smallest credible pilot is a shadow scoring exercise on a time-separated, access-controlled institutional cohort. Agree the application cutoff, feature provenance, delayed outcome window and relevant segments; compare a fixed incumbent policy/model with the candidate on the same population. Inspect calibration, discrimination, stability, and a policy-specific approval/loss tradeoff before using scores to change approvals. Measure request latency and feature freshness separately.

The value proposition remains structural until the institution supplies approval volumes, product/loan economics, loss definitions, current performance and policy constraints. No approval uplift, loss reduction or financial return is claimed.

## Model-to-service handoff

The solution owns history aggregation, ordered engineered features, fitted boosters and the accepted ensemble. A serving canary can validate prediction equivalence on a declared engineered-feature request contract; that does not demonstrate an end-to-end raw application/credit-history service unless feature assembly is implemented and tested too.

The shared infrastructure can host the model and observe its operational behavior. Model validity still depends on the client's point-in-time feature and evaluation contracts. This division keeps the commercial story attached to the actual decision rather than to infrastructure installation.

## Source scope

Sponsor objectives, measured competition results and proposed institutional validation are distinct evidence classes. This is an independently developed competition solution, not a Home Credit engagement or a validated lending policy.

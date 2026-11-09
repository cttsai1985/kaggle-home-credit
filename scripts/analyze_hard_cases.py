"""Describe retained Home Credit OOF ranking errors without fitting or changing AUC."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

COHORT_FRACTION = 0.10
PROFILE_FEATURES = (
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pairwise_burden(target, probability):
    """Exact binary AUC and each class member's opposite-class misranking fraction.

    A tie contributes half a misranking. Sorting/search avoids an n_pos*n_neg
    matrix. The mean burden WITHIN EITHER class equals 1-AUC; do not add them.
    """
    target, probability = np.asarray(target), np.asarray(probability, dtype=float)
    if target.ndim != 1 or target.shape != probability.shape or set(np.unique(target)) != {0, 1}:
        raise ValueError("Expected aligned binary labels with both classes")
    if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    positive, negative = target == 1, target == 0
    positives, negatives = np.sort(probability[positive]), np.sort(probability[negative])
    lower = np.searchsorted(negatives, probability[positive], side="left")
    upper = np.searchsorted(negatives, probability[positive], side="right")
    positive_errors = len(negatives) - upper + (upper - lower) / 2
    lower = np.searchsorted(positives, probability[negative], side="left")
    upper = np.searchsorted(positives, probability[negative], side="right")
    negative_errors = lower + (upper - lower) / 2
    burden = np.empty(len(target), dtype=float)
    burden[positive] = positive_errors / len(negatives)
    burden[negative] = negative_errors / len(positives)
    # Twice the error count stays integral, including half-weighted ties.
    auc = 1 - float(positive_errors.sum() / (len(positives) * len(negatives)))
    if abs(float(burden[negative].mean()) - (1 - auc)) > 1e-12:
        raise ValueError("Positive and negative pairwise accounting disagree")
    return auc, burden


def hardest_by_class(target, burden):
    """Top ceil(10% * class size) per class, with saved OOF row order for ties."""
    selected = np.zeros(len(target), dtype=bool)
    for label in [0, 1]:
        indices = np.flatnonzero(np.asarray(target) == label)
        count = int(np.ceil(COHORT_FRACTION * len(indices)))
        chosen = indices[np.argsort(-np.asarray(burden)[indices], kind="stable")[:count]]
        selected[chosen] = True
    return selected


def align_source(oof, applications):
    for name, frame in [("OOF", oof), ("applications", applications)]:
        if (
            "SK_ID_CURR" not in frame
            or frame.SK_ID_CURR.isna().any()
            or not frame.SK_ID_CURR.is_unique
        ):
            raise ValueError(f"{name} IDs must be unique and nonmissing")
    eligible = applications.loc[applications.CODE_GENDER.ne("XNA")].copy()
    if set(oof.SK_ID_CURR) != set(eligible.SK_ID_CURR):
        raise ValueError("OOF population differs from the declared eligible application IDs")
    aligned = eligible.set_index("SK_ID_CURR").loc[oof.SK_ID_CURR]
    if not np.array_equal(oof.TARGET.to_numpy(), aligned.TARGET.to_numpy()):
        raise ValueError("OOF labels differ from the exact source applications")
    return aligned


def quantiles(values):
    return dict(
        zip(
            ["p10", "p25", "median", "p75", "p90"],
            [float(value) for value in np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])],
            strict=True,
        )
    )


def feature_profile(features, selected):
    result = {}
    for name in PROFILE_FEATURES:
        if name not in features:
            continue
        values = pd.to_numeric(features[name], errors="raise").to_numpy(dtype=float)
        finite = np.isfinite(values)
        cohorts = {}
        for cohort, mask in [("hard", selected), ("remaining", ~selected)]:
            valid = values[mask & finite]
            cohorts[cohort] = {
                "rows_with_value": len(valid),
                "missing_fraction": float((~finite[mask]).mean()),
                "median": float(np.median(valid)) if len(valid) else None,
            }
        result[name] = cohorts
    return result


def analyze(run_dir, applications_path, cache=None):
    run_dir, applications_path = Path(run_dir), Path(applications_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Analyze only a completed model run")
    for name in ["oof.csv", "metrics.json", "fold-metrics.json"]:
        if sha256(run_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"Retained artifact hash differs: {name}")
    expected_input = manifest["features"]["input_sha256"]["application_train.csv"]
    if sha256(applications_path) != expected_input:
        raise ValueError("Application source differs from the producer snapshot")
    applications = pd.read_csv(applications_path, usecols=["SK_ID_CURR", "TARGET", "CODE_GENDER"])
    oof = pd.read_csv(run_dir / "oof.csv")
    if list(oof.columns) != ["SK_ID_CURR", "TARGET", "probability", "fold"]:
        raise ValueError("Unexpected retained OOF schema")
    align_source(oof, applications)
    folds = manifest["config"]["folds"]
    if oof.fold.isna().any() or set(oof.fold) != set(range(1, folds + 1)):
        raise ValueError("Invalid retained fold coverage")
    target, probability = oof.TARGET.to_numpy(), oof.probability.to_numpy()
    auc, burden = pairwise_burden(target, probability)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    retained_auc = float(metrics["oof_roc_auc"])
    if abs(auc - retained_auc) > 1e-12:
        raise ValueError("Recomputed full-population AUC differs from retained evidence")
    hard = hardest_by_class(target, burden)
    classes = {}
    for label, name in [(1, "payment_difficulty"), (0, "other_applications")]:
        selected = target == label
        hard_class = selected & hard
        total = burden[selected].sum()
        classes[name] = {
            "label": label,
            "rows": int(selected.sum()),
            "mean_pairwise_misranking_fraction": float(burden[selected].mean()),
            "probability_quantiles": quantiles(probability[selected]),
            "burden_quantiles": quantiles(burden[selected]),
            "hard_cohort_rows": int(hard_class.sum()),
            "actual_class_fraction": float(hard_class.sum() / selected.sum()),
            "hard_cohort_probability_quantiles": quantiles(probability[hard_class]),
            "hard_cohort_mean_burden": float(burden[hard_class].mean()),
            "class_pairwise_error_share_in_hard_cohort": float(burden[hard_class].sum() / total)
            if total
            else None,
        }
    retained_folds = json.loads((run_dir / "fold-metrics.json").read_text())
    expected_fold_auc = {row["fold"]: row["auc"] for row in retained_folds}
    if len(expected_fold_auc) != folds or set(expected_fold_auc) != set(range(1, folds + 1)):
        raise ValueError("Retained fold metrics are incomplete or duplicated")
    fold_rows = []
    for fold in range(1, folds + 1):
        selected = oof.fold.eq(fold).to_numpy()
        fold_auc, _ = pairwise_burden(target[selected], probability[selected])
        if abs(fold_auc - expected_fold_auc[fold]) > 1e-12:
            raise ValueError("Recomputed fold AUC differs from retained evidence")
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(selected.sum()),
                "positives": int(target[selected].sum()),
                "auc": fold_auc,
                "global_burden_mean_by_class": {
                    str(label): float(burden[selected & (target == label)].mean())
                    for label in [0, 1]
                },
                "hard_cohort_rows_by_class": {
                    str(label): int((selected & hard & (target == label)).sum()) for label in [0, 1]
                },
            }
        )
    profile, cache_hash = {}, None
    if cache is not None:
        cache = Path(cache)
        cache_hash = sha256(cache / "selected.parquet")
        if cache_hash != manifest["features"]["cache_sha256"]:
            raise ValueError("Feature cache differs from the producer")
        # Read only seven supplied features plus identity; no full feature-table materialization.
        import pyarrow
        import pyarrow.parquet as pq

        pyarrow.set_cpu_count(1)
        available = set(pq.ParquetFile(cache / "selected.parquet").schema.names)
        columns = ["SK_ID_CURR", *[name for name in PROFILE_FEATURES if name in available]]
        frame = pd.read_parquet(cache / "selected.parquet", columns=columns)
        if frame.SK_ID_CURR.isna().any() or not frame.SK_ID_CURR.is_unique:
            raise ValueError("Feature-cache IDs must be unique and nonmissing")
        if not set(oof.SK_ID_CURR).issubset(set(frame.SK_ID_CURR)):
            raise ValueError("Feature cache does not cover the retained application population")
        features = frame.set_index("SK_ID_CURR").loc[oof.SK_ID_CURR]
        for label, name in [(1, "payment_difficulty"), (0, "other_applications")]:
            selected = target == label
            profile[name] = feature_profile(features.loc[selected], hard[selected])
    return {
        "schema_version": 1,
        "analyzed_at": datetime.now(UTC).isoformat(),
        "scope": "Complete retained stratified OOF population; no fitting, filtering or submission",
        "source_sha256": {
            "analysis_script": sha256(__file__),
            "producer_manifest": sha256(manifest_path),
            "application_train.csv": sha256(applications_path),
            "oof.csv": sha256(run_dir / "oof.csv"),
            "metrics.json": sha256(run_dir / "metrics.json"),
            "profile_feature_cache": cache_hash,
        },
        "population": {
            "rows": len(oof),
            "positives": int(target.sum()),
            "negatives": int((target == 0).sum()),
            "eligibility": "source applications excluding CODE_GENDER=XNA; retained recipe",
            "exact_application_ids_and_labels_verified": True,
            "all_folds_retained": folds,
        },
        "headline": {
            "roc_auc": auc,
            "retained_roc_auc": retained_auc,
            "absolute_metric_delta": abs(auc - retained_auc),
        },
        "diagnostic": {
            "name": "opposite-class misranking fraction per application; half-weighted ties",
            "mean_within_either_class_equals": "1 - full-population AUC",
            "cohort_fraction_declared_within_each_class": COHORT_FRACTION,
            "tie_break": "retained OOF file row order; stable descending burden sort",
            "positive_and_negative_error_shares_must_not_be_added": True,
            "hard_cohorts_require_observed_labels_and_are_not_unseen_case_selectors": True,
        },
        "classes": classes,
        "folds": fold_rows,
        "selected_feature_associations": profile,
        "limitations": [
            "Ranking diagnostics do not establish lending cutoffs, calibration or causal effects.",
            "Both class-specific views describe the same errors; their shares are not additive.",
            "Competition OOF does not establish future lending performance.",
            "All rows and headline AUC are retained; no hard application is removed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--applications", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; retained evidence is not overwritten")
    receipt = analyze(args.run_dir, args.applications, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "rows": receipt["population"]["rows"],
                "roc_auc": receipt["headline"]["roc_auc"],
                "hard_positive_rows": receipt["classes"]["payment_difficulty"]["hard_cohort_rows"],
                "hard_negative_rows": receipt["classes"]["other_applications"]["hard_cohort_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()

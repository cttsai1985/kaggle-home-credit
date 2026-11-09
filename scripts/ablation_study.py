"""Controlled application-only versus relational features on fixed applicant folds."""

import argparse
import gc
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from home_credit.features import application_train_test
from home_credit.pipeline import (
    FEATURE_EXCLUDE,
    fold_assignments,
    read_features,
    sha256,
    source_identity,
    write_json,
)


def run(cache, data_dir, output, threads):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frame, cache_receipt = read_features(cache)
    applications = application_train_test(data_dir)
    application_columns = set(applications.columns) - FEATURE_EXCLUDE
    del applications
    gc.collect()
    train = frame.loc[frame.TARGET.notna()].reset_index(drop=True)
    del frame
    target = train.TARGET.astype("uint8").to_numpy()
    full = [name for name in train if name not in FEATURE_EXCLUDE]
    application = [name for name in full if name in application_columns]
    if not application or not set(application) < set(full):
        raise ValueError("Application ablation must be a nonempty strict feature subset")
    variants = {"application_only": application, "application_plus_relational": full}
    config = json.loads(Path("configs/baseline.json").read_text())
    config.update(threads=threads, max_rounds=100, early_stopping_rounds=None)
    assignments = fold_assignments(target, config)
    assignment_path = output / "fold-assignments.csv"
    train[["SK_ID_CURR"]].assign(fold=assignments).to_csv(assignment_path, index=False)
    write_json(output / "config.json", config)
    write_json(output / "feature-groups.json", variants)
    receipt = {
        "schema_version": 1,
        "project_id": "home-credit",
        "status": "running",
        "study": "fixed-budget application-only versus selected relational features",
        "source": source_identity(),
        "script_sha256": sha256(Path(__file__)),
        "features": cache_receipt,
        "assignment_sha256": sha256(assignment_path),
        "rows": len(train),
        "positive_rows": int(target.sum()),
        "config": config,
        "model_seeds": [0, 1],
        "feature_counts": {name: len(columns) for name, columns in variants.items()},
        "feature_group_sha256": sha256(output / "feature-groups.json"),
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow")
            },
        },
        "interpretation": {
            "controlled_change": (
                "Only selected relational feature inclusion changes within each seed. "
                "All eligible rows, fold assignments, LightGBM parameters and fixed "
                "100-round budget are matched. No early stopping or scored test data."
            ),
            "variance": (
                "Two predeclared model seeds on identical folds measure descriptive "
                "training-seed sensitivity, not a confidence interval."
            ),
            "scope": (
                "Bounded fixed-budget ablation, not the full early-stopped benchmark. "
                "The inherited cache includes unlabeled transductive summaries; "
                "this does not establish prospective train-only credit validation."
            ),
        },
        "runs": [],
    }
    write_json(output / "manifest.json", receipt)
    for seed in receipt["model_seeds"]:
        for variant, columns in variants.items():
            run_started = time.monotonic()
            x = train[columns]
            oof = np.full(len(train), np.nan)
            rows = []
            params = {
                **config["parameters"],
                "seed": seed,
                "bagging_seed": seed,
                "feature_fraction_seed": seed,
                "data_random_seed": seed,
                "num_threads": threads,
                "deterministic": True,
                "force_col_wise": True,
            }
            for fold in range(1, config["folds"] + 1):
                fit = np.flatnonzero(assignments != fold)
                valid = np.flatnonzero(assignments == fold)
                dataset = lgb.Dataset(
                    x.iloc[fit],
                    label=target[fit],
                    feature_name=[f"feature_{i:04d}" for i in range(len(columns))],
                )
                model = lgb.train(params, dataset, num_boost_round=100)
                prediction = model.predict(x.iloc[valid], num_threads=threads)
                oof[valid] = prediction
                model_path = output / f"model-{seed}-{variant}-fold-{fold}.txt"
                model.save_model(str(model_path))
                rows.append(
                    {
                        "fold": fold,
                        "validation_rows": len(valid),
                        "validation_positives": int(target[valid].sum()),
                        "roc_auc": float(roc_auc_score(target[valid], prediction)),
                        "iterations": model.current_iteration(),
                        "model_sha256": sha256(model_path),
                    }
                )
                print(json.dumps({"seed": seed, "variant": variant, **rows[-1]}), flush=True)
                del model, dataset
                gc.collect()
            if not np.isfinite(oof).all():
                raise ValueError("Incomplete OOF predictions")
            path = output / f"oof-{seed}-{variant}.csv"
            train[["SK_ID_CURR", "TARGET"]].assign(fold=assignments, probability=oof).to_csv(
                path, index=False
            )
            scores = [row["roc_auc"] for row in rows]
            result = {
                "model_seed": seed,
                "variant": variant,
                "features": len(columns),
                "pooled_oof_roc_auc": float(roc_auc_score(target, oof)),
                "fold_mean": float(np.mean(scores)),
                "fold_sample_sd": float(np.std(scores, ddof=1)),
                "folds": rows,
                "oof_sha256": sha256(path),
                "parameters": params,
                "elapsed_seconds": time.monotonic() - run_started,
            }
            receipt["runs"].append(result)
            write_json(output / "manifest.json", receipt)
            print(
                json.dumps(
                    {
                        "completed": variant,
                        "seed": seed,
                        "auc": result["pooled_oof_roc_auc"],
                        "seconds": result["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    summaries = {}
    for variant in variants:
        scores = [r["pooled_oof_roc_auc"] for r in receipt["runs"] if r["variant"] == variant]
        summaries[variant] = {
            "mean": float(np.mean(scores)),
            "sample_sd": float(np.std(scores, ddof=1)),
            "min": min(scores),
            "max": max(scores),
            "seeds": len(scores),
        }
    paired = []
    for seed in receipt["model_seeds"]:
        lookup = {r["variant"]: r for r in receipt["runs"] if r["model_seed"] == seed}
        paired.append(
            {
                "model_seed": seed,
                "add_relational_auc_delta": lookup["application_plus_relational"][
                    "pooled_oof_roc_auc"
                ]
                - lookup["application_only"]["pooled_oof_roc_auc"],
            }
        )
    receipt.update(
        status="completed",
        summaries=summaries,
        paired_deltas=paired,
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "manifest.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    run(args.cache, args.data_dir, args.output, args.threads)

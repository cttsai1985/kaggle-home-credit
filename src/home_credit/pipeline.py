"""Cross-validated LightGBM with applicant-level predictions and portable artifacts."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from home_credit.features import build_features

LEGACY_COMMIT = "a378d5f"
FEATURE_EXCLUDE = {"TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_identity() -> dict:
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    source_tree = package.parent.name == "src" and (root / "pyproject.toml").is_file()
    commit, dirty = None, None
    if source_tree and shutil.which("git"):
        identity = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        state = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True
        )
        if identity.returncode == 0:
            commit = identity.stdout.strip()
            dirty = bool(state.stdout.strip()) if state.returncode == 0 else None
    return {
        "legacy_commit": LEGACY_COMMIT,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_kind": "source-tree" if source_tree else "installed-package",
        "package_version": importlib.metadata.version("home-credit-risk"),
        "source_files": {
            "src/home_credit/" + str(p.relative_to(package)): sha256(p)
            for p in sorted(package.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json"}
        },
    }


def prepare(data_dir: Path, output: Path, *, selected=True, num_rows=None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frame = build_features(data_dir, selected=selected, num_rows=num_rows)
    frame.to_parquet(output / "selected.parquet", index=False)
    manifest = {
        "schema_version": 1,
        "recipe": "historical-selected" if selected else "historical-all-features",
        "source": source_identity(),
        "rows": len(frame),
        "columns": len(frame.columns),
        "train_rows": int(frame.TARGET.notna().sum()),
        "test_rows": int(frame.TARGET.isna().sum()),
        "row_limit": num_rows,
        "feature_source_sha256": sha256(Path(__file__).with_name("features.py")),
        "input_sha256": {p.name: sha256(p) for p in sorted(data_dir.glob("*.csv"))},
        "cache_sha256": sha256(output / "selected.parquet"),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def read_features(cache: Path) -> tuple[pd.DataFrame, dict]:
    receipt = json.loads((cache / "manifest.json").read_text())
    if sha256(cache / "selected.parquet") != receipt["cache_sha256"]:
        raise ValueError("Feature cache checksum mismatch")
    frame = pd.read_parquet(cache / "selected.parquet")
    if frame.SK_ID_CURR.isna().any() or frame.SK_ID_CURR.duplicated().any():
        raise ValueError("Expected one feature row per unique applicant ID")
    labeled = frame.TARGET.dropna()
    if set(labeled.unique()) != {0, 1}:
        raise ValueError("Training requires both binary target classes")
    return frame, receipt


def fold_assignments(targets, config: dict) -> np.ndarray:
    """One deterministic validation fold per applicant; never search split seeds."""
    targets = np.asarray(targets)
    folds = config["folds"]
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("At least two folds are required")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("Both binary target classes are required")
    if min(np.bincount(targets.astype(int), minlength=2)) < folds:
        raise ValueError("Each target class needs at least one applicant per fold")
    splitter_type = StratifiedKFold if config["stratified"] else KFold
    splitter = splitter_type(n_splits=folds, shuffle=True, random_state=config["split_seed"])
    result = np.zeros(len(targets), dtype=np.int16)
    for number, (_, valid) in enumerate(splitter.split(np.zeros(len(targets)), targets), 1):
        if (result[valid] != 0).any():
            raise ValueError("Validation assignments overlap")
        result[valid] = number
    if (result == 0).any():
        raise ValueError("Validation assignment coverage is incomplete")
    return result


def cv_report(frame: pd.DataFrame, oof: pd.DataFrame, config: dict, best_iterations=None) -> dict:
    """Independently inspect OOF identity, split, class balance and score dispersion."""
    train = frame.loc[frame.TARGET.notna()].reset_index(drop=True)
    test = frame.loc[frame.TARGET.isna()].reset_index(drop=True)
    if train.SK_ID_CURR.duplicated().any() or oof.SK_ID_CURR.duplicated().any():
        raise ValueError("Duplicate applicant identities in CV inputs")
    if oof.SK_ID_CURR.tolist() != train.SK_ID_CURR.tolist():
        raise ValueError("OOF applicant identities/order or coverage differ")
    targets = train.TARGET.astype(int).to_numpy()
    if not np.array_equal(oof.TARGET.to_numpy(), targets):
        raise ValueError("OOF targets differ from the feature-cache targets")
    probabilities = oof.probability.to_numpy()
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Invalid OOF probabilities")
    expected = fold_assignments(targets, config)
    if not np.array_equal(expected, fold_assignments(targets, config)):
        raise ValueError("Configured split is not repeatable")
    if not np.array_equal(oof.fold.to_numpy(), expected):
        raise ValueError("OOF assignments do not match the configured split")
    ids = train.SK_ID_CURR.to_numpy()
    global_rate = float(targets.mean())
    rows = []
    for number in range(1, config["folds"] + 1):
        valid = expected == number
        fit = ~valid
        overlap = len(set(ids[fit]) & set(ids[valid]))
        if overlap:
            raise ValueError("Training and validation applicant IDs overlap")
        positives = int(targets[valid].sum())
        rows.append(
            {
                "fold": number,
                "training_rows": int(fit.sum()),
                "validation_rows": int(valid.sum()),
                "positive_rows": positives,
                "negative_rows": int(valid.sum()) - positives,
                "positive_rate": float(targets[valid].mean()),
                "positive_rate_delta_pp": float((targets[valid].mean() - global_rate) * 100),
                "roc_auc": float(roc_auc_score(targets[valid], probabilities[valid])),
                "best_iteration": (best_iterations or {}).get(number),
                "id_overlap": overlap,
            }
        )
    scores = np.array([row["roc_auc"] for row in rows])
    candidates = ["NEW_EXT_SOURCES_MEAN", "NEW_CREDIT_TO_ANNUITY_RATIO", "DAYS_BIRTH"]
    covariates = [name for name in candidates if name in frame]
    if not covariates:
        covariates = [
            name
            for name in frame
            if name not in FEATURE_EXCLUDE and pd.api.types.is_numeric_dtype(frame[name].dtype)
        ][:3]

    def summary(values):
        observed = values.dropna()
        observed = observed[np.isfinite(observed)]
        return {
            "rows": len(values),
            "observed_rows": len(observed),
            "missing_fraction": float(1 - len(observed) / len(values)) if len(values) else None,
            "mean": float(observed.mean()) if len(observed) else None,
            "median": float(observed.median()) if len(observed) else None,
            "p10": float(observed.quantile(0.1)) if len(observed) else None,
            "p90": float(observed.quantile(0.9)) if len(observed) else None,
        }

    distributions = {
        name: {
            "training": summary(train[name]),
            "test": summary(test[name]),
            "validation_folds": {
                str(i): summary(train.loc[expected == i, name])
                for i in range(1, config["folds"] + 1)
            },
        }
        for name in covariates
    }
    assignment_bytes = (
        pd.DataFrame({"SK_ID_CURR": ids, "fold": expected}).to_csv(index=False).encode()
    )
    weakest = rows[int(np.argmin(scores))]
    return {
        "schema_version": 1,
        "project_id": "home-credit",
        "splitter": "StratifiedKFold" if config["stratified"] else "KFold",
        "split_seed": config["split_seed"],
        "folds": config["folds"],
        "assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "assignment_repeatable": True,
        "eligible_rows": len(train),
        "oof_rows": len(oof),
        "oof_coverage": 1.0,
        "id_overlap": 0,
        "positive_rows": int(targets.sum()),
        "positive_rate": global_rate,
        "pooled_oof_roc_auc": float(roc_auc_score(targets, probabilities)),
        "fold_auc_mean": float(scores.mean()),
        "fold_auc_std": float(scores.std(ddof=1)),
        "fold_auc_min": float(scores.min()),
        "fold_auc_max": float(scores.max()),
        "fold_metrics": rows,
        "weakest_fold": weakest,
        "covariate_distributions": distributions,
        "interpretation": "Fold dispersion is descriptive, not a confidence interval; "
        "folds share training data. "
        "Covariate summaries diagnose possible shifts "
        "without proving their cause. "
        "Feature construction retains the declared transductive boundary.",
    }


def analyze(cache: Path, run_dir: Path, output: Path | None = None) -> dict:
    """Write a post-training diagnostic without mutating the completed model run."""
    frame, feature_receipt = read_features(cache)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Analyze a completed model run")
    if feature_receipt["cache_sha256"] != manifest["features"]["cache_sha256"]:
        raise ValueError("Analysis requires the recorded feature cache")
    for name in ["oof.csv", "config.json", "fold-metrics.json"]:
        if sha256(run_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"Analysis artifact checksum mismatch: {name}")
    oof = pd.read_csv(run_dir / "oof.csv")
    config = json.loads((run_dir / "config.json").read_text())
    iterations = {
        row["fold"]: row["best_iteration"]
        for row in json.loads((run_dir / "fold-metrics.json").read_text())
    }
    report = cv_report(frame, oof, config, iterations)
    report["provenance"] = {
        "model_manifest_sha256": sha256(run_dir / "manifest.json"),
        "oof_sha256": sha256(run_dir / "oof.csv"),
        "cache_sha256": feature_receipt["cache_sha256"],
        "logical_run_id": manifest["logical_run_id"],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    return report


def fit_cv(cache: Path, run_dir: Path, config: dict) -> dict:
    """Fit disjoint applicant folds and average their complete test predictions."""
    if config["threads"] < 1 or config["folds"] < 2:
        raise ValueError("Positive thread count and at least two folds required")
    frame, cache_receipt = read_features(cache)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "project_id": "home-credit",
        "run_kind": config.get("run_kind", "real_data_model_run"),
        "logical_run_id": str(uuid.uuid4()),
        "status": "running",
        "profile": config["name"],
        "source": source_identity(),
        "config": config,
        "features": cache_receipt,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow"]
            },
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    try:
        train = frame.loc[frame.TARGET.notna()].reset_index(drop=True)
        test = frame.loc[frame.TARGET.isna()].reset_index(drop=True)
        features = [name for name in frame if name not in FEATURE_EXCLUDE]
        write_json(run_dir / "features.json", features)
        write_json(run_dir / "config.json", config)
        x, y = train[features], train.TARGET.astype("uint8").to_numpy()
        configured_assignments = fold_assignments(y, config)
        oof = np.full(len(train), np.nan)
        test_probability = np.zeros(len(test))
        observed_folds = np.full(len(train), -1, dtype=int)
        scores, importance = [], []
        params = {
            **config["parameters"],
            "num_threads": config["threads"],
            "deterministic": True,
            "force_col_wise": True,
        }
        for number in range(1, config["folds"] + 1):
            fit = np.flatnonzero(configured_assignments != number)
            valid = np.flatnonzero(configured_assignments == number)
            assert not set(fit) & set(valid)
            train_set = lgb.Dataset(
                x.iloc[fit],
                label=y[fit],
                free_raw_data=True,
                feature_name=[f"feature_{i:04d}" for i in range(len(features))],
            )
            valid_set = lgb.Dataset(
                x.iloc[valid],
                label=y[valid],
                reference=train_set,
                feature_name=[f"feature_{i:04d}" for i in range(len(features))],
            )
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=config["max_rounds"],
                valid_sets=[valid_set],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(config["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(period=200),
                ],
            )
            booster.save_model(str(run_dir / f"fold-{number}.txt"))
            oof[valid] = booster.predict(x.iloc[valid], num_threads=config["threads"])
            if len(test):
                test_probability += (
                    booster.predict(test[features], num_threads=config["threads"]) / config["folds"]
                )
            observed_folds[valid] = number
            scores.append(
                {
                    "fold": number,
                    "auc": float(roc_auc_score(y[valid], oof[valid])),
                    "best_iteration": booster.best_iteration,
                    "train_rows": len(fit),
                    "validation_rows": len(valid),
                }
            )
            importance.append(
                pd.DataFrame(
                    {
                        "feature": features,
                        "gain": booster.feature_importance("gain"),
                        "fold": number,
                    }
                )
            )
            write_json(run_dir / "fold-metrics.json", scores)
            print(json.dumps(scores[-1]), flush=True)
        if not np.isfinite(oof).all() or not (observed_folds > 0).all():
            raise ValueError("OOF coverage incomplete")
        oof_frame = pd.DataFrame(
            {
                "SK_ID_CURR": train.SK_ID_CURR,
                "TARGET": y,
                "probability": oof,
                "fold": observed_folds,
            }
        )
        oof_frame.to_csv(run_dir / "oof.csv", index=False)
        submission = pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR, "TARGET": test_probability})
        validate_submission(submission, test.SK_ID_CURR)
        submission.to_csv(run_dir / "submission.csv", index=False)
        pd.concat(importance, ignore_index=True).to_csv(run_dir / "importance.csv", index=False)
        metrics = {
            "oof_roc_auc": float(roc_auc_score(y, oof)),
            "fold_auc_mean": float(np.mean([r["auc"] for r in scores])),
            "train_rows": len(train),
            "test_rows": len(test),
            "features": len(features),
        }
        write_json(run_dir / "metrics.json", metrics)
        manifest.update(
            status="completed",
            elapsed_seconds=time.monotonic() - started,
            metrics=metrics,
            artifacts={
                p.name: sha256(p)
                for p in sorted(run_dir.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        )
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(run_dir / "manifest.json", manifest)
        raise


def validate_submission(submission: pd.DataFrame, expected_ids) -> None:
    if list(submission.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Unexpected submission columns")
    if submission.SK_ID_CURR.tolist() != list(expected_ids):
        raise ValueError("Submission ID identity or ordering differs")
    values = submission.TARGET.to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Submission probabilities must be finite and within [0, 1]")


def evaluate(cache: Path, run_dir: Path) -> dict:
    """Reconstruct all saved OOF predictions from the independent model files."""
    frame, receipt = read_features(cache)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if receipt["cache_sha256"] != manifest["features"]["cache_sha256"]:
        raise ValueError("Evaluation requires the original feature cache")
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or (run_dir / name).is_symlink():
            raise ValueError("Unsafe artifact manifest")
        if sha256(run_dir / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    train = frame.loc[frame.TARGET.notna()].reset_index(drop=True)
    oof = pd.read_csv(run_dir / "oof.csv")
    if train.SK_ID_CURR.tolist() != oof.SK_ID_CURR.tolist():
        raise ValueError("Saved OOF applicant ordering differs")
    features = json.loads((run_dir / "features.json").read_text())
    reconstructed = np.full(len(train), np.nan)
    for fold in range(1, manifest["config"]["folds"] + 1):
        selected = oof.fold == fold
        model = lgb.Booster(model_file=str(run_dir / f"fold-{fold}.txt"))
        reconstructed[selected] = model.predict(
            train.loc[selected, features], num_threads=manifest["config"]["threads"]
        )
    np.testing.assert_allclose(reconstructed, oof.probability, atol=1e-14, rtol=0)
    return {
        "oof_roc_auc": float(roc_auc_score(train.TARGET, reconstructed)),
        "rows": len(train),
        "saved_predictions_match": True,
    }


def predict(cache: Path, run_dir: Path, output: Path) -> dict:
    frame, receipt = read_features(cache)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if receipt["cache_sha256"] != manifest["features"]["cache_sha256"]:
        raise ValueError("Prediction requires the recorded feature cache")
    test = frame.loc[frame.TARGET.isna()]
    features = json.loads((run_dir / "features.json").read_text())
    probabilities = np.zeros(len(test))
    for fold in range(1, manifest["config"]["folds"] + 1):
        model_path = run_dir / f"fold-{fold}.txt"
        if sha256(model_path) != manifest["artifacts"][model_path.name]:
            raise ValueError("Model checksum mismatch")
        model = lgb.Booster(model_file=str(model_path))
        probabilities += (
            model.predict(test[features], num_threads=manifest["config"]["threads"])
            / manifest["config"]["folds"]
        )
    result = pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR, "TARGET": probabilities})
    validate_submission(result, test.SK_ID_CURR)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    receipt = {
        "rows": len(result),
        "output_sha256": sha256(output),
        "manifest_sha256": sha256(run_dir / "manifest.json"),
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt

"""Cross-validated LightGBM with applicant-level predictions and portable artifacts."""

import hashlib
import importlib.metadata
import json
import platform
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
    root = Path(__file__).resolve().parents[2]
    identity = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True
    )
    return {
        "legacy_commit": LEGACY_COMMIT,
        "git_commit": identity.stdout.strip() if identity.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()),
        "source_files": {
            str(p.relative_to(root)): sha256(p)
            for p in sorted((root / "src").rglob("*"))
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


def fit_cv(cache: Path, run_dir: Path, config: dict) -> dict:
    """Fit disjoint applicant folds and average their complete test predictions."""
    if config["threads"] < 1 or config["folds"] < 2:
        raise ValueError("Positive thread count and at least two folds required")
    frame, cache_receipt = read_features(cache)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
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
        splitter_type = StratifiedKFold if config["stratified"] else KFold
        splitter = splitter_type(
            n_splits=config["folds"], shuffle=True, random_state=config["split_seed"]
        )
        oof = np.full(len(train), np.nan)
        test_probability = np.zeros(len(test))
        fold_assignments = np.full(len(train), -1, dtype=int)
        scores, importance = [], []
        params = {
            **config["parameters"],
            "num_threads": config["threads"],
            "deterministic": True,
            "force_col_wise": True,
        }
        for number, (fit, valid) in enumerate(splitter.split(x, y), start=1):
            assert not set(fit) & set(valid)
            train_set = lgb.Dataset(x.iloc[fit], label=y[fit], free_raw_data=True)
            valid_set = lgb.Dataset(x.iloc[valid], label=y[valid], reference=train_set)
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
            fold_assignments[valid] = number
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
        if not np.isfinite(oof).all() or not (fold_assignments > 0).all():
            raise ValueError("OOF coverage incomplete")
        oof_frame = pd.DataFrame(
            {
                "SK_ID_CURR": train.SK_ID_CURR,
                "TARGET": y,
                "probability": oof,
                "fold": fold_assignments,
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

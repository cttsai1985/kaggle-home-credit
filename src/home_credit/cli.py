"""Explicit data, feature, training and evidence commands."""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from home_credit.pipeline import evaluate, fit_cv, predict, prepare, sha256, write_json

COMPETITION = "home-credit-default-risk"


def download(data_dir: Path) -> dict:
    executable = Path(sys.executable).with_name("kaggle")
    if not executable.is_file():
        raise RuntimeError(
            "Install the data dependency group and authenticate with kaggle auth login"
        )
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN"):
        auth = subprocess.run(
            [str(executable), "auth", "print-access-token"], capture_output=True, text=True
        )
        if auth.returncode or not auth.stdout.strip() or "\n" in auth.stdout.strip():
            raise RuntimeError("Authenticate with kaggle auth login")
        env["KAGGLE_API_TOKEN"] = auth.stdout.strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / (COMPETITION + ".zip")
    if not archive_path.is_file():
        result = subprocess.run(
            [str(executable), "competitions", "download", COMPETITION, "-p", str(data_dir), "-q"],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("Kaggle download failed; check authentication and accepted rules")
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or (info.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe archive member")
        # Never replace an edited existing payload silently.
        for info in archive.infolist():
            if not (data_dir / info.filename).exists():
                archive.extract(info, data_dir)
    result = {
        "competition": COMPETITION,
        "archive_sha256": sha256(archive_path),
        "input_sha256": {p.name: sha256(p) for p in sorted(data_dir.glob("*.csv"))},
    }
    write_json(data_dir / "download-manifest.json", result)
    return result


def validate(data_dir: Path) -> dict:
    train = pd.read_csv(data_dir / "application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
    test = pd.read_csv(data_dir / "application_test.csv", usecols=["SK_ID_CURR"])
    if train.SK_ID_CURR.duplicated().any() or test.SK_ID_CURR.duplicated().any():
        raise ValueError("Duplicate application identifiers")
    if set(train.SK_ID_CURR) & set(test.SK_ID_CURR):
        raise ValueError("Training and test applicant identities overlap")
    if train.TARGET.isna().any() or set(train.TARGET.unique()) != {0, 1}:
        raise ValueError("Invalid binary targets")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if sample.SK_ID_CURR.tolist() != test.SK_ID_CURR.tolist():
        raise ValueError("Official test and sample submission ordering differs")
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "target_rate": float(train.TARGET.mean()),
        "sample_id_order_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "validate"]:
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, required=True)
    sub = commands.add_parser("features")
    sub.add_argument("--data-dir", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--all-features", action="store_true")
    sub.add_argument("--row-limit", type=int)
    sub = commands.add_parser("train")
    sub.add_argument("--cache", type=Path, required=True)
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--config", type=Path, default=Path("configs/reference.json"))
    for name in ["evaluate", "predict"]:
        sub = commands.add_parser(name)
        sub.add_argument("--cache", type=Path, required=True)
        sub.add_argument("--run-dir", type=Path, required=True)
        if name == "predict":
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("track")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    if args.command == "download":
        result = download(args.data_dir)
    elif args.command == "validate":
        result = validate(args.data_dir)
    elif args.command == "features":
        result = prepare(
            args.data_dir, args.output, selected=not args.all_features, num_rows=args.row_limit
        )
    elif args.command == "train":
        result = fit_cv(args.cache, args.run_dir, json.loads(args.config.read_text()))
    elif args.command == "evaluate":
        result = evaluate(args.cache, args.run_dir)
    elif args.command == "predict":
        result = predict(args.cache, args.run_dir, args.output)
    else:
        from home_credit.tracking import track

        if not args.tracking_uri:
            parser.error("Set MLFLOW_TRACKING_URI or pass --tracking-uri")
        result = track(args.run_dir, args.tracking_uri)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

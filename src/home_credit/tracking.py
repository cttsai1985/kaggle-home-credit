"""Import a completed local or retrieved Kaggle run into an existing MLflow server."""

import hashlib
import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from home_credit.pipeline import sha256, write_json


def track(run_dir: Path, tracking_uri: str) -> dict:
    import mlflow

    if urlparse(tracking_uri).username:
        raise ValueError("Pass credentials through the runtime, not the tracking URI")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only completed runs may be imported")
    project_id = manifest.get("project_id", "home-credit")
    if project_id != "home-credit":
        raise ValueError("This client imports only home-credit runs")
    allowed = {
        "config.json",
        "features.json",
        "fold-metrics.json",
        "importance.csv",
        "metrics.json",
        "oof.csv",
        "submission.csv",
        *(f"fold-{i}.txt" for i in range(1, manifest["config"]["folds"] + 1)),
    }
    if set(manifest["artifacts"]) != allowed:
        raise ValueError("Unexpected artifact manifest")
    logical_id = str(uuid.UUID(manifest["logical_run_id"]))
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or (run_dir / name).is_symlink():
            raise ValueError("Unsafe artifact manifest")
        if sha256(run_dir / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("kaggle/home-credit")
    if experiment is None:
        raise RuntimeError(
            "Experiment kaggle/home-credit is missing; bootstrap it through mlops-platform "
            "before importing runs"
        )
    digest = sha256(run_dir / "manifest.json")
    previous = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.logical_run_id = '{logical_id}'",
        output_format="list",
    )
    if previous:
        if (
            previous[0].info.status != "FINISHED"
            or previous[0].data.tags.get("manifest_sha256") != digest
        ):
            raise ValueError("Conflicting or incomplete run import")
        return {"status": "already_imported", "mlflow_run_id": previous[0].info.run_id}
    source = manifest.get("source", {})
    asset_source = manifest.get("execution", {}).get("code_asset", {}).get("source", {})
    tags = {
        "project_id": project_id,
        "run_kind": manifest.get(
            "run_kind",
            "real_data_model_run"
            if manifest.get("features", {}).get("input_sha256")
            else "unclassified_run",
        ),
        "logical_run_id": logical_id,
        "manifest_sha256": digest,
    }
    source_commit = source.get("git_commit") or asset_source.get("git_commit")
    if source_commit:
        tags["source_commit"] = source_commit
    if source.get("source_files"):
        tags["source_sha256"] = hashlib.sha256(
            json.dumps(source["source_files"], sort_keys=True).encode()
        ).hexdigest()
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=manifest["profile"], tags=tags
    ) as run:
        config = manifest["config"]
        mlflow.log_params({k: v for k, v in config.items() if k != "parameters"})
        mlflow.log_params(config["parameters"])
        mlflow.log_metrics(manifest["metrics"])
        for name in [*manifest["artifacts"], "manifest.json"]:
            mlflow.log_artifact(str(run_dir / name))
        result = {"status": "imported", "mlflow_run_id": run.info.run_id}
    write_json(run_dir / "tracking.json", result)
    return result

"""Import a completed local or retrieved Kaggle run into an existing MLflow server."""

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
    logical_id = str(uuid.UUID(manifest["logical_run_id"]))
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or (run_dir / name).is_symlink():
            raise ValueError("Unsafe artifact manifest")
        if sha256(run_dir / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment("kaggle/home-credit-default-risk")
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
    with mlflow.start_run(
        run_name=manifest["profile"], tags={"logical_run_id": logical_id, "manifest_sha256": digest}
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

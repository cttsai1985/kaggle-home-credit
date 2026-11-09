"""Serve the evaluated LightGBM ensemble on ordered, engineered application features.

Feature construction from relational histories remains a client workflow. This
adapter deliberately does not describe a raw application or credit-history API.
"""

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _verify_payload(path: Path) -> dict:
    contract = json.loads((path / "serving-contract.json").read_text())
    if contract.get("schema_version") != 1 or contract.get("request_kind") != "engineered_features":
        raise ValueError("Unsupported Home Credit serving contract")
    if not isinstance(contract.get("folds"), int) or not 1 <= contract["folds"] <= 100:
        raise ValueError("Invalid saved ensemble size")
    expected = {"features.json", "producer-manifest.json"}
    expected.update(f"fold-{fold}.txt" for fold in range(1, contract["folds"] + 1))
    if set(contract.get("artifacts", {})) != expected:
        raise ValueError("Incomplete or unexpected serving artifact set")
    for name, digest in contract["artifacts"].items():
        item = path / name
        if Path(name).name != name or item.is_symlink() or not item.is_file():
            raise ValueError("Unsafe serving artifact")
        if _sha256(item) != digest:
            raise ValueError(f"Serving artifact checksum mismatch: {name}")
    features = json.loads((path / "features.json").read_text())
    if not features or any(not isinstance(name, str) or not name for name in features):
        raise ValueError("Invalid engineered feature names")
    if len(set(features)) != len(features) or len(features) != contract["feature_count"]:
        raise ValueError("Duplicate or inconsistent engineered features")
    producer_path = path / "producer-manifest.json"
    if _sha256(producer_path) != contract["producer_manifest_sha256"]:
        raise ValueError("Producer identity differs from the serving contract")
    producer = json.loads(producer_path.read_text())
    if producer.get("logical_run_id") != contract["producer_logical_run_id"]:
        raise ValueError("Producer logical identity differs from the serving contract")
    if producer.get("status") != "completed" or producer["config"]["folds"] != contract["folds"]:
        raise ValueError("Serving requires the declared completed producer")
    for name in expected - {"producer-manifest.json"}:
        if contract["artifacts"][name] != producer["artifacts"].get(name):
            raise ValueError("Serving payload differs from the producer")
    return {**contract, "features": features}


class HomeCreditEnsemble(mlflow.pyfunc.PythonModel):
    """Mean prediction of all retained folds; preprocessing is already materialized."""

    def load_context(self, context):
        path = Path(context.artifacts["ensemble"])
        self.contract = _verify_payload(path)
        self.features = self.contract["features"]
        self.models = [
            lgb.Booster(model_file=str(path / f"fold-{fold}.txt"))
            for fold in range(1, self.contract["folds"] + 1)
        ]
        if any(model.num_feature() != len(self.features) for model in self.models):
            raise ValueError("Saved model feature count differs from the request contract")

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame) or model_input.columns.has_duplicates:
            raise ValueError("Provide a DataFrame with unique engineered feature columns")
        if set(model_input.columns) != set(self.features):
            raise ValueError("Request must contain exactly the saved engineered features")
        if any(not pd.api.types.is_numeric_dtype(model_input[name]) for name in self.features):
            raise ValueError("Engineered features must be numeric; missing values are allowed")
        values = model_input.loc[:, self.features].astype("float64")
        if np.isinf(values.to_numpy()).any():
            raise ValueError("Engineered features must not contain infinity")
        if len(values) == 0:
            raise ValueError("Provide at least one application feature row")
        probabilities = np.zeros(len(values))
        for model in self.models:
            probabilities += model.predict(values, num_threads=2) / len(self.models)
        if (
            not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
        ):
            raise ValueError("Ensemble output is not a bounded repayment-difficulty probability")
        return pd.DataFrame({"TARGET": probabilities}, index=model_input.index)


def export_model(run_dir: Path, output: Path, example: pd.DataFrame) -> dict:
    """Export immutable MLflow packaging and check parity; do not fit any model."""
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    outputs = [
        output,
        *[
            output.with_name(output.name + suffix)
            for suffix in [".receipt.json", ".request.json", ".expected.json"]
        ],
    ]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError("Serving output already exists; choose a new version directory")
    manifest_path = run_dir / "manifest.json"
    producer = json.loads(manifest_path.read_text())
    if producer.get("status") != "completed":
        raise ValueError("Export requires a completed model run")
    if not isinstance(producer.get("config", {}).get("folds"), int):
        raise ValueError("Producer does not declare an ensemble")
    for name, digest in producer["artifacts"].items():
        item = run_dir / name
        if Path(name).name != name or item.is_symlink() or not item.is_file():
            raise ValueError("Unsafe producer artifact")
        if _sha256(item) != digest:
            raise ValueError(f"Producer artifact checksum mismatch: {name}")
    features = json.loads((run_dir / "features.json").read_text())
    if not isinstance(example, pd.DataFrame) or set(example.columns) != set(features):
        raise ValueError("Input example must contain the exact engineered feature set")
    example = example.loc[:, features].astype("float64")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".home-credit-export-", dir=output.parent) as directory:
        payload = Path(directory) / "ensemble"
        payload.mkdir()
        shutil.copy2(manifest_path, payload / "producer-manifest.json")
        shutil.copy2(run_dir / "features.json", payload / "features.json")
        for fold in range(1, producer["config"]["folds"] + 1):
            name = f"fold-{fold}.txt"
            shutil.copy2(run_dir / name, payload / name)
        contract = {
            "schema_version": 1,
            "project_id": "home-credit",
            "request_kind": "engineered_features",
            "raw_relational_feature_assembly_included": False,
            "feature_count": len(features),
            "folds": producer["config"]["folds"],
            "producer_manifest_sha256": _sha256(manifest_path),
            "producer_logical_run_id": producer.get("logical_run_id"),
            "prediction": "equal mean of all saved fold boosters",
            "threads_per_prediction": 2,
            "artifacts": {p.name: _sha256(p) for p in sorted(payload.iterdir())},
        }
        _write_json(payload / "serving-contract.json", contract)
        adapter = HomeCreditEnsemble()
        adapter.load_context(type("Context", (), {"artifacts": {"ensemble": str(payload)}})())
        expected = adapter.predict(None, example)
        signature = ModelSignature(
            inputs=Schema([ColSpec(DataType.double, name) for name in features]),
            outputs=Schema([ColSpec(DataType.double, "TARGET")]),
        )
        requirements = [
            f"{name}=={importlib.metadata.version(name)}"
            for name in ["mlflow", "lightgbm", "numpy", "pandas", "cloudpickle"]
        ]
        candidate = Path(directory) / "model"
        mlflow.pyfunc.save_model(
            path=str(candidate),
            python_model=HomeCreditEnsemble(),
            artifacts={"ensemble": str(payload)},
            signature=signature,
            input_example=example,
            code_paths=[str(Path(__file__).resolve().parent)],
            pip_requirements=requirements,
        )
        reloaded = mlflow.pyfunc.load_model(str(candidate))
        actual = reloaded.predict(example)
        delta = float(np.max(np.abs(actual.TARGET.to_numpy() - expected.TARGET.to_numpy())))
        if delta > 1e-12:
            raise ValueError("Packaged predictions differ from the saved ensemble")
        if output.exists() or output.is_symlink():
            raise ValueError("Serving output appeared during export; refusing to overwrite it")
        candidate.rename(output)
    receipt = {
        "schema_version": 1,
        "status": "verified_packaged_model",
        "request_kind": "engineered_features",
        "producer_manifest_sha256": _sha256(manifest_path),
        "folds_packaged": producer["config"]["folds"],
        "feature_count": len(features),
        "parity_rows": len(example),
        "maximum_prediction_delta": delta,
        "tolerance": 1e-12,
        "mlflow_version": importlib.metadata.version("mlflow"),
        "real_serving_endpoint_verified": False,
        "raw_relational_feature_assembly_included": False,
    }
    request = {
        "dataframe_split": {
            "columns": features,
            "data": example.astype(object).where(example.notna(), None).values.tolist(),
        }
    }
    _write_json(output.with_name(output.name + ".request.json"), request)
    _write_json(
        output.with_name(output.name + ".expected.json"),
        {"predictions": [{"TARGET": float(value)} for value in expected.TARGET]},
    )
    _write_json(output.with_name(output.name + ".receipt.json"), receipt)
    return receipt

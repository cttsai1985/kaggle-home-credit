"""Serving packages preserve the saved ensemble and engineered-feature contract."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

from home_credit.serving import HomeCreditEnsemble, _sha256, export_model


@pytest.fixture
def producer(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    values = np.linspace(-2, 2, 40)
    example = pd.DataFrame({"amount_ratio": values, "previous_late": (values > 0).astype(float)})
    features = list(example.columns)
    (run / "features.json").write_text(json.dumps(features))
    models = []
    for fold in [1, 2]:
        model = lgb.train(
            {
                "objective": "binary",
                "verbosity": -1,
                "num_threads": 1,
                "min_data_in_leaf": 2,
                "seed": fold,
            },
            lgb.Dataset(example, label=(values > 0).astype(int)),
            num_boost_round=3,
        )
        model.save_model(str(run / f"fold-{fold}.txt"))
        models.append(model)
    manifest = {
        "status": "completed",
        "run_kind": "infrastructure_fixture",
        "logical_run_id": "serving-test-fixture",
        "config": {"folds": 2},
        "artifacts": {p.name: _sha256(p) for p in run.iterdir()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    sample = example.iloc[:4].copy()
    sample.iloc[0, 0] = np.nan
    return run, sample, models


def test_export_preserves_ensemble_and_bundle_survives_fresh_process(producer, tmp_path):
    run, example, models = producer
    output = tmp_path / "model"
    receipt = export_model(run, output, example)
    assert receipt["folds_packaged"] == 2
    assert receipt["maximum_prediction_delta"] == 0
    assert receipt["raw_relational_feature_assembly_included"] is False
    request_payload = json.loads(output.with_name(output.name + ".request.json").read_text())
    assert request_payload["dataframe_split"]["columns"] == list(example.columns)
    assert request_payload["dataframe_split"]["data"][0][0] is None
    packaged = mlflow.pyfunc.load_model(str(output))
    expected = np.mean([m.predict(example, num_threads=1) for m in models], axis=0)
    np.testing.assert_allclose(packaged.predict(example).TARGET, expected, atol=1e-12, rtol=0)
    request = tmp_path / "input.json"
    request.write_text(example.to_json(orient="split"))
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = """
import io,json,sys
from pathlib import Path
import pandas as pd
import mlflow.pyfunc
model=mlflow.pyfunc.load_model(sys.argv[1])
frame=pd.read_json(io.StringIO(Path(sys.argv[2]).read_text()),orient='split')
import home_credit.serving as serving
assert Path(serving.__file__).is_relative_to(Path(sys.argv[1])/'code')
print(json.dumps(model.predict(frame).TARGET.tolist()))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(output), str(request)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    np.testing.assert_allclose(json.loads(result.stdout), expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="already exists"):
        export_model(run, output, example)


def test_payload_and_request_guards(producer, tmp_path):
    run, example, _ = producer
    output = tmp_path / "model"
    export_model(run, output, example)
    context = SimpleNamespace(artifacts={"ensemble": str(output / "artifacts" / "ensemble")})
    adapter = HomeCreditEnsemble()
    adapter.load_context(context)
    reordered = example[example.columns[::-1]]
    np.testing.assert_array_equal(adapter.predict(None, reordered), adapter.predict(None, example))
    missing = example.copy()
    missing.iloc[0, 0] = np.nan
    assert np.isfinite(adapter.predict(None, missing).TARGET).all()
    for invalid in [example.drop(columns="amount_ratio"), example.assign(raw_account_id=123)]:
        with pytest.raises(ValueError, match="exactly"):
            adapter.predict(None, invalid)
    with pytest.raises(ValueError, match="numeric"):
        adapter.predict(None, example.assign(amount_ratio="raw"))
    infinite = example.copy()
    infinite.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        adapter.predict(None, infinite)
    artifact = Path(context.artifacts["ensemble"]) / "fold-1.txt"
    artifact.write_text(artifact.read_text() + "\ncorrupted\n")
    with pytest.raises(ValueError, match="checksum"):
        HomeCreditEnsemble().load_context(context)


def test_export_rejects_tampered_producer_before_writing(producer, tmp_path):
    run, example, _ = producer
    (run / "features.json").write_text('["different-feature"]')
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="checksum"):
        export_model(run, output, example)
    assert not output.exists()

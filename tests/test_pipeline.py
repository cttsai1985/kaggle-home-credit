import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from home_credit.features import installments_payments, one_hot_encoder
from home_credit.pipeline import evaluate, fit_cv, predict, sha256, validate_submission, write_json


def test_string_categories_and_missing_values_are_encoded():
    source = pd.DataFrame({"kind": pd.Series(["A", "B", None], dtype="str"), "amount": [1, 2, 3]})
    encoded, categories = one_hot_encoder(source)
    assert set(categories) == {"kind_A", "kind_B", "kind_nan"}
    assert encoded.kind_nan.tolist() == [0, 0, 1]
    assert encoded.amount.tolist() == [1, 2, 3]


def test_installments_aggregate_before_join_and_clip_days(tmp_path):
    pd.DataFrame(
        {
            "SK_ID_CURR": [10, 10, 20],
            "SK_ID_PREV": [1, 1, 2],
            "NUM_INSTALMENT_VERSION": [1, 1, 2],
            "AMT_PAYMENT": [80.0, 100.0, 10.0],
            "AMT_INSTALMENT": [100.0, 100.0, 10.0],
            "DAYS_ENTRY_PAYMENT": [-8.0, -15.0, -5.0],
            "DAYS_INSTALMENT": [-10.0, -10.0, -7.0],
        }
    ).to_csv(tmp_path / "installments_payments.csv", index=False)
    result = installments_payments(tmp_path)
    assert result.index.is_unique
    assert result.loc[10, "INSTAL_COUNT"] == 2
    assert result.loc[10, "INSTAL_DPD_SUM"] == 2
    assert result.loc[10, "INSTAL_DBD_SUM"] == 5
    assert result.loc[10, "INSTAL_PAYMENT_DIFF_SUM"] == 20


def make_cache(path: Path):
    path.mkdir()
    rng = np.random.default_rng(42)
    x = rng.normal(size=420)
    y = (x + rng.normal(0, 0.5, size=420) > 0).astype(float)
    y[360:] = np.nan
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(420),
            "TARGET": y,
            "signal: value": x,
            "noise": rng.normal(size=420),
        }
    )
    frame.to_parquet(path / "selected.parquet", index=False)
    write_json(path / "manifest.json", {"cache_sha256": sha256(path / "selected.parquet")})
    return frame


def test_full_oof_coverage_reload_and_test_order(tmp_path):
    cache = tmp_path / "cache"
    frame = make_cache(cache)
    config = json.loads(Path("configs/fixture.json").read_text())
    run = tmp_path / "run"
    manifest = fit_cv(cache, run, config)
    assert manifest["status"] == "completed"
    assert manifest["project_id"] == "home-credit"
    assert manifest["run_kind"] == "synthetic_fixture"
    assert manifest["metrics"]["train_rows"] == 360
    assert manifest["metrics"]["oof_roc_auc"] > 0.8
    oof = pd.read_csv(run / "oof.csv")
    assert oof.SK_ID_CURR.is_unique and len(oof) == 360
    assert oof.fold.value_counts().to_dict() == {1: 120, 2: 120, 3: 120}
    assert json.loads((run / "features.json").read_text()) == ["signal: value", "noise"]
    result = evaluate(cache, run)
    assert result["saved_predictions_match"]
    assert result["oof_roc_auc"] == manifest["metrics"]["oof_roc_auc"]
    output = tmp_path / "reloaded-submission.csv"
    receipt = predict(cache, run, output)
    assert receipt["rows"] == 60
    submission = pd.read_csv(output)
    validate_submission(submission, frame.loc[frame.TARGET.isna(), "SK_ID_CURR"])
    np.testing.assert_allclose(submission.TARGET, pd.read_csv(run / "submission.csv").TARGET)
    with pytest.raises(ValueError, match="ordering"):
        validate_submission(submission.iloc[::-1], submission.SK_ID_CURR)
    (run / "fold-1.txt").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        evaluate(cache, run)


def test_mlflow_import_uploads_artifacts_once_and_rejects_conflicts(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    from home_credit.tracking import track

    cache = tmp_path / "cache"
    make_cache(cache)
    run = tmp_path / "run"
    config = json.loads(Path("configs/fixture.json").read_text())
    fit_cv(cache, run, config)
    # Completed older benchmark bundles do not have the new identity field.
    legacy_manifest = json.loads((run / "manifest.json").read_text())
    legacy_manifest.pop("project_id")
    write_json(run / "manifest.json", legacy_manifest)
    original_uri = mlflow.get_tracking_uri()
    uri = "sqlite:///" + str(tmp_path / "tracking.sqlite")
    try:
        mlflow.set_tracking_uri(uri)
        client = mlflow.MlflowClient()
        with pytest.raises(RuntimeError, match="bootstrap"):
            track(run, uri)
        assert client.get_experiment_by_name("kaggle/home-credit") is None
        experiment = client.create_experiment(
            "kaggle/home-credit", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        first = track(run, uri)
        second = track(run, uri)
        assert first["status"] == "imported" and second["status"] == "already_imported"
        assert first["mlflow_run_id"] == second["mlflow_run_id"]
        assert len(client.search_runs([experiment])) == 1
        imported = client.get_run(first["mlflow_run_id"])
        assert imported.data.tags["project_id"] == "home-credit"
        assert imported.data.tags["run_kind"] == "synthetic_fixture"
        assert "source_commit" in imported.data.tags and "source_sha256" in imported.data.tags
        downloaded = client.download_artifacts(first["mlflow_run_id"], "fold-1.txt")
        assert Path(downloaded).read_bytes() == (run / "fold-1.txt").read_bytes()
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["profile"] = "conflicting-profile"
        write_json(manifest_path, manifest)
        with pytest.raises(ValueError, match="Conflicting"):
            track(run, uri)
        manifest["project_id"] = "another-project"
        write_json(manifest_path, manifest)
        with pytest.raises(ValueError, match="home-credit"):
            track(run, uri)
        manifest["project_id"] = "home-credit"
        manifest["artifacts"]["unexpected.txt"] = "not-an-allowed-artifact"
        write_json(manifest_path, manifest)
        with pytest.raises(ValueError, match="Unexpected artifact"):
            track(run, uri)
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_installed_package_provenance_does_not_require_git(tmp_path, monkeypatch):
    import home_credit.pipeline as pipeline

    package = tmp_path / "site-packages" / "home_credit"
    package.mkdir(parents=True)
    source = package / "pipeline.py"
    source.write_text("# installed source\n")
    monkeypatch.setattr(pipeline, "__file__", str(source))
    identity = pipeline.source_identity()
    assert identity["source_kind"] == "installed-package"
    assert identity["git_commit"] is None and identity["git_dirty"] is None
    assert identity["source_files"] == {"src/home_credit/pipeline.py": sha256(source)}


def test_source_tree_without_git_keeps_package_hashes(tmp_path, monkeypatch):
    import home_credit.pipeline as pipeline

    package = tmp_path / "src" / "home_credit"
    package.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("# source project\n")
    source = package / "pipeline.py"
    source.write_text("# source bytes\n")
    monkeypatch.setattr(pipeline, "__file__", str(source))
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: None)
    identity = pipeline.source_identity()
    assert identity["source_kind"] == "source-tree"
    assert identity["git_commit"] is None and identity["git_dirty"] is None
    assert identity["source_files"] == {"src/home_credit/pipeline.py": sha256(source)}


def test_stratified_folds_are_repeatable_complete_and_label_balanced():
    from home_credit.pipeline import fold_assignments

    targets = np.array([0] * 90 + [1] * 10)
    config = {"folds": 5, "stratified": True, "split_seed": 1001}
    first = fold_assignments(targets, config)
    np.testing.assert_array_equal(first, fold_assignments(targets, config))
    assert set(first) == {1, 2, 3, 4, 5}
    assert len(first) == 100
    for fold in range(1, 6):
        assert np.count_nonzero(first == fold) == 20
        assert targets[first == fold].sum() == 2
        assert not set(np.flatnonzero(first == fold)) & set(np.flatnonzero(first != fold))
    with pytest.raises(ValueError, match="one applicant per fold"):
        fold_assignments(np.array([0] * 90 + [1] * 4), config)


def test_cv_report_recomputes_pooled_auc_dispersion_and_covariate_context():
    from home_credit.pipeline import cv_report, fold_assignments

    targets = np.array([0, 0, 1, 1])
    config = {"folds": 2, "stratified": True, "split_seed": 1001}
    assignments = fold_assignments(targets, config)
    probabilities = np.where(
        assignments == 1, np.where(targets == 1, 0.9, 0.1), np.where(targets == 1, 0.1, 0.9)
    )
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": range(6),
            "TARGET": [0, 0, 1, 1, np.nan, np.nan],
            "signal": [0.1, 0.2, 0.8, 0.9, 1.0, 1.0],
        }
    )
    oof = pd.DataFrame(
        {
            "SK_ID_CURR": range(4),
            "TARGET": targets,
            "probability": probabilities,
            "fold": assignments,
        }
    )
    report = cv_report(frame, oof, config, {1: 20, 2: 30})
    assert report["pooled_oof_roc_auc"] == 0.5
    assert report["fold_auc_mean"] == 0.5
    assert report["fold_auc_std"] == pytest.approx(np.sqrt(0.5))
    assert report["fold_auc_min"] == 0 and report["fold_auc_max"] == 1
    assert report["weakest_fold"]["fold"] == 2
    assert report["weakest_fold"]["best_iteration"] == 30
    assert report["oof_coverage"] == 1 and report["id_overlap"] == 0
    assert report["covariate_distributions"]["signal"]["training"]["mean"] == 0.5
    assert report["covariate_distributions"]["signal"]["test"]["mean"] == 1.0
    changed = oof.copy()
    changed.loc[0, "fold"] = 3 - changed.loc[0, "fold"]
    with pytest.raises(ValueError, match="configured split"):
        cv_report(frame, changed, config)
    duplicated = oof.copy()
    duplicated.loc[0, "SK_ID_CURR"] = duplicated.loc[1, "SK_ID_CURR"]
    with pytest.raises(ValueError, match="Duplicate"):
        cv_report(frame, duplicated, config)

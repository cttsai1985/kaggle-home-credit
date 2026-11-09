"""Known ranking examples verify exact pair accounting without pair matrices."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "credit_hard_cases", ROOT / "scripts/analyze_hard_cases.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_known_auc_with_half_weighted_ties_and_both_class_views():
    target = np.array([1, 1, 0, 0])
    auc, burden = audit.pairwise_burden(target, [0.9, 0.5, 0.5, 0.1])
    assert auc == 0.875
    np.testing.assert_array_equal(burden, [0.0, 0.25, 0.25, 0.0])
    assert burden[target == 1].mean() == burden[target == 0].mean() == 1 - auc
    auc, burden = audit.pairwise_burden([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert auc == 0.75
    np.testing.assert_array_equal(burden, [0.0, 0.5, 0.5, 0.0])


def test_perfect_reversed_and_stable_within_class_cohorts():
    target = np.array([0] * 10 + [1] * 10)
    auc, burden = audit.pairwise_burden(target, [0.1] * 10 + [0.9] * 10)
    assert auc == 1.0
    np.testing.assert_array_equal(burden, 0)
    assert np.flatnonzero(audit.hardest_by_class(target, burden)).tolist() == [0, 10]
    auc, burden = audit.pairwise_burden(target, [0.9] * 10 + [0.1] * 10)
    assert auc == 0.0
    np.testing.assert_array_equal(burden, 1)
    for labels, values in [([1, 1], [0.1, 0.2]), ([0, 1], [0.2, np.inf])]:
        with pytest.raises(ValueError):
            audit.pairwise_burden(labels, values)


def test_exact_application_membership_labels_and_duplicate_rejection():
    applications = pd.DataFrame(
        {"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "CODE_GENDER": ["F", "M", "XNA"]}
    )
    oof = pd.DataFrame({"SK_ID_CURR": [2, 1], "TARGET": [1, 0]})
    assert audit.align_source(oof, applications).index.tolist() == [2, 1]
    with pytest.raises(ValueError, match="unique"):
        audit.align_source(pd.concat([oof, oof.iloc[:1]]), applications)
    with pytest.raises(ValueError, match="population"):
        audit.align_source(oof.iloc[:1], applications)
    with pytest.raises(ValueError, match="labels"):
        audit.align_source(oof.assign(TARGET=[0, 0]), applications)

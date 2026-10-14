import pandas as pd
import pytest

from ml import data_validation as dv

requires_raw = pytest.mark.skipif(
    not (dv.RAW_DIR / dv.PRIMARY_FILE).is_file(), reason="raw BRFSS CSVs not present in data/raw/"
)


def test_feature_list_excludes_target():
    assert len(dv.FEATURES) == 21
    assert dv.TARGET not in dv.FEATURES
    groups = dv.BINARY_FEATURES + list(dv.ORDINAL_FEATURES) + list(dv.COUNT_FEATURES) + dv.CONTINUOUS_FEATURES
    assert sorted(groups) == sorted(dv.FEATURES)


def test_multiset_excess_counts_unmatched_rows():
    population = pd.DataFrame({"a": [1, 1, 2], "b": [0, 0, 1]})
    sample = pd.DataFrame({"a": [1, 1, 1, 3], "b": [0, 0, 0, 0]})
    # Third (1, 0) exceeds the population's two copies; (3, 0) is absent.
    assert dv.multiset_excess(sample, population, ["a", "b"]) == 2


@requires_raw
def test_primary_dataset_passes_all_checks():
    df = dv.load_csv(dv.PRIMARY_FILE)
    checks = dv.validate_schema(df, dv.PRIMARY_FILE, dv.TARGET, dv.EXPECTED_SHAPE)
    checks += dv.target_checks(df, dv.PRIMARY_FILE, dv.TARGET, {0, 1})
    assert [c for c in checks if c.status == "fail"] == []
    assert df[dv.TARGET].value_counts().to_dict() == {0: 218_334, 1: 35_346}


@requires_raw
def test_raw_files_match_manifest_and_are_unchanged():
    checks, _ = dv.integrity_checks([dv.PRIMARY_FILE, dv.BALANCED_FILE, dv.MULTICLASS_FILE])
    assert all(c.status == "pass" for c in checks), checks


@requires_raw
def test_binary_target_is_diabetes_only():
    primary = dv.load_csv(dv.PRIMARY_FILE)
    multi = dv.load_csv(dv.MULTICLASS_FILE)
    balanced = dv.load_csv(dv.BALANCED_FILE)
    rel, _, _ = dv.relationship_analysis(primary, multi, balanced)
    assert rel["012_features_identical_row_by_row"]
    assert rel["binary_equals_012_eq_2"]
    assert rel["5050_rows_not_matchable_to_primary_rows"] == 0

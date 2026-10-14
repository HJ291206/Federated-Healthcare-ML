"""Validate the BRFSS 2015 diabetes CSVs in data/raw/ without modifying them.

Run from the repository root:
    python -m ml.data_validation

Writes:
    results/dataset_summary.csv          one row per file
    results/class_distribution.csv       class counts per dataset
    results/validation_checks.csv        every check with pass/warn/fail/info
    results/label_mapping_012_vs_binary.csv
    results/validation_report.json       all of the above plus relationship analysis

Exit code is 1 if any check has status "fail".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
MANIFEST = RAW_DIR / "MANIFEST.sha256"

PRIMARY_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"
BALANCED_FILE = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
MULTICLASS_FILE = "diabetes_012_health_indicators_BRFSS2015.csv"
EXCLUDED_FILE = "online_retail.csv"

TARGET = "Diabetes_binary"
MULTICLASS_TARGET = "Diabetes_012"
EXPECTED_SHAPE = (253_680, 22)

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex",
]
# Inclusive code ranges documented in the UCI variable table (dataset id 891).
ORDINAL_FEATURES = {"GenHlth": (1, 5), "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)}
COUNT_FEATURES = {"MentHlth": (0, 30), "PhysHlth": (0, 30)}
CONTINUOUS_FEATURES = ["BMI"]

# Column order as it appears in the CSV.
EXPECTED_COLUMNS = [
    TARGET, "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
]
FEATURES = [c for c in EXPECTED_COLUMNS if c != TARGET]

DOCUMENTED_TARGETS = {
    PRIMARY_FILE: "0 = no diabetes; 1 = prediabetes or diabetes (Kaggle + UCI metadata)",
    BALANCED_FILE: "0 = no diabetes; 1 = prediabetes or diabetes; 50-50 balanced (Kaggle metadata)",
    MULTICLASS_FILE: "0 = no diabetes or only during pregnancy; 1 = prediabetes; 2 = diabetes (Kaggle metadata)",
}


@dataclass
class Check:
    dataset: str
    check: str
    status: str  # pass | fail | warn | info
    detail: str


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Copy the original CSV into data/raw/; "
            "this project never substitutes synthetic data."
        )
    return pd.read_csv(path)


def read_manifest() -> dict[str, str]:
    if not MANIFEST.is_file():
        return {}
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip()] = digest
    return entries


def write_manifest(names: list[str]) -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"{MANIFEST} already exists; refusing to overwrite.")
    lines = [f"{sha256(RAW_DIR / n)}  {n}" for n in names if (RAW_DIR / n).is_file()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def integrity_checks(names: list[str]) -> tuple[list[Check], dict[str, str]]:
    manifest = read_manifest()
    checks, hashes = [], {}
    for name in names:
        path = RAW_DIR / name
        if not path.is_file():
            checks.append(Check(name, "file_present", "fail", f"missing: {path}"))
            continue
        hashes[name] = digest = sha256(path)
        if name not in manifest:
            checks.append(Check(name, "sha256_matches_manifest", "warn", f"not in manifest; sha256={digest}"))
        elif manifest[name] != digest:
            checks.append(Check(name, "sha256_matches_manifest", "fail",
                                f"expected {manifest[name]}, got {digest}"))
        else:
            checks.append(Check(name, "sha256_matches_manifest", "pass", digest))
    return checks, hashes


def validate_schema(df: pd.DataFrame, name: str, target: str, expected_shape: tuple[int, int] | None) -> list[Check]:
    checks: list[Check] = []
    expected_cols = [target] + FEATURES

    if expected_shape is not None:
        checks.append(Check(name, "shape", "pass" if df.shape == expected_shape else "fail",
                            f"observed {df.shape}, expected {expected_shape}"))
    else:
        checks.append(Check(name, "shape", "info", f"observed {df.shape}"))

    cols = list(df.columns)
    if cols == expected_cols:
        checks.append(Check(name, "column_names_and_order", "pass", f"{len(cols)} columns as expected"))
    else:
        missing = sorted(set(expected_cols) - set(cols))
        extra = sorted(set(cols) - set(expected_cols))
        status = "fail" if missing or extra else "warn"
        checks.append(Check(name, "column_names_and_order", status,
                            f"missing={missing} extra={extra} order_differs={not missing and not extra}"))
        if missing:
            return checks

    dtypes = df.dtypes.astype(str).value_counts().to_dict()
    all_numeric = all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes)
    checks.append(Check(name, "dtypes_numeric", "pass" if all_numeric else "fail", json.dumps(dtypes)))

    if all_numeric:
        non_integer = int((df.to_numpy() % 1 != 0).sum())
        checks.append(Check(name, "all_values_integer_coded", "pass" if non_integer == 0 else "fail",
                            f"{non_integer} non-integer cells (stored as {', '.join(dtypes)})"))

    n_missing = int(df.isna().sum().sum())
    per_col = {c: int(v) for c, v in df.isna().sum().items() if v}
    checks.append(Check(name, "missing_values", "pass" if n_missing == 0 else "fail",
                        f"{n_missing} missing cells {per_col if per_col else ''}".strip()))

    checks.append(Check(name, "target_excluded_from_features",
                        "pass" if target not in FEATURES and len(FEATURES) == 21 else "fail",
                        f"{len(FEATURES)} input features; target '{target}' not among them"))

    for col in BINARY_FEATURES:
        bad = sorted(set(df[col].unique()) - {0, 1})
        if bad:
            checks.append(Check(name, f"domain:{col}", "fail", f"values outside {{0,1}}: {bad[:10]}"))
    for col, (lo, hi) in {**ORDINAL_FEATURES, **COUNT_FEATURES}.items():
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out:
            checks.append(Check(name, f"domain:{col}", "fail", f"{n_out} values outside [{lo},{hi}]"))
    domain_fails = [c for c in checks if c.check.startswith("domain:")]
    if not domain_fails:
        checks.append(Check(name, "feature_value_domains", "pass",
                            "binary features in {0,1}; ordinal/count features within documented ranges"))

    bmi = df["BMI"]
    checks.append(Check(name, "bmi_plausibility", "info",
                        f"range [{bmi.min():g}, {bmi.max():g}]; BMI<15: {int((bmi < 15).sum())}, "
                        f"BMI>60: {int((bmi > 60).sum())}, BMI>=90: {int((bmi >= 90).sum())}. "
                        "No documented valid range; values retained."))

    n_dup = int(df.duplicated().sum())
    checks.append(Check(name, "exact_duplicate_rows", "info",
                        f"{n_dup} rows ({n_dup / len(df):.2%}) repeat an earlier row exactly. "
                        "Retained: 21 coarse integer codes make identical profiles from different "
                        "respondents plausible. Not removed."))

    feat_dup = df.duplicated(subset=FEATURES, keep=False)
    conflicting_groups = df[feat_dup].groupby(FEATURES, sort=False)[target].nunique()
    n_conflict_groups = int((conflicting_groups > 1).sum())
    conflict_rows = int(df[feat_dup].groupby(FEATURES, sort=False)[target]
                        .transform("nunique").gt(1).sum())
    checks.append(Check(name, "identical_features_different_target", "info",
                        f"{n_conflict_groups} feature profiles appear with more than one target value, "
                        f"covering {conflict_rows} rows. Indicates irreducible label noise."))
    return checks


def class_distribution(df: pd.DataFrame, name: str, target: str, meanings: dict[int, str]) -> pd.DataFrame:
    counts = df[target].value_counts().sort_index()
    return pd.DataFrame({
        "dataset": name,
        "target_column": target,
        "class": counts.index.astype(int),
        "meaning_observed": [meanings.get(int(k), "") for k in counts.index],
        "count": counts.to_numpy(),
        "share": (counts / counts.sum()).round(6).to_numpy(),
    })


def target_checks(df: pd.DataFrame, name: str, target: str, expected: set[int]) -> list[Check]:
    values = set(int(v) for v in df[target].unique())
    counts = df[target].value_counts()
    ratio = counts.max() / counts.min()
    minority_share = counts.min() / counts.sum()
    return [
        Check(name, "target_column_present", "pass" if target in df.columns else "fail", target),
        Check(name, "target_values", "pass" if values == expected else "fail",
              f"observed {sorted(values)}, expected {sorted(expected)}"),
        Check(name, "class_balance", "info",
              f"majority:minority = {ratio:.2f}:1; minority share = {minority_share:.2%}; "
              f"{'balanced' if ratio <= 1.05 else 'imbalanced'}"),
    ]


def multiset_excess(sample: pd.DataFrame, population: pd.DataFrame, cols: list[str]) -> int:
    """Rows of `sample` that cannot be matched one-to-one to rows of `population` on `cols`."""
    s = sample.value_counts(subset=cols).rename("n_sample")
    p = population.value_counts(subset=cols).rename("n_population")
    joined = s.to_frame().join(p, how="left").fillna(0)
    return int((joined["n_sample"] - joined["n_population"]).clip(lower=0).sum())


def relationship_analysis(primary: pd.DataFrame, multi: pd.DataFrame, balanced: pd.DataFrame) -> tuple[dict, list[Check], pd.DataFrame]:
    rel: dict = {}
    checks: list[Check] = []

    same_features = bool(np.array_equal(primary[FEATURES].to_numpy(), multi[FEATURES].to_numpy()))
    rel["012_features_identical_row_by_row"] = same_features
    crosstab = pd.crosstab(multi[MULTICLASS_TARGET].astype(int), primary[TARGET].astype(int))
    crosstab.index.name, crosstab.columns.name = MULTICLASS_TARGET, TARGET
    rel["012_vs_binary_crosstab"] = {int(k): {int(c): int(v) for c, v in row.items()}
                                     for k, row in crosstab.iterrows()}

    binary_is_diabetes_only = bool(same_features and (primary[TARGET] == (multi[MULTICLASS_TARGET] == 2)).all())
    prediabetes_in_positive = int(((multi[MULTICLASS_TARGET] == 1) & (primary[TARGET] == 1)).sum())
    rel["binary_equals_012_eq_2"] = binary_is_diabetes_only
    rel["prediabetes_rows_labelled_positive_in_binary"] = prediabetes_in_positive
    checks.append(Check(MULTICLASS_FILE, "same_respondents_as_primary", "info",
                        f"features identical row by row: {same_features}"))
    checks.append(Check(PRIMARY_FILE, "target_definition_vs_documentation",
                        "warn" if binary_is_diabetes_only else "pass",
                        "Observed: Diabetes_binary == 1 exactly when Diabetes_012 == 2 (diabetes); "
                        f"all {int(crosstab.loc[1].sum()) if 1 in crosstab.index else 0} prediabetes rows "
                        "are labelled 0. Documentation says 1 = 'prediabetes or diabetes'. "
                        "Labels are NOT changed; the observed definition is used and reported."
                        if binary_is_diabetes_only else "Observed mapping consistent with documentation."))

    bal_pos, bal_neg = balanced[balanced[TARGET] == 1], balanced[balanced[TARGET] == 0]
    rel["5050_rows_not_matchable_to_primary_rows"] = multiset_excess(balanced, primary, EXPECTED_COLUMNS)
    rel["5050_positives_not_matchable_to_012_diabetes"] = multiset_excess(
        bal_pos, multi[multi[MULTICLASS_TARGET] == 2], FEATURES)
    rel["5050_negatives_not_matchable_to_012_no_diabetes"] = multiset_excess(
        bal_neg, multi[multi[MULTICLASS_TARGET] == 0], FEATURES)
    rel["5050_positive_count_equals_012_diabetes_count"] = bool(
        len(bal_pos) == int((multi[MULTICLASS_TARGET] == 2).sum()))
    subset = rel["5050_rows_not_matchable_to_primary_rows"] == 0
    checks.append(Check(BALANCED_FILE, "subsample_of_primary", "warn" if subset else "info",
                        f"{rel['5050_rows_not_matchable_to_primary_rows']} of {len(balanced)} rows cannot be "
                        "matched to a primary row. "
                        + ("Consistent with being a subsample of the primary file: never mix it with "
                           "primary-file train/test splits (leakage)." if subset else "")))
    checks.append(Check(BALANCED_FILE, "target_definition_vs_documentation", "warn",
                        f"{len(bal_pos)} positives = number of Diabetes_012 == 2 rows: "
                        f"{rel['5050_positive_count_equals_012_diabetes_count']}; positives unmatched to "
                        f"diabetes rows: {rel['5050_positives_not_matchable_to_012_diabetes']}; negatives "
                        f"unmatched to no-diabetes rows: {rel['5050_negatives_not_matchable_to_012_no_diabetes']}. "
                        "Observed positive class = diabetes only, not 'prediabetes or diabetes'."))
    return rel, checks, crosstab


def run(write_manifest_flag: bool = False) -> int:
    names = [PRIMARY_FILE, BALANCED_FILE, MULTICLASS_FILE, EXCLUDED_FILE]
    if write_manifest_flag:
        write_manifest(names)
    checks, hashes = integrity_checks(names)
    if any(c.status == "fail" and c.check == "file_present" for c in checks if c.dataset != EXCLUDED_FILE):
        for c in checks:
            print(f"[{c.status.upper():4}] {c.dataset}: {c.check} - {c.detail}")
        return 1

    primary = load_csv(PRIMARY_FILE)
    balanced = load_csv(BALANCED_FILE)
    multi = load_csv(MULTICLASS_FILE)

    checks += validate_schema(primary, PRIMARY_FILE, TARGET, EXPECTED_SHAPE)
    checks += target_checks(primary, PRIMARY_FILE, TARGET, {0, 1})
    checks += validate_schema(balanced, BALANCED_FILE, TARGET, None)
    checks += target_checks(balanced, BALANCED_FILE, TARGET, {0, 1})
    checks += validate_schema(multi, MULTICLASS_FILE, MULTICLASS_TARGET, EXPECTED_SHAPE)
    checks += target_checks(multi, MULTICLASS_FILE, MULTICLASS_TARGET, {0, 1, 2})
    checks.append(Check(EXCLUDED_FILE, "excluded_from_healthcare_model", "info",
                        "Retail transactions, unrelated to health; hashed only, not loaded."))

    rel, rel_checks, crosstab = relationship_analysis(primary, multi, balanced)
    checks += rel_checks

    binary_meaning = {0: "no diabetes (includes prediabetes)", 1: "diabetes"} if rel["binary_equals_012_eq_2"] \
        else {0: "no diabetes", 1: "prediabetes or diabetes"}
    classes = pd.concat([
        class_distribution(primary, PRIMARY_FILE, TARGET, binary_meaning),
        class_distribution(balanced, BALANCED_FILE, TARGET, binary_meaning),
        class_distribution(multi, MULTICLASS_FILE, MULTICLASS_TARGET,
                           {0: "no diabetes or only during pregnancy", 1: "prediabetes", 2: "diabetes"}),
    ], ignore_index=True)

    observed_def = {
        PRIMARY_FILE: "1 = diabetes (Diabetes_012 == 2); prediabetes coded 0" if rel["binary_equals_012_eq_2"]
        else "matches documentation",
        BALANCED_FILE: "1 = diabetes; 0 = no diabetes, including at least "
                       f"{rel['5050_negatives_not_matchable_to_012_no_diabetes']} prediabetes profiles",
        MULTICLASS_FILE: "0/1/2 as documented",
    }
    rows = []
    for name, df, target, role in [(PRIMARY_FILE, primary, TARGET, "primary"),
                                   (BALANCED_FILE, balanced, TARGET, "alternative"),
                                   (MULTICLASS_FILE, multi, MULTICLASS_TARGET, "alternative")]:
        counts = df[target].value_counts().sort_index()
        rows.append({
            "dataset": name, "role": role, "sha256": hashes[name],
            "n_rows": len(df), "n_columns": df.shape[1], "n_features": len(FEATURES),
            "target_column": target,
            "target_definition_documented": DOCUMENTED_TARGETS[name],
            "target_definition_observed": observed_def[name],
            "class_counts": json.dumps({int(k): int(v) for k, v in counts.items()}),
            "positive_or_max_class_share": round(float(counts.iloc[-1] / counts.sum()), 6),
            "majority_to_minority_ratio": round(float(counts.max() / counts.min()), 4),
            "n_missing_cells": int(df.isna().sum().sum()),
            "n_exact_duplicate_rows": int(df.duplicated().sum()),
            "duplicate_share": round(float(df.duplicated().mean()), 6),
        })
    rows.append({"dataset": EXCLUDED_FILE, "role": "excluded (not health data)",
                 "sha256": hashes.get(EXCLUDED_FILE, "")})
    summary = pd.DataFrame(rows)
    int_cols = ["n_rows", "n_columns", "n_features", "n_missing_cells", "n_exact_duplicate_rows"]
    summary[int_cols] = summary[int_cols].astype("Int64")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    classes.to_csv(RESULTS_DIR / "class_distribution.csv", index=False)
    checks_df = pd.DataFrame([asdict(c) for c in checks])
    checks_df.to_csv(RESULTS_DIR / "validation_checks.csv", index=False)
    crosstab.to_csv(RESULTS_DIR / "label_mapping_012_vs_binary.csv")
    (RESULTS_DIR / "validation_report.json").write_text(json.dumps({
        "features": FEATURES, "target": TARGET,
        "feature_groups": {"binary": BINARY_FEATURES, "ordinal": ORDINAL_FEATURES,
                           "count": COUNT_FEATURES, "continuous": CONTINUOUS_FEATURES},
        "relationships": rel,
        "checks": [asdict(c) for c in checks],
    }, indent=2), encoding="utf-8")

    for c in checks:
        print(f"[{c.status.upper():4}] {c.dataset}: {c.check} - {c.detail}")
    print("\nClass distribution:\n" + classes.to_string(index=False))
    print("\nDiabetes_012 (rows) vs Diabetes_binary (columns):\n" + crosstab.to_string())
    n_fail = int((checks_df.status == "fail").sum())
    print(f"\n{n_fail} failed, {int((checks_df.status == 'warn').sum())} warnings. Outputs in {RESULTS_DIR}")
    return 1 if n_fail else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Record SHA-256 checksums of data/raw files (refuses to overwrite).")
    args = parser.parse_args()
    sys.exit(run(args.write_manifest))


if __name__ == "__main__":
    main()

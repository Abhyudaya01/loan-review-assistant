"""Exercise leakage boundaries and reproducible outputs using offline synthetic data."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from loan_review.data.features import (
    ID,
    PROTECTED_ATTRIBUTES,
    TARGET,
    history_aggregates,
)
from loan_review.data.pipeline import (
    PipelineConfig,
    prepare_data,
    run_pipeline,
    split_applications,
)
from loan_review.data.proxy import proxy_report


def test_stratified_disjoint_splits_and_protected_separation(raw_frames):
    prepared = prepare_data(*raw_frames)
    assert [s.features.height for s in prepared.splits.values()] == [140, 30, 30]
    id_sets = [set(s.labels[ID]) for s in prepared.splits.values()]
    assert len(set.union(*id_sets)) == 200
    assert sum(len(ids) for ids in id_sets) == len(set.union(*id_sets))
    for split in prepared.splits.values():
        assert not ({ID, TARGET, *PROTECTED_ATTRIBUTES} & set(split.features.columns))
        assert not any("TARGET" in column for column in split.features.columns)
        assert split.labels[TARGET].mean() == pytest.approx(0.2)
        assert split.audit_attributes.columns == [ID, *PROTECTED_ATTRIBUTES]
        assert split.audit_attributes[ID].to_list() == split.labels[ID].to_list()
        expected = raw_frames[0].join(split.labels.select(ID), on=ID, how="semi")
        assert_frame_equal(
            split.audit_attributes, expected.select(ID, *PROTECTED_ATTRIBUTES).sort(ID)
        )


def test_aggregates_sentinel_ratios_and_missing_history(raw_frames):
    prepared = prepare_data(*raw_frames)
    combined = pl.concat(
        [
            split.features.hstack(split.labels.select(ID))
            for split in prepared.splits.values()
        ]
    ).sort(ID)
    row = combined.row(0, named=True)
    assert row["BUREAU_COUNT"] == 2
    assert row["BUREAU_AMT_CREDIT_SUM_SUM"] == 40
    assert row["BUREAU_AMT_CREDIT_SUM_MEAN"] == 20
    assert row["BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX"] == 8
    assert row["BUREAU_CREDIT_DAY_OVERDUE_MAX"] == 4
    assert row["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] is None
    assert row["PREVIOUS_APPLICATION_AMT_CREDIT_SUM"] == 300
    assert row["PREVIOUS_APPLICATION_APPROVED_COUNT"] == 1
    assert row["PREVIOUS_APPLICATION_REFUSED_COUNT"] == 1
    assert row["DAYS_EMPLOYED_SENTINEL"] == 1
    assert np.isnan(row["DAYS_EMPLOYED"])
    assert row["PAYMENT_TO_INCOME"] is None
    assert row["CREDIT_TO_INCOME"] is None
    assert combined.row(1, named=True)["CREDIT_TO_GOODS"] is None
    valid = combined.row(2, named=True)
    assert valid["PAYMENT_TO_INCOME"] == pytest.approx(1002 / 50002)
    assert valid["CREDIT_TO_INCOME"] == pytest.approx(20002 / 50002)
    assert valid["CREDIT_TO_GOODS"] == pytest.approx(20002 / 18002)
    assert valid["DAYS_EMPLOYED"] == -102
    absent = combined.row(199, named=True)
    assert absent["BUREAU_COUNT"] == absent["PREVIOUS_APPLICATION_COUNT"] == 0
    assert absent["BUREAU_AMT_CREDIT_SUM_SUM"] is None
    assert absent["PREVIOUS_APPLICATION_AMT_CREDIT_MEAN"] is None


def test_fixed_seed_and_application_row_order_are_deterministic(raw_frames):
    first = prepare_data(*raw_frames, PipelineConfig(seed=17))
    second = prepare_data(
        raw_frames[0].reverse(), *raw_frames[1:], PipelineConfig(seed=17)
    )
    for name in first.splits:
        assert_frame_equal(first.splits[name].features, second.splits[name].features)
        assert_frame_equal(first.splits[name].labels, second.splits[name].labels)
    assert first.manifest == second.manifest
    assert_frame_equal(first.proxy_report, second.proxy_report)
    other = prepare_data(*raw_frames, PipelineConfig(seed=18))
    assert (
        first.splits["train"].labels[ID].to_list()
        != other.splits["train"].labels[ID].to_list()
    )


def test_holdout_changes_cannot_affect_training_features_or_proxy_report(raw_frames):
    first = prepare_data(*raw_frames)
    held_out = pl.concat([first.splits[n].labels.select(ID) for n in ("val", "test")])[
        ID
    ].to_list()
    application, bureau, previous = raw_frames
    application = application.with_columns(
        pl.when(pl.col(ID).is_in(held_out))
        .then(1e10)
        .otherwise(pl.col("AMT_INCOME_TOTAL"))
        .alias("AMT_INCOME_TOTAL"),
        pl.when(pl.col(ID).is_in(held_out))
        .then(pl.lit("changed"))
        .otherwise(pl.col("CODE_GENDER"))
        .alias("CODE_GENDER"),
    )
    bureau = bureau.with_columns(
        pl.when(pl.col(ID).is_in(held_out))
        .then(1e12)
        .otherwise(pl.col("AMT_CREDIT_SUM"))
        .alias("AMT_CREDIT_SUM")
    )
    previous = previous.with_columns(
        pl.when(pl.col(ID).is_in(held_out))
        .then(1e12)
        .otherwise(pl.col("AMT_CREDIT"))
        .alias("AMT_CREDIT")
    )
    second = prepare_data(application, bureau, previous)
    assert_frame_equal(first.splits["train"].features, second.splits["train"].features)
    assert_frame_equal(first.proxy_report, second.proxy_report)


def test_history_is_scoped_before_aggregation(raw_frames, monkeypatch):
    from loan_review.data import features

    assignments = split_applications(raw_frames[0], seed=42)
    allowed = [set(frame[ID]) for frame in assignments.values()]
    calls = []

    def inspect(history, *, source):
        ids = set(history[ID])
        assert any(ids <= split_ids for split_ids in allowed)
        calls.append(source)
        return history_aggregates(history, source=source)

    monkeypatch.setattr(features, "history_aggregates", inspect)
    prepare_data(*raw_frames)
    assert calls == ["bureau", "previous_application"] * 3


def test_proxy_screen_flags_numeric_categorical_and_nonlinear_associations():
    n = 400
    age = np.linspace(-1, 1, n)
    features = pl.DataFrame(
        {
            "AGE_PROXY": age,
            "NONLINEAR_PROXY": age**2,
            "GENDER_PROXY": ["a" if i % 2 else "b" for i in range(n)],
            "FAMILY_PROXY": ["x" if i % 3 else "y" for i in range(n)],
            "CONSTANT": [1] * n,
            "MISSING": pl.Series([None] * n, dtype=pl.Float64),
        }
    )
    audit = pl.DataFrame(
        {
            ID: range(n),
            "DAYS_BIRTH": age * 1000 - 15000,
            "CODE_GENDER": ["F" if i % 2 else "M" for i in range(n)],
            "NAME_FAMILY_STATUS": ["Single" if i % 3 else "Married" for i in range(n)],
        }
    )
    report = proxy_report(features, audit, seed=42)
    for feature, attribute in (
        ("AGE_PROXY", "DAYS_BIRTH"),
        ("GENDER_PROXY", "CODE_GENDER"),
        ("FAMILY_PROXY", "NAME_FAMILY_STATUS"),
    ):
        row = report.filter(
            (pl.col("feature") == feature)
            & (pl.col("protected_attribute") == attribute)
        ).row(0, named=True)
        assert row["correlation_flag"]
        assert row["mutual_information_flag"]
        assert row["flagged"]
    nonlinear = report.filter(
        (pl.col("feature") == "NONLINEAR_PROXY")
        & (pl.col("protected_attribute") == "DAYS_BIRTH")
    ).row(0, named=True)
    assert not nonlinear["correlation_flag"]
    assert nonlinear["mutual_information_flag"]
    assert not report.filter(pl.col("feature").is_in(["CONSTANT", "MISSING"]))[
        "flagged"
    ].any()
    assert (
        report.filter(pl.col("feature") == "MISSING")["mutual_information"].null_count()
        == 3
    )
    # Sampling and nearest-neighbor jitter also use the configured seed.
    assert_frame_equal(
        proxy_report(features, audit, seed=9, max_rows=50),
        proxy_report(features, audit, seed=9, max_rows=50),
    )


def test_versioned_parquet_manifest_and_csv_outputs(raw_frames, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, frame in zip(
        ("application_train", "bureau", "previous_application"), raw_frames, strict=True
    ):
        frame.write_csv(raw / f"{name}.csv")
    output, manifest = (
        tmp_path / "artifacts",
        tmp_path / "configs" / "feature_manifest.yaml",
    )
    run_dir = run_pipeline(raw, output, manifest)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["split_rows"] == {"train": 140, "val": 30, "test": 30}
    assert metadata["proxy_screen_split"] == "train"
    expected = prepare_data(*raw_frames)
    for name, split in expected.splits.items():
        assert_frame_equal(
            pl.read_parquet(run_dir / f"{name}_features.parquet"), split.features
        )
        assert_frame_equal(
            pl.read_parquet(run_dir / f"{name}_labels.parquet"), split.labels
        )
        assert_frame_equal(
            pl.read_parquet(run_dir / f"{name}_audit_attributes.parquet"),
            split.audit_attributes,
        )
    text = manifest.read_text()
    assert text == (run_dir / "feature_manifest.yaml").read_text()
    # Scalars are JSON-quoted YAML; every feature's required fields appear exactly once.
    assert text.count("  - name:") == len(expected.manifest)
    for feature in expected.manifest:
        for key in ("name", "source", "dtype", "description"):
            assert f"{key}: {json.dumps(getattr(feature, key))}" in text
    report = pl.read_csv(output / "proxy_report.csv")
    assert set(report["split"]) == {"train"}
    assert report["n_observations"].max() <= 140
    assert report["artifact_version"].unique().to_list() == [run_dir.name]
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    assert run_pipeline(raw, output, manifest) == run_dir
    assert before == {path.name: path.read_bytes() for path in run_dir.iterdir()}
    changed = run_pipeline(raw, output, manifest, PipelineConfig(seed=43))
    assert changed != run_dir
    assert before == {path.name: path.read_bytes() for path in run_dir.iterdir()}


@pytest.mark.parametrize(
    "failure", ["duplicate", "null_id", "bad_target", "missing", "tiny"]
)
def test_invalid_application_data_fails_explicitly(raw_frames, failure):
    application, bureau, previous = raw_frames
    if failure == "duplicate":
        application = pl.concat([application, application.head(1)])
    elif failure == "null_id":
        application = application.with_columns(pl.lit(None, dtype=pl.Int64).alias(ID))
    elif failure == "bad_target":
        application = application.with_columns(pl.lit("0").alias(TARGET))
    elif failure == "missing":
        application = application.drop("DAYS_BIRTH")
    else:
        application = application.head(4)
    with pytest.raises(ValueError):
        prepare_data(application, bureau, previous)


def test_empty_history_preserves_all_applicants(raw_frames):
    application, bureau, previous = raw_frames
    result = prepare_data(application, bureau.head(0), previous.head(0))
    assert (
        sum(split.features.height for split in result.splits.values())
        == application.height
    )
    for split in result.splits.values():
        assert split.features["BUREAU_COUNT"].sum() == 0
        assert (
            split.features["BUREAU_AMT_CREDIT_SUM_SUM"].null_count()
            == split.features.height
        )


def test_configuration_rejects_coercion_and_invalid_thresholds():
    for payload in (
        {"seed": "42"},
        {"seed": -1},
        {"correlation_threshold": 1.1},
        {"mutual_information_threshold": float("nan")},
        {"proxy_max_rows": 1},
    ):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(payload)


def test_missing_csv_reports_required_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="application_train.csv"):
        run_pipeline(tmp_path)

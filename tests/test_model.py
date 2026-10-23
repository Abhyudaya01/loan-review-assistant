"""Verify offline model selection boundaries, explanations, and portable artifacts."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError
from sklearn.isotonic import IsotonicRegression

from loan_review.data.features import EXCLUDED_FEATURES, ID, TARGET, FeatureInfo
from loan_review.data.pipeline import manifest_yaml
from loan_review.model.bundle import content_hash, file_hash, load_model
from loan_review.model.calibration import IsotonicCalibrator
from loan_review.model.config import TrainingConfig, load_thresholds
from loan_review.model.evaluation import (
    evaluate_test,
    probability_metrics,
    reliability_bins,
)
from loan_review.model.explanations import explain_applications
from loan_review.model.pipeline import run_training
from loan_review.model.preprocessing import FeatureEncoder
from loan_review.model.reasons import load_reason_codes, top_reasons
from loan_review.model.training import train_model


@pytest.fixture(scope="module")
def model_data():
    rng = np.random.default_rng(237)
    n = 1000
    burden = rng.uniform(0.05, 0.8, size=n)
    external = rng.uniform(0, 1, size=n)
    income_type = np.where(rng.random(n) > 0.4, "Working", "Commercial associate")
    logits = -2 + 5 * burden - 3 * external
    target = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.int64)
    features = pl.DataFrame(
        {
            "PAYMENT_TO_INCOME": burden,
            "CREDIT_TO_INCOME": burden * 3 + rng.uniform(0, 0.2, size=n),
            "EXT_SOURCE_1": external,
            "NAME_INCOME_TYPE": income_type,
        }
    )
    target_series = pl.Series(TARGET, target)
    return {
        "train": (features[:600], target_series[:600]),
        "val": (features[600:800], target_series[600:800]),
        "test": (features[800:], target_series[800:]),
    }


@pytest.fixture(scope="module")
def trained(model_data):
    return train_model(
        *model_data["train"],
        *model_data["val"],
        data_hash="synthetic-training-data",
        git_sha="synthetic-test-sha",
        reason_codes=load_reason_codes(),
        thresholds=load_thresholds(),
        config=TrainingConfig(n_trials=2, num_boost_round=60, early_stopping_rounds=8),
    )


def test_reason_coverage_for_catalog_and_every_model_feature(trained):
    catalog = yaml.safe_load(Path("configs/feature_manifest.yaml").read_text())
    dictionary = load_reason_codes()
    dictionary.require_coverage([feature["name"] for feature in catalog["features"]])
    dictionary.require_coverage(trained.encoder.feature_names)
    assert not set(dictionary.features).intersection(EXCLUDED_FEATURES)
    with pytest.raises(ValueError, match="without reason codes"):
        dictionary.require_coverage(["UNREVIEWED_NEW_FEATURE"])


def test_top_reasons_deduplicate_filter_signed_groups_and_keep_evidence():
    row = {
        "PAYMENT_TO_INCOME": 0.7,
        "CREDIT_TO_INCOME": -0.2,
        "EXT_SOURCE_1": 0.4,
        "EXT_SOURCE_2": 0.2,
        "AMT_CREDIT": -0.8,
        "AMT_GOODS_PRICE": 0.1,
        "DAYS_EMPLOYED": 0.0,
    }
    reasons = top_reasons(row)
    assert [reason.code for reason in reasons] == ["EXTERNAL_SCORE", "PAYMENT_BURDEN"]
    assert reasons[0].net_shap_value == pytest.approx(0.6)
    assert reasons[1].net_shap_value == pytest.approx(0.5)
    assert len(reasons[0].supporting_features) == 2
    assert reasons[1].offsetting_features[0].shap_value == -0.2
    assert all(
        item.shap_value > 0 for reason in reasons for item in reason.supporting_features
    )
    assert top_reasons({name: -abs(value) for name, value in row.items()}) == []
    assert top_reasons(row, k=1) == reasons[:1]
    assert top_reasons(row, k=0) == []
    assert top_reasons(dict(reversed(list(row.items())))) == reasons
    with pytest.raises(ValueError, match="without reason codes"):
        top_reasons({"UNKNOWN": 1.0})
    with pytest.raises(ValidationError):
        top_reasons({"AMT_CREDIT": float("nan")})


def test_calibrated_probabilities_are_bounded_and_match_sklearn(model_data, trained):
    validation, labels = model_data["val"]
    raw_validation = trained.predict_raw_probability(validation)
    reference = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
        raw_validation, labels.to_numpy()
    )
    raw_grid = np.linspace(0, 1, 101)
    actual = trained.calibrator.predict(raw_grid)
    np.testing.assert_allclose(actual, reference.predict(raw_grid))
    assert (np.diff(actual) >= 0).all()
    probabilities = trained.predict_proba(model_data["test"][0])
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    with pytest.raises(ValueError):
        trained.calibrator.predict(np.array([1.1]))
    # A constant-score model still has a valid one-knot isotonic calibrator.
    constant = IsotonicCalibrator.fit(np.full(4, 0.5), np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(
        constant.predict(np.array([0.0, 0.5, 1.0])), [0.5, 0.5, 0.5]
    )


def test_encoder_uses_only_training_categories_and_rejects_protected_columns(
    model_data,
):
    train = model_data["train"][0]
    encoder = FeatureEncoder.fit(train)
    novel = model_data["val"][0].with_columns(
        pl.lit("VAL_ONLY").alias("NAME_INCOME_TYPE")
    )
    matrix = encoder.transform(novel)
    assert (matrix[:, encoder.categorical_indices[0]] == -1).all()
    assert "VAL_ONLY" not in encoder.categories["NAME_INCOME_TYPE"]
    for protected in ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", ID, TARGET):
        with pytest.raises(ValidationError):
            FeatureEncoder.fit(train.with_columns(pl.lit(1).alias(protected)))
    with pytest.raises(ValueError, match="columns/order"):
        encoder.transform(train.select(list(reversed(train.columns))))


def test_model_version_and_scores_survive_reload(trained, model_data, tmp_path):
    path = trained.save(tmp_path)
    restored = load_model(path)
    assert restored.model_version == trained.model_version
    assert restored.model_version == content_hash(restored.metadata["version_inputs"])
    assert restored.metadata["version_inputs"]["data_hash"] == "synthetic-training-data"
    assert restored.metadata["version_inputs"]["git_sha"] == "synthetic-test-sha"
    np.testing.assert_allclose(
        restored.predict_proba(model_data["test"][0]),
        trained.predict_proba(model_data["test"][0]),
        rtol=0,
        atol=0,
    )
    assert restored.save(tmp_path) == path
    assert load_model(path).model_version == trained.model_version
    explainer = json.loads((path / "explainer_metadata.json").read_text())
    assert explainer["model_output"] == "raw"
    assert explainer["units"] == "default_log_odds"
    assert explainer["features"] == trained.encoder.feature_names
    (path / "calibrator.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model(path)


def test_tuning_is_bounded_seeded_and_reproducible(trained, model_data):
    again = train_model(
        *model_data["train"],
        *model_data["val"],
        data_hash="synthetic-training-data",
        git_sha="synthetic-test-sha",
        reason_codes=load_reason_codes(),
        thresholds=load_thresholds(),
        config=TrainingConfig(n_trials=2, num_boost_round=60, early_stopping_rounds=8),
    )
    assert again.model_version == trained.model_version
    assert again.metadata["trials"] == trained.metadata["trials"]
    assert len(again.metadata["trials"]) == 2
    assert 1 <= trained.booster.current_iteration() <= 60
    for settings in (
        {"n_trials": 31},
        {"n_trials": 0},
        {"seed": "42"},
        {"num_boost_round": 10, "early_stopping_rounds": 10},
    ):
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate(settings)


def test_treeshap_additivity_ordering_and_reasons(trained, model_data):
    sample = model_data["test"][0].head(10)
    explanations = explain_applications(trained, sample, list(range(10)))
    raw = trained.booster.predict(
        trained.encoder.transform(sample), raw_score=True, num_threads=1
    )
    for index, explanation in enumerate(explanations):
        values = [item.shap_value for item in explanation.contributions]
        assert values == sorted(values, reverse=True)
        assert explanation.base_value + sum(values) == pytest.approx(raw[index])
        assert explanation.application_id == index
        assert explanation.model_version == trained.model_version
        for reason in top_reasons(
            explanation.shap_row, reason_codes=trained.reason_codes
        ):
            assert reason.net_shap_value > 0
            assert all(item.shap_value > 0 for item in reason.supporting_features)


def test_score_band_boundaries_and_no_decisions(trained, model_data):
    thresholds = load_thresholds()
    assert [
        thresholds.band(value) for value in (0.0, 0.049, 0.05, 0.149, 0.15, 1.0)
    ] == ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]
    for value in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            thresholds.band(value)
    scored = trained.score(model_data["test"][0])
    assert set(scored["risk_band"]) <= {"LOW", "MEDIUM", "HIGH"}
    assert "decision" not in scored.columns


def test_final_report_metrics_and_reliability_artifacts(trained, model_data, tmp_path):
    path = evaluate_test(
        trained,
        *model_data["test"],
        report_root=tmp_path,
        test_data_hash="synthetic-test",
    )
    report = json.loads((path / "metrics.json").read_text())
    assert report["evaluation_split"] == "test"
    assert report["rows"] == 200
    assert set(report["calibrated"]) == {"auc", "ks", "brier"}
    assert all(0 <= value <= 1 for value in report["calibrated"].values())
    assert (path / "reliability.png").read_bytes().startswith(b"\x89PNG")
    bins = pl.read_csv(path / "reliability_bins.csv")
    assert bins.group_by("probability_type").agg(pl.col("count").sum())[
        "count"
    ].to_list() == [200, 200]
    perfect = probability_metrics(
        np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])
    )
    assert perfect == {"auc": 1.0, "ks": 1.0, "brier": 0.0}
    edges = reliability_bins(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert edges["count"].sum() == 2


def _write_phase_one(directory, model_data):
    directory.mkdir()
    for index, (name, (features, target)) in enumerate(model_data.items()):
        features.write_parquet(directory / f"{name}_features.parquet")
        pl.DataFrame(
            {ID: np.arange(features.height) + index * 10_000, TARGET: target}
        ).write_parquet(directory / f"{name}_labels.parquet")
    features = model_data["train"][0]
    (directory / "feature_manifest.yaml").write_text(
        manifest_yaml(
            [
                FeatureInfo(name, "synthetic", str(dtype), "Synthetic test feature")
                for name, dtype in features.schema.items()
            ],
            "synthetic-phase-one",
        )
    )
    outputs = {path.name: file_hash(path) for path in directory.iterdir()}
    (directory / "metadata.json").write_text(json.dumps({"outputs": outputs}))


def test_test_files_are_read_only_after_frozen_model_and_never_affect_fit(
    model_data, tmp_path, monkeypatch
):
    from loan_review.model import pipeline

    data_dir = tmp_path / "phase-one"
    _write_phase_one(data_dir, model_data)
    output = tmp_path / "artifacts"
    original_read = pipeline._read_split
    calls = []

    def guarded_read(directory, name, metadata):
        if name == "test":
            assert list((output / "models").glob("*/model.txt"))
            assert list((output / "models").glob("*/calibrator.json"))
        calls.append(name)
        return original_read(directory, name, metadata)

    monkeypatch.setattr(pipeline, "_read_split", guarded_read)
    config = TrainingConfig(n_trials=1, num_boost_round=30, early_stopping_rounds=5)
    model_dir, report_dir = run_training(data_dir, output, config=config)
    assert calls == ["train", "val", "test"]
    assert report_dir.is_relative_to(output / "model_report")
    # Change test features/labels, then update the input integrity record.
    test = model_data["test"][0].with_columns(pl.lit(0.99).alias("EXT_SOURCE_1"))
    test.write_parquet(data_dir / "test_features.parquet")
    labels = pl.read_parquet(data_dir / "test_labels.parquet").with_columns(
        (1 - pl.col(TARGET)).alias(TARGET)
    )
    labels.write_parquet(data_dir / "test_labels.parquet")
    meta_path = data_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    for name in ("test_features.parquet", "test_labels.parquet"):
        metadata["outputs"][name] = file_hash(data_dir / name)
    meta_path.write_text(json.dumps(metadata))
    second_model, second_report = run_training(data_dir, output, config=config)
    assert second_model == model_dir
    assert second_report != report_dir
    assert load_model(second_model).model_version == load_model(model_dir).model_version


def test_actual_phase_one_aggregate_outputs_train_and_reload(raw_frames, tmp_path):
    from loan_review.data.pipeline import run_pipeline

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name, frame in zip(
        ("application_train", "bureau", "previous_application"), raw_frames, strict=True
    ):
        frame.write_csv(raw_dir / f"{name}.csv")
    phase_one = run_pipeline(
        raw_dir, tmp_path / "data-artifacts", tmp_path / "manifest.yaml"
    )
    model_dir, report_dir = run_training(
        phase_one,
        tmp_path / "model-artifacts",
        config=TrainingConfig(n_trials=1, num_boost_round=20, early_stopping_rounds=4),
    )
    model = load_model(model_dir)
    features = pl.read_parquet(phase_one / "test_features.parquet")
    model.reason_codes.require_coverage(features.columns)
    assert "BUREAU_AMT_CREDIT_SUM_MEAN" in model.encoder.feature_names
    assert "DAYS_EMPLOYED_SENTINEL" in model.encoder.feature_names
    assert len(model.predict_proba(features)) == 30
    assert (report_dir / "metrics.json").is_file()
    explanation = explain_applications(model, features.head(1))[0]
    assert set(explanation.shap_row) == set(features.columns)

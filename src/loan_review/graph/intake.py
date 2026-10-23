"""Deterministic application intake and Phase 1 compatible single-row features.

Incoming free text is untrusted. Intake returns it only as data in the normalized
application, and feature construction excludes it. This module makes no LLM calls
and never interpolates application text into an LLM system prompt.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import ValidationError

from loan_review.data.features import (
    ID,
    PROTECTED_ATTRIBUTES,
    FeatureInfo,
    build_features,
)
from loan_review.graph.schemas import (
    FieldError,
    IntakeResult,
    LoanApplication,
    optional_field_error,
)
from loan_review.model.preprocessing import FeatureEncoder

_DTYPES: dict[str, pl.DataType] = {
    name: getattr(pl, name)
    for name in (
        "Float64",
        "Float32",
        "Int64",
        "Int32",
        "Int16",
        "Int8",
        "UInt64",
        "UInt32",
        "UInt16",
        "UInt8",
        "String",
        "Boolean",
        "Null",
    )
}


def _dtype(name: str) -> pl.DataType:
    """Accept only Phase 1 scalar Parquet dtypes from a saved manifest."""
    try:
        return _DTYPES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported feature dtype in manifest: {name}") from error


class FeatureBuildError(ValueError):
    """Identify a missing or mismatched feature in the saved model schema."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


def _error_path(issue: Mapping[str, Any]) -> str:
    """Translate Pydantic paths into a stable field name without input values."""
    location = issue.get("loc", ())
    if location:
        return ".".join(str(part) for part in location)
    message = str(issue.get("msg", ""))
    for phrase in ("unsupported raw application column: ", " must be ", " is outside"):
        if phrase in message:
            if phrase == "unsupported raw application column: ":
                return message.split(phrase, 1)[1].strip()
            return message.split(phrase, 1)[0].removeprefix("Value error, ").strip()
    return "application"


def intake_node(
    payload: object, feature_manifest: Sequence[FeatureInfo] | None = None
) -> IntakeResult:
    """Validate and normalize once, returning field-level errors without coercion.

    Normalize only bounded untrusted employer text by trimming surrounding space;
    model inputs preserve their raw values for Phase 1 parity. Submitted TARGET
    and unrecognized fields are rejected. Errors never echo submitted values.
    """
    if isinstance(payload, LoanApplication):
        payload = payload.model_dump()
    if not isinstance(payload, Mapping):
        return IntakeResult(
            valid=False,
            errors=[
                FieldError(
                    field="application",
                    code="mapping_type",
                    message="Application must be an object",
                )
            ],
            normalized=None,
        )
    problems = [
        FieldError(field=str(name), code="invalid_optional_field", message=message)
        for name, value in payload.items()
        if name not in LoanApplication.model_fields
        if (message := optional_field_error(str(name), value)) is not None
    ]
    normalized: LoanApplication | None = None
    try:
        normalized = LoanApplication.model_validate(dict(payload))
    except ValidationError as error:
        validated_errors = [
            FieldError(
                field=_error_path(issue),
                code=str(issue["type"]),
                message=str(issue["msg"]),
            )
            for issue in error.errors(include_input=False, include_context=False)
        ]
        seen = {problem.field for problem in problems}
        problems.extend(
            problem for problem in validated_errors if problem.field not in seen
        )
    if feature_manifest is not None:
        missing = sorted(
            item.name
            for item in feature_manifest
            if item.source == "application_train.csv" and item.name not in payload
        )
        seen = {problem.field for problem in problems}
        problems.extend(
            FieldError(
                field=name,
                code="missing_model_feature",
                message="Required model input is missing",
            )
            for name in missing
            if name not in seen
        )
    if problems:
        return IntakeResult(valid=False, errors=problems, normalized=None)
    assert normalized is not None
    return IntakeResult(valid=True, errors=[], normalized=normalized)


def load_feature_manifest(path: Path) -> list[FeatureInfo]:
    """Load the exact Phase 1 manifest; reject the provisional pre-data catalog."""
    content = yaml.safe_load(path.read_text())
    if (
        not isinstance(content, dict)
        or content.get("status") == "catalog_pending_source_data"
    ):
        raise ValueError("A generated, versioned Phase 1 feature manifest is required")
    if not isinstance(content.get("version"), str) or not isinstance(
        content.get("features"), list
    ):
        raise ValueError("Feature manifest must contain version and features")
    features: list[FeatureInfo] = []
    for record in content["features"]:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "source",
            "dtype",
            "description",
        }:
            raise ValueError(
                "Each manifest feature needs name, source, dtype, description"
            )
        if not all(isinstance(value, str) and value for value in record.values()):
            raise ValueError("Manifest feature fields must be nonempty strings")
        _dtype(record["dtype"])
        features.append(FeatureInfo(**record))
    if not features or len({feature.name for feature in features}) != len(features):
        raise ValueError("Manifest must have unique features")
    return features


def _history_for_applicant(
    history: pl.DataFrame | None, application_id: int, source: str
) -> pl.DataFrame:
    """Require typed historical keys and restrict rows to this applicant."""
    if history is None:
        return pl.DataFrame(schema={ID: pl.Int64})
    if ID not in history.columns or not history.schema[ID].is_integer():
        raise FeatureBuildError(ID, f"{source}.{ID} must be an integer column")
    if history[ID].null_count():
        raise FeatureBuildError(ID, f"{source}.{ID} must not contain nulls")
    return history.filter(pl.col(ID) == application_id)


def build_application_features(
    application: LoanApplication,
    feature_manifest: Sequence[FeatureInfo],
    *,
    bureau: pl.DataFrame | None = None,
    previous_application: pl.DataFrame | None = None,
    encoder: FeatureEncoder | None = None,
) -> pl.DataFrame:
    """Build the exact saved-model row with Phase 1's feature implementation.

    The manifest must come from the model's versioned Phase 1 run. All original
    application features expected by that manifest are required at serving time;
    missing inputs raise a named error rather than becoming silent nulls.
    Historical tables are optional because an applicant may have no history;
    missing history statistics are null and counts are zero, as in Phase 1. The
    optional fitted encoder verifies column order and type compatibility without
    learning categories from this application.
    """
    if not feature_manifest:
        raise ValueError("A nonempty versioned feature manifest is required")
    names = [item.name for item in feature_manifest]
    if len(set(names)) != len(names):
        raise ValueError("The feature manifest contains duplicate names")
    if encoder is not None and names != encoder.feature_names:
        raise ValueError("Feature manifest and fitted model column order disagree")
    # Revalidate dumped data so even model_copy(update=...) cannot bypass intake.
    application = LoanApplication.model_validate(application.model_dump())
    values = application.raw_feature_values()
    raw_columns = [
        item for item in feature_manifest if item.source == "application_train.csv"
    ]
    for item in raw_columns:
        if item.name not in values:
            raise FeatureBuildError(
                item.name, f"Missing required model input: {item.name}"
            )
    fields: dict[str, pl.Series] = {
        ID: pl.Series(ID, [application.SK_ID_CURR], dtype=pl.Int64)
    }
    for name in PROTECTED_ATTRIBUTES:
        fields[name] = pl.Series(name, [values[name]])
    for item in raw_columns:
        if item.name == ID or item.name in PROTECTED_ATTRIBUTES:
            raise FeatureBuildError(
                item.name, "Protected and identifier columns are not model features"
            )
        try:
            dtype = (
                pl.DataType.from_python(type(values[item.name]))
                if False
                else _dtype(item.dtype)
            )
            fields[item.name] = pl.Series(
                item.name, [values[item.name]], dtype=dtype, strict=True
            )
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise FeatureBuildError(
                item.name, f"Input does not match saved dtype {item.dtype}"
            ) from error
    # DAYS_EMPLOYED may be represented in the manifest as Float64 after sentinel
    # replacement; it is raw signed integer here and Phase 1 converts it below.
    if "DAYS_EMPLOYED" not in fields:
        raise FeatureBuildError(
            "DAYS_EMPLOYED", "Missing required model input: DAYS_EMPLOYED"
        )
    frame = pl.DataFrame(fields)
    bureau_frame = _history_for_applicant(bureau, application.SK_ID_CURR, "bureau")
    previous_frame = _history_for_applicant(
        previous_application, application.SK_ID_CURR, "previous_application"
    )
    built, _ = build_features(frame, bureau_frame, previous_frame)
    for item in feature_manifest:
        if item.name not in built.columns:
            if item.source not in ("bureau.csv", "previous_application.csv"):
                raise FeatureBuildError(
                    item.name, f"Missing computed feature: {item.name}"
                )
            value = 0 if item.name.endswith("_COUNT") else None
            built = built.with_columns(pl.lit(value).alias(item.name))
    ordered = built.select(names)
    for item in feature_manifest:
        if str(ordered.schema[item.name]) != item.dtype:
            try:
                ordered = ordered.with_columns(
                    pl.col(item.name).cast(_dtype(item.dtype), strict=True)
                )
            except pl.exceptions.PolarsError as error:
                raise FeatureBuildError(
                    item.name, f"Computed feature differs from saved dtype {item.dtype}"
                ) from error
    if encoder is not None:
        encoder.transform(ordered)
    return ordered

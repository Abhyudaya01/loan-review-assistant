"""Prepare reproducible Home Credit datasets with applicant-isolated holdouts.

No out-of-time split is possible: application_train has no absolute application
dates (its day fields are relative offsets). A seeded random split therefore
measures same-population generalization, not robustness to future population,
policy, or economic drift. Historical tables also lack an absolute as-of anchor,
so a temporal leakage audit cannot be established from this split alone.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split

from loan_review.data.features import (
    ID,
    PROTECTED_ATTRIBUTES,
    TARGET,
    FeatureInfo,
    build_features,
)
from loan_review.data.proxy import proxy_report

PIPELINE_VERSION = "1.0.0"
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_NUMERIC = (
    "DAYS_EMPLOYED",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
)


class PipelineConfig(BaseModel):
    """Validate reproducibility and proxy-screen settings without coercion."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    correlation_threshold: Annotated[
        float, Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ] = 0.3
    mutual_information_threshold: Annotated[
        float, Field(ge=0.0, allow_inf_nan=False)
    ] = 0.1
    proxy_max_rows: Annotated[int, Field(ge=4)] = 20_000


@dataclass(frozen=True)
class DatasetSplit:
    """Keep model columns separate from row-aligned IDs/labels and audit attributes."""

    features: pl.DataFrame
    labels: pl.DataFrame  # SK_ID_CURR, TARGET; defines features' row order.
    audit_attributes: pl.DataFrame  # SK_ID_CURR plus protected attributes.


@dataclass(frozen=True)
class PreparedData:
    """Contain the three splits, exact emitted feature schema, and training screen."""

    splits: dict[str, DatasetSplit]
    manifest: list[FeatureInfo]
    proxy_report: pl.DataFrame
    config: PipelineConfig


def _validate_ids(frame: pl.DataFrame, source: str, *, unique: bool) -> None:
    if ID not in frame or not frame.schema[ID].is_integer():
        raise ValueError(f"{source}.{ID} must be an integer column")
    if frame[ID].null_count():
        raise ValueError(f"{source}.{ID} must not contain nulls")
    if unique and frame[ID].n_unique() != frame.height:
        raise ValueError(f"{source}.{ID} must be unique before splitting")


def split_applications(
    applications: pl.DataFrame, *, seed: int
) -> dict[str, pl.DataFrame]:
    """Assign 70/15/15 stratified IDs before any aggregation or fitted operation.

    Integer rounding can shift fractions by one row. Rows are sorted by ID before
    seeded shuffling, so CSV row order does not affect membership. See the module
    docstring for why this is not an out-of-time evaluation.
    """
    required = {ID, TARGET, *PROTECTED_ATTRIBUTES, *REQUIRED_NUMERIC}
    missing = required.difference(applications.columns)
    if missing:
        raise ValueError(f"application_train is missing columns: {sorted(missing)}")
    _validate_ids(applications, "application_train", unique=True)
    target = applications[TARGET]
    if (
        not target.dtype.is_integer()
        or target.null_count()
        or set(target.unique().to_list()) != {0, 1}
    ):
        raise ValueError(
            "TARGET must contain both integer classes 0 and 1 without nulls"
        )
    for column in (*REQUIRED_NUMERIC, "DAYS_BIRTH"):
        if not applications.schema[column].is_numeric():
            raise ValueError(f"application_train.{column} must be numeric")
    for column in ("CODE_GENDER", "NAME_FAMILY_STATUS"):
        if applications.schema[column] != pl.String:
            raise ValueError(f"application_train.{column} must be a string")
    ordered = applications.sort(ID)
    try:
        train, holdout = train_test_split(
            list(range(ordered.height)),
            test_size=0.3,
            stratify=ordered[TARGET].to_list(),
            random_state=seed,
        )
        val, test = train_test_split(
            holdout,
            test_size=0.5,
            stratify=ordered[holdout][TARGET].to_list(),
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError(
            "Insufficient class counts for stratified 70/15/15 splits"
        ) from error
    splits = {
        name: ordered[indices].sort(ID)
        for name, indices in zip(SPLIT_NAMES, (train, val, test), strict=True)
    }
    if any(
        set(frame[TARGET].unique().to_list()) != {0, 1} for frame in splits.values()
    ):
        raise ValueError("Each split must contain both TARGET classes; add more rows")
    return splits


def _prepare_splits(
    assignments: dict[str, pl.DataFrame],
    bureau: pl.DataFrame,
    previous: pl.DataFrame,
    config: PipelineConfig,
) -> PreparedData:
    for frame, source in ((bureau, "bureau"), (previous, "previous_application")):
        _validate_ids(frame, source, unique=False)
    splits: dict[str, DatasetSplit] = {}
    manifest: list[FeatureInfo] = []
    for name, applications in assignments.items():
        features, split_manifest = build_features(applications, bureau, previous)
        if manifest and split_manifest != manifest:
            raise ValueError("Feature schemas differ across splits")
        manifest = split_manifest
        splits[name] = DatasetSplit(
            features,
            applications.select(ID, TARGET),
            applications.select(ID, *PROTECTED_ATTRIBUTES),
        )
    training = splits["train"]
    report = proxy_report(
        training.features,
        training.audit_attributes,
        seed=config.seed,
        correlation_threshold=config.correlation_threshold,
        mutual_information_threshold=config.mutual_information_threshold,
        max_rows=config.proxy_max_rows,
    )
    return PreparedData(splits, manifest, report, config)


def prepare_data(
    applications: pl.DataFrame,
    bureau: pl.DataFrame,
    previous: pl.DataFrame,
    config: PipelineConfig | None = None,
) -> PreparedData:
    """Split applications first, then aggregate histories independently per split."""
    config = config or PipelineConfig()
    assignments = split_applications(applications, seed=config.seed)
    return _prepare_splits(assignments, bureau, previous, config)


def manifest_yaml(features: list[FeatureInfo], artifact_version: str) -> str:
    """Serialize the flat manifest using JSON-quoted YAML scalars."""
    lines = [
        f"version: {json.dumps(artifact_version)}",
        f"pipeline_version: {json.dumps(PIPELINE_VERSION)}",
        "features:",
    ]
    for feature in features:
        for index, (key, value) in enumerate(asdict(feature).items()):
            prefix = "  - " if index == 0 else "    "
            lines.append(f"{prefix}{key}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("artifacts"),
    manifest_path: Path = Path("configs/feature_manifest.yaml"),
    config: PipelineConfig | None = None,
) -> Path:
    """Read CSVs and write Parquet plus content-versioned metadata and proxy reports.

    Each artifacts/<version>/ directory contains immutable-by-convention run
    outputs, including its manifest snapshot and file hashes. The requested
    configs manifest and artifacts/proxy_report.csv are latest-run copies. IDs
    live in each split's labels and audit files; feature Parquets contain only
    model inputs and align row-for-row with their corresponding labels.
    """
    config = config or PipelineConfig()
    paths = {
        name: raw_dir / f"{name}.csv"
        for name in (
            "application_train",
            "bureau",
            "previous_application",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing source CSV: {path}")
    input_hashes = {path.name: _sha256(path) for path in paths.values()}
    applications = pl.read_csv(paths["application_train"], infer_schema_length=None)
    assignments = split_applications(applications, seed=config.seed)
    # Histories are not even loaded until application membership is fixed.
    bureau = pl.read_csv(paths["bureau"], infer_schema_length=None)
    previous = pl.read_csv(paths["previous_application"], infer_schema_length=None)
    prepared = _prepare_splits(assignments, bureau, previous, config)
    if input_hashes != {path.name: _sha256(path) for path in paths.values()}:
        raise ValueError("Source CSVs changed during the run; retry with stable inputs")
    provenance = {
        "pipeline_version": PIPELINE_VERSION,
        "config": config.model_dump(),
        "inputs": input_hashes,
        "code": {
            path.name: _sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "libraries": {
            name: version(name) for name in ("polars", "numpy", "scikit-learn")
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()[:16]
    artifact_version = f"data-v{PIPELINE_VERSION}-{fingerprint}"
    run_dir = output_dir / artifact_version
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, split in prepared.splits.items():
        split.features.write_parquet(run_dir / f"{name}_features.parquet")
        split.labels.write_parquet(run_dir / f"{name}_labels.parquet")
        split.audit_attributes.write_parquet(
            run_dir / f"{name}_audit_attributes.parquet"
        )
    # Include split membership in the complete keyed audit frame for convenience.
    audit = pl.concat(
        [
            split.audit_attributes.with_columns(pl.lit(name).alias("split"))
            for name, split in prepared.splits.items()
        ]
    ).sort(ID)
    audit.write_parquet(run_dir / "audit_attributes.parquet")
    report = prepared.proxy_report.with_columns(
        pl.lit(artifact_version).alias("artifact_version"),
        pl.lit("train").alias("split"),
        pl.lit(config.correlation_threshold).alias("correlation_threshold"),
        pl.lit(config.mutual_information_threshold).alias(
            "mutual_information_threshold"
        ),
    )
    report.write_csv(run_dir / "proxy_report.csv")
    manifest_text = manifest_yaml(prepared.manifest, artifact_version)
    (run_dir / "feature_manifest.yaml").write_text(manifest_text)
    metadata = {
        **provenance,
        "artifact_version": artifact_version,
        "split_rows": {
            name: split.features.height for name, split in prepared.splits.items()
        },
        "proxy_screen_split": "train",
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "metadata.json"
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text)
    report.write_csv(output_dir / "proxy_report.csv")
    return run_dir


def main() -> None:
    """Run the offline data pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Prepare Home Credit data and proxy diagnostics."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/feature_manifest.yaml")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correlation-threshold", type=float, default=0.3)
    parser.add_argument("--mi-threshold", type=float, default=0.1)
    parser.add_argument("--proxy-max-rows", type=int, default=20_000)
    args = parser.parse_args()
    config = PipelineConfig(
        seed=args.seed,
        correlation_threshold=args.correlation_threshold,
        mutual_information_threshold=args.mi_threshold,
        proxy_max_rows=args.proxy_max_rows,
    )
    print(run_pipeline(args.raw_dir, args.output_dir, args.manifest, config))

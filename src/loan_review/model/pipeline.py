"""Train on Phase 1 Parquets, freeze the model, then evaluate the untouched test set."""

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from loan_review.data.features import ID, TARGET
from loan_review.model.bundle import content_hash, file_hash
from loan_review.model.config import TrainingConfig, load_thresholds
from loan_review.model.evaluation import evaluate_test
from loan_review.model.reasons import load_reason_codes
from loan_review.model.training import train_model


def _read_split(
    directory: Path, name: str, metadata: dict[str, Any]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Verify Phase 1 file hashes before loading a requested split."""
    for suffix in ("features", "labels"):
        filename = f"{name}_{suffix}.parquet"
        if file_hash(directory / filename) != metadata["outputs"][filename]:
            raise ValueError(f"Phase 1 checksum mismatch: {filename}")
    features = pl.read_parquet(directory / f"{name}_features.parquet")
    labels = pl.read_parquet(directory / f"{name}_labels.parquet")
    if labels.columns != [ID, TARGET] or labels.height != features.height:
        raise ValueError(f"{name} features/labels are not aligned")
    if (
        not labels.schema[ID].is_integer()
        or labels[ID].null_count()
        or labels[ID].n_unique() != labels.height
    ):
        raise ValueError(f"{name} applicant IDs must be unique nonnull integers")
    return features, labels


def run_training(
    data_dir: Path,
    artifact_root: Path = Path("artifacts"),
    *,
    reason_codes_path: Path = Path("configs/reason_codes.yaml"),
    thresholds_path: Path = Path("configs/thresholds.yaml"),
    config: TrainingConfig | None = None,
) -> tuple[Path, Path]:
    """Load test files only after saving the model trained on a Phase 1 run."""
    metadata = json.loads((data_dir / "metadata.json").read_text())
    manifest_path = data_dir / "feature_manifest.yaml"
    if file_hash(manifest_path) != metadata["outputs"][manifest_path.name]:
        raise ValueError("Phase 1 feature manifest checksum mismatch")
    manifest = yaml.safe_load(manifest_path.read_text())
    train_features, train_labels = _read_split(data_dir, "train", metadata)
    val_features, val_labels = _read_split(data_dir, "val", metadata)
    if set(train_labels[ID]).intersection(val_labels[ID]):
        raise ValueError("Applicants overlap between train and validation")
    if train_features.columns != [feature["name"] for feature in manifest["features"]]:
        raise ValueError("Phase 1 manifest and model features disagree")
    training_files = (
        "train_features.parquet",
        "train_labels.parquet",
        "val_features.parquet",
        "val_labels.parquet",
    )
    data_hash = content_hash(
        {name: metadata["outputs"][name] for name in training_files}
    )
    bundle = train_model(
        train_features,
        train_labels[TARGET],
        val_features,
        val_labels[TARGET],
        data_hash=data_hash,
        reason_codes=load_reason_codes(reason_codes_path),
        thresholds=load_thresholds(thresholds_path),
        config=config,
    )
    model_dir = bundle.save(artifact_root)
    # This is the first point at which test Parquets are hashed, opened, or scored.
    test_features, test_labels = _read_split(data_dir, "test", metadata)
    if set(test_labels[ID]).intersection(set(train_labels[ID]) | set(val_labels[ID])):
        raise ValueError("Test applicants overlap with train or validation")
    test_hash = content_hash(
        {
            name: metadata["outputs"][name]
            for name in ("test_features.parquet", "test_labels.parquet")
        }
    )
    report_dir = evaluate_test(
        bundle,
        test_features,
        test_labels[TARGET],
        report_root=artifact_root / "model_report",
        test_data_hash=test_hash,
    )
    return model_dir, report_dir


def main() -> None:
    """Run bounded tuning and evaluation on an explicit Phase 1 directory."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the loan-review risk model."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--reason-codes", type=Path, default=Path("configs/reason_codes.yaml")
    )
    parser.add_argument(
        "--thresholds", type=Path, default=Path("configs/thresholds.yaml")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    args = parser.parse_args()
    model, report = run_training(
        args.data_dir,
        args.artifact_root,
        reason_codes_path=args.reason_codes,
        thresholds_path=args.thresholds,
        config=TrainingConfig(
            seed=args.seed,
            n_trials=args.trials,
            num_boost_round=args.rounds,
            early_stopping_rounds=args.early_stopping_rounds,
        ),
    )
    print(json.dumps({"model": str(model), "report": str(report)}, indent=2))

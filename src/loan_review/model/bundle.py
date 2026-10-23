"""Persist, verify, and reload a complete versioned scoring bundle without pickle."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from loan_review.model.calibration import IsotonicCalibrator, validate_probabilities
from loan_review.model.config import RiskThresholds
from loan_review.model.preprocessing import FeatureEncoder, FloatArray
from loan_review.model.reasons import ReasonCodes


def content_hash(value: object) -> str:
    """Hash canonical JSON for stable version identities across reloads."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    """Hash file contents without loading large artifacts into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    """Record HEAD, or mark a repository with no commits as UNCOMMITTED."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


@dataclass(frozen=True)
class ModelBundle:
    """Keep the model, encoder, calibrator, and fixed configs together."""

    booster: lgb.Booster
    encoder: FeatureEncoder
    calibrator: IsotonicCalibrator
    thresholds: RiskThresholds
    reason_codes: ReasonCodes
    metadata: dict[str, Any]

    @property
    def model_version(self) -> str:
        """Return the stored identity, unchanged by saving or loading."""
        return str(self.metadata["model_version"])

    def predict_raw_probability(self, features: pl.DataFrame) -> FloatArray:
        """Predict default probability before isotonic calibration."""
        values = np.asarray(
            self.booster.predict(self.encoder.transform(features), num_threads=1),
            dtype=np.float64,
        )
        validate_probabilities(values)
        return values

    def predict_proba(self, features: pl.DataFrame) -> FloatArray:
        """Return calibrated P(TARGET=1), never a credit decision."""
        return self.calibrator.predict(self.predict_raw_probability(features))

    def score(self, features: pl.DataFrame) -> pl.DataFrame:
        """Return row-aligned probabilities and descriptive reviewer risk bands."""
        raw = self.predict_raw_probability(features)
        calibrated = self.calibrator.predict(raw)
        return pl.DataFrame(
            {
                "raw_default_probability": raw,
                "default_probability": calibrated,
                "risk_band": [
                    self.thresholds.band(float(value)) for value in calibrated
                ],
                "model_version": [self.model_version] * len(calibrated),
            }
        )

    def save(self, artifact_root: Path = Path("artifacts")) -> Path:
        """Save native model text, JSON configuration, and integrity hashes."""
        if content_hash(self.metadata["version_inputs"]) != self.model_version:
            raise ValueError("Model version does not match its provenance")
        directory = artifact_root / "models" / self.model_version
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"))
        for filename, value in (
            ("encoder.json", self.encoder),
            ("calibrator.json", self.calibrator),
            ("thresholds.json", self.thresholds),
            ("reason_codes.json", self.reason_codes),
        ):
            (directory / filename).write_text(value.model_dump_json(indent=2) + "\n")
        explainer_metadata = {
            "model_version": self.model_version,
            "algorithm": "TreeSHAP",
            "model_output": "raw",
            "feature_perturbation": "tree_path_dependent",
            "positive_class": 1,
            "units": "default_log_odds",
            "background": "training leaf counts",
            "features": self.encoder.feature_names,
            "caveat": (
                "Correlated features can redistribute SHAP contributions; "
                "explanations are not causal and do not explain isotonic probabilities."
            ),
        }
        (directory / "explainer_metadata.json").write_text(
            json.dumps(explainer_metadata, indent=2) + "\n"
        )
        hashes = {
            path.name: file_hash(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "metadata.json"
        }
        (directory / "metadata.json").write_text(
            json.dumps(
                {**self.metadata, "files": hashes},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        return directory


def load_model(directory: Path) -> ModelBundle:
    """Verify hashes and reload without recomputing or changing model_version."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if content_hash(metadata["version_inputs"]) != metadata["model_version"]:
        raise ValueError("Stored model version does not match its provenance")
    required = {
        "model.txt",
        "encoder.json",
        "calibrator.json",
        "thresholds.json",
        "reason_codes.json",
        "explainer_metadata.json",
    }
    if not required.issubset(metadata["files"]):
        raise ValueError("Missing integrity hashes for model bundle files")
    for name in required:
        if file_hash(directory / name) != metadata["files"][name]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    encoder = FeatureEncoder.model_validate_json(
        (directory / "encoder.json").read_text()
    )
    calibrator = IsotonicCalibrator.model_validate_json(
        (directory / "calibrator.json").read_text()
    )
    thresholds = RiskThresholds.model_validate_json(
        (directory / "thresholds.json").read_text()
    )
    reasons = ReasonCodes.model_validate_json(
        (directory / "reason_codes.json").read_text()
    )
    reasons.require_coverage(encoder.feature_names)
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    if booster.feature_name() != encoder.feature_names:
        raise ValueError("Model and encoder feature schemas disagree")
    return ModelBundle(booster, encoder, calibrator, thresholds, reasons, metadata)

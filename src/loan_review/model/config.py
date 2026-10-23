"""Validate versioned training settings and reviewer-only probability bands."""

from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject unexpected fields, nonfinite floats, and implicit string coercion."""

    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, allow_inf_nan=False
    )


class TrainingConfig(StrictModel):
    """Bound tuning effort and fix every stochastic training seed."""

    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 42
    n_trials: Annotated[int, Field(ge=1, le=30)] = 10
    num_boost_round: Annotated[int, Field(ge=2, le=2000)] = 500
    early_stopping_rounds: Annotated[int, Field(ge=1)] = 40

    @model_validator(mode="after")
    def validate_stopping(self) -> Self:
        if self.early_stopping_rounds >= self.num_boost_round:
            raise ValueError("early_stopping_rounds must be less than num_boost_round")
        return self


class RiskThresholds(StrictModel):
    """Define fixed descriptive risk bands, never credit decisions."""

    version: str
    purpose: str
    probability: Literal["isotonic_calibrated_default_probability"]
    low_upper: Annotated[float, Field(gt=0, lt=1)]
    medium_upper: Annotated[float, Field(gt=0, lt=1)]

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.low_upper >= self.medium_upper:
            raise ValueError("Risk-band boundaries must be strictly increasing")
        return self

    def band(self, probability: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
        """Label calibrated default probability without making a credit decision."""
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probability must be finite and within [0, 1]")
        if probability < self.low_upper:
            return "LOW"
        if probability < self.medium_upper:
            return "MEDIUM"
        return "HIGH"


def load_thresholds(path: Path = Path("configs/thresholds.yaml")) -> RiskThresholds:
    """Load and strictly validate reviewer risk bands."""
    return RiskThresholds.model_validate(yaml.safe_load(path.read_text()))

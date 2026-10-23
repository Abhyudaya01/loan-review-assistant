"""Fit isotonic regression on validation and persist portable interpolation knots."""

from typing import Self

import numpy as np
from pydantic import model_validator
from sklearn.isotonic import IsotonicRegression

from loan_review.model.config import StrictModel
from loan_review.model.preprocessing import FloatArray


def validate_probabilities(values: FloatArray) -> None:
    """Reject invalid probabilities rather than silently clipping scoring errors."""
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("Probabilities must be a finite vector within [0, 1]")


class IsotonicCalibrator(StrictModel):
    """Store sklearn isotonic knots and reproduce out_of_bounds='clip' predictions."""

    x_thresholds: list[float]
    y_thresholds: list[float]

    @model_validator(mode="after")
    def validate_knots(self) -> Self:
        x, y = np.asarray(self.x_thresholds), np.asarray(self.y_thresholds)
        if not len(x) or len(x) != len(y):
            raise ValueError("Calibrator knots must be nonempty and aligned")
        validate_probabilities(x)
        validate_probabilities(y)
        if (np.diff(x) <= 0).any() or (np.diff(y) < 0).any():
            raise ValueError(
                "Isotonic knots must have increasing x and nondecreasing y"
            )
        return self

    @classmethod
    def fit(
        cls, validation_probabilities: FloatArray, validation_labels: FloatArray
    ) -> Self:
        """Fit validation-only isotonic knots; reports disclose reuse for tuning."""
        validate_probabilities(validation_probabilities)
        if len(validation_labels) != len(validation_probabilities) or set(
            validation_labels
        ) != {0, 1}:
            raise ValueError(
                "Calibration labels must be aligned and contain both binary classes"
            )
        fitted = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            validation_probabilities,
            validation_labels,
        )
        return cls(
            x_thresholds=fitted.X_thresholds_.tolist(),
            y_thresholds=fitted.y_thresholds_.tolist(),
        )

    def predict(self, probabilities: FloatArray) -> FloatArray:
        """Interpolate saved knots and clip only beyond the fitted input range."""
        validate_probabilities(probabilities)
        result = np.asarray(
            np.interp(probabilities, self.x_thresholds, self.y_thresholds),
            dtype=np.float64,
        )
        validate_probabilities(result)
        return result

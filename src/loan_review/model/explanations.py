"""Explain default log-odds with TreeSHAP and deterministic contribution ordering.

Correlated features can exchange SHAP attribution under small data/model changes
or different dependence assumptions. Rankings are not stable causal explanations.
This explainer describes the raw binary LightGBM margin, not the nonlinear
isotonic-calibrated probability. Positive values push toward TARGET=1 (default).
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
import shap

from loan_review.model.bundle import ModelBundle
from loan_review.model.config import StrictModel
from loan_review.model.reasons import FeatureContribution


class ApplicationExplanation(StrictModel):
    """Keep baseline and sorted evidence for one row, optionally keyed by applicant."""

    row_index: int
    application_id: int | None
    model_version: str
    base_value: float
    raw_log_odds: float
    contributions: list[FeatureContribution]

    @property
    def shap_row(self) -> dict[str, float]:
        """Expose a feature-name mapping suitable for top_reasons."""
        return {item.feature: item.shap_value for item in self.contributions}


def explain_applications(
    bundle: ModelBundle,
    features: pl.DataFrame,
    application_ids: Sequence[int] | None = None,
) -> list[ApplicationExplanation]:
    """Sort each application's contributions by push toward default."""
    if application_ids is not None and len(application_ids) != features.height:
        raise ValueError("Application IDs must align with feature rows")
    matrix = bundle.encoder.transform(features)
    explainer = shap.TreeExplainer(
        bundle.booster, model_output="raw", feature_perturbation="tree_path_dependent"
    )
    values = explainer.shap_values(matrix, check_additivity=True)
    base = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    if isinstance(values, list):  # Older supported SHAP versions return two classes.
        values = values[1]
    contributions = np.asarray(values, dtype=np.float64)
    if contributions.ndim == 3 and contributions.shape[-1] == 2:
        contributions = contributions[:, :, 1]
    baseline = float(base[-1])
    raw_scores = np.asarray(
        bundle.booster.predict(matrix, raw_score=True, num_threads=1), dtype=np.float64
    )
    if contributions.shape != matrix.shape or not np.isfinite(contributions).all():
        raise ValueError("TreeSHAP returned invalid contribution dimensions or values")
    if not np.allclose(
        baseline + contributions.sum(axis=1), raw_scores, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("SHAP contributions do not add up to the raw default log-odds")
    return [
        ApplicationExplanation(
            row_index=index,
            application_id=None if application_ids is None else application_ids[index],
            model_version=bundle.model_version,
            base_value=baseline,
            raw_log_odds=float(raw_scores[index]),
            contributions=sorted(
                [
                    FeatureContribution(feature=name, shap_value=float(value))
                    for name, value in zip(
                        bundle.encoder.feature_names, row, strict=True
                    )
                ],
                key=lambda item: (-item.shap_value, item.feature),
            ),
        )
        for index, row in enumerate(contributions)
    ]

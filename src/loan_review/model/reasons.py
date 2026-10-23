"""Select fixed reason codes from signed SHAP evidence, without an LLM.

SHAP allocations can be unstable when features are correlated: similar models or
background assumptions may distribute credit differently across substitutes.
These contributions are associative, not causal. Grouping related features reduces
redundant reasons but does not remove that instability. Positive SHAP means an
increase in the underlying model's default log-odds relative to its baseline; it
is not a claim about a feature's monotonic direction or calibrated probability.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import model_validator

from loan_review.data.features import EXCLUDED_FEATURES
from loan_review.model.config import StrictModel


class ReasonDefinition(StrictModel):
    """Describe a fixed reviewer-facing reason and its SHAP sign convention."""

    code: str
    short_text: str
    direction: Literal["positive_shap_increases_default_log_odds"]


class ReasonCodes(StrictModel):
    """Map every allowed model feature explicitly; fail closed on unknown features."""

    version: str
    purpose: str
    features: dict[str, ReasonDefinition]

    @model_validator(mode="after")
    def validate_definitions(self) -> Self:
        if EXCLUDED_FEATURES.intersection(self.features):
            raise ValueError(
                "Reason dictionary must not include protected attributes or identifiers"
            )
        by_code: dict[str, ReasonDefinition] = {}
        for definition in self.features.values():
            if not definition.code.strip() or not definition.short_text.strip():
                raise ValueError("Reason codes and short text must be nonempty")
            if definition.code in by_code and by_code[definition.code] != definition:
                raise ValueError(f"Conflicting definitions for code {definition.code}")
            by_code[definition.code] = definition
        return self

    def require_coverage(self, features: list[str]) -> None:
        """Reject training or explanation if any model feature has no fixed reason."""
        missing = set(features).difference(self.features)
        if missing:
            raise ValueError(f"Features without reason codes: {sorted(missing)}")


class FeatureContribution(StrictModel):
    """Record a feature's signed contribution in default log-odds units."""

    feature: str
    shap_value: float


class ReasonEvidence(ReasonDefinition):
    """Support one deduplicated reason with positive contributions and any offsets."""

    net_shap_value: float
    supporting_features: list[FeatureContribution]
    offsetting_features: list[FeatureContribution]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.net_shap_value <= 0 or not self.supporting_features:
            raise ValueError("A reason must have positive net evidence and support")
        if any(item.shap_value <= 0 for item in self.supporting_features):
            raise ValueError("Supporting features must increase risk")
        if any(item.shap_value >= 0 for item in self.offsetting_features):
            raise ValueError("Offsetting features must reduce risk")
        return self


def load_reason_codes(path: Path = Path("configs/reason_codes.yaml")) -> ReasonCodes:
    """Read the versioned fixed dictionary using strict schema validation."""
    return ReasonCodes.model_validate(yaml.safe_load(path.read_text()))


def top_reasons(
    shap_row: Mapping[str, float],
    k: int = 4,
    *,
    reason_codes: ReasonCodes | None = None,
) -> list[ReasonEvidence]:
    """Return up to k codes with positive net SHAP, sorted by push toward default.

    Sum all signed contributions for each code before filtering, so a small
    positive feature cannot hide larger negative evidence for the same reason.
    Supporting features are strictly positive; offsets are exposed separately.
    Zero and negative groups are omitted. Ties use code/feature names. The LLM
    does not select, rank, or modify these codes. See module correlation caveat.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    dictionary = reason_codes if reason_codes is not None else load_reason_codes()
    dictionary.require_coverage(list(shap_row))
    grouped: dict[str, list[FeatureContribution]] = {}
    definitions: dict[str, ReasonDefinition] = {}
    for name, value in shap_row.items():
        contribution = FeatureContribution(feature=name, shap_value=value)
        definition = dictionary.features[name]
        grouped.setdefault(definition.code, []).append(contribution)
        definitions[definition.code] = definition
    reasons: list[ReasonEvidence] = []
    for code, contributions in grouped.items():
        # Stable summation order makes output independent of mapping insertion order.
        contributions.sort(key=lambda item: (-item.shap_value, item.feature))
        net = sum(item.shap_value for item in contributions)
        if net <= 0:
            continue
        reasons.append(
            ReasonEvidence(
                **definitions[code].model_dump(),
                net_shap_value=net,
                supporting_features=[
                    item for item in contributions if item.shap_value > 0
                ],
                offsetting_features=[
                    item for item in contributions if item.shap_value < 0
                ],
            )
        )
    return sorted(reasons, key=lambda reason: (-reason.net_shap_value, reason.code))[:k]

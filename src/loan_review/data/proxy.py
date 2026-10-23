"""Screen training features for association with separate protected audit attributes."""

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

from loan_review.data.features import PROTECTED_ATTRIBUTES

FloatArray = NDArray[np.float64]
REPORT_SCHEMA = {
    "feature": pl.String,
    "protected_attribute": pl.String,
    "n_observations": pl.Int64,
    "correlation": pl.Float64,
    "correlation_comparison": pl.String,
    "mutual_information": pl.Float64,
    "correlation_flag": pl.Boolean,
    "mutual_information_flag": pl.Boolean,
    "flagged": pl.Boolean,
    "status": pl.String,
}


def _encode(series: pl.Series) -> tuple[FloatArray, bool]:
    """Encode categories only for MI; numeric low-cardinality integers are discrete."""
    if series.dtype.is_numeric():
        values = np.asarray(series.to_list(), dtype=np.float64)
        discrete = np.unique(values).size <= 2 or (
            series.dtype.is_integer() and np.unique(values).size <= 20
        )
        return values, discrete
    _, codes = np.unique(np.asarray(series.to_list(), dtype=str), return_inverse=True)
    return codes.astype(np.float64), True


def _correlation_columns(series: pl.Series) -> list[tuple[str, FloatArray]]:
    """One-hot categorical values so arbitrary category codes never imply ordering."""
    values, _ = _encode(series)
    if series.dtype.is_numeric():
        return [(series.name, values)]
    categories = sorted(set(str(value) for value in series.to_list()))
    return [
        (f"{series.name}={category}", (values == index).astype(np.float64))
        for index, category in enumerate(categories)
    ]


def _max_correlation(x: pl.Series, y: pl.Series) -> tuple[float | None, str | None]:
    best: float | None = None
    comparison: str | None = None
    for x_name, x_values in _correlation_columns(x):
        for y_name, y_values in _correlation_columns(y):
            if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
                continue
            value = float(np.corrcoef(x_values, y_values)[0, 1])
            if np.isfinite(value) and (best is None or abs(value) > abs(best)):
                best, comparison = value, f"{x_name} vs {y_name}"
    return best, comparison


def _mutual_information(x: pl.Series, y: pl.Series, seed: int) -> float | None:
    x_values, x_discrete = _encode(x)
    y_values, _ = _encode(y)
    # DAYS_BIRTH is continuous; gender and family status are categorical.
    y_discrete = y.name != "DAYS_BIRTH"
    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return 0.0
    if x_discrete and y_discrete:
        return float(mutual_info_score(x_values, y_values))
    # Nearest-neighbor estimators need repeated observations in discrete groups.
    if x_discrete or y_discrete:
        groups = x_values if x_discrete else y_values
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        repeated = counts[inverse] >= 2
        x_values, y_values = x_values[repeated], y_values[repeated]
    if len(x_values) < 4:
        return None
    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return None
    estimator = mutual_info_classif if y_discrete else mutual_info_regression
    return float(
        estimator(
            x_values.reshape(-1, 1),
            y_values,
            discrete_features=x_discrete,
            n_neighbors=min(3, len(x_values) - 1),
            random_state=seed,
        )[0]
    )


def proxy_report(
    features: pl.DataFrame,
    audit_attributes: pl.DataFrame,
    *,
    seed: int,
    correlation_threshold: float = 0.3,
    mutual_information_threshold: float = 0.1,
    max_rows: int = 20_000,
) -> pl.DataFrame:
    """Screen row-aligned TRAINING frames; never fit on validation or test data.

    MI is measured in nats. Numeric correlations use Pearson; categorical sides
    use one-hot indicators and report the largest absolute pairwise correlation.
    Missing/nonfinite pairs are excluded, not imputed. Numeric integers with at
    most 20 values (or any binary numeric variable) are discrete for MI. A seeded
    sample caps runtime. All pairs are reported, with explicit flags and coverage;
    an unestimable statistic is null, not evidence of no association. These are
    screening heuristics, not proof of fairness or grounds for a credit decision.
    """
    if features.height != audit_attributes.height:
        raise ValueError("Features and audit attributes must be row-aligned")
    if features.height > max_rows:
        indices = np.sort(
            np.random.default_rng(seed).choice(
                features.height,
                size=max_rows,
                replace=False,
            )
        ).tolist()
        features, audit_attributes = features[indices], audit_attributes[indices]
    rows: list[dict[str, object]] = []
    for feature in features.columns:
        for attribute in PROTECTED_ATTRIBUTES:
            pairs = pl.DataFrame(
                {feature: features[feature], attribute: audit_attributes[attribute]}
            )
            pairs = pairs.drop_nulls()
            for column in (feature, attribute):
                if pairs.schema[column].is_numeric():
                    pairs = pairs.filter(pl.col(column).is_finite())
            correlation, comparison, mi = None, None, None
            status = "insufficient_data"
            if pairs.height >= 4:
                x, y = pairs[feature], pairs[attribute]
                correlation, comparison = _max_correlation(x, y)
                mi = _mutual_information(x, y, seed)
                status = (
                    "ok"
                    if correlation is not None and mi is not None
                    else "constant_or_unestimable"
                )
            corr_flag = (
                correlation is not None and abs(correlation) > correlation_threshold
            )
            mi_flag = mi is not None and mi > mutual_information_threshold
            rows.append(
                {
                    "feature": feature,
                    "protected_attribute": attribute,
                    "n_observations": pairs.height,
                    "correlation": correlation,
                    "correlation_comparison": comparison,
                    "mutual_information": mi,
                    "correlation_flag": corr_flag,
                    "mutual_information_flag": mi_flag,
                    "flagged": corr_flag or mi_flag,
                    "status": status,
                }
            )
    return pl.DataFrame(rows, schema=REPORT_SCHEMA).sort(
        "feature", "protected_attribute"
    )

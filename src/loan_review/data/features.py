"""Build applicant-local features without targets or population-level fitting."""

from dataclasses import dataclass

import polars as pl

ID = "SK_ID_CURR"
TARGET = "TARGET"
PROTECTED_ATTRIBUTES = ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS")
EXCLUDED_FEATURES = {ID, TARGET, *PROTECTED_ATTRIBUTES, "AGE", "age"}

# Explicit allowlists prevent labels or unexpected history columns being aggregated.
BUREAU_AGGREGATIONS: dict[str, tuple[str, ...]] = {
    "AMT_CREDIT_SUM": ("sum", "mean", "max"),
    "AMT_CREDIT_SUM_DEBT": ("sum", "mean", "max"),
    "AMT_CREDIT_SUM_LIMIT": ("sum", "mean"),
    "AMT_CREDIT_SUM_OVERDUE": ("sum", "mean", "max"),
    "AMT_CREDIT_MAX_OVERDUE": ("max",),
    "CREDIT_DAY_OVERDUE": ("mean", "max"),
    "CNT_CREDIT_PROLONG": ("sum",),
    "DAYS_CREDIT": ("mean", "min", "max"),
    "DAYS_CREDIT_ENDDATE": ("mean", "min", "max"),
    "DAYS_ENDDATE_FACT": ("mean", "min", "max"),
    "AMT_ANNUITY": ("sum", "mean"),
}
PREVIOUS_AGGREGATIONS: dict[str, tuple[str, ...]] = {
    "AMT_ANNUITY": ("sum", "mean", "max"),
    "AMT_APPLICATION": ("sum", "mean", "max"),
    "AMT_CREDIT": ("sum", "mean", "max"),
    "AMT_DOWN_PAYMENT": ("sum", "mean"),
    "AMT_GOODS_PRICE": ("sum", "mean", "max"),
    "CNT_PAYMENT": ("mean", "max"),
    "RATE_DOWN_PAYMENT": ("mean",),
    "DAYS_DECISION": ("mean", "min", "max"),
}
RATIOS = {
    "PAYMENT_TO_INCOME": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_TO_INCOME": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "CREDIT_TO_GOODS": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
}


@dataclass(frozen=True)
class FeatureInfo:
    """Describe one emitted model feature in the versioned manifest."""

    name: str
    source: str
    dtype: str
    description: str


def history_aggregates(
    history: pl.DataFrame, *, source: str
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Aggregate only the history rows supplied for an already assigned split.

    Counts are zero for applicants without history after the caller's left join.
    Other statistics stay missing. All-missing sums remain null rather than zero.
    No categories are learned and no target-derived columns are read.
    """
    statuses: tuple[str, ...]
    if source == "bureau":
        specs = BUREAU_AGGREGATIONS
        status_column = "CREDIT_ACTIVE"
        statuses = ("Active", "Closed")
    elif source == "previous_application":
        specs = PREVIOUS_AGGREGATIONS
        status_column = "NAME_CONTRACT_STATUS"
        statuses = ("Approved", "Refused", "Canceled", "Unused offer")
    else:
        raise ValueError(f"Unknown history source: {source}")
    prefix = source.upper()
    count = f"{prefix}_COUNT"
    expressions = [pl.len().cast(pl.Int64).alias(count)]
    descriptions = {count: f"Number of {source} records for this applicant."}
    for column, operations in specs.items():
        if column not in history.columns:
            continue
        if not history.schema[column].is_numeric():
            raise ValueError(f"{source}.{column} must be numeric")
        value = pl.col(column).cast(pl.Float64)
        value = pl.when(value.is_finite()).then(value).otherwise(None)
        for operation in operations:
            name = f"{prefix}_{column}_{operation.upper()}"
            statistic = getattr(value, operation)()
            expressions.append(
                pl.when(value.count() > 0).then(statistic).otherwise(None).alias(name)
            )
            descriptions[name] = f"{operation} of {source}.{column} per applicant."
    if status_column in history.columns:
        for status in statuses:
            name = f"{prefix}_{status.upper().replace(' ', '_')}_COUNT"
            expressions.append(
                (pl.col(status_column) == status)
                .fill_null(False)
                .sum()
                .cast(pl.Int64)
                .alias(name)
            )
            descriptions[name] = (
                f"Number of {source} records with {status_column}={status}."
            )
    return history.group_by(ID).agg(expressions).sort(ID), descriptions


def build_features(
    applications: pl.DataFrame,
    bureau: pl.DataFrame,
    previous: pl.DataFrame,
) -> tuple[pl.DataFrame, list[FeatureInfo]]:
    """Engineer one split using only its applicants and their historical records."""
    frame = applications.drop(
        [c for c in EXCLUDED_FEATURES if c != ID and c in applications]
    )
    reserved = {"DAYS_EMPLOYED_SENTINEL", *RATIOS}
    if reserved.intersection(frame.columns) or any(
        name.startswith(("BUREAU_", "PREVIOUS_APPLICATION_")) for name in frame.columns
    ):
        raise ValueError("Application columns collide with engineered feature names")
    descriptions = {
        name: f"Original application_train.{name}; no fitted transformation."
        for name in frame.columns
        if name != ID
    }
    sources = dict.fromkeys(descriptions, "application_train.csv")
    frame = frame.with_columns(
        (pl.col("DAYS_EMPLOYED") == 365243)
        .fill_null(False)
        .cast(pl.Int8)
        .alias("DAYS_EMPLOYED_SENTINEL"),
        pl.when(pl.col("DAYS_EMPLOYED") == 365243)
        .then(float("nan"))
        .otherwise(pl.col("DAYS_EMPLOYED"))
        .cast(pl.Float64)
        .alias("DAYS_EMPLOYED"),
    )
    descriptions["DAYS_EMPLOYED"] = (
        "Days employed; the 365243 sentinel is replaced with NaN."
    )
    descriptions["DAYS_EMPLOYED_SENTINEL"] = (
        "1 when original DAYS_EMPLOYED equals 365243."
    )
    sources["DAYS_EMPLOYED_SENTINEL"] = "application_train.csv:DAYS_EMPLOYED"
    for name, (numerator, denominator) in RATIOS.items():
        top, bottom = pl.col(numerator), pl.col(denominator)
        ratio = top / bottom
        frame = frame.with_columns(
            pl.when(
                top.is_finite() & bottom.is_finite() & (bottom > 0) & ratio.is_finite()
            )
            .then(ratio)
            .otherwise(None)
            .alias(name)
        )
        descriptions[name] = (
            f"{numerator}/{denominator}; invalid or nonpositive denominator is null."
        )
        sources[name] = f"application_train.csv:{numerator},{denominator}"
    for history, source in ((bureau, "bureau"), (previous, "previous_application")):
        # Filter before aggregation to exclude other splits and unknown applicants.
        scoped = history.join(applications.select(ID), on=ID, how="semi")
        aggregated, details = history_aggregates(scoped, source=source)
        frame = frame.join(aggregated, on=ID, how="left", validate="1:1")
        counts = [name for name in details if name.endswith("_COUNT")]
        frame = frame.with_columns(pl.col(counts).fill_null(0))
        descriptions.update(details)
        sources.update(dict.fromkeys(details, f"{source}.csv"))
    frame = frame.sort(ID).drop(ID)
    manifest = [
        FeatureInfo(name, sources[name], str(dtype), descriptions[name])
        for name, dtype in frame.schema.items()
    ]
    return frame, manifest

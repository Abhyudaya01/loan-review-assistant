"""Strict intake contracts and typed placeholders for later review graph nodes.

Free text is untrusted application data. Only data-feature fields may reach the
feature builder; no free text is interpolated into an LLM system prompt here.
"""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from loan_review.data.features import EXCLUDED_FEATURES, RATIOS

# Supported optional Home Credit application_train columns. These are the raw
# fields accepted in addition to the required fields below. Engineered features,
# history aggregates, TARGET, and protected feature aliases are never accepted.
APPLICATION_CATEGORICAL = frozenset(
    {
        "NAME_CONTRACT_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "ORGANIZATION_TYPE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    }
)
APPLICATION_NUMERIC = frozenset(
    {
        "CNT_CHILDREN",
        "REGION_POPULATION_RELATIVE",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "HOUR_APPR_PROCESS_START",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "TOTALAREA_MODE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        *(f"FLAG_DOCUMENT_{index}" for index in range(2, 22)),
        *(
            f"{name}_{suffix}"
            for name in (
                "APARTMENTS",
                "BASEMENTAREA",
                "YEARS_BEGINEXPLUATATION",
                "YEARS_BUILD",
                "COMMONAREA",
                "ELEVATORS",
                "ENTRANCES",
                "FLOORSMAX",
                "FLOORSMIN",
                "LANDAREA",
                "LIVINGAPARTMENTS",
                "LIVINGAREA",
                "NONLIVINGAPARTMENTS",
                "NONLIVINGAREA",
            )
            for suffix in ("AVG", "MODE", "MEDI")
        ),
    }
)
OPTIONAL_APPLICATION_FIELDS = APPLICATION_CATEGORICAL | APPLICATION_NUMERIC
RESERVED = EXCLUDED_FEATURES | set(RATIOS) | {"DAYS_EMPLOYED_SENTINEL", "employer_name"}


def optional_field_error(name: str, value: object) -> str | None:
    """Validate one flat optional raw column independently of required fields."""
    if name not in OPTIONAL_APPLICATION_FIELDS:
        return f"unsupported raw application column: {name}"
    if value is None:
        return None
    if name in APPLICATION_CATEGORICAL:
        if not isinstance(value, str) or not value or len(value) > 128:
            return f"{name} must be a nonempty category of at most 128 characters"
        if name in {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"} and value not in {"Y", "N"}:
            return f"{name} must be Y or N"
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name} must be numeric"
    elif not -1e12 <= value <= 1e12:
        return f"{name} is outside the supported numeric range"
    return None


class StrictSchema(BaseModel):
    """Reject unknown fields and string-to-number coercion in graph contracts."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class LoanApplication(BaseModel):
    """Validate one raw Home Credit application without a known TARGET label.

    Required fields support Phase 1 sentinel/ratio features and separate fairness
    audit attributes. Other supported application_train fields remain flat Pydantic
    extras with strict type/range validation; those needed by the saved model must
    be present when the feature builder checks its exact feature schema. Employer
    name is a length-limited untrusted note, never a model feature or system prompt.
    """

    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)
    __pydantic_extra__: dict[str, StrictInt | StrictFloat | StrictStr | None] = Field(
        init=False
    )

    SK_ID_CURR: Annotated[StrictInt, Field(gt=0)]
    AMT_ANNUITY: Annotated[StrictFloat, Field(ge=0)]
    AMT_INCOME_TOTAL: Annotated[StrictFloat, Field(ge=0)]
    AMT_CREDIT: Annotated[StrictFloat, Field(gt=0)]
    AMT_GOODS_PRICE: Annotated[StrictFloat, Field(ge=0)]
    DAYS_EMPLOYED: StrictInt
    CODE_GENDER: Literal["F", "M", "XNA"]
    DAYS_BIRTH: Annotated[StrictInt, Field(ge=-36500, le=-18 * 365)]
    NAME_FAMILY_STATUS: Literal[
        "Married",
        "Single / not married",
        "Civil marriage",
        "Separated",
        "Widow",
        "Unknown",
        "Single",  # Synthetic fixture shorthand.
    ]
    employer_name: Annotated[StrictStr | None, Field(max_length=120)] = None

    @field_validator("DAYS_EMPLOYED")
    @classmethod
    def employed_days(cls, value: int) -> int:
        """Accept past-day offsets and the public dataset's 365243 sentinel."""
        if value > 0 and value != 365243:
            raise ValueError("must be nonpositive or the 365243 sentinel")
        if value < -36500:
            raise ValueError("must be at least -36500 days")
        return value

    @field_validator("employer_name")
    @classmethod
    def clean_untrusted_text(cls, value: str | None) -> str | None:
        """Trim and bound free text while retaining it as untrusted data only."""
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_optional_columns(self) -> Self:
        for name, value in (self.__pydantic_extra__ or {}).items():
            message = optional_field_error(name, value)
            if message is not None:
                raise ValueError(message)
        return self

    def raw_feature_values(self) -> dict[str, int | float | str | None]:
        """Return only raw Phase 1 columns, excluding free text and TARGET."""
        values = self.model_dump(exclude={"employer_name"})
        return {str(key): value for key, value in values.items()}


class FieldError(StrictSchema):
    """Identify one invalid input path without including sensitive submitted values."""

    field: str
    code: str
    message: str


class IntakeResult(StrictSchema):
    """Return either a normalized application or field-level validation errors."""

    valid: bool
    errors: list[FieldError]
    normalized: LoanApplication | None

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        if self.valid != (self.normalized is not None) or self.valid == bool(
            self.errors
        ):
            raise ValueError("valid intake requires normalized data and no errors")
        return self


class RiskResult(StrictSchema):
    """Placeholder for a validated model probability and reviewer risk band."""

    model_version: str | None = None
    default_probability: Annotated[float | None, Field(ge=0, le=1)] = None
    risk_band: Literal["LOW", "MEDIUM", "HIGH"] | None = None


class PolicyResult(StrictSchema):
    """Placeholder for deterministic policy checks and failed rule codes."""

    policy_version: str | None = None
    failed_rule_codes: list[str] = Field(default_factory=list)
    error_flag: bool = False


class Explanation(StrictSchema):
    """Placeholder for fixed reason codes and bounded verbalization text."""

    reason_codes: list[str] = Field(default_factory=list)
    text: Annotated[str | None, Field(max_length=2000)] = None


class ReviewerDecision(StrictSchema):
    """Placeholder for a decision entered by an identified human reviewer."""

    reviewer_id: str | None = None
    decision: Literal["APPROVE", "DECLINE", "REFER"] | None = None
    decided_at: datetime | None = None


class AuditRecord(StrictSchema):
    """Placeholder for the future single hash-chained decision record."""

    application_id: int | None = None
    model_version: str | None = None
    config_versions: dict[str, str] = Field(default_factory=dict)
    previous_hash: str | None = None
    record_hash: str | None = None

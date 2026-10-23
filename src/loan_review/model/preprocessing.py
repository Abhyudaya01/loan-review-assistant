"""Learn categorical vocabularies on train only; keep native LightGBM missing values."""

from typing import Self

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import model_validator

from loan_review.data.features import EXCLUDED_FEATURES
from loan_review.model.config import StrictModel

FloatArray = NDArray[np.float64]


class FeatureEncoder(StrictModel):
    """Persist ordered feature names and training-only categorical vocabularies."""

    feature_names: list[str]
    categories: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("Feature names must be nonempty and unique")
        if EXCLUDED_FEATURES.intersection(self.feature_names):
            raise ValueError(
                "IDs, targets, and protected attributes cannot be model features"
            )
        if not set(self.categories).issubset(self.feature_names):
            raise ValueError("Category mappings contain unknown features")
        if any(values != sorted(set(values)) for values in self.categories.values()):
            raise ValueError("Categorical vocabularies must be sorted and unique")
        return self

    @classmethod
    def fit(cls, train: pl.DataFrame) -> Self:
        """Fit only training categories; numeric features require no imputation."""
        categories: dict[str, list[str]] = {}
        for name, dtype in train.schema.items():
            if dtype == pl.String:
                categories[name] = sorted(train[name].drop_nulls().unique().to_list())
            elif not (dtype.is_numeric() or dtype in (pl.Boolean, pl.Null)):
                raise ValueError(f"Unsupported feature dtype: {name}={dtype}")
        return cls(feature_names=train.columns, categories=categories)

    @property
    def categorical_indices(self) -> list[int]:
        """Identify encoded columns that LightGBM must treat as categorical."""
        return [
            i for i, name in enumerate(self.feature_names) if name in self.categories
        ]

    def transform(self, frame: pl.DataFrame) -> FloatArray:
        """Encode categories; unseen/null values use LightGBM's -1 missing code."""
        if frame.columns != self.feature_names:
            raise ValueError("Feature columns/order differ from the fitted schema")
        columns: list[FloatArray] = []
        for name in self.feature_names:
            series = frame[name]
            if name in self.categories:
                if series.dtype not in (pl.String, pl.Null):
                    raise ValueError(f"{name} must retain its categorical string dtype")
                lookup = {
                    value: index for index, value in enumerate(self.categories[name])
                }
                values = np.array(
                    [lookup.get(value, -1) for value in series.to_list()],
                    dtype=np.float64,
                )
            else:
                if not (
                    series.dtype.is_numeric() or series.dtype in (pl.Boolean, pl.Null)
                ):
                    raise ValueError(f"{name} must remain numeric")
                values = np.asarray(
                    series.cast(pl.Float64).to_numpy(), dtype=np.float64
                ).copy()
                values[~np.isfinite(values)] = np.nan
            columns.append(values)
        return np.column_stack(columns)

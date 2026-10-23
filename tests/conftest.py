"""Share offline Home Credit fixtures across data and model integration tests."""

import polars as pl
import pytest

from loan_review.data.features import ID, TARGET


@pytest.fixture
def raw_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = 200
    applications = pl.DataFrame(
        {
            ID: list(range(100_000, 100_000 + n)),
            TARGET: [int(i % 5 == 0) for i in range(n)],
            "CODE_GENDER": ["F" if i % 2 else "M" for i in range(n)],
            "DAYS_BIRTH": [-8000 - i * 37 for i in range(n)],
            "NAME_FAMILY_STATUS": ["Single" if i % 3 else "Married" for i in range(n)],
            "DAYS_EMPLOYED": [365243 if i % 7 == 0 else -100 - i for i in range(n)],
            "AMT_ANNUITY": [1000.0 + i for i in range(n)],
            "AMT_INCOME_TOTAL": [0.0 if i == 0 else 50_000.0 + i for i in range(n)],
            "AMT_CREDIT": [20_000.0 + i for i in range(n)],
            "AMT_GOODS_PRICE": [0.0 if i == 1 else 18_000.0 + i for i in range(n)],
            "NAME_EDUCATION_TYPE": [
                "Higher" if i % 4 else "Secondary" for i in range(n)
            ],
            "FLAG_OWN_CAR": ["Y" if i % 2 else "N" for i in range(n)],
            "EXT_SOURCE_1": [None if i % 9 == 0 else i / n for i in range(n)],
        }
    )
    # Two records for half the applicants; the others have no linked history.
    ids = [100_000 + i for i in range(n // 2) for _ in range(2)] + [999_999]
    size = len(ids)
    bureau = pl.DataFrame(
        {
            ID: ids,
            "SK_ID_BUREAU": list(range(size)),
            "AMT_CREDIT_SUM": [10.0, 30.0] * (n // 2) + [1e12],
            "AMT_CREDIT_SUM_DEBT": [None, None] * (n // 2) + [1.0],
            "AMT_CREDIT_MAX_OVERDUE": [2.0, 8.0] * (n // 2) + [1e12],
            "CREDIT_DAY_OVERDUE": [1, 4] * (n // 2) + [9999],
            "CREDIT_ACTIVE": ["Active", "Closed"] * (n // 2) + ["Active"],
            "TARGET": [1] * size,  # Must never become an aggregate.
        }
    )
    previous = pl.DataFrame(
        {
            ID: ids,
            "SK_ID_PREV": list(range(size)),
            "AMT_CREDIT": [100.0, 200.0] * (n // 2) + [1e12],
            "AMT_ANNUITY": [10.0, 20.0] * (n // 2) + [1e12],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused"] * (n // 2) + ["Approved"],
            "TARGET": [0] * size,
        }
    )
    return applications, bureau, previous

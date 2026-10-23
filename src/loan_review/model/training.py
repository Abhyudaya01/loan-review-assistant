"""Train LightGBM with bounded seeded tuning and validation-only calibration."""

from importlib.metadata import version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import roc_auc_score

from loan_review.model.bundle import ModelBundle, content_hash, file_hash, git_revision
from loan_review.model.calibration import IsotonicCalibrator
from loan_review.model.config import RiskThresholds, TrainingConfig
from loan_review.model.preprocessing import FeatureEncoder, FloatArray
from loan_review.model.reasons import ReasonCodes


def binary_labels(labels: pl.Series, expected_rows: int) -> FloatArray:
    """Require aligned, nonmissing binary integer targets with both classes."""
    if (
        len(labels) != expected_rows
        or not labels.dtype.is_integer()
        or labels.null_count()
        or set(labels.unique().to_list()) != {0, 1}
    ):
        raise ValueError(
            "Labels must be aligned binary integers containing both 0 and 1"
        )
    return np.asarray(labels.to_numpy(), dtype=np.float64)


def train_model(
    train_features: pl.DataFrame,
    train_labels: pl.Series,
    val_features: pl.DataFrame,
    val_labels: pl.Series,
    *,
    data_hash: str,
    reason_codes: ReasonCodes,
    thresholds: RiskThresholds,
    config: TrainingConfig | None = None,
    git_sha: str | None = None,
) -> ModelBundle:
    """Fit without accepting test data; val drives AUC tuning, stopping, and isotonic.

    Reusing validation for selection and calibration can overfit that holdout;
    only the untouched test report estimates final performance. Isotonic can be
    particularly variable with small validation sets. No tuning uses test scores.
    """
    config = config or TrainingConfig()
    if not data_hash:
        raise ValueError("A source data hash is required for model provenance")
    reason_codes.require_coverage(train_features.columns)
    encoder = FeatureEncoder.fit(train_features)
    train_x, val_x = encoder.transform(train_features), encoder.transform(val_features)
    train_y = binary_labels(train_labels, train_features.height)
    val_y = binary_labels(val_labels, val_features.height)
    base_params: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": config.seed,
        "bagging_seed": config.seed,
        "feature_fraction_seed": config.seed,
        "data_random_seed": config.seed,
        "feature_pre_filter": False,
    }
    best_booster: lgb.Booster | None = None
    best_auc = -1.0
    best_params: dict[str, Any] = {}

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_booster, best_auc, best_params
        params = {
            **base_params,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }
        train_set = lgb.Dataset(
            train_x,
            label=train_y,
            feature_name=encoder.feature_names,
            categorical_feature=encoder.categorical_indices,
        )
        val_set = lgb.Dataset(
            val_x,
            label=val_y,
            reference=train_set,
            feature_name=encoder.feature_names,
            categorical_feature=encoder.categorical_indices,
        )
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=config.num_boost_round,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(
                    config.early_stopping_rounds, first_metric_only=True, verbose=False
                )
            ],
        )
        auc = float(
            roc_auc_score(
                val_y,
                booster.predict(
                    val_x, num_iteration=booster.best_iteration, num_threads=1
                ),
            )
        )
        trial.set_user_attr("best_iteration", booster.best_iteration)
        if auc > best_auc:
            best_auc, best_params = auc, params
            # Freeze only selected iterations; later inference cannot use extra trees.
            best_booster = lgb.Booster(
                model_str=booster.model_to_string(num_iteration=booster.best_iteration)
            )
        return auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(
        objective, n_trials=config.n_trials, n_jobs=1, show_progress_bar=False
    )
    if best_booster is None:
        raise RuntimeError("Tuning produced no fitted model")
    val_probabilities = np.asarray(
        best_booster.predict(val_x, num_threads=1), dtype=np.float64
    )
    calibrator = IsotonicCalibrator.fit(val_probabilities, val_y)
    version_inputs = {
        "data_hash": data_hash,
        "params": best_params,
        "git_sha": git_sha if git_sha is not None else git_revision(),
        "training_config": config.model_dump(),
        "best_iteration": best_booster.current_iteration(),
        "encoder": encoder.model_dump(),
        "calibrator": calibrator.model_dump(),
        "reason_codes_hash": content_hash(reason_codes.model_dump()),
        "thresholds_hash": content_hash(thresholds.model_dump()),
        "code_hashes": {
            path.name: file_hash(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "libraries": {
            name: version(name)
            for name in (
                "lightgbm",
                "optuna",
                "numpy",
                "polars",
                "scikit-learn",
                "shap",
            )
        },
    }
    metadata = {
        "model_version": content_hash(version_inputs),
        "version_inputs": version_inputs,
        "validation_auc": best_auc,
        "best_trial": study.best_trial.number,
        "trials": [
            {
                "number": trial.number,
                "validation_auc": trial.value,
                "params": trial.params,
                "best_iteration": trial.user_attrs["best_iteration"],
            }
            for trial in study.trials
        ],
        "calibration_split": "val",
        "selection_split": "val",
        "training_split": "train",
        "caveat": (
            "Validation is reused for tuning, stopping, and isotonic calibration; "
            "test is reserved for final evaluation."
        ),
    }
    return ModelBundle(
        best_booster, encoder, calibrator, thresholds, reason_codes, metadata
    )

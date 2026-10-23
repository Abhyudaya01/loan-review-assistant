"""Evaluate a frozen model on test once; never feed report metrics back into tuning."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from loan_review.model.bundle import ModelBundle
from loan_review.model.calibration import validate_probabilities
from loan_review.model.preprocessing import FloatArray
from loan_review.model.training import binary_labels


def probability_metrics(
    labels: FloatArray, probabilities: FloatArray
) -> dict[str, float]:
    """Compute AUC, two-sided empirical KS, and mean squared probability error."""
    validate_probabilities(probabilities)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "ks": float(np.max(np.abs(tpr - fpr))),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def reliability_bins(labels: FloatArray, probabilities: FloatArray) -> pl.DataFrame:
    """Report nonempty fixed-width bins, including probability 1 in the final bin."""
    validate_probabilities(probabilities)
    membership = np.minimum((probabilities * 10).astype(int), 9)
    rows = []
    for index in range(10):
        mask = membership == index
        if mask.any():
            rows.append(
                {
                    "bin_lower": index / 10,
                    "bin_upper": (index + 1) / 10,
                    "count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()),
                    "observed_default_rate": float(labels[mask].mean()),
                }
            )
    return pl.DataFrame(rows)


def evaluate_test(
    bundle: ModelBundle,
    features: pl.DataFrame,
    labels: pl.Series,
    *,
    report_root: Path = Path("artifacts/model_report"),
    test_data_hash: str,
) -> Path:
    """Write a frozen-model test report, raw/calibrated metrics, and reliability plot.

    Validation was reused for early stopping, selection, and isotonic fitting.
    Test estimates are evaluation only. Random Home Credit holdouts cannot assess
    out-of-time drift because absolute application dates are unavailable.
    """
    target = binary_labels(labels, features.height)
    raw = bundle.predict_raw_probability(features)
    calibrated = bundle.calibrator.predict(raw)
    report = {
        "model_version": bundle.model_version,
        "test_data_hash": test_data_hash,
        "evaluation_split": "test",
        "rows": len(target),
        "raw": probability_metrics(target, raw),
        "calibrated": probability_metrics(target, calibrated),
        "risk_bands": "Illustrative reviewer inputs only; not credit decisions.",
        "limitations": [
            "Validation is reused for tuning, stopping, and isotonic calibration.",
            (
                "Isotonic calibration may overfit small validation sets "
                "and may introduce ties that change AUC."
            ),
            (
                "Random splits do not measure temporal generalization; "
                "absolute application dates are absent."
            ),
        ],
    }
    # A separate test hash prevents overwriting reports for a different holdout.
    directory = report_root / bundle.model_version / test_data_hash
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    figure = Figure(figsize=(7, 6), layout="constrained")
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], "--", color="0.5", label="Perfect calibration")
    bins = []
    for name, probabilities, color in (
        ("Raw", raw, "#5B6770"),
        ("Isotonic", calibrated, "#007C91"),
    ):
        table = reliability_bins(target, probabilities)
        bins.append(table.with_columns(pl.lit(name.lower()).alias("probability_type")))
        axes.plot(
            table["mean_probability"],
            table["observed_default_rate"],
            marker="o",
            label=name,
            color=color,
        )
    axes.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted default probability",
        ylabel="Observed default rate",
        title="Test-set reliability · 10 fixed-width bins",
    )
    axes.legend()
    axes.grid(alpha=0.2)
    figure.savefig(directory / "reliability.png", dpi=160)
    pl.concat(bins).write_csv(directory / "reliability_bins.csv")
    return directory

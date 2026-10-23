# Loan Review Assistant

Human-led consumer loan review assistant using the public Home Credit Default
Risk dataset. Data preparation, risk modeling, calibration, and explanations are
implemented; review orchestration is scaffolded.
The LLM will verbalize supplied reasons; a human makes every final credit decision.

## Development

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv python install
uv sync --locked
make lint type test
```

Python is pinned to 3.11. `uv.lock` fixes dependency versions. The `uv_build`
build backend packages the `src` layout; it is a build dependency only.
SQLite is included with Python; a Postgres driver will be selected when persistence
is implemented.

## Layout

- `src/loan_review/`: data, model, policy, LLM, graph, audit, fairness, evals, API.
- `configs/`: feature manifest and future policy rules, reason codes, and thresholds.
- `data/raw/`, `artifacts/`: local, gitignored data and versioned outputs.
- `tests/`: offline synthetic data checks and a deterministic FakeLLM.
- `notebooks/`: future exploratory analysis.

Other modules currently document their intended responsibilities only. `FakeLLM`
returns a fixed Pydantic response from a strict Pydantic request without a network
call. Retry, human routing, decisioning, and audit behavior are not implemented.

## Prepare data

Place `application_train.csv`, `bureau.csv`, and `previous_application.csv` from
Home Credit Default Risk in `data/raw/`, then run:

```sh
uv run --locked python -m loan_review.data --seed 42
```

The command prints its output directory: `artifacts/data-v1.0.0-<content-hash>/`.
Each run contains:

- `train_features.parquet`, `val_features.parquet`, `test_features.parquet`: model
  columns only, with no IDs, target, gender, birth/age, or family status columns.
- Corresponding `*_labels.parquet`: `SK_ID_CURR`, `TARGET`, aligned row-for-row
  with features. Use these files to associate feature rows with applicants.
- Corresponding `*_audit_attributes.parquet`, plus combined
  `audit_attributes.parquet`: protected attributes keyed by `SK_ID_CURR`. The
  combined frame also records split membership.
- `feature_manifest.yaml`: exact feature names, source, dtype, and descriptions.
- `proxy_report.csv`: training-only association statistics, coverage, thresholds,
  and flags. It includes every feature–attribute pair; filter `flagged == true`
  for the screening findings.
- `metadata.json`: seed/settings, split sizes, pipeline/library versions, and
  SHA-256 hashes of source CSVs, pipeline code, and output files.

The manifest is also written to `configs/feature_manifest.yaml`, and the latest
report to `artifacts/proxy_report.csv`. The checked-in manifest starts as an
explicitly labeled pre-data catalog; a real run replaces it with the exact schema.
Changed inputs, code, dependencies, or settings produce a new run version.

Applications are stratified on `TARGET` into 70/15/15 splits before historical
aggregation. Each history table is restricted to that split's applicant IDs before
aggregation. Missing history gives zero record/status counts and null statistics;
all-missing numeric sums remain null. Aggregations use explicit column allowlists
and never use targets. Application categories remain strings for later model
preprocessing; no imputer, scaler, or encoder is fitted here.

`DAYS_EMPLOYED == 365243` becomes NaN with an indicator flag. Ratios use annuity /
income, credit / income, and credit / goods price. Missing, nonfinite, zero, or
negative denominators yield null. Protected attributes are carried only in the
audit frames and used for the proxy screen.

No out-of-time validation is possible because application dates are absent.
Random holdouts cannot measure future population or economic drift, and the
relative historical dates do not establish an absolute as-of leakage audit.

### Proxy settings

```sh
uv run --locked python -m loan_review.data \
  --correlation-threshold 0.3 --mi-threshold 0.1 --proxy-max-rows 20000
```

The default screen uses a seeded sample of at most 20,000 training applicants.
It flags absolute Pearson correlation above 0.3 or mutual information above 0.1
nats. Categorical correlations use the maximum absolute correlation between
one-hot indicators, avoiding arbitrary ordinal category codes. Mutual information
uses categorical targets for gender/family status and a continuous target for
birth days. Missing/nonfinite pairs are excluded; constants and unestimable
statistics are identified explicitly. These heuristics do not establish fairness
or automatically remove features. Validation and test attributes never influence
the screen. NumPy and scikit-learn are explicit dependencies for these statistics
and stratified splitting; Polars supplies Parquet support without another engine.

## Train, calibrate, and evaluate

Use the versioned directory printed by the data pipeline:

```sh
uv run --locked python -m loan_review.model \
  --data-dir artifacts/data-v1.0.0-<content-hash> --trials 10 --seed 42
```

Optuna performs 10 sequential trials by default, with a validated maximum of 30.
It searches learning rate, leaf count, minimum leaf size, and L2 regularization.
LightGBM trains on `train`, selects trials by validation AUC, and uses early
stopping on `val`. Training uses deterministic CPU settings and one thread.
Categorical vocabularies are fitted on training rows only; unseen categories map
to LightGBM's missing-category value. Numeric nulls/NaNs remain missing.

The selected model is calibrated with isotonic regression fitted on validation
probabilities and labels. Validation is reused for selection, stopping, and
calibration, which can overfit that holdout, particularly on small datasets.
The test files are first loaded after the trained model and calibrator are saved.
Test metrics never select parameters, calibration, or risk-band thresholds.

The command prints two directories:

- `artifacts/models/<model_version>/`: native `model.txt`, portable isotonic
  `calibrator.json`, training-only `encoder.json`, fixed reason/threshold snapshots,
  `explainer_metadata.json`, and integrity/provenance `metadata.json`.
- `artifacts/model_report/<model_version>/<test-data-hash>/`: `metrics.json` with
  raw and calibrated AUC, KS, and Brier scores; `reliability.png`; and
  `reliability_bins.csv` with bin counts and observed default rates.

The model version is a canonical SHA-256 hash containing the train/validation
data hash, effective parameters, and Git SHA, plus calibration, encoder, config,
source-code, and library fingerprints. A repository without commits records
`UNCOMMITTED` explicitly and still fingerprints source files. Reloading verifies
file integrity and preserves the version. No pickle is needed: calibration is
saved as isotonic interpolation knots with clipping outside the fitted range.

### Reviewer bands and explanations

`configs/thresholds.yaml` defines illustrative bands on calibrated default
probability: LOW below 0.05, MEDIUM from 0.05 to below 0.15, HIGH from 0.15 through
1. These fixed, versioned labels are inputs to a human reviewer, never credit
decisions or validated operational cutoffs.

`configs/reason_codes.yaml` explicitly maps the 175 known unprotected Home Credit
application/aggregate features to fixed reason codes. Training and explanation
fail if a model feature has no mapping; future features require an explicit
dictionary update. `direction` refers to positive SHAP contribution to default
log-odds, not an assumed monotonic relationship with a raw feature's value.

```python
from pathlib import Path
import polars as pl

from loan_review.model.bundle import load_model
from loan_review.model.explanations import explain_applications
from loan_review.model.reasons import top_reasons

model = load_model(Path("artifacts/models/<model_version>"))
features = pl.read_parquet("artifacts/data-v1.0.0-<content-hash>/test_features.parquet")
scores = model.score(features)  # probabilities and reviewer bands; no decisions
explanation = explain_applications(model, features.head(1))[0]
reasons = top_reasons(explanation.shap_row, k=4, reason_codes=model.reason_codes)
```

TreeSHAP uses `model_output="raw"` and `tree_path_dependent` with training leaf
counts. Contributions are sorted by push toward default and checked to sum with
the baseline to the model's raw log-odds. They do not decompose the nonlinear
isotonic-calibrated probability. Correlated features can exchange contributions
and change rankings; SHAP is associative, not causal.

`top_reasons` groups signed SHAP values by fixed code, keeps only groups with a
positive net contribution, and returns up to four deduplicated reasons by default.
Each includes positive supporting features/values and negative offsets. The LLM
never selects reasons. Optuna is added for the requested tuning, Matplotlib for
the exported reliability plot, and PyYAML for validated configuration loading.

## Validate one application at intake

`loan_review.graph.intake_node` validates a single raw application with Pydantic
v2 and returns `IntakeResult(valid, errors, normalized)`. Errors identify fields
without echoing submitted values. The required raw columns are `SK_ID_CURR`, the
five financial/employment inputs used by Phase 1, and the three protected audit
attributes. Supported optional Home Credit application columns remain flat fields;
the generated feature manifest determines which ones the trained model requires.
`TARGET` is never accepted from a live application.

`employer_name` is optional, limited to 120 characters, and treated as untrusted
data. It is trimmed on intake and excluded from both model features and system
prompts. There is no LLM call in intake. Unknown columns and unexpected numeric
strings fail validation.

```python
from pathlib import Path
import polars as pl

from loan_review.graph import (
    build_application_features, intake_node, load_feature_manifest,
)
from loan_review.model.bundle import load_model

phase_one = Path("artifacts/data-v1.0.0-<content-hash>")
manifest = load_feature_manifest(phase_one / "feature_manifest.yaml")
model = load_model(Path("artifacts/models/<model_version>"))
result = intake_node(raw_application_mapping, manifest)
if result.valid and result.normalized is not None:
    features = build_application_features(
        result.normalized, manifest, bureau=bureau_rows,
        previous_application=previous_application_rows, encoder=model.encoder,
    )
    scores = model.score(features)
```

The feature builder calls the same Phase 1 aggregation and ratio logic used for
batch training, then checks the saved manifest's exact column order and dtypes.
It treats absent history as zero counts and null statistics, and raises a named
error if a required raw model input is missing. Passing the fitted encoder also
checks compatibility without learning new categories from this application.

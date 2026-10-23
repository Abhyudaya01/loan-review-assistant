# Configuration

Place versioned YAML policy rules, reason-code dictionaries, and thresholds here
when implemented. Their versions must be recorded in audit records.

`feature_manifest.yaml` is initially a pre-data catalog of core and engineered
features. Running `python -m loan_review.data` replaces it with the exact output
schema and artifact version and saves a snapshot beside the Parquet outputs.

`thresholds.yaml` provides fixed illustrative LOW/MEDIUM/HIGH reviewer bands.
`reason_codes.yaml` maps every known model feature to a fixed code, short text,
and positive-SHAP direction. Training rejects uncovered features. Both configs
are versioned, hashed into model provenance, and snapshotted with each model.

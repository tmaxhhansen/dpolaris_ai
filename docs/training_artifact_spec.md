# Training Artifact Spec (v1.0.0)

This document defines the **Training Observability Contract** between:

- `dPolaris_ai` (trainer/predictor backend)
- `dPolaris Java app` (controller/UI)

Every training run writes a self-contained run folder:

```text
runs/<runId>/
  artifact.json
  manifest.json
  run_summary.json
  data_summary.json
  feature_summary.json
  split_summary.json
  model_summary.json
  metrics_summary.json
  backtest_summary.json
  diagnostics_summary.json
  artifacts/
    ...copied reproducibility files (model, metadata, logs, reports)...
```

`training_artifact_version` is semantic versioned and currently:

```json
{
  "training_artifact_version": "1.0.0"
}
```

## Contract Sections

`RunSummary`:
- `run_id`, timestamps, `duration_seconds`
- git metadata: `git_commit_hash`, `git_branch`
- runtime metadata: `environment`, `hostname`, `user`
- training identity: `model_type`, `target`, `horizon`, `tickers`, `timeframes`

`DataSummary`:
- `sources_used`, `start`, `end`, `bars_count`
- `missingness_report`
- `corporate_actions_applied`, `adjustments`
- `outliers_detected`
- `drop_or_repair_decisions`

`FeatureSummary`:
- `feature_registry_version`
- `features` (name + params)
- `missingness_per_feature`
- `normalization_method`
- `leakage_checks_status`

`SplitSummary`:
- walk-forward window metadata
- train/val/test ranges
- sample sizes

`ModelSummary`:
- `algorithm`
- `hyperparameters`
- `feature_importance`
- `calibration_method`

`MetricsSummary`:
- `classification`
- `regression`
- `trading`
- `calibration`

`BacktestSummary`:
- assumptions (`slippage`, spread, fees, etc.)
- equity-curve stats
- trade-list artifact link

`DiagnosticsSummary`:
- drift baseline stats
- regime distribution
- error analysis
- top failure cases

## Schemas

JSON schema files are located in:

- `/Users/darrenwon/my-git/dpolaris_ai/schemas/training_artifact.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/run_summary.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/data_summary.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/feature_summary.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/split_summary.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/model_summary.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/metrics_summary.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/backtest_summary.schema.json`
- `/Users/darrenwon/my-git/dpolaris_ai/schemas/diagnostics_summary.schema.json`

## Backend Endpoints

- `GET /runs`: list runs
- `GET /runs/{id}`: full run payload
- `GET /runs/{id}/artifacts`: list files in run folder
- `GET /runs/{id}/artifact/{name}`: download specific file
- `GET /runs/compare?run_ids=<id1,id2,...>`: compare key metrics

## Backward Compatibility

Loader normalizes legacy/sparse payloads into the v1 section layout:

- accepts camel-case section keys (for old clients)
- fills missing sections with defaults
- preserves old `training_artifact_version` while exposing stable keys to clients

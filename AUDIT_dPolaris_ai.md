# AUDIT: dPolaris_ai (Engineering-Grade, Audit-Only)

Date: 2026-02-10  
Repo: `/Users/darrenwon/my-git/dpolaris_ai`  
Commit audited: `e707483` (branch `main`)  
Scope: full repository audit (code/config/docs/tests/scripts), no implementation changes in this pass.

## Executive Summary

1. The repo has a strong foundation for observability artifacts (`runs/<runId>`) and a broad test suite (`48 passed`), but critical production paths still bypass key precision controls.
2. The current training path used by `@train` and deep-learning worker still relies on legacy `ml/features.py` and drops non-price joined data, so macro/fundamental/sentiment/regime features are largely not in the actual model training path.
3. Walk-forward/no-lookahead is enforced for classic trainer (`ml/trainer_precision.py`), but deep-learning path uses a single time split and records `"leakage_checks_status": "passed"` without equivalent leakage validation.
4. Artifact contract exists and is versioned (`1.0.0`), but required audit files like `calibration_report.json` and `leakage_checks.json` are not emitted as standalone artifacts, and backtest CSV artifacts are not guaranteed in run folders.
5. Backtesting realism code exists (`backtest/engine.py`) and is good, but it is not the default metric path for training; default execution assumptions include `commission_bps=0.0`, which can still produce optimistic outcomes unless overridden.

---

## 1) Inventory & Architecture Map

### Top-level repo overview

- `api/`: FastAPI server, train/predict endpoints, deep-learning job queue, run artifact endpoints.
- `core/`: AI command routing (`@train`, `@predict`, etc.), config/db/memory orchestration.
- `cli/`: CLI entrypoints (`train`, `predict`, `server`, etc.).
- `ml/`: feature engine, classic trainer, deep-learning trainer/worker, evaluation, training artifacts, inspector.
- `data/`: canonical schema, quality gates, connectors, causal alignment, dataset builder.
- `features/`: plugin-style feature library (technical/macro/fundamental/sentiment/regime).
- `backtest/`: execution-aware backtest engine + reporting.
- `monitoring/`: drift/performance monitoring, retraining decisions, self-critique logging.
- `registry/`: model version registry with snapshots.
- `risk/`: first-class risk manager (sizing/exposure/kill-switch constraints).
- `daemon/`: scheduler for recurring operational jobs.
- `schemas/`: JSON schemas for run artifact contract.
- `docs/`: precision plan + artifact spec.
- `tests/`: unit/integration tests for causality, artifacts, backtest, monitoring, risk, inspector.

### Main entrypoints

- Training (classic):
  - API: `POST /api/train/{symbol}` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1283`
  - AI command: `_cmd_train` in `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py:512`
  - Trainer: `ModelTrainer.train_full_pipeline` in `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:681`
- Training (deep learning):
  - API async job enqueue: `POST /api/jobs/deep-learning/train` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1320`
  - Worker/subprocess: `_run_deep_learning_job` + `_execute_deep_learning_subprocess` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:627` and `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:181`
  - Worker module: `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning_worker.py:102`
- Inference:
  - Classic predict API: `POST /api/predict/{symbol}` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1276`
  - DL predict API: `POST /api/deep-learning/predict/{symbol}` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1520`
  - Signal synthesis: `POST /api/signals/{symbol}` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1876`
  - Prediction inspector: `GET /predict/inspect` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1640`
- Backtest:
  - Engine class: `/Users/darrenwon/my-git/dpolaris_ai/backtest/engine.py:183`
  - Reporting: `/Users/darrenwon/my-git/dpolaris_ai/backtest/reporting.py:117`
  - Note: not directly wired into trainer’s primary evaluation path.
- Monitoring:
  - Drift monitor: `/Users/darrenwon/my-git/dpolaris_ai/monitoring/drift.py:248`
  - Performance monitor: `/Users/darrenwon/my-git/dpolaris_ai/monitoring/drift.py:287`
  - Retraining scheduler: `/Users/darrenwon/my-git/dpolaris_ai/monitoring/drift.py:348`
  - Self-critique logger: `/Users/darrenwon/my-git/dpolaris_ai/monitoring/drift.py:403`
- Registry:
  - Model registry: `/Users/darrenwon/my-git/dpolaris_ai/registry/model_registry.py:16`

### Data flow diagram (actual implemented path)

```text
Raw sources:
  yfinance price/fundamentals (data/connectors/yfinance.py, tools/market_data.py)
    ->
Canonicalization:
  canonicalize_price_frame + optional split/dividend adjustment
  (data/schema.py)
    ->
Quality gate:
  duplicates/missing/negative/stale/outlier/min-history report
  (data/quality.py)
    ->
Causal alignment:
  as-of joins for fundamentals/macro/news
  (data/alignment.py, data/dataset_builder.py)
    ->
Training frame currently used:
  reduced to OHLCV + date for both classic and DL paths
  (core/ai.py, ml/deep_learning_worker.py)
    ->
Feature engineering:
  legacy FeatureEngine (ml/features.py)
    ->
Splits:
  classic: TimeSeriesSplit walk-forward (trainer_precision.py)
  deep learning: single chronological split (deep_learning.py)
    ->
Model training:
  classic sklearn/xgboost/lightgbm/logistic
  deep learning lstm/transformer
    ->
Calibration + metrics:
  classic: OOF platt calibration + standardized metrics (ml/evaluation.py)
  deep learning: calibration fit on test split (deep_learning.py)
    ->
Artifacts:
  write_training_artifact -> runs/<runId>/*
  (ml/training_artifacts.py)
    ->
Prediction services:
  /api/predict, /api/deep-learning/predict, /api/signals, /predict/inspect
  (api/server.py, ml/predictor.py, ml/prediction_inspector.py)
```

---

## 2) Training Pipeline Audit (Train-button path)

### Exact train trigger path

- `POST /api/train/{symbol}` calls `ai.chat("@train SYMBOL")`: `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1283`
- `_cmd_train`:
  - builds unified dataset: `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py:523`
  - then **keeps only** `timestamp/open/high/low/close/volume`, renames to `date`: `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py:535`
  - trains via `ModelTrainer.train_full_pipeline`: `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py:541`

### Config loading + snapshot

- Precision config loaded with defaults+YAML merge:
  - loader: `/Users/darrenwon/my-git/dpolaris_ai/ml/precision_config.py:125`
  - default path: `/Users/darrenwon/my-git/dpolaris_ai/ml/precision_config.py:15`
  - config file: `/Users/darrenwon/my-git/dpolaris_ai/data/config/prediction_precision.yaml:1`
- Config snapshot generation in artifact writer:
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:263`
  - persisted: `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:666`

### Splits (random vs time-based, walk-forward default)

- Classic trainer uses walk-forward `TimeSeriesSplit`:
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:555`
  - fallback to walk-forward if unsupported method configured: `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:731`
- Deep-learning trainer uses a single chronological split (not walk-forward):
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:310`

### Label/target definition

- Target generation in `FeatureEngine._add_targets`:
  - future return: `close.shift(-horizon) / close - 1`: `/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:305`
  - direction label: `(future_return > 0).astype(int)`: `/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:309`

### Model training + save location

- Classic save path and metadata:
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:443`
  - writes `model.pkl`, `scaler.pkl`, `metadata.json`: `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:464` to `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:487`
- Model registry write (best-effort):
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:491`

### Calibration + produced metrics

- Classic:
  - platt calibration fit/apply on OOF predictions: `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:615` and `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:618`
  - metrics stack from `ml/evaluation.py` classification/regression/trading.
- Deep learning:
  - calibration fit in evaluation on test split: `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:516`
  - stored in metrics: `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:526`

### Backtest execution assumptions in training

- Classic primary training score uses `compute_trading_metrics` from predicted probabilities + realized returns:
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:639`
- Full execution backtest engine exists (`backtest/engine.py`) but is **not** called by trainer pipeline.

---

## 3) Data Correctness & Causality (No Lookahead)

### Implemented causal/time controls

- Strict as-of join (`right timestamp <= left timestamp`):
  - `/Users/darrenwon/my-git/dpolaris_ai/data/alignment.py:74`
- Canonical schema + timezone-aware timestamps:
  - `/Users/darrenwon/my-git/dpolaris_ai/data/schema.py:14`
  - `/Users/darrenwon/my-git/dpolaris_ai/data/schema.py:75`
- Split/dividend adjustment helper:
  - `/Users/darrenwon/my-git/dpolaris_ai/data/schema.py:125`
- Data quality gate + JSON report:
  - `/Users/darrenwon/my-git/dpolaris_ai/data/quality.py:107`
  - report write: `/Users/darrenwon/my-git/dpolaris_ai/data/quality.py:246`

### Leakage Risk Table

| Risk | Location | Severity | Why it matters | Suggested fix |
|---|---|---|---|---|
| Deep-learning artifact marks leakage check as passed without running explicit no-lookahead validation | `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:807` (`train_full_pipeline`) | High | Can falsely certify model readiness and violate audit guarantees | Run `validate_no_lookahead_features` (or equivalent DL-specific check) and persist actual result in artifact |
| Quality gate failures are reported but not enforced (training can continue after failed min-history/quality checks) | `/Users/darrenwon/my-git/dpolaris_ai/data/dataset_builder.py:82`, `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py:541`, `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning_worker.py:177` | High | Pipeline can train on known-bad datasets; precision and reproducibility degrade | Add fail-fast policy in dataset builder/training entrypoints when required checks fail |
| Train path discards joined macro/fundamental/news columns and uses OHLCV-only frame | `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py:535`, `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning_worker.py:49` | High | System appears richer than deployed model; user expects additional data influence that is not real | Feed aligned enriched dataset into feature library used by trainer, not reduced OHLCV-only frame |
| Deep-learning validation is single split, not walk-forward | `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:310` | High | Performance estimates can be unstable/non-robust across regimes | Implement walk-forward folds for DL path or clearly segregate DL mode as experimental |
| No-lookahead validator assumes `date` column directly | `/Users/darrenwon/my-git/dpolaris_ai/ml/validation.py:45` | Medium | Fragile with `timestamp`-based inputs; can break validation behavior | Make validator time-column-agnostic (`timestamp/date/datetime`) |
| Reproducibility "re-executable" check is inferred from snapshot/hash presence, not actual replay | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:420` | Medium | Score can overstate true reproducibility | Add optional deterministic replay check and mark score with hard evidence |
| Deep-learning calibration fit on test split used for model metadata | `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:516` | Medium | Calibration quality may look better than true forward deployment | Calibrate via CV/OOF or dedicated calibration holdout |
| Backtest engine default commission is zero | `/Users/darrenwon/my-git/dpolaris_ai/backtest/engine.py:23` | Medium | Can produce optimistic PnL unless caller overrides config | Use conservative non-zero defaults and guardrail warnings |

---

## 4) Feature Pipeline & Registry Audit

### What is actually used in training today

- Active production feature engine for classic and DL training is `FeatureEngine` in `/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:16`.
- It generates price/vol/volume/technical/time features and targets in one class.
- Training pipelines call this engine through:
  - classic: `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:713`
  - deep learning: `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:285`

### Feature registry status

- Plugin registry exists:
  - `/Users/darrenwon/my-git/dpolaris_ai/features/registry.py:34`
- Plugin catalog metadata exists:
  - `/Users/darrenwon/my-git/dpolaris_ai/features/registry.py:84`
- Default registry includes technical+fundamentals+macro+sentiment+regime plugins:
  - `/Users/darrenwon/my-git/dpolaris_ai/features/technical.py:490`
- But these plugin features are currently used mainly by tests, not main trainer path (`FeatureEngine` path is dominant).

### Feature Catalog (derived from code)

#### Active training feature families (`ml/features.py`)

- Returns: `return_1d`, `return_5d`, `return_10d`, `return_20d` (`/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:92`)
- Trend/MA ratios: `price_sma*`, `sma*_ratio`, `ema_*`, momentum/roc (`/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:97` to `/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:127`)
- Volatility: `hvol_*`, `atr_14`, `atr_percent`, `bb_*`, `kc_width` (`/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:141`)
- Volume: `vol_sma_*`, `vol_ratio_*`, `obv*`, `vwap_20`, `ad_line` (`/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:173`)
- Technical: `rsi_14`, `macd*`, `stoch*`, `williams_r`, `cci`, `adx`, `trend_strength`, etc. (`/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:203`)
- Time features: day/week/month/quarter flags (`/Users/darrenwon/my-git/dpolaris_ai/ml/features.py:271`)

#### Plugin feature families implemented but not primary-path integrated

- `returns`, `trend`, `momentum`, `volatility`, `volume`, `structure`, `gaps`, `relative_strength`, plus fundamentals/macro/sentiment/regime plugin registration in `/Users/darrenwon/my-git/dpolaris_ai/features/technical.py:490`.

### Feature hygiene findings

- `feature_registry_version` and `missingness_per_feature` fields exist in artifact schema/writer, but trainer currently fills missingness as `{}`:
  - classic artifact payload: `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:820` and `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:822`
  - deep-learning artifact payload: `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:803` and `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:805`
- No implemented production pruning pipeline found for zero-variance/high-correlation duplicate suppression tied to run artifact fields.

---

## 5) Artifacts Compliance Audit

### Contract implementation

- Artifact version constant: `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:30`
- Writer entry: `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:583`
- File writes:
  - section JSON files + snapshots + manifest: `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:657` to `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:681`
- API retrieval endpoints:
  - `/runs`, `/runs/{id}`, `/runs/{id}/artifacts`, `/runs/{id}/artifact/{name}`, `/runs/compare` in `/Users/darrenwon/my-git/dpolaris_ai/api/server.py:1391`

### Artifacts Completeness Matrix (against required set)

| Required artifact | Generated? | Where | Always / Sometimes / Never | Notes |
|---|---|---|---|---|
| `run_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:658` | Always (if writer called) | Contract-compliant |
| `data_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:659` | Always | Contract-compliant |
| `feature_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:660` | Always | Contract-compliant |
| `split_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:661` | Always | Contract-compliant |
| `model_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:662` | Always | Contract-compliant |
| `metrics_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:663` | Always | Contract-compliant |
| `backtest_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:664` | Always | Often sparse depending on caller payload |
| `diagnostics_summary.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:665` | Always | Often sparse |
| `config_snapshot.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:666` | Always | Good |
| `dependency_snapshot.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:667` | Always | Good |
| `data_hashes.json` | Yes | `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:668` | Always | Good |
| `calibration_report.json` | No standalone file | N/A | Never | Calibration is embedded in `metrics_summary` only |
| `trade_log.csv` | Not as `trade_log.csv` in run artifact writer | Backtest engine writes `trades.csv` at `/Users/darrenwon/my-git/dpolaris_ai/backtest/engine.py:1077` | Sometimes (only backtest engine runs) | Naming mismatch + not guaranteed copied to run artifact |
| `equity_curve.csv` | Via backtest engine only | `/Users/darrenwon/my-git/dpolaris_ai/backtest/engine.py:1075` | Sometimes | Not guaranteed in `runs/<runId>` |
| `leakage_checks.json` | No standalone file | N/A | Never | Only string status in `feature_summary` |
| Plots | Optional | `/Users/darrenwon/my-git/dpolaris_ai/backtest/reporting.py:24` | Sometimes | Requires report generation + matplotlib |

---

## 6) Evaluation Integrity

### What is solid

- Standardized classification/regression/trading metrics implemented centrally:
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/evaluation.py:90`
- Calibration reliability outputs implemented (curve + Brier + calibration error):
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/evaluation.py:117`
- Primary objective scoring is configurable:
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/evaluation.py:347`
  - config defaults in `/Users/darrenwon/my-git/dpolaris_ai/data/config/prediction_precision.yaml:34`

### Integrity concerns

- Classic training uses simplified probability-to-return proxy, not full order-level execution engine:
  - `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:633`
  - backtest engine is separate and not integrated into trainer path.
- Deep-learning path is less rigorous than classic path:
  - single split, not walk-forward (`/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:310`)
  - leakage status hardcoded passed in artifact (`/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:807`).
- Backtest engine default commission is zero:
  - `/Users/darrenwon/my-git/dpolaris_ai/backtest/engine.py:23`.

---

## 7) Tests & CI

### Test baseline

- Executed in this audit: `.venv/bin/python -m pytest -q`  
- Result: `48 passed, 1 warning in 8.73s`.

### Existing high-value tests

- No-lookahead checks:
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_no_lookahead.py:10`
- Walk-forward default:
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_walk_forward_default.py:7`
- Artifact contract + backward compatibility:
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_training_artifact_contract.py:84`
- Data alignment/quality:
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_data_alignment.py:28`
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_data_quality_behavior.py:32`
- Backtest realism/costs:
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_backtest_engine_reality.py:50`
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_backtest_costs.py:8`
- Inspector causality:
  - `/Users/darrenwon/my-git/dpolaris_ai/tests/test_prediction_inspector_api.py:40`

### Test coverage gaps

1. No test currently fails when deep-learning artifact claims leakage pass without actual leakage check.
2. No fail-fast test that enforces stopping training when quality gates fail (`minimum_history.passed == false`).
3. No integration test proving trainer pipeline includes the full enriched feature stack (macro/fundamental/sentiment/regime) in production path.
4. No contract test requiring standalone `calibration_report.json` and `leakage_checks.json`.
5. No CI workflow found (no `.github/workflows/*`) to enforce tests on every change.

### Add these tests first

1. `test_deep_learning_leakage_status_is_measured_not_hardcoded` against `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py`.
2. `test_training_fails_when_quality_gate_min_history_fails` against `/Users/darrenwon/my-git/dpolaris_ai/data/dataset_builder.py`.
3. `test_train_pipeline_uses_enriched_features_not_ohlcv_only` against `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py`.
4. `test_run_artifact_includes_standalone_leakage_and_calibration_reports` against `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py`.
5. `test_deep_learning_validation_method_default_walk_forward_or_explicit_noncompliant_flag` against `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py`.

---

## Key Risks (Ranked)

1. **High**: Deep-learning observability incorrectly signals leakage safety (`leakage_checks_status` hardcoded passed).  
2. **High**: Quality gate outcomes are not enforcement gates; bad data can still train models.  
3. **High**: Enriched data/features are mostly not used by actual train paths (OHLCV-only reduction).  
4. **High**: Deep-learning validation is single split (no walk-forward default), reducing reliability.  
5. **Medium**: Artifact contract misses standalone leakage/calibration files and guaranteed trade/equity artifacts.  
6. **Medium**: Reproducibility score overstates re-executability (presence-based, not replay-based).  
7. **Medium**: Backtest engine default assumptions can still be optimistic (`commission_bps=0.0`).  
8. **Medium**: No CI gate to prevent regression in leakage/contract behavior.

---

## 30/60/90-Day Remediation Plan

### 30 days (stabilize integrity)

- Enforce quality gates as hard stop in training entrypoints.
- Replace deep-learning leakage status placeholder with measured result.
- Emit explicit non-compliance markers when path is not walk-forward validated.
- Raise artifact contract to require standalone leakage/calibration reports.

### 60 days (unify real feature path)

- Integrate plugin feature library into production trainer path.
- Ensure macro/fundamental/sentiment/regime features are causally joined and actually consumed by models.
- Add fold-stability outputs (feature importance variance) into artifacts.
- Wire full backtest engine outputs into run artifacts for completed runs.

### 90 days (production-readiness)

- Add deterministic replay command and include replay result in reproducibility score.
- Implement CI workflow running full leakage/contract/backtest/integration suite.
- Add guardrail endpoint and readiness computation directly from artifacts.
- Freeze training artifact schema v1.1 with backward-compat policy and migration tests.

---

## Definition of Done: Production-Ready Precision

- [ ] Train paths (classic + DL) enforce causal and quality checks before fitting.
- [ ] No training run can report leakage pass without a concrete leakage check artifact.
- [ ] Walk-forward (or explicit approved alternative) is encoded and auditable per run.
- [ ] Artifact folder includes reproducibility snapshots + standalone leakage/calibration reports.
- [ ] Backtest assumptions are non-zero/default-conservative and attached to every scored run.
- [ ] Enriched features (macro/fundamental/sentiment/regime) are provably in the production model input.
- [ ] Reproducibility score includes actual replay success, not snapshot presence only.
- [ ] CI enforces all critical tests on every merge.

---

## Top 10 Files To Review First

1. `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py`  
Reason: real `@train` production path and OHLCV-only reduction.

2. `/Users/darrenwon/my-git/dpolaris_ai/api/server.py`  
Reason: API train/predict/job routes, run artifact retrieval, signal synthesis.

3. `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py`  
Reason: canonical classic training/evaluation/calibration pipeline.

4. `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py`  
Reason: DL split/calibration/artifact generation and current compliance gaps.

5. `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning_worker.py`  
Reason: subprocess runtime behavior, data ingestion path, stability guards.

6. `/Users/darrenwon/my-git/dpolaris_ai/ml/features.py`  
Reason: actual feature/target math used by production trainers.

7. `/Users/darrenwon/my-git/dpolaris_ai/data/dataset_builder.py`  
Reason: source canonicalization + quality/alignment handoff point.

8. `/Users/darrenwon/my-git/dpolaris_ai/data/quality.py`  
Reason: quality policies, repair/drop logic, and missing fail-fast behavior.

9. `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py`  
Reason: observability contract + reproducibility semantics.

10. `/Users/darrenwon/my-git/dpolaris_ai/backtest/engine.py`  
Reason: true execution-friction backtest mechanics and artifact outputs.

---

## Immediate Fixes (Low Effort / High Impact)

1. Replace hardcoded deep-learning leakage status.
   - Location: `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning.py:807`
   - Impact: prevents false “green” readiness.

2. Enforce minimum-history/data-quality gate failure as training blocker.
   - Locations: `/Users/darrenwon/my-git/dpolaris_ai/data/dataset_builder.py:82`, `/Users/darrenwon/my-git/dpolaris_ai/core/ai.py:541`, `/Users/darrenwon/my-git/dpolaris_ai/ml/deep_learning_worker.py:177`
   - Impact: prevents garbage-in model runs.

3. Make no-lookahead validator time-column robust (`timestamp`/`date`/`datetime`).
   - Location: `/Users/darrenwon/my-git/dpolaris_ai/ml/validation.py:45`
   - Impact: avoids silent validation fragility.

4. Use conservative default commissions in execution config.
   - Location: `/Users/darrenwon/my-git/dpolaris_ai/backtest/engine.py:23`
   - Impact: reduces accidental optimistic backtests.

5. Emit standalone `leakage_checks.json` and `calibration_report.json` in run folder.
   - Location: `/Users/darrenwon/my-git/dpolaris_ai/ml/training_artifacts.py:657`
   - Impact: improves auditability and Java UI consistency.

6. Fix ticker metadata in classic artifact payload to reflect actual symbol, not model name token.
   - Location: `/Users/darrenwon/my-git/dpolaris_ai/ml/trainer_precision.py:806`
   - Impact: improves run list/filter reliability and comparability.

---

## Final Audit Position

The system is close to a strong auditable architecture, but it is not yet reliably “precise + reproducible + non-leaky” across all active train paths.  
The biggest gap is mismatch between implemented infrastructure and what the production training path actually consumes/enforces.


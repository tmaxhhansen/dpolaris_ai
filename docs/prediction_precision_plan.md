# Prediction Precision Plan

## Current State (Repo Audit)

### Project Structure
- `api/`: FastAPI endpoints, async deep-learning training queue/job logs
- `ml/`: feature engineering, classic trainer/predictor, deep-learning trainer/worker
- `tools/`: market/option data access (Yahoo Finance via `yfinance`)
- `core/`: AI command router (`@train`, `@predict`), config, DB, memory
- `cli/`: command-line entrypoints
- `daemon/`: scheduled retraining/scanning jobs

### Data Sources
- Primary market data: `tools/market_data.py` (`yfinance`) for OHLCV + options chain.
- Training uses historical bars only (`date/open/high/low/close/volume`).

### Feature Pipeline
- `ml/features.py` builds:
  - price/return/momentum and MA ratios
  - volatility (HV/ATR/Bollinger)
  - volume/OBV/VWAP
  - technical (RSI/MACD/Stoch/ADX/CCI)
  - time features (day/month/quarter/week)
- Targets:
  - `target_direction` (classification: up/down over horizon)
  - `target_return` (regression-style future return)
  - additional target columns (`target_magnitude`, volatility, max gain/loss)

### Models and Training Entrypoints
- Classic ML (`ml/trainer.py` -> `ModelTrainer`):
  - random forest, gradient boosting, xgboost, lightgbm, logistic
  - CLI/AI path: `@train SYMBOL` in `core/ai.py`
- Deep learning (`ml/deep_learning.py` + worker):
  - LSTM / Transformer
  - API path used by app Train button: `POST /api/jobs/deep-learning/train`
  - direct API path: `POST /api/deep-learning/train/{symbol}`

### Prediction Generation
- Classic: `ml/predictor.py` (`Predictor.predict` / `generate_trading_signal`)
- Deep learning: `DeepLearningTrainer.predict`, exposed at `/api/deep-learning/predict/{symbol}`
- Trade setup synthesis: `/api/signals/{symbol}` combines model output + latest feature state.

### Previous Evaluation Gaps
- Classification metrics were basic and inconsistent between classic/deep learning.
- No standardized regression/trading metric stack.
- Backtest-like results did not consistently include transaction cost + slippage.
- Walk-forward validation was not guaranteed as default in classic pipeline.
- Probability calibration output was not standardized/persisted.

### Leakage Risk Areas (Before Hardening)
- Potential preprocessing leakage risk in sequence scaling (deep-learning path) if fit on full split.
- No automated no-lookahead assertion test for feature generation.
- Validation mode could be switched away from walk-forward without explicit precision policy.

## Precision Objective

Primary objective is now config-driven:
- Default: maximize Sharpe subject to max drawdown cap.
- Fallback contributes classification quality (F1) to avoid degenerate high-turnover Sharpe artifacts.

Config lives in: `data/config/prediction_precision.yaml`.

## Implemented Enhancements

1. **Standardized metric engine**
   - Added `ml/evaluation.py`:
     - Classification: accuracy, precision, recall, F1, AUC, reliability bins, ECE-style calibration error, Brier score
     - Regression: MAE, RMSE, MAPE, directional accuracy, correlation, threshold hit-rates
     - Trading: CAGR, Sharpe, Sortino, max drawdown, profit factor, win rate, avg win/loss, exposure, turnover, slippage+cost-adjusted PnL
     - Primary-score computation
   - Added Platt calibration fit/apply utilities.

2. **Precision config + defaults**
   - Added `ml/precision_config.py`.
   - Added `data/config/prediction_precision.yaml` defining:
     - targets/horizon
     - walk-forward defaults
     - metric bins/thresholds
     - backtest assumptions
     - calibration mode
     - primary score constraints

3. **Walk-forward default + calibrated confidence**
   - Added precision trainer implementation: `ml/trainer_precision.py`.
   - `ml/trainer.py` now aliases `ModelTrainer` to precision-enhanced trainer.
   - Classic model training now:
     - runs walk-forward validation by default
     - computes full metric suites
     - stores probability calibration in model metadata
     - keeps backward-compatible top-level metrics keys for existing UI/API consumers

4. **Prediction confidence quality**
   - Updated `ml/predictor.py` to apply stored calibration metadata automatically.
   - Deep-learning predict path now supports calibration metadata application.

5. **Leakage hardening**
   - Added `ml/validation.py` with no-lookahead validator and seed management.
   - Deep-learning data scaling updated to fit scaler on train split only.

## Acceptance-Test Scope

Added tests assert:
- no-lookahead feature behavior
- cost/slippage effect on backtest PnL
- walk-forward default in precision config/training output
- calibrated confidence fields returned by predictor
- reproducible training outputs under fixed seed


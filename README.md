# dpolaris_ai

## Mac Control Center Integration

Use the venv Python entrypoint on macOS:

```bash
./.venv/bin/python -m cli.main server --host 127.0.0.1 --port 8420
```

The server sets `LLM_PROVIDER=none` by default for this command if not already set.

### Runtime files

- Managed backend PID: `~/dpolaris_data/run/backend.pid`
- Managed backend heartbeat/status: `~/dpolaris_data/run/backend.heartbeat.json`

### Backend control endpoints

- `POST /api/control/backend/start?force=false`
- `POST /api/control/backend/stop?force=false`
- `POST /api/control/backend/restart?force=false`
- `GET /api/control/backend/status`

`GET /api/control/backend/status` includes:

- `python_executable`
- `pid`
- `running`
- `uptime` and `uptime_seconds`
- `last_health`
- `current_health`
- `last_heartbeat`
- `port_owner_pid` / `port_conflict`

Example response (abbreviated):

```json
{
  "status": "ok",
  "managed": true,
  "running": true,
  "pid": 12345,
  "python_executable": "/Users/you/my-git/dpolaris_ai/.venv/bin/python",
  "uptime": "3m 12s",
  "last_health": {"ok": true, "timestamp": "2026-02-15T10:24:05Z"},
  "current_health": {"ok": true, "timestamp": "2026-02-15T10:24:06Z"}
}
```

### Port ownership safety and `force=true`

If port `8420` is already owned by a non-managed process, `/start` and `/restart` return HTTP `409` with structured error details and do not kill anything by default.

With `force=true`, the backend will kill the port owner only when command-line/cwd checks match an allowlist:

- command line contains `-m cli.main server`
- repository path context contains `dpolaris_ai`

Arbitrary processes are never force-killed.

### Orchestrator control surface (for dPolaris_ops)

- `GET /api/control/orchestrator/status`
- `POST /api/control/orchestrator/start`
- `POST /api/control/orchestrator/stop`

These provide a stable API surface for external control services.

### Java call pattern (recommended)

1. Try `GET /api/control/backend/status`
2. If backend unavailable, start process with `./.venv/bin/python -m cli.main server --host 127.0.0.1 --port 8420`
3. Poll `GET /health` until HTTP `200`
4. Use `POST /api/control/backend/restart` and `POST /api/control/backend/stop` for lifecycle operations

### Smoke test

```bash
python scripts/smoke_control_center.py --host 127.0.0.1 --port 8420
```

### Universe API (Nasdaq 500 / Watchlist / Combined)

Universe source of truth is filesystem JSON under:
- `~/dpolaris_data/universe/` by default
- `DPOLARIS_UNIVERSE_DIR` when explicitly set

If files are missing, deterministic defaults are generated on first request so UI tabs never hard-fail.

```bash
curl http://127.0.0.1:8420/api/universe/list
curl http://127.0.0.1:8420/api/universe/nasdaq500
curl http://127.0.0.1:8420/api/universe/watchlist
curl http://127.0.0.1:8420/api/universe/combined
curl -X POST 'http://127.0.0.1:8420/api/watchlist/add?symbol=AAPL'
curl -X POST 'http://127.0.0.1:8420/api/watchlist/remove?symbol=AAPL'
curl -X POST 'http://127.0.0.1:8420/api/universe/rebuild?force=true'
```

Expected:
- `/api/universe/list` returns names including `nasdaq500`, `watchlist`, `combined`
- `nasdaq500` and `combined` return non-empty `tickers`; `watchlist` can be empty until user adds symbols
- ticker rows include metadata keys for Java table rendering:
  - `symbol`, `name`, `sector`, `market_cap`, `avg_volume_7d`, `change_pct_1d`, `mentions`, `last_analysis_date`

Notes:
- NASDAQ 500 candidates are parsed from NasdaqTrader `nasdaqlisted.txt`, then ranked by market cap (yfinance metadata).
- Watchlist source of truth is `~/dpolaris_data/watchlist.json` (user-managed, persisted across launches).
- WSB 100 uses a mentions provider interface:
  - `PRAW` mode when `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` are configured
  - fallback cached JSON mode at `~/dpolaris_data/mentions/wsb_posts.json`
  - install optional dependency for live Reddit mode: `pip install praw`
- Combined is a deduped union of NASDAQ 500 + watchlist tickers.

### News API (provider fallback, no-key safe)

```bash
curl 'http://127.0.0.1:8420/api/news/AAPL?limit=20'
```

Response shape:
- `symbol`, `provider`, `count`, `items[]`, `warnings[]`, `cached`, `updated_at`
- each item has: `source`, `title`, `url`, `published_at`

Provider order:
1. Finnhub (`FINNHUB_API_KEY`)
2. Marketaux (`MARKETAUX_API_KEY`)
3. NewsAPI (`NEWSAPI_API_KEY`)
4. disabled fallback (empty list, warning only)

### Analysis Report Pipeline (LLM-free)

Deep-learning analysis reports are persisted as durable artifacts under:
- `~/dpolaris_data/analysis/*.json`

The backend generates a fixed multi-section report (Overview, Price/Volume Snapshot, Technical Indicators, Chart Patterns, Model Signals, News, Risk Notes, Next Steps) without requiring Anthropic keys.

```bash
# Fast report generation (no retraining required)
curl -X POST 'http://127.0.0.1:8420/api/analyze/report?symbol=AAPL'

# List saved reports (newest first)
curl 'http://127.0.0.1:8420/api/analysis/list?limit=200'

# List reports for one symbol
curl 'http://127.0.0.1:8420/api/analysis/by-symbol/AAPL?limit=50'

# Fetch one full report payload
curl 'http://127.0.0.1:8420/api/analysis/<analysis_id>'
```

Notes:
- Default news provider is keyless disabled fallback (returns empty list with warnings).
- Set `FINNHUB_API_KEY`, `MARKETAUX_API_KEY`, or `NEWSAPI_API_KEY` to enable live headline ingestion.

## Windows Orchestrator Notes

Always run server and orchestrator with the venv interpreter, not system Python.

### Start API server

```powershell
C:\my-git\dpolaris_ai\.venv\Scripts\python.exe -m cli.main server --host 127.0.0.1 --port 8420
```

### Start orchestrator

```powershell
C:\my-git\dpolaris_ai\.venv\Scripts\python.exe -m cli.main orchestrator --host 127.0.0.1 --port 8420 --interval-health 60 --interval-scan 30m --dry-run
```

### Task Scheduler recommendation

When creating a Scheduled Task:

- Program/script:
  - `C:\my-git\dpolaris_ai\.venv\Scripts\python.exe`
- Add arguments:
  - `-m cli.main orchestrator --host 127.0.0.1 --port 8420 --interval-health 60 --interval-scan 30m`
- Start in:
  - `C:\my-git\dpolaris_ai`

Do not target `C:\Users\...\Python311\python.exe`; that can break orchestrator ownership/interpreter consistency.

### Verification (Windows)

```powershell
pwsh -File C:\my-git\dpolaris_ai\scripts\verify_orchestrator_windows.ps1
```

## Deep Learning on Windows (Torch)

Use venv Python only:

```powershell
C:\my-git\dpolaris_ai\.venv\Scripts\python.exe -m pip install -r C:\my-git\dpolaris_ai\requirements-windows.txt
```

### GPU preferred (RTX / CUDA)

```powershell
pwsh -File C:\my-git\dpolaris_ai\scripts\install_torch_gpu.ps1
```

### CPU fallback

If GPU wheels are unavailable, CPU torch from default index is supported:

```powershell
C:\my-git\dpolaris_ai\.venv\Scripts\python.exe -m pip install torch
```

### Deep-learning smoke

```powershell
pwsh -File C:\my-git\dpolaris_ai\scripts\smoke_deep_learning_job.ps1
```

## Stock Metadata & Analysis API (Java Integration)

These endpoints provide stock metadata and analysis history for the Java control center's Deep Learning table.

### Stock Metadata Endpoint

Get sector, market cap, 7-day average volume, and 1-day change % for multiple symbols:

```bash
# macOS / Linux
curl 'http://127.0.0.1:8420/api/stocks/metadata?symbols=AAPL,MSFT,NVDA'

# Windows PowerShell
Invoke-RestMethod -Uri 'http://127.0.0.1:8420/api/stocks/metadata?symbols=AAPL,MSFT,NVDA'
```

Response:
```json
{
  "AAPL": {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "market_cap": 3000000000000,
    "avg_volume_7d": 50000000,
    "change_percent_1d": 1.25,
    "as_of": "2026-02-15T10:30:00",
    "source": "yfinance",
    "error": null
  }
}
```

### Analysis Last Date Endpoint

Get the timestamp of the most recent training run for each symbol:

```bash
# macOS / Linux
curl 'http://127.0.0.1:8420/api/analysis/last?symbols=AAPL,NVDA'

# Windows PowerShell
Invoke-RestMethod -Uri 'http://127.0.0.1:8420/api/analysis/last?symbols=AAPL,NVDA'
```

Response:
```json
{
  "AAPL": {
    "last_analysis_at": "2026-02-14T15:30:00Z",
    "run_id": "run_20260214_153000",
    "model_type": "lstm",
    "status": "completed"
  },
  "NVDA": {
    "last_analysis_at": null,
    "run_id": null
  }
}
```

### Analysis Detail Endpoint

Get detailed analysis artifacts for a symbol:

```bash
# macOS / Linux
curl 'http://127.0.0.1:8420/api/analysis/detail/AAPL'

# Windows PowerShell
Invoke-RestMethod -Uri 'http://127.0.0.1:8420/api/analysis/detail/AAPL'
```

Response:
```json
{
  "symbol": "AAPL",
  "last_analysis_at": "2026-02-14T15:30:00Z",
  "run_id": "run_20260214_153000",
  "model_type": "lstm",
  "status": "completed",
  "artifacts": [
    {"type": "dl_training", "title": "Deep Learning (LSTM) Summary", "data": {...}},
    {"type": "metrics", "title": "Model Metrics", "data": {...}},
    {"type": "data_quality", "title": "Data Quality", "data": {...}}
  ]
}
```

### Smoke Test

```bash
# macOS / Linux
./scripts/smoke_metadata_analysis.sh

# Quick test (fewer symbols)
./scripts/smoke_metadata_analysis.sh --quick
```

# dpolaris_ai

## Windows Orchestrator Notes

Always run server and orchestrator with the venv interpreter, not system Python.

### Start API server

```powershell
C:\my-git\dpolaris_ai\.venv\Scripts\python.exe -m cli.main server --host 127.0.0.1 --port 8420
```

### One-command clean start (Windows, deterministic port ownership)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_server_clean.ps1
```

Quick checks:

```powershell
irm http://127.0.0.1:8420/health
irm http://127.0.0.1:8420/api/debug/port-owner
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

## Universe API (Control Center)

List universes:

```powershell
irm http://127.0.0.1:8420/api/universe/list
```

Expected list shape:

```json
{
  "universes": [
    {"name": "all", "count": 123, "path": "dynamic:all", "updated_at": "2026-01-01T00:00:00+00:00"}
  ]
}
```

Quick smoke:

```powershell
irm http://127.0.0.1:8420/api/universe/list
irm http://127.0.0.1:8420/api/scan/universe/list
irm http://127.0.0.1:8420/api/universe/all
pwsh -File C:\my-git\dpolaris_ai\scripts\smoke_universe.ps1
pwsh -File C:\my-git\dpolaris_ai\scripts\smoke_universe_list.ps1
```

Fetch one universe:

```powershell
irm http://127.0.0.1:8420/api/universe/combined_1000
```

Expected response shape:

```json
{
  "name": "combined_1000",
  "path": "C:\\my-git\\dpolaris_ai\\universe\\combined_1000.json",
  "tickers": ["AAPL", "MSFT"],
  "count": 2
}
```

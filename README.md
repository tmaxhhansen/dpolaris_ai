# dpolaris_ai

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


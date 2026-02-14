#!/usr/bin/env pwsh
param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8420,
    [int]$TimeoutSec = 20
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Missing venv python: $PythonExe"
}

$RunDir = Join-Path $HOME "dpolaris_data\run"
$LogDir = Join-Path $HOME "dpolaris_data\logs"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$OrchPidFile = Join-Path $RunDir "orchestrator.pid"
$BackendPidFile = Join-Path $RunDir "backend.pid"
$OrchLog = Join-Path $LogDir "verify_orchestrator.log"

function Get-PortOwnerPid([int]$PortNum) {
    $rows = netstat -ano -p tcp | Select-String ":$PortNum" | Select-String "LISTENING"
    foreach ($row in $rows) {
        $parts = ($row.ToString() -split "\s+") | Where-Object { $_ -ne "" }
        if ($parts.Length -ge 5) {
            $pid = 0
            if ([int]::TryParse($parts[-1], [ref]$pid)) {
                return $pid
            }
        }
    }
    return $null
}

function Kill-ManagedPidFile([string]$Path) {
    if (-not (Test-Path $Path)) { return }
    $txt = (Get-Content $Path -Raw).Trim()
    if (-not $txt) { return }
    $pid = 0
    if (-not [int]::TryParse($txt, [ref]$pid)) { return }
    try {
        Stop-Process -Id $pid -Force -ErrorAction Stop
    } catch {}
}

Kill-ManagedPidFile $BackendPidFile
Kill-ManagedPidFile $OrchPidFile

$owner = Get-PortOwnerPid -PortNum $Port
if ($owner) {
    throw "Port $Port is in use by PID $owner. Refusing to kill unknown owner."
}

Write-Host "Starting orchestrator with venv python: $PythonExe"
$proc = Start-Process -FilePath $PythonExe -ArgumentList @(
    "-m", "cli.main", "orchestrator",
    "--host", $Host,
    "--port", "$Port",
    "--interval-health", "60",
    "--interval-scan", "30m",
    "--dry-run"
) -WorkingDirectory $RepoRoot -PassThru -RedirectStandardOutput $OrchLog -RedirectStandardError $OrchLog

$healthUrl = "http://$Host`:$Port/health"
$statusUrl = "http://$Host`:$Port/api/orchestrator/status"
$deadline = (Get-Date).AddSeconds($TimeoutSec)
$healthOk = $false
$statusOk = $false
$statusPayload = $null

while ((Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
        if ($health.status -eq "healthy") {
            $healthOk = $true
        }
    } catch {}

    try {
        $statusPayload = Invoke-RestMethod -Uri $statusUrl -Method Get -TimeoutSec 3
        if ($statusPayload.running -eq $true -and $statusPayload.backend_state.pid) {
            $statusOk = $true
            break
        }
    } catch {}

    Start-Sleep -Milliseconds 700
}

if (-not $healthOk) {
    throw "FAIL: /health did not become healthy within timeout."
}
if (-not $statusOk) {
    throw "FAIL: /api/orchestrator/status did not report running=true with backend pid."
}

$backendPid = [int]$statusPayload.backend_state.pid
$backendProc = Get-CimInstance Win32_Process -Filter "ProcessId=$backendPid" -ErrorAction SilentlyContinue
if ($null -eq $backendProc) {
    throw "FAIL: backend pid $backendPid not found."
}

$cmd = [string]$backendProc.CommandLine
if ($cmd -notmatch "\\\.venv\\Scripts\\python\.exe") {
    throw "FAIL: backend command line does not reference .venv\\Scripts\\python.exe : $cmd"
}

Write-Host "PASS: orchestrator status and backend interpreter look correct."
Write-Host "Orchestrator PID: $($proc.Id)"
Write-Host "Backend PID: $backendPid"

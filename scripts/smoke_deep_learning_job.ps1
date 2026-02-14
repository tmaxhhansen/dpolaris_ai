#!/usr/bin/env pwsh
param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8420,
    [int]$TimeoutSec = 180
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Missing venv python at $PythonExe"
}

$outLog = Join-Path $PSScriptRoot "smoke_dl_server_out.log"
$errLog = Join-Path $PSScriptRoot "smoke_dl_server_err.log"

Write-Host "Starting server..."
$env:LLM_PROVIDER = "none"
$server = Start-Process -FilePath $PythonExe -ArgumentList @(
    "-m", "cli.main", "server", "--host", $ApiHost, "--port", "$Port"
) -WorkingDirectory $RepoRoot -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog

try {
    $healthUrl = "http://$ApiHost`:$Port/health"
    $statusUrl = "http://$ApiHost`:$Port/api/deep-learning/status"
    $enqueueUrl = "http://$ApiHost`:$Port/api/jobs/deep-learning/train"

    $deadline = (Get-Date).AddSeconds(30)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
            if ($h.status -eq "healthy") {
                $ready = $true
                break
            }
        } catch {}
        Start-Sleep -Milliseconds 600
    }
    if (-not $ready) {
        throw "Server did not become healthy"
    }

    $dl = Invoke-RestMethod -Uri $statusUrl -Method Get -TimeoutSec 5
    Write-Host ("DL status: torch_importable={0}, cuda_available={1}" -f $dl.torch_importable, $dl.cuda_available)

    $payload = @{ symbol = "SPY"; model_type = "lstm"; epochs = 1 } | ConvertTo-Json
    $job = Invoke-RestMethod -Uri $enqueueUrl -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 10
    $jobId = [string]$job.id
    if (-not $jobId) { throw "Missing job id from enqueue response" }
    Write-Host "Queued job: $jobId"

    $jobUrl = "http://$ApiHost`:$Port/api/jobs/$jobId"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $state = Invoke-RestMethod -Uri $jobUrl -Method Get -TimeoutSec 5
        $status = [string]$state.status
        if ($status -eq "completed") {
            Write-Host ("PASS: deep-learning job completed (device={0})" -f $state.result.device)
            exit 0
        }
        if ($status -eq "failed") {
            $errCode = ""
            if ($state.result -and $state.result.error_code) {
                $errCode = [string]$state.result.error_code
            }
            Write-Host ("FAIL: deep-learning job failed. error={0} error_code={1}" -f $state.error, $errCode)
            exit 2
        }
        Start-Sleep -Seconds 2
    }

    throw "Timed out waiting for deep-learning job completion"
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    }
}


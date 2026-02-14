$ErrorActionPreference = "Stop"

$repoRoot = "C:\my-git\dpolaris_ai"
$pythonExe = "C:\my-git\dpolaris_ai\.venv\Scripts\python.exe"
$resetScript = Join-Path $repoRoot "scripts\reset_8420.ps1"
$logDir = "C:\Users\darre\dpolaris_data\logs"
$stdoutLog = Join-Path $logDir "server_out.log"
$stderrLog = Join-Path $logDir "server_err.log"
$healthUrl = "http://127.0.0.1:8420/health"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host ("FAIL missing interpreter: {0}" -f $pythonExe)
    exit 1
}

if (-not (Test-Path -LiteralPath $resetScript)) {
    Write-Host ("FAIL missing reset script: {0}" -f $resetScript)
    exit 1
}

if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Host "Resetting port 8420 state..."
powershell -ExecutionPolicy Bypass -File $resetScript
if ($LASTEXITCODE -ne 0) {
    Write-Host ("FAIL reset_8420.ps1 returned exit code {0}" -f $LASTEXITCODE)
    exit $LASTEXITCODE
}

Write-Host "Starting backend server with clean environment..."
$cmdArgs = '/c set "LLM_PROVIDER=none" && "{0}" -m cli.main server --host 127.0.0.1 --port 8420 1>>"{1}" 2>>"{2}"' -f $pythonExe, $stdoutLog, $stderrLog
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdArgs -WorkingDirectory $repoRoot -PassThru

Write-Host ("Started PID: {0}" -f $proc.Id)
Write-Host ("Health URL: {0}" -f $healthUrl)
Write-Host ("Stdout log: {0}" -f $stdoutLog)
Write-Host ("Stderr log: {0}" -f $stderrLog)
exit 0

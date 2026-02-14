$ErrorActionPreference = "Stop"

$repoRoot = "C:\my-git\dpolaris_ai"
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $repoRoot "requirements-windows.txt"

function Resolve-Python {
    if (Get-Command python3.11 -ErrorAction SilentlyContinue) { return "python3.11" }
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py -3.11" }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    throw "Python not found in PATH."
}

Set-Location $repoRoot

if (-not (Test-Path -LiteralPath $requirements)) {
    throw "Missing requirements file: $requirements"
}

$pyCmd = Resolve-Python
Write-Host ("Using bootstrap python: {0}" -f $pyCmd)

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating virtual environment..."
    if ($pyCmd -eq "py -3.11") {
        py -3.11 -m venv $venvDir
    } else {
        & $pyCmd -m venv $venvDir
    }
}

Write-Host "Installing requirements-windows.txt..."
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r $requirements

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host ("Next: {0} -m cli.main server --host 127.0.0.1 --port 8420" -f $venvPython)
Write-Host "Then test:"
Write-Host "  irm http://127.0.0.1:8420/api/universe/list"
Write-Host "  irm http://127.0.0.1:8420/api/universe/combined_1000"

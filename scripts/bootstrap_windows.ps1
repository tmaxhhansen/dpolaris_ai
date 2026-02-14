$ErrorActionPreference = "Stop"

$repoRoot = "C:\my-git\dpolaris_ai"
$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $repoRoot "requirements-windows.txt"

function Resolve-PythonCommand {
    if (Get-Command python3.11 -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python3.11"; Args = @() }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $null = & py -3.11 -c "import sys; print(sys.version)"
            if ($LASTEXITCODE -eq 0) {
                return @{ Cmd = "py"; Args = @("-3.11") }
            }
        } catch {
            # Ignore and fall back.
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python"; Args = @() }
    }

    throw "Python was not found in PATH. Install Python 3.11+ and retry."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)] [hashtable]$Py,
        [Parameter(Mandatory = $true)] [string[]]$Args
    )
    & $Py.Cmd @($Py.Args + $Args)
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Py.Cmd) $($Py.Args -join ' ') $($Args -join ' ')"
    }
}

Set-Location $repoRoot

if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "Missing requirements file: $requirementsPath"
}

$py = Resolve-PythonCommand
Write-Host ("Using bootstrap Python: {0} {1}" -f $py.Cmd, ($py.Args -join " "))

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating virtual environment (.venv)..."
    Invoke-Python -Py $py -Args @("-m", "venv", $venvPath)
} else {
    Write-Host "Virtual environment already exists."
}

Write-Host "Upgrading pip/setuptools/wheel..."
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip tooling." }

Write-Host "Installing requirements-windows.txt..."
& $venvPython -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements-windows.txt." }

Write-Host "Checking CUDA availability (torch)..."
$cudaCheck = @'
import json
try:
    import torch
    available = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if available else None
    print(json.dumps({
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": available,
        "cuda_device": device,
    }))
except Exception as exc:
    print(json.dumps({
        "torch_error": str(exc),
        "hint": "Install torch in this venv. For GPU builds, install the official CUDA wheel that matches your system.",
    }))
'@

& $venvPython -c $cudaCheck
if ($LASTEXITCODE -ne 0) { throw "CUDA check snippet failed." }

Write-Host "Bootstrap complete."
Write-Host ("Run server: {0} -m cli.main server --host 127.0.0.1 --port 8420" -f $venvPython)

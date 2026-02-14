#!/usr/bin/env pwsh
param(
    [string]$CudaChannel = "cu121"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Missing venv python at $PythonExe"
}

$IndexUrl = "https://download.pytorch.org/whl/$CudaChannel"
Write-Host "Installing PyTorch GPU wheels from $IndexUrl"

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install --index-url $IndexUrl torch torchvision torchaudio

Write-Host "Verifying torch + CUDA availability..."
& $PythonExe -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"


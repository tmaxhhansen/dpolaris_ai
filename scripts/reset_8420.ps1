$ErrorActionPreference = "Stop"

$repoRoot = "C:\my-git\dpolaris_ai"
$targetPort = 8420
$signatureA = "-m cli.main server"
$signatureB = "--port 8420"

function Get-PortOwnerPid {
    param([int]$Port)
    $lines = netstat -ano -p tcp | Select-String -Pattern (":{0}\s+.*LISTENING\s+" -f $Port)
    foreach ($line in $lines) {
        $parts = ($line.Line -replace "\s+", " ").Trim().Split(" ")
        if ($parts.Length -ge 5) {
            $pidText = $parts[$parts.Length - 1]
            if ($pidText -match "^\d+$") {
                return [int]$pidText
            }
        }
    }
    return $null
}

Write-Host "Scanning for repo-owned python server on port $targetPort ..."
$killed = @()

$procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'"
foreach ($proc in $procs) {
    $cmdline = [string]($proc.CommandLine)
    if (-not $cmdline) { continue }
    if ($cmdline -like "*$repoRoot*" -and $cmdline -like "*$signatureA*" -and $cmdline -like "*$signatureB*") {
        Write-Host ("Stopping PID {0}" -f $proc.ProcessId)
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            $killed += $proc.ProcessId
        } catch {
            Write-Host ("Warning: failed to stop PID {0}: {1}" -f $proc.ProcessId, $_.Exception.Message)
        }
    }
}

if ($killed.Count -eq 0) {
    Write-Host "No matching repo server python processes found."
} else {
    Write-Host ("Stopped {0} process(es): {1}" -f $killed.Count, ($killed -join ", "))
}

$timeoutSeconds = 20
$elapsed = 0
while ($elapsed -lt $timeoutSeconds) {
    $ownerPid = Get-PortOwnerPid -Port $targetPort
    if ($null -eq $ownerPid) {
        Write-Host "Port 8420 is free."
        exit 0
    }
    Start-Sleep -Seconds 1
    $elapsed += 1
}

$finalOwner = Get-PortOwnerPid -Port $targetPort
if ($null -ne $finalOwner) {
    Write-Host ("Port 8420 is still in use by PID {0}." -f $finalOwner)
    exit 1
}

Write-Host "Port 8420 is free."
exit 0

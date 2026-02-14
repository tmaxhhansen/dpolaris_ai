param(
    [string]$BaseUrl = "http://127.0.0.1:8420"
)

$ErrorActionPreference = "Stop"

function Fail([string]$message) {
    Write-Host ("FAIL: {0}" -f $message)
    exit 1
}

try {
    $health = Invoke-RestMethod -Uri ($BaseUrl + "/health") -TimeoutSec 8
} catch {
    Fail ("health request failed: " + $_.Exception.Message)
}

if ($health.status -ne "healthy") {
    Fail ("unexpected health response: " + ($health | ConvertTo-Json -Compress))
}
Write-Host "PASS: /health"

try {
    $list = Invoke-RestMethod -Uri ($BaseUrl + "/api/universe/list") -TimeoutSec 8
} catch {
    Fail ("/api/universe/list failed: " + $_.Exception.Message)
}

if (-not ($list.PSObject.Properties.Name -contains "universes")) {
    Fail "/api/universe/list missing universes field"
}
Write-Host ("PASS: /api/universe/list count={0}" -f [int]$list.count)

$targetName = $null
if ($list.universes -and $list.universes.Count -gt 0) {
    $first = $list.universes[0]
    if ($first -is [string]) {
        $targetName = $first
    } elseif ($first.PSObject.Properties.Name -contains "name") {
        $targetName = [string]$first.name
    }
}

if ([string]::IsNullOrWhiteSpace($targetName)) {
    Write-Host "PASS: no universe entries available; skipping /api/universe/{name} check"
    exit 0
}

try {
    $universe = Invoke-RestMethod -Uri ($BaseUrl + "/api/universe/" + $targetName) -TimeoutSec 12
} catch {
    Fail ("/api/universe/$targetName failed: " + $_.Exception.Message)
}

if (-not ($universe.PSObject.Properties.Name -contains "tickers")) {
    Fail ("/api/universe/$targetName missing tickers field")
}

Write-Host ("PASS: /api/universe/{0} count={1}" -f $targetName, [int]$universe.count)
Write-Host "PASS: smoke_universe"
exit 0

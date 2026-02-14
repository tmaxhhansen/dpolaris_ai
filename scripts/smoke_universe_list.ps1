$ErrorActionPreference = "Stop"

function Test-UniverseListEndpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 10
    } catch {
        Write-Host ("FAIL {0} - request error: {1}" -f $Url, $_.Exception.Message)
        return $false
    }

    if ($response -is [System.Array]) {
        Write-Host ("PASS {0} - returned array with {1} item(s)." -f $Url, $response.Count)
        return $true
    }

    Write-Host ("FAIL {0} - expected JSON array but got {1}." -f $Url, $response.GetType().FullName)
    return $false
}

$allPassed = $true
$allPassed = (Test-UniverseListEndpoint -Url "http://127.0.0.1:8420/api/universe/list") -and $allPassed
$allPassed = (Test-UniverseListEndpoint -Url "http://127.0.0.1:8420/api/scan/universe/list") -and $allPassed

if ($allPassed) {
    Write-Host "PASS smoke_universe_list"
    exit 0
}

Write-Host "FAIL smoke_universe_list"
exit 1

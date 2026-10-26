[CmdletBinding()]
param(
    [string]$OutputFile = "artifacts/selfplay/production.jsonl",
    [ValidateSet("random", "greedy", "mcts")]
    [string]$PolicyA = "mcts",
    [ValidateSet("random", "greedy", "mcts")]
    [string]$PolicyB = "greedy",
    [int]$Seed = 20261022,
    [ValidateSet(1, 2, 3)]
    [int]$Level = 3,
    [int]$Simulations = 32,
    [double]$TimeLimitSeconds = 1,
    [double]$MinutesPerBatch = 60,
    [int]$RestartDelaySeconds = 5
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$tableturf = Join-Path $projectRoot ".venv/Scripts/tableturf.exe"
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputFile)) {
    [System.IO.Path]::GetFullPath($OutputFile)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputFile))
}

if (-not (Test-Path -LiteralPath $tableturf)) {
    throw "tableturf executable not found: $tableturf"
}

$resume = (Test-Path -LiteralPath $outputPath) -and
    ((Get-Item -LiteralPath $outputPath).Length -gt 0)

while ($true) {
    $arguments = @(
        "selfplay",
        "--games", "100000000",
        "--seed", "$Seed",
        "--level", "$Level",
        "--policy-a", $PolicyA,
        "--policy-b", $PolicyB,
        "--simulations", "$Simulations",
        "--time-limit", "$TimeLimitSeconds",
        "--max-minutes", "$MinutesPerBatch",
        "--output", $outputPath
    )
    if ($resume) {
        $arguments += "--resume"
    }

    & $tableturf @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 130) {
        Write-Host "Loop interrupted; completed games remain saved."
        break
    }
    if ($exitCode -ne 0) {
        throw "tableturf selfplay failed with exit code $exitCode"
    }

    $resume = $true
    Start-Sleep -Seconds $RestartDelaySeconds
}


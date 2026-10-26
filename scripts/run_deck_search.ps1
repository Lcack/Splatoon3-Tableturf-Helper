[CmdletBinding()]
param(
    [string]$OutputDir = "artifacts/deck_search/main",
    [int]$Decks = 120,
    [int]$GamesPerDeck = 20,
    [int]$Seed = 20261022,
    [int]$Simulations = 32,
    [double]$TimeLimitSeconds = 1,
    [double]$MinutesPerBatch = 240
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$tableturf = Join-Path $projectRoot ".venv/Scripts/tableturf.exe"
$pythonw = Join-Path $projectRoot ".venv/Scripts/pythonw.exe"
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDir))
}

if (-not (Test-Path -LiteralPath $tableturf)) {
    throw "tableturf executable not found: $tableturf"
}
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

Start-Process `
    -FilePath $pythonw `
    -ArgumentList @(
        "-m", "tableturf_rebuild", "deck-search-monitor",
        "--output", $outputPath
    ) `
    -WorkingDirectory $projectRoot | Out-Null

$resume = Test-Path -LiteralPath (Join-Path $outputPath "meta.json")
while ($true) {
    $arguments = @(
        "deck-search",
        "--decks", "$Decks",
        "--games-per-deck", "$GamesPerDeck",
        "--seed", "$Seed",
        "--simulations", "$Simulations",
        "--time-limit", "$TimeLimitSeconds",
        "--max-minutes", "$MinutesPerBatch",
        "--no-monitor",
        "--output", $outputPath
    )
    if ($resume) {
        $arguments += "--resume"
    }

    & $tableturf @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 130) {
        Write-Host "Deck search interrupted; completed games remain saved."
        break
    }
    if ($exitCode -ne 0) {
        throw "deck-search failed with exit code $exitCode"
    }

    $progressPath = Join-Path $outputPath "progress.json"
    if (Test-Path -LiteralPath $progressPath) {
        $status = (Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json).status
        if ($status -eq "complete") {
            Write-Host "Deck search complete."
            break
        }
    }
    $resume = $true
    Start-Sleep -Seconds 3
}


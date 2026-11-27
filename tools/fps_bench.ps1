<#
.SYNOPSIS
  Record game FPS with the translator off, idle and actively translating, then report the impact.

.DESCRIPTION
  Uses PresentMon 2.x (https://github.com/GameTechDev/PresentMon/releases) to capture frame
  times of the game process for -Seconds in each scenario. Between scenarios the script pauses
  so you can put the translator in the right state:

    off          translator not running
    idle         translator running, nobody talking
    translating  translator running, foreign speech playing (e.g. a looping clip through Discord)

  Stand still in the same spot (a practice range works well) for every run so the scene is
  comparable. Results go to bench\fps\<timestamp>\ and are summarised by tools\fps_summary.py.

.EXAMPLE
  .\tools\fps_bench.ps1 -Game VALORANT-Win64-Shipping.exe -PresentMon C:\Tools\PresentMon.exe
#>
param(
    [Parameter(Mandatory)] [string] $Game,
    [string] $PresentMon = "PresentMon.exe",
    [int] $Seconds = 60,
    [string[]] $Scenarios = @("off", "idle", "translating"),
    [double] $MaxImpact = 5
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$out = Join-Path $repo ("bench\fps\" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force $out | Out-Null

$files = @()
foreach ($s in $Scenarios) {
    Write-Host ""
    Write-Host "Scenario '$s': set the translator up for this run, focus the game, then press Enter here." -ForegroundColor Cyan
    [void] (Read-Host)
    Write-Host "Recording $Seconds s in 5 s..."
    Start-Sleep 5
    $csv = Join-Path $out "$s.csv"
    & $PresentMon --process_name $Game --output_file $csv --timed $Seconds `
        --terminate_after_timed --stop_existing_session --no_console_stats
    if (-not (Test-Path $csv)) { throw "PresentMon wrote no file for '$s'. Is '$Game' running? Run as administrator." }
    $files += $csv
}

Write-Host ""
python (Join-Path $PSScriptRoot "fps_summary.py") @files --process $Game --max-impact $MaxImpact
exit $LASTEXITCODE

# Builds dist\LingoOverlay\ (PyInstaller) and dist\LingoOverlay-Setup-<version>.exe (Inno Setup).
# Needs: the runtime deps (requirements.txt) and PyInstaller in the active Python, and
# Inno Setup 6 (https://jrsoftware.org/isinfo.php; or `winget install JRSoftware.InnoSetup`).
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
param([string]$Python = "python")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$version = & $Python -c "import lingo_overlay; print(lingo_overlay.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Could not import lingo_overlay with $Python" }
Write-Host "Building Lingo Overlay $version"

& $Python -m PyInstaller packaging\lingo_overlay.spec --noconfirm --distpath dist --workpath build
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @("$env:ProgramFiles\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
              "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found. Install it, or ship dist\LingoOverlay\ as a zip." }

& $iscc "/DAppVersion=$version" packaging\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Write-Host "Done: dist\LingoOverlay-Setup-$version.exe"

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'Creating the local Python environment...'
    python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed.' }
}
& $pythonPath (Join-Path $projectRoot 'scripts\check_runtime.py')
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
Push-Location $projectRoot
try {
    & $pythonPath -m streamlit run app.py
} finally {
    Pop-Location
}

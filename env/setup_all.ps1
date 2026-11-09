# =============================================================================
#  setup_all.ps1  -  slides 42-44 in one go (Windows)
#
#    1. download Miniforge installer from GitHub (conda-forge)      [slide 42]
#    2. silent install: Just Me, default path, NOT added to PATH     [slide 42]
#    3. create conda env "mlstudy" + rdkit/sklearn/xgboost/...       [slide 43]
#    4. pip install torch (CPU), torch_geometric, shap               [slide 43]
#    5. register Jupyter kernel, run check_env.py                    [slides 44, 46]
#
#  How to run (PowerShell, normal user, no admin needed):
#    cd "C:\Users\mthog\CEPL ML\env"
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#    .\setup_all.ps1
#
#  Re-running is safe: finished steps are skipped.
#  Korean user name?  ->  .\setup_all.ps1 -InstallDir C:\miniforge3   (slide 42/46)
# =============================================================================
param(
    [string]$InstallDir = "$env:USERPROFILE\miniforge3",
    [string]$EnvName    = "mlstudy",
    [switch]$SkipCheck
)
$ErrorActionPreference = 'Stop'
$condaBat = Join-Path $InstallDir 'condabin\conda.bat'
$here     = Split-Path -Parent $MyInvocation.MyCommand.Path

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

# --- 1-2. Miniforge -----------------------------------------------------------
if (Test-Path $condaBat) {
    Step "Miniforge already installed at $InstallDir (skip)"
} else {
    $url       = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'
    $installer = Join-Path $env:TEMP 'Miniforge3-Windows-x86_64.exe'
    Step "Downloading Miniforge installer -> $installer"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    Step "Installing Miniforge silently (Just Me, no PATH change) -> $InstallDir"
    # NSIS switches: /S silent, /D=<dir> must be LAST and unquoted
    $installerArgs = @('/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', '/S', "/D=$InstallDir")
    $p = Start-Process -FilePath $installer -ArgumentList $installerArgs -Wait -PassThru
    if (-not (Test-Path $condaBat)) { throw "Miniforge install failed (exit $($p.ExitCode)). Install manually: slide 42." }
    Write-Host "Miniforge installed. 'Miniforge Prompt' is now in the Start menu."
}

# --- 3. conda env ---------------------------------------------------------------
$envs = & $condaBat env list 2>$null
if ($envs -match "^\s*$EnvName\s") {
    Step "conda env '$EnvName' already exists (skip create)"
} else {
    Step "conda create -n $EnvName ... (5-10 min)"
    & $condaBat create -n $EnvName -c conda-forge python=3.11 rdkit scikit-learn xgboost pandas matplotlib jupyter ipykernel -y
    if ($LASTEXITCODE -ne 0) { throw "conda create failed - see slide 46 table (00_설치안내.md section 6)" }
}

# --- 4. pip packages (inside the env) --------------------------------------------
Step "pip install torch (CPU build)"
& $condaBat run --no-capture-output -n $EnvName pip install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "pip install torch failed" }

Step "pip install torch_geometric shap"
& $condaBat run --no-capture-output -n $EnvName pip install torch_geometric shap
if ($LASTEXITCODE -ne 0) { throw "pip install torch_geometric shap failed" }

# --- 5. kernel + check ---------------------------------------------------------------
Step "register Jupyter kernel '$EnvName'"
& $condaBat run --no-capture-output -n $EnvName python -m ipykernel install --user --name $EnvName --display-name "Python ($EnvName)"

if (-not $SkipCheck) {
    Step "check_env.py (slide 44)"
    & $condaBat run --no-capture-output -n $EnvName python (Join-Path $here 'check_env.py')
}

Write-Host ""
Write-Host "All done." -ForegroundColor Green
Write-Host "Every study session (slide 45):  Miniforge Prompt ->  conda activate $EnvName  ->  cd <practice folder>  ->  jupyter notebook"
Write-Host "   or double-click  start_jupyter.bat"

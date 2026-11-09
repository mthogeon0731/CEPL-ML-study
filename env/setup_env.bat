@echo off
setlocal
REM ============================================================================
REM  setup_env.bat  -  create the "mlstudy" conda environment (slides p.43)
REM
REM  Run from:  Start menu > "Miniforge Prompt"  (conda must be on PATH there)
REM      cd /d "C:\Users\mthog\OneDrive\바탕 화면\CEPL ML\env"
REM      setup_env.bat
REM
REM  Steps = the 4 lines on slide 43 + Jupyter kernel registration (slide 46,
REM  row 5) + installation check (slide 44).  Safe to re-run: skips an
REM  existing "mlstudy" environment.
REM ============================================================================

where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] conda not found. Open "Miniforge Prompt" from the Start menu and run this file there. ^(slide 46, row 1^)
  echo         No Miniforge yet?  Run setup_all.ps1 instead ^(installs Miniforge, then this^).
  pause
  exit /b 1
)

conda env list | findstr /b /r /c:"mlstudy[ ]" >nul
if not errorlevel 1 (
  echo [skip] environment "mlstudy" already exists - skipping conda create
) else (
  echo [1/4] conda create -n mlstudy ... ^(5-10 min^)
  call conda create -n mlstudy -c conda-forge python=3.11 rdkit scikit-learn xgboost pandas matplotlib jupyter ipykernel -y
  if errorlevel 1 goto :err
)

echo [2/4] conda activate mlstudy
call conda activate mlstudy
if errorlevel 1 goto :err

echo [3/4] pip install torch ^(CPU build^)
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :err

echo [4/4] pip install torch_geometric shap
pip install torch_geometric shap
if errorlevel 1 goto :err

echo [+]   register Jupyter kernel "mlstudy"
python -m ipykernel install --user --name mlstudy --display-name "Python (mlstudy)"

echo [+]   installation check ^(slide 44^)
python "%~dp0check_env.py"

echo.
echo Done.  Next time: start_jupyter.bat  ^(or the 3 lines on slide 45^).
pause
exit /b 0

:err
echo.
echo [ERROR] A step failed. Look up the FIRST line of the error message in
echo         00_install guide ^(00_*.md^) section 6 = slide 46 troubleshooting table.
pause
exit /b 1

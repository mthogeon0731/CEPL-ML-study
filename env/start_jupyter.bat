@echo off
setlocal
REM ============================================================================
REM  start_jupyter.bat  -  the 3 lines of slide 45, "every study session":
REM      conda activate mlstudy
REM      cd <practice folder>
REM      jupyter notebook
REM  Double-click this file, or run it from Miniforge Prompt.
REM  Practice folder = ..\261105_Week1_Practice  (unzip the lab's practice zip there)
REM ============================================================================

set "PRACTICE=%~dp0..\261105_Week1_Practice"
set "CONDA_BAT=%USERPROFILE%\miniforge3\condabin\conda.bat"
if exist "C:\miniforge3\condabin\conda.bat" set "CONDA_BAT=C:\miniforge3\condabin\conda.bat"

where conda >nul 2>&1
if errorlevel 1 (
  if not exist "%CONDA_BAT%" (
    echo [ERROR] conda not found. Install Miniforge first: setup_all.ps1  ^(slide 42^)
    pause
    exit /b 1
  )
  call "%CONDA_BAT%" activate mlstudy
) else (
  call conda activate mlstudy
)
if errorlevel 1 (
  echo [ERROR] "conda activate mlstudy" failed - create the environment first: setup_env.bat ^(slide 43^)
  pause
  exit /b 1
)

if not exist "%PRACTICE%\feature.py" (
  echo [WARN] practice folder not ready: %PRACTICE%
  echo        unzip 261105_Week1_Practice.zip there first ^(slide 47^). Opening Jupyter in the CEPL ML folder instead.
  set "PRACTICE=%~dp0.."
)
cd /d "%PRACTICE%"
echo Starting Jupyter Notebook in: %CD%
jupyter notebook

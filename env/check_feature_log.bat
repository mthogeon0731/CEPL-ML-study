@echo off
setlocal
REM check_feature_log.bat - slide 44 check inside the lab's practice folder: python feature.py (expects piperidine x: (6, 147) ...)
set "LOG=%~dp0check_feature_log.txt"
set "PRACTICE=%~dp0..\261105_Week1_Practice"
set "CONDA_BAT=%USERPROFILE%\miniforge3\condabin\conda.bat"
if exist "C:\miniforge3\condabin\conda.bat" set "CONDA_BAT=C:\miniforge3\condabin\conda.bat"
> "%LOG%" echo [check_feature_log] %DATE% %TIME%
if not exist "%PRACTICE%\feature.py" (
  >> "%LOG%" echo feature.py not found in %PRACTICE%
  goto :end
)
cd /d "%PRACTICE%"
>> "%LOG%" echo cwd: %CD%
>> "%LOG%" echo --- python -c "import rdkit, sklearn, xgboost; print('ok')" ---
call "%CONDA_BAT%" run --no-capture-output -n mlstudy python -c "import rdkit, sklearn, xgboost; print('ok')" >> "%LOG%" 2>&1
>> "%LOG%" echo --- python feature.py ---
call "%CONDA_BAT%" run --no-capture-output -n mlstudy python feature.py >> "%LOG%" 2>&1
>> "%LOG%" echo exit code %ERRORLEVEL%
>> "%LOG%" echo --- import datamanager, models (no training) ---
call "%CONDA_BAT%" run --no-capture-output -n mlstudy python -c "import datamanager, models, train; print('imports ok')" >> "%LOG%" 2>&1
>> "%LOG%" echo exit code %ERRORLEVEL%
:end
type "%LOG%"
timeout /t 8 >nul

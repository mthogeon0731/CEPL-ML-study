@echo off
setlocal
REM check_env_log.bat - double-click: runs check_env.py inside the mlstudy env and writes check_env_log.txt next to it.
set "LOG=%~dp0check_env_log.txt"
set "CONDA_BAT=%USERPROFILE%\miniforge3\condabin\conda.bat"
if exist "C:\miniforge3\condabin\conda.bat" set "CONDA_BAT=C:\miniforge3\condabin\conda.bat"
> "%LOG%" echo [check_env_log] %DATE% %TIME%
if not exist "%CONDA_BAT%" (
  >> "%LOG%" echo conda.bat not found: %CONDA_BAT%  - Miniforge not installed yet
  goto :end
)
>> "%LOG%" echo conda: %CONDA_BAT%
call "%CONDA_BAT%" --version >> "%LOG%" 2>&1
call "%CONDA_BAT%" env list >> "%LOG%" 2>&1
>> "%LOG%" echo --- check_env.py (mlstudy) ---
call "%CONDA_BAT%" run --no-capture-output -n mlstudy python "%~dp0check_env.py" >> "%LOG%" 2>&1
>> "%LOG%" echo exit code %ERRORLEVEL%
>> "%LOG%" echo --- jupyter kernels ---
call "%CONDA_BAT%" run --no-capture-output -n mlstudy jupyter kernelspec list >> "%LOG%" 2>&1
:end
type "%LOG%"
timeout /t 8 >nul

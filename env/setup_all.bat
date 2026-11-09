@echo off
REM setup_all.bat - double-click to run setup_all.ps1 (slides 42-44) in a PowerShell window that stays open.
REM Same as:  powershell -ExecutionPolicy Bypass -File setup_all.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0setup_all.ps1"

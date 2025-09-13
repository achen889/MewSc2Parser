@echo off
REM === mew_ready.bat ===
REM Opens a new Command Prompt in this folder
REM and immediately activates the venv (mew_venv).

setlocal

REM Change to the directory where this bat file resides
cd /d "%~dp0"

REM Path to venv
set VENV_PATH=%cd%\mew_venv

REM Check that venv exists
if not exist "%VENV_PATH%\Scripts\activate" (
    echo [!] Virtual environment not found at:
    echo     %VENV_PATH%
    echo Run mew_venv.bat first to create it.
    pause
    exit /b 1
)

REM Open new console and activate venv first thing
start cmd /k "cd /d %cd% && call \"%VENV_PATH%\Scripts\activate\""

endlocal
exit

@echo off
echo ====================================================
echo   MEW VENV SETUP SCRIPT (Python 3.11)
echo ====================================================
echo.

REM Step 1: Create virtual environment
echo [1/4] Creating virtual environment: mew_venv ...
py -3.11 -m venv mew_venv
if errorlevel 1 (
    echo [!] Failed to create venv. Make sure Python 3.11 is installed.
    pause
    exit /b 1
)
echo [+] Virtual environment created.
echo.

REM Step 2: Activate virtual environment
echo [2/4] Activating virtual environment ...
call mew_venv\Scripts\activate
if errorlevel 1 (
    echo [!] Failed to activate venv.
    pause
    exit /b 1
)
echo [+] venv activated.
echo.

REM Step 3: Install dependencies from requirements.txt
echo [3/4] Installing core requirements ...
python -m pip install --upgrade pip
python -m pip install -r "requirements.txt"
if errorlevel 1 (
    echo [!] Some packages from requirements.txt failed to install.
) else (
    echo [+] requirements.txt installed successfully.
)
echo.

REM Step 4: Install AI-related dependencies
echo [4/4] Installing AI-related requirements ...
python -m pip install -r "requirements_ai.txt"
if errorlevel 1 (
    echo [!] Some packages from requirements_ai.txt failed to install.
) else (
    echo [+] requirements_ai.txt installed successfully.
)
echo.

echo ====================================================
echo   SETUP COMPLETE
echo   Virtual environment: mew_venv
echo   To activate again later:
echo      call mew_venv\Scripts\activate
echo ====================================================
echo.

REM Pause so user can read the output
pause

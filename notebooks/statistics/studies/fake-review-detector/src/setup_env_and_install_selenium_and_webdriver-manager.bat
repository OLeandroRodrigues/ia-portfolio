@echo off
cd /d "%~dp0"

REM Create the virtual environment  if it doesn't exist
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Active the virtual environment
call venv\Scripts\activate

REM Update pip and install packages
echo 📦 Installing required packages...
python -m pip install --upgrade pip
python -m pip install selenium webdriver-manager

echo.
echo ✅ Setup complete.
echo To activate the virtual environment later, run:
echo     venv\Scripts\activate

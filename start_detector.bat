@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "dataset" mkdir dataset
if not exist "alerts" mkdir alerts

echo Starting detector...
start "DriveAI Detector" cmd /k python app.py
endlocal

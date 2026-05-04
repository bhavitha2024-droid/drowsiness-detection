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

echo Starting dashboard at http://127.0.0.1:8501
start "DriveAI Dashboard" cmd /k streamlit run dashboard_streamlit.py --server.address 127.0.0.1 --server.port 8501
endlocal

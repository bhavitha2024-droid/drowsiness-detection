@echo off
setlocal

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
  taskkill /PID %%a /F >nul 2>&1
)

for /f "skip=1 tokens=2 delims==; " %%a in ('wmic process where "name='python.exe' and (commandline like '%%app.py%%' or commandline like '%%dashboard_streamlit.py%%')" get processid /value 2^>nul') do (
  if not "%%a"=="" taskkill /PID %%a /F >nul 2>&1
)

echo Attempted to stop DriveAI detector and dashboard processes.
endlocal

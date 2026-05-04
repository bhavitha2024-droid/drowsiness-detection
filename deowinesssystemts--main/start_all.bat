@echo off
setlocal
cd /d "%~dp0"

call start_detector.bat
timeout /t 5 /nobreak >nul
call start_dashboard.bat

echo.
echo DriveAI is starting.
echo Detector window: desktop OpenCV window
echo Dashboard URL: http://127.0.0.1:8501
echo.
endlocal

@echo off
setlocal
cd /d "%~dp0"
echo Resonix AI PC server for mobile access
echo.
echo Open this address on this PC:
echo http://127.0.0.1:8000/
echo.
echo On a phone connected to the same Wi-Fi, use your PC IP address, for example:
echo http://192.168.x.x:8000/
echo.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    python main.py
)

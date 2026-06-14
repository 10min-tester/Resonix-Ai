@echo off
setlocal
cd /d "%~dp0"
echo Resonix AI Lite server
echo.
echo Open this address in Chrome or Edge:
echo http://127.0.0.1:8088/
echo.
echo Mobile devices on the same Wi-Fi can use your PC IP address, for example:
echo http://192.168.x.x:8088/
echo.
python -m http.server 8088 --bind 0.0.0.0

@echo off
setlocal
cd /d "%~dp0"
echo Resonix AI Lite server
echo.
echo Open this address in Chrome or Edge:
echo http://127.0.0.1:8088/
echo.
python -m http.server 8088

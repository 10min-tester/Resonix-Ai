@echo off
setlocal
cd /d "%~dp0"
python quality_guard_smoke.py
endlocal

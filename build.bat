@echo off
chcp 65001 > nul

echo ===========================================
echo Resonix AI PyInstaller Build Script
echo ===========================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment was not found at .venv\Scripts\activate.bat
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
echo [INFO] Virtual environment activated.

if not exist "models" (
    mkdir models
    echo [WARN] Created models directory. Add ONNX model files before building.
)

if not exist "frontend\index.html" (
    echo [ERROR] frontend\index.html was not found.
    pause
    exit /b 1
)

echo [INFO] Starting PyInstaller build...
pyinstaller --clean build.spec

if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo [INFO] Build complete: dist\ResonixAI\ResonixAI.exe
pause

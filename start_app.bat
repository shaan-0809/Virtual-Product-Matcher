@echo off
chcp 65001 >nul
title Visual Product Matcher - Windows Startup

echo.
echo ========================================
echo    Visual Product Matcher
echo    Windows Startup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ℹ️  No virtual environment found, using system Python
)

echo.
echo 🚀 Starting Visual Product Matcher...
echo.

REM Start the application using the startup script
python start_app.py --host 0.0.0.0

echo.
echo 👋 Application stopped
pause

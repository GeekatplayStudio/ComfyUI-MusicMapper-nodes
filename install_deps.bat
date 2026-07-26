@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="
if exist "%~dp0..\..\..\python_embeded\python.exe" set "PYTHON_CMD=%~dp0..\..\..\python_embeded\python.exe"
if not defined PYTHON_CMD if exist "%~dp0..\..\..\python_embedded\python.exe" set "PYTHON_CMD=%~dp0..\..\..\python_embedded\python.exe"
if not defined PYTHON_CMD set "PYTHON_CMD=py -3"

%PYTHON_CMD% -V >nul 2>&1
if errorlevel 1 set "PYTHON_CMD=python"

%PYTHON_CMD% -V >nul 2>&1
if errorlevel 1 (
    echo Could not find a usable Python interpreter.
    echo Please run "pip install -r requirements.txt" manually in your ComfyUI python environment.
    exit /b 1
)

echo ===================================================
echo Geekatplay Studio - MusicMapper Dependency Installer
echo Using Python: %PYTHON_CMD%
echo ===================================================
echo Installing dependencies from requirements.txt...
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Installation failed.
    exit /b %errorlevel%
)

echo.
echo ===================================================
echo Installation complete successfully!
echo Please restart ComfyUI.
echo ===================================================
pause

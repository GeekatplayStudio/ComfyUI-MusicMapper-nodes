@echo off
setlocal

cd /d "%~dp0"

echo ===================================================
echo Geekatplay Studio - MusicMapper Dependency Installer
echo Created by Vladimir Chopine
echo ===================================================
echo.

set "PYTHON_CMD="
if exist "%~dp0..\..\..\python_embeded\python.exe" set "PYTHON_CMD=%~dp0..\..\..\python_embeded\python.exe"
if not defined PYTHON_CMD if exist "%~dp0..\..\..\python_embedded\python.exe" set "PYTHON_CMD=%~dp0..\..\..\python_embedded\python.exe"
if not defined PYTHON_CMD set "PYTHON_CMD=py -3"

%PYTHON_CMD% -V >nul 2>&1
if errorlevel 1 set "PYTHON_CMD=python"

%PYTHON_CMD% -V >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Could not find a usable Python interpreter.
    echo Please run "pip install -r requirements.txt" manually in your ComfyUI python environment.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_CMD%
echo Installing audio analysis dependencies from requirements.txt...
echo.
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Installation failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ---------------------------------------------------
echo Checking Ollama installation status...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [TIP] Ollama is not installed or not in PATH.
    echo If you want local LLM music prompt generation, download Ollama from:
    echo https://ollama.com/download/windows
    echo Then run: ollama pull llama3
    echo (MusicMapper will also work offline using its built-in rules engine!)
) else (
    echo [OK] Ollama is detected on your system.
)
echo ---------------------------------------------------

echo.
echo ===================================================
echo Installation complete successfully!
echo Please restart ComfyUI to load the Geekatplay Studio nodes.
echo ===================================================
pause

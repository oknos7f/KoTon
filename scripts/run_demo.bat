@echo off
REM ==============================================================================
REM KoTon Real-Time Security Detection System - Windows Quickstart Launcher
REM ==============================================================================

cd /d "%~dp0.."

echo ==================================================================
echo   KoTon: Real-Time GNN Provenance Intrusion Detection System
echo   Windows / WSL2 3-Window WebGL SOC Platform
echo ==================================================================

if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found in .venv. Using system python...
)

if not exist "checkpoints\koton_hgt_best.pt" (
    echo [*] Training GNN model checkpoint...
    python scripts\train_darpa_tc.py --arch hgt --epochs 5 --batch-size 16
)

echo.
echo [*] Starting KoTon Flask Server...
echo [*] Access at: http://localhost:5000/split
echo.

python scripts\run_demo.py

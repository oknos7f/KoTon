#!/usr/bin/env bash
# ==============================================================================
# KoTon Real-Time Security Detection System - WSL2 / Linux Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=================================================================="
echo "  KoTon: Real-Time GNN Provenance Intrusion Detection System"
echo "  WSL2 Linux Host Ingestion & 3-Window WebGL SOC Platform"
echo "=================================================================="

# 1. Check or activate Python virtual environment
if [ -d ".venv" ]; then
    echo "[*] Activating existing virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[*] Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "[*] Creating new Python virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[*] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# 2. Check if trained model checkpoint exists
CHECKPOINT="$PROJECT_ROOT/checkpoints/koton_hgt_best.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[!] Trained GNN checkpoint not found. Training model now on DARPA TC stream..."
    python3 scripts/train_darpa_tc.py --arch hgt --epochs 5 --batch-size 16
fi

# 3. Check Linux auditd status in WSL2
if command -v auditctl &> /dev/null; then
    echo "[*] Checking auditd service status..."
    if sudo service auditd status &> /dev/null || sudo /etc/init.d/auditd status &> /dev/null; then
        echo "[+] Linux auditd service is active."
    else
        echo "[*] Starting auditd service in WSL2..."
        sudo service auditd start 2>/dev/null || true
    fi
else
    echo "[!] auditctl not found. Running in hybrid/simulation mode."
    echo "    (To enable real Linux syscall ingestion, run: sudo bash scripts/setup_wsl.sh)"
fi

# 4. Start Flask Multi-Window Web Server
echo ""
echo "=================================================================="
echo "  Starting KoTon Server on http://0.0.0.0:5000"
echo "  Access from Windows browser at:"
echo "    - Unified Split View:  http://localhost:5000/split"
echo "    - Main Hub:            http://localhost:5000/"
echo "    - Window 1 (Victim):   http://localhost:5000/app"
echo "    - Window 2 (Attacker): http://localhost:5000/terminal"
echo "    - Window 3 (SOC 3D):   http://localhost:5000/dashboard"
echo "=================================================================="
echo ""

python3 scripts/run_demo.py

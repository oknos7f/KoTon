#!/usr/bin/env bash
# ==============================================================================
# KoTon WSL2 Linux Auditd Configuration & Setup Script
# ==============================================================================

set -e

echo "[*] Initializing KoTon environment for WSL2..."

# 1. Install auditd & python dependencies if needed
if command -v apt-get &> /dev/null; then
    echo "[*] Updating package list & installing auditd..."
    sudo apt-get update -y
    sudo apt-get install -y auditd audispd-plugins python3-venv python3-pip
fi

# 2. Start auditd service
echo "[*] Starting Linux auditd service..."
sudo service auditd start || sudo /etc/init.d/auditd start

# 3. Configure audit rules for provenance tracking
echo "[*] Setting audit rules for process exec, network connect, and sensitive files..."
sudo auditctl -D # Clear existing rules
sudo auditctl -b 8192

# Rule 1: Track process executions (execve)
sudo auditctl -a always,exit -F arch=b64 -S execve -k koton_exec

# Rule 2: Track network connections (connect)
sudo auditctl -a always,exit -F arch=b64 -S connect -k koton_net

# Rule 3: Track access to sensitive system files
sudo auditctl -w /etc/passwd -p rwxa -k koton_sensitive
sudo auditctl -w /etc/shadow -p rwxa -k koton_sensitive
sudo auditctl -w /tmp -p wa -k koton_tmp

echo "[+] Audit rules successfully loaded:"
sudo auditctl -l

echo ""
echo "[+] Setup complete! You can now run:"
echo "    python3 scripts/run_demo.py"
echo "    Then access http://localhost:5000 in your browser."

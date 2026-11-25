#!/usr/bin/env bash
set -euo pipefail

STEP_DELAY="${KOTON_ATTACK_DELAY:-5.0}"
SUBSTEP_DELAY="${KOTON_SUBSTEP_DELAY:-1.2}"
WEB_SIM_DIR="/var/www/html/koton_sim"
if [ ! -w "/var/www/html" ] 2>/dev/null; then
    WEB_SIM_DIR="/tmp/koton_webroot"
fi
EXFIL_PORT=4444

print_usage() {
    cat <<'EOF'
Usage: ./demo_attack.sh [OPTIONS]

Options:
  (no args)   Run all 5 scenarios sequentially at a slow observable pace
  --1         Scenario 1: Port Probe & Directory Enumeration (Class 1: scans)
  --2         Scenario 2: Webshell Drop & Child RCE Chain (Class 2: webshell/reverse_shell)
  --3         Scenario 3: Password Hash Dictionary Cracking (Class 3: cracking)
  --4         Scenario 4: Privilege Escalation & Shadow Read (Class 4: privilege_escalation)
  --5         Scenario 5: Sensitive Archive & Socket Exfiltration (Class 5: service_stop/dnsteal)
  --all       Explicitly run all 5 scenarios sequentially
  -h, --help  Show this help message
EOF
}

prepare_sandbox() {
    mkdir -p "${WEB_SIM_DIR}"
    if [ ! -f /tmp/mock_shadow ]; then
        echo "root:\$6\$rounds=656000\$mockhash:19740:0:99999:7:::" > /tmp/mock_shadow
        chmod 0644 /tmp/mock_shadow
    fi
}

run_stage_1() {
    echo "[Stage 1/5] Reconnaissance & Port/Directory Scan (Class 1: scans)..."
    for port in 22 80 443 3306 8080 9000; do
        python3 - "${port}" <<'PY'
import socket, sys
p = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.15)
try:
    s.connect(("127.0.0.1", p))
except OSError:
    pass
finally:
    s.close()
PY
        sleep "${SUBSTEP_DELAY}"
    done
    ls -la /etc /var/www /tmp >/dev/null 2>&1 || true
    echo "  -> Stage 1 completed."
}

run_stage_2() {
    echo "[Stage 2/5] Webshell Drop & Child RCE Chain (Class 2: webshell/reverse_shell)..."
    cat > "${WEB_SIM_DIR}/cmd_webshell.sh" <<'SH'
#!/bin/sh
id
uname -a
cat /etc/hostname
SH
    chmod +x "${WEB_SIM_DIR}/cmd_webshell.sh"
    sleep "${SUBSTEP_DELAY}"
    /bin/sh -c "${WEB_SIM_DIR}/cmd_webshell.sh >/tmp/webshell_out.txt 2>&1"
    sleep "${SUBSTEP_DELAY}"
    cat /tmp/webshell_out.txt >/dev/null 2>&1 || true
    echo "  -> Stage 2 completed."
}

run_stage_3() {
    echo "[Stage 3/5] Credential Hash Dictionary Cracking (Class 3: cracking)..."
    python3 - <<'PY'
import hashlib
target_plain = "admin2026"
target_hash = hashlib.sha256(target_plain.encode()).hexdigest()
with open("/tmp/hash_dump.txt", "w", encoding="utf-8") as fp:
    fp.write(f"svc_admin:{target_hash}\n")
PY
    sleep "${SUBSTEP_DELAY}"
    python3 - <<'PY'
import hashlib
candidates = ["password", "123456", "ubuntu", "qwerty", "koton", "admin2026"]
with open("/tmp/hash_dump.txt", "r", encoding="utf-8") as fp:
    line = fp.read().strip()
user, target_h = line.split(":", 1)
for word in candidates:
    if hashlib.sha256(word.encode()).hexdigest() == target_h:
        with open("/tmp/cracked_creds.txt", "w", encoding="utf-8") as out:
            out.write(f"{user}:{word}\n")
        break
PY
    sleep "${SUBSTEP_DELAY}"
    echo "  -> Stage 3 completed."
}

run_stage_4() {
    echo "[Stage 4/5] Privilege Escalation & Credential Access (Class 4: privilege_escalation)..."
    sudo -n -l >/dev/null 2>&1 || true
    sleep "${SUBSTEP_DELAY}"
    find /usr/bin -maxdepth 1 -perm -4000 >/tmp/suid_bins.txt 2>/dev/null || true
    sleep "${SUBSTEP_DELAY}"
    cat /etc/passwd >/dev/null
    sleep "${SUBSTEP_DELAY}"
    cat /tmp/mock_shadow >/dev/null
    echo "  -> Stage 4 completed."
}

run_stage_5() {
    echo "[Stage 5/5] Sensitive Archive & Socket Exfiltration (Class 5: service_stop/dnsteal)..."
    tar -czf /tmp/exfil_customers_secret.tar.gz \
        /tmp/mock_shadow /tmp/suid_bins.txt /tmp/webshell_out.txt /tmp/cracked_creds.txt 2>/dev/null || \
        tar -czf /tmp/exfil_customers_secret.tar.gz /tmp/mock_shadow 2>/dev/null || true
    sleep "${SUBSTEP_DELAY}"
    python3 - "${EXFIL_PORT}" <<'PY'
import socket, sys, threading
port = int(sys.argv[1])

def dummy_listener(ready_evt):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    srv.settimeout(5.0)
    ready_evt.set()
    try:
        conn, _ = srv.accept()
        while conn.recv(4096):
            pass
        conn.close()
    except OSError:
        pass
    finally:
        srv.close()

evt = threading.Event()
t = threading.Thread(target=dummy_listener, args=(evt,), daemon=True)
t.start()
evt.wait(timeout=2.0)

with open("/tmp/exfil_customers_secret.tar.gz", "rb") as fp:
    payload = fp.read()

cli = socket.create_connection(("127.0.0.1", port), timeout=2.0)
cli.sendall(payload)
cli.close()
t.join(timeout=2.0)
PY
    echo "  -> Stage 5 completed."
}

TARGETS=()
for arg in "$@"; do
    case "${arg}" in
        --1|-1|1) TARGETS+=("1") ;;
        --2|-2|2) TARGETS+=("2") ;;
        --3|-3|3) TARGETS+=("3") ;;
        --4|-4|4) TARGETS+=("4") ;;
        --5|-5|5) TARGETS+=("5") ;;
        --all|-a) TARGETS=("1" "2" "3" "4" "5") ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "[Error] Unknown option: ${arg}" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

echo "============================================================"
echo " [KoTon Live Scenario Simulator] Multi-Stage Provenance"
echo " Host: $(hostname) | Kernel: $(uname -r) | User: $(id -un)"
echo "============================================================"

prepare_sandbox

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "[Mode] Sequential Full Chain (slow observable mode: stage_delay=${STEP_DELAY}s, substep_delay=${SUBSTEP_DELAY}s)"
    TARGETS=("1" "2" "3" "4" "5")
    TOTAL="${#TARGETS[@]}"
    IDX=0
    for stage_id in "${TARGETS[@]}"; do
        IDX=$((IDX + 1))
        "run_stage_${stage_id}"
        if [ "${IDX}" -lt "${TOTAL}" ]; then
            sleep "${STEP_DELAY}"
        fi
    done
else
    echo "[Mode] Selected Scenario IDs: ${TARGETS[*]}"
    TOTAL="${#TARGETS[@]}"
    IDX=0
    for stage_id in "${TARGETS[@]}"; do
        IDX=$((IDX + 1))
        "run_stage_${stage_id}"
        if [ "${IDX}" -lt "${TOTAL}" ]; then
            sleep "${SUBSTEP_DELAY}"
        fi
    done
fi

echo "[KoTon Live Scenario Simulator] Execution finished."

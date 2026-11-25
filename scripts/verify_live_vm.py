import logging
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.app import ensure_runtime_started

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_live_vm")


def ssh_cmd(ssh_key: Path, remote_command: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Execute a command inside the isolated QEMU/KVM Ubuntu VM over SSH port 2222."""
    return subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=3",
            "-i",
            str(ssh_key),
            "-p",
            "2222",
            "ubuntu@127.0.0.1",
            remote_command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def main() -> None:
    """Boot Ubuntu VM, sync parameterized demo_attack.sh and idle service, and verify live streaming."""
    rt = ensure_runtime_started(start_listeners=False)
    collector = rt["collector"]
    engine = rt["engine"]
    engine.reset_session_aliases()
    collector.start()

    subprocess.run(["bash", str(PROJECT_ROOT / "scripts" / "start_ubuntu_vm.sh")], check=True)
    ssh_key = Path.home() / ".koton_vm" / "id_ed25519"

    logger.info("Waiting for Ubuntu VM SSH (127.0.0.1:2222)...")
    ssh_ready = False
    for _ in range(45):
        try:
            res = ssh_cmd(ssh_key, "uname -r", timeout=6)
            if res.returncode == 0:
                ssh_ready = True
                logger.info("Connected to Ubuntu VM! Guest kernel: %s", res.stdout.strip())
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2.0)

    assert ssh_ready, "Timed out waiting for Ubuntu VM SSH."

    idle_provision_cmd = (
        "sudo mkdir -p /var/www/html && sudo chown ubuntu:ubuntu /var/www/html; "
        "cat <<'EOF' | sudo tee /usr/local/bin/koton_idle_heartbeat.sh >/dev/null\n"
        "#!/usr/bin/env bash\n"
        "while true; do\n"
        "    cat /etc/hostname >/dev/null 2>&1 || true\n"
        "    sleep 3\n"
        "    cat /etc/os-release >/dev/null 2>&1 || true\n"
        "    sleep 4\n"
        "    ls /tmp >/dev/null 2>&1 || true\n"
        "    sleep 5\n"
        "done\n"
        "EOF\n"
        "sudo chmod 0755 /usr/local/bin/koton_idle_heartbeat.sh; "
        "cat <<'EOF' | sudo tee /etc/systemd/system/koton-idle-activity.service >/dev/null\n"
        "[Unit]\n"
        "Description=KoTon Benign Idle Telemetry Heartbeat\n"
        "After=auditd.service koton-audit-forwarder.service\n\n"
        "[Service]\n"
        "Type=simple\n"
        "User=ubuntu\n"
        "ExecStart=/usr/local/bin/koton_idle_heartbeat.sh\n"
        "Restart=always\n"
        "RestartSec=3\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
        "EOF\n"
        "sudo service auditd restart || sudo systemctl restart auditd || true; "
        "sudo /usr/sbin/augenrules --load || sudo /sbin/auditctl -R /etc/audit/rules.d/99-koton.rules || true; "
        "sudo chmod 0644 /var/log/audit/audit.log || true; "
        "sudo systemctl daemon-reload && "
        "sudo systemctl enable koton-audit-forwarder.service koton-idle-activity.service && "
        "sudo systemctl restart koton-audit-forwarder.service koton-idle-activity.service"
    )
    ssh_cmd(ssh_key, idle_provision_cmd, timeout=90)

    rules_res = ssh_cmd(ssh_key, "sudo /usr/sbin/auditctl -l || sudo /sbin/auditctl -l", timeout=15)
    logger.info("Active VM kernel auditd rules:\n%s", rules_res.stdout.strip())
    assert "No rules" not in rules_res.stdout and "koton_exec" in rules_res.stdout, (
        f"Expected active kernel syscall audit rules, got: {rules_res.stdout}"
    )

    subprocess.run(
        [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-i",
            str(ssh_key),
            "-P",
            "2222",
            str(PROJECT_ROOT / "scripts" / "demo_attack.sh"),
            "ubuntu@127.0.0.1:/home/ubuntu/demo_attack.sh",
        ],
        check=True,
    )
    ssh_cmd(ssh_key, "chmod +x /home/ubuntu/demo_attack.sh && sed -i 's/\\r$//' /home/ubuntu/demo_attack.sh", timeout=15)

    stream_ready = False
    for _ in range(20):
        ssh_cmd(ssh_key, "cat /etc/hostname >/dev/null", timeout=10)
        time.sleep(1.0)
        if collector.drain_raw_logs():
            stream_ready = True
            break
    assert stream_ready, "VM auditd forwarder stream did not reach the host collector!"
    logger.info("Idle heartbeat & auditd forwarder stream confirmed on host collector.")

    _ = collector.drain_edges()
    _ = collector.drain_raw_logs()
    engine.reset_session_aliases()

    logger.info("Testing parameterized scenario '/home/ubuntu/demo_attack.sh --1'...")
    s1_res = ssh_cmd(
        ssh_key,
        "KOTON_SUBSTEP_DELAY=0.05 /home/ubuntu/demo_attack.sh --1",
        timeout=30,
    )
    assert "Stage 1/5" in s1_res.stdout and "Stage 2/5" not in s1_res.stdout, (
        f"Expected only Stage 1 to execute for --1, got:\n{s1_res.stdout}"
    )
    time.sleep(2.0)
    s1_edges = collector.drain_edges()
    s1_logs = collector.drain_raw_logs()
    assert len(s1_edges) > 0 and len(s1_logs) > 0, "Expected live edges and raw logs from --1 scenario!"
    engine.ingest_streaming_edges(s1_edges)
    s1_tick = engine.run_streaming_tick()
    assert s1_tick is not None
    logger.info(
        "[Stage --1 Verified] Raw logs=%d | Added nodes=%d | Tick=%.2f ms",
        len(s1_logs),
        len(s1_tick["graph_delta"]["added_nodes"]),
        float(s1_tick["telemetry"]["latency_ms"]),
    )

    logger.info("Testing full sequential '/home/ubuntu/demo_attack.sh' (Stages 1..5)...")
    full_res = ssh_cmd(
        ssh_key,
        "KOTON_ATTACK_DELAY=0.15 KOTON_SUBSTEP_DELAY=0.04 /home/ubuntu/demo_attack.sh",
        timeout=60,
    )
    logger.info("VM Full Scenario Output:\n%s", full_res.stdout.strip())
    assert "Stage 5/5" in full_res.stdout

    time.sleep(1.0)
    live_edges = collector.drain_edges()
    engine.ingest_streaming_edges(live_edges)
    burst_out = engine.run_streaming_tick()
    tick_out = engine.run_streaming_tick()
    assert burst_out is not None and tick_out is not None
    burst_tel = burst_out["telemetry"]
    tel = tick_out["telemetry"]
    alert = tick_out["attack_alert"]

    logger.info(
        "[LIVE VM END-TO-END PASSED] Cold Burst=%.2f ms | Steady Tick=%.2f ms | Nodes=%d | Edges=%d | Online CPR=%.2f%% | Attack Detected=%s (%s, %d subgraph nodes)",
        float(burst_tel["latency_ms"]),
        float(tel["latency_ms"]),
        int(tel["num_nodes"]),
        int(tel["num_edges"]),
        float(tel["online_cpr_pct"]),
        alert["has_attack"],
        alert["dominant_attack_class"],
        len(alert["subgraph_node_ids"]),
    )
    try:
        assert float(burst_tel["latency_ms"]) < 500.0
        assert float(tel["latency_ms"]) < 250.0
        assert alert["has_attack"] is True
    finally:
        collector.stop()
        ssh_cmd(ssh_key, "sync && sudo poweroff || true", timeout=10)


if __name__ == "__main__":
    main()

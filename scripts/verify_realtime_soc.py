import json
import logging
from pathlib import Path
import socket
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.collector.audit_stream import AuditStreamCollector
from web.app import RUNTIME, app, ensure_runtime_started

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_realtime_soc")


def verify_hud_icons() -> None:
    """Verify user-supplied entity PNG icons exist under web/images/ without modifying them."""
    sizes = {}
    for name in ("process.png", "file.png", "socket.png"):
        p = PROJECT_ROOT / "web" / "images" / name
        assert p.is_file(), f"Missing HUD icon: {p}"
        with Image.open(p) as im:
            assert im.width > 0 and im.height > 0, f"Invalid icon dimensions for {name}: {im.size}"
            sizes[name] = im.size
    logger.info("[SOC Check 1/5 PASSED] User HUD icons preserved and verified: %s", sizes)


def verify_empty_startup_and_flask_endpoints() -> None:
    """Verify web server starts with an empty 0/0 cloud, serves user icons, and handles live simulation."""
    rt = ensure_runtime_started(start_listeners=False)
    rt["engine"].reset_session_aliases()
    rt["latest_tick_result"] = None

    with app.test_client() as client:
        r_idx = client.get("/")
        assert r_idx.status_code == 200 and b"live-log-stream" in r_idx.data

        r_snap_empty = client.get("/api/snapshot")
        assert r_snap_empty.status_code == 200
        empty_json = json.loads(r_snap_empty.data.decode("utf-8"))
        assert empty_json["nodes"] == [] and empty_json["edges"] == [], (
            "Expected empty cloud on initial startup!"
        )

        for icon_name in ("process.png", "file.png", "socket.png"):
            r_img = client.get(f"/images/{icon_name}")
            assert r_img.status_code == 200 and len(r_img.data) > 100

        r_thr = client.post("/api/threshold", json={"alpha": 0.05})
        assert r_thr.status_code == 200
        thr_json = json.loads(r_thr.data.decode("utf-8"))
        assert float(thr_json["latency_ms"]) < 10.0

    logger.info("[SOC Check 2/5 PASSED] Initial empty cloud (0/0), HUD icon serving, and REST API verified.")


def verify_live_tcp_collector_logs_and_significance_monotonicity() -> None:
    """Verify TCP collector, raw log draining, <250ms tick, and monotonic Significance state transitions."""
    rt = ensure_runtime_started(start_listeners=False)
    taxonomy = rt["engine"].taxonomy

    test_collector = AuditStreamCollector(
        taxonomy=taxonomy,
        tcp_host="127.0.0.1",
        tcp_port=19999,
        unix_socket_path="/tmp/koton_verify_audit.sock",
        default_host_id="ubuntu-vm",
        carryover_sec=0.15,
    )
    test_collector.start()
    time.sleep(0.25)

    now = time.time()
    stream_lines = [
        f'ubuntu-vm\ttype=SYSCALL msg=audit({now:.3f}:500): arch=c000003e syscall=257 success=yes pid=2900 ppid=1 uid=1000 exe="/usr/bin/cat" comm="cat"',
        f'ubuntu-vm\ttype=PATH msg=audit({now:.3f}:500): item=0 name="/etc/hostname" nametype=NORMAL',
        f'ubuntu-vm\ttype=EOE msg=audit({now:.3f}:500):',
        f'ubuntu-vm\ttype=SYSCALL msg=audit({now + 0.04:.3f}:501): arch=c000003e syscall=42 success=yes pid=3001 ppid=1 uid=1000 exe="/usr/bin/python3" comm="python3"',
        f'ubuntu-vm\ttype=SOCKADDR msg=audit({now + 0.04:.3f}:501): saddr=02001F907F0000010000000000000000',
        f'ubuntu-vm\ttype=EOE msg=audit({now + 0.04:.3f}:501):',
        f'ubuntu-vm\ttype=SYSCALL msg=audit({now + 0.08:.3f}:502): arch=c000003e syscall=0 success=yes pid=3005 ppid=1 uid=33 exe="/usr/sbin/apache2" comm="apache2"',
        f'ubuntu-vm\ttype=PATH msg=audit({now + 0.08:.3f}:502): item=0 name="/var/www/html/index.html" nametype=NORMAL',
        f'ubuntu-vm\ttype=EOE msg=audit({now + 0.08:.3f}:502):',
        f'ubuntu-vm\ttype=SYSCALL msg=audit({now + 0.14:.3f}:503): arch=c000003e syscall=0 success=yes pid=3005 ppid=1 uid=33 exe="/usr/sbin/apache2" comm="apache2"',
        f'ubuntu-vm\ttype=PATH msg=audit({now + 0.14:.3f}:503): item=0 name="/var/www/html/index.html" nametype=NORMAL',
        f'ubuntu-vm\ttype=EOE msg=audit({now + 0.14:.3f}:503):',
        f'ubuntu-vm\ttype=SYSCALL msg=audit({now + 0.25:.3f}:504): arch=c000003e syscall=59 success=yes pid=3010 ppid=3005 uid=33 exe="/bin/sh" comm="sh"',
        f'ubuntu-vm\ttype=EXECVE msg=audit({now + 0.25:.3f}:504): argc=3 a0="/bin/sh" a1="-c" a2="/tmp/koton_webroot/cmd_webshell.sh"',
        f'ubuntu-vm\ttype=PATH msg=audit({now + 0.25:.3f}:504): item=0 name="/tmp/koton_webroot/cmd_webshell.sh" nametype=NORMAL',
        f'ubuntu-vm\ttype=EOE msg=audit({now + 0.25:.3f}:504):',
        f'ubuntu-vm\ttype=SYSCALL msg=audit({now + 0.40:.3f}:505): arch=c000003e syscall=59 success=yes pid=3025 ppid=3010 uid=0 exe="/usr/bin/sudo" comm="sudo"',
        f'ubuntu-vm\ttype=EXECVE msg=audit({now + 0.40:.3f}:505): argc=3 a0="sudo" a1="cat" a2="/etc/shadow"',
        f'ubuntu-vm\ttype=PATH msg=audit({now + 0.40:.3f}:505): item=0 name="/etc/shadow" nametype=NORMAL',
        f'ubuntu-vm\ttype=EOE msg=audit({now + 0.40:.3f}:505):',
        f'ubuntu-vm\ttype=SYSCALL msg=audit({now + 0.60:.3f}:506): arch=c000003e syscall=42 success=yes pid=3030 ppid=3025 uid=0 exe="/usr/bin/curl" comm="curl"',
        f'ubuntu-vm\ttype=SOCKADDR msg=audit({now + 0.60:.3f}:506): saddr=0200115C7F0000010000000000000000',
        f'ubuntu-vm\ttype=PATH msg=audit({now + 0.60:.3f}:506): item=0 name="/tmp/exfil_customers_secret.tar.gz" nametype=NORMAL',
        f'ubuntu-vm\ttype=EOE msg=audit({now + 0.60:.3f}:506):',
    ]

    with socket.create_connection(("127.0.0.1", 19999), timeout=3.0) as cli:
        cli.sendall(("\n".join(stream_lines) + "\n").encode("utf-8"))
    time.sleep(0.25)

    drained_logs = test_collector.drain_raw_logs()
    drained = test_collector.drain_edges()
    test_collector.stop()
    assert len(drained_logs) == len(stream_lines), "Raw auditd log stream buffer missed lines!"
    assert len(drained) >= 7, f"Expected >=7 provenance edges from TCP stream, got {len(drained)}"
    logger.info(
        "[SOC Check 3/5 PASSED] AuditStreamCollector streamed %d raw logs and %d provenance edges.",
        len(drained_logs),
        len(drained),
    )

    engine = rt["engine"]
    engine.reset_session_aliases()
    engine.update_threshold_zero_reinference(alpha=0.05)
    engine.ingest_streaming_edges(drained)
    assert engine.online_cpr_merged >= 1, "Online CPR did not merge <1s duplicate read edges!"

    _ = engine.run_streaming_tick()
    tick_res = engine.run_streaming_tick()
    assert tick_res is not None
    latency_ms = float(tick_res["telemetry"]["latency_ms"])
    assert latency_ms < 250.0, f"Streaming tick exceeded 250ms SLA: {latency_ms} ms"
    logger.info(
        "[SOC Check 4/5 PASSED] Streaming tick latency=%.2f ms | Online CPR merged=%d.",
        latency_ms,
        engine.online_cpr_merged,
    )

    res_mid = engine.update_threshold_zero_reinference(alpha=0.05)
    mid_active = sum(1 for n in engine._prev_emitted_nodes.values() if n["is_malicious"] and n["attack_class"] != "benign")

    res_strict = engine.update_threshold_zero_reinference(alpha=0.002)
    strict_active = sum(1 for n in engine._prev_emitted_nodes.values() if n["is_malicious"] and n["attack_class"] != "benign")

    res_loose = engine.update_threshold_zero_reinference(alpha=0.24)
    loose_active = sum(1 for n in engine._prev_emitted_nodes.values() if n["is_malicious"] and n["attack_class"] != "benign")

    assert strict_active >= 1, "Strict significance should still highlight high-confidence attack nodes!"
    assert strict_active < mid_active < loose_active, (
        f"Expected monotonic active node transitions (strict < mid < loose), got {strict_active}, {mid_active}, {loose_active}"
    )
    assert float(res_strict["latency_ms"]) < 5.0
    logger.info(
        "[SOC Check 5/5 PASSED] Zero-Reinference (%.3f ms) monotonic transitions verified: alpha=0.002 -> %d active | alpha=0.050 -> %d active | alpha=0.240 -> %d active.",
        float(res_strict["latency_ms"]),
        strict_active,
        mid_active,
        loose_active,
    )
    RUNTIME["engine"].reset_session_aliases()
    RUNTIME["latest_tick_result"] = None


def main() -> None:
    """Run all real-time SOC verification checks."""
    verify_hud_icons()
    verify_empty_startup_and_flask_endpoints()
    verify_live_tcp_collector_logs_and_significance_monotonicity()
    logger.info("ALL REAL-TIME SOC VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()

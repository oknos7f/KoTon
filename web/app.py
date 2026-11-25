import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
import torch
import yaml

from src.collector.audit_stream import AuditStreamCollector
from src.data.taxonomy import LabelTaxonomy
from src.graph.encoder import HeterogeneousFeatureEncoder
from src.graph.sanitizer import PathSanitizer
from src.models.hetero_gnn import EdgeAwareMultiTaskHeteroGNN
from src.serving.engine import KoTonInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("koton.web")


def create_runtime_components() -> Dict[str, object]:
    """Load configuration, trained HeteroGNN checkpoint, Conformal profile, collector, and streaming engine."""
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    scenario = cfg["scenarios"]["default_scenario"]
    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    checkpoints_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taxonomy = LabelTaxonomy(PROJECT_ROOT / "configs" / "taxonomy.yaml", scenario=scenario)
    sanitizer = PathSanitizer(scenarios=cfg["scenarios"]["enabled"] + cfg["scenarios"]["excluded"])

    encoder = HeterogeneousFeatureEncoder(
        sanitizer=sanitizer,
        proc_exe_dim=int(cfg["features"]["process_exe_hfh_dim"]),
        proc_cmd_dim=int(cfg["features"]["process_cmd_emb_dim"]),
        file_path_dim=int(cfg["features"]["file_path_hfh_dim"]),
        sock_ip_dim=int(cfg["features"]["socket_ip_hfh_dim"]),
        sensitive_prefixes=cfg["features"]["sensitive_prefixes"],
        seed=int(cfg["project"]["seed"]),
    )
    ft_path = artifacts_dir / f"{scenario}_cmd_fasttext.model"
    if ft_path.is_file():
        encoder.load_cmdline_embedder(ft_path)
    else:
        encoder.fit_cmdline_embedder(["/sbin/init", "/bin/sh", "/usr/bin/sudo"])

    model = EdgeAwareMultiTaskHeteroGNN(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        gat_out_channels=int(cfg["model"]["gat_out_channels"]),
        gat_heads=int(cfg["model"]["gat_heads"]),
        num_layers=int(cfg["model"]["num_layers"]),
        edge_dim=int(cfg["model"]["edge_dim"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
        mask_rate=float(cfg["model"]["mask_rate"]),
        sce_gamma=float(cfg["model"]["sce_gamma"]),
        focal_gamma=float(cfg["model"]["focal_gamma"]),
        cls_loss_weight=float(cfg["model"]["cls_loss_weight"]),
        class_weights=torch.ones(int(cfg["model"]["num_classes"]), dtype=torch.float32),
    )

    ckpt_path = checkpoints_dir / f"{scenario}_hetero_gnn.pt"
    recon_scale = 1.0
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        recon_scale = float(ckpt.get("recon_calibration_scale", 1.0))
        logger.info("Loaded trained checkpoint from %s (device=%s)", ckpt_path, device)

    profile_path = artifacts_dir / "threshold_profile.json"
    engine = KoTonInferenceEngine(
        model=model,
        taxonomy=taxonomy,
        encoder=encoder,
        recon_scale=recon_scale,
        window_size_sec=float(cfg["graph"]["window_size_sec"]),
        micro_tick_sec=5.0,
        threshold_profile_path=profile_path if profile_path.is_file() else None,
        device=device,
    )

    tcp_port = int(os.environ.get("KOTON_COLLECTOR_PORT", "9999"))
    collector = AuditStreamCollector(
        taxonomy=taxonomy,
        tcp_host="0.0.0.0",
        tcp_port=tcp_port,
        default_host_id="ubuntu-vm",
    )

    return {
        "cfg": cfg,
        "engine": engine,
        "collector": collector,
        "latest_tick_result": None,
    }


app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "web" / "templates"),
    static_folder=str(PROJECT_ROOT / "web" / "static"),
)
app.config["SECRET_KEY"] = "koton-realtime-soc"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

RUNTIME: Dict[str, object] = {}
WORKER_STARTED = False
WORKER_LOCK = threading.Lock()


def ensure_runtime_started(start_listeners: bool = True) -> Dict[str, object]:
    """Lazily initialize runtime components and start background micro-tick streaming loop."""
    global WORKER_STARTED
    with WORKER_LOCK:
        if not RUNTIME:
            RUNTIME.update(create_runtime_components())
        if start_listeners and not WORKER_STARTED:
            collector: AuditStreamCollector = RUNTIME["collector"]
            collector.start()
            t = threading.Thread(target=_streaming_tick_loop, name="KoTonMicroTickWorker", daemon=True)
            t.start()
            WORKER_STARTED = True
    return RUNTIME


def _streaming_tick_loop() -> None:
    """Stream incoming raw auditd logs, incremental graph deltas, and 2-Hop alerts via SocketIO."""
    collector: AuditStreamCollector = RUNTIME["collector"]
    engine: KoTonInferenceEngine = RUNTIME["engine"]
    last_tick_wall = 0.0

    while True:
        raw_logs = collector.drain_raw_logs(max_items=60)
        if raw_logs:
            socketio.emit("audit_log_batch", {"logs": raw_logs})

        new_edges = collector.drain_edges()
        now = time.time()
        should_tick = False

        if new_edges:
            engine.ingest_streaming_edges(new_edges)
            should_tick = True
        elif (now - last_tick_wall) >= engine.micro_tick_sec and RUNTIME.get("latest_tick_result") is not None:
            should_tick = True

        if should_tick:
            tick_out = engine.run_streaming_tick()
            last_tick_wall = now
            if tick_out is not None:
                RUNTIME["latest_tick_result"] = tick_out
                socketio.emit(
                    "graph_delta",
                    {
                        "timestamp": tick_out["timestamp"],
                        "delta": tick_out["graph_delta"],
                        "telemetry": tick_out["telemetry"],
                    },
                )
                if tick_out["attack_alert"]["has_attack"]:
                    socketio.emit(
                        "attack_alert",
                        {
                            "timestamp": tick_out["timestamp"],
                            "alert": tick_out["attack_alert"],
                            "telemetry": tick_out["telemetry"],
                        },
                    )

        time.sleep(0.20)


@app.route("/")
def index() -> str:
    """Render Dual-Window 3D Provenance SOC Dashboard."""
    ensure_runtime_started(start_listeners=True)
    return render_template("index.html")


@app.route("/images/<path:filename>")
def serve_hud_icon(filename: str):
    """Serve 256x256 entity HUD PNG icons from web/images/."""
    return send_from_directory(str(PROJECT_ROOT / "web" / "images"), filename)


@app.route("/api/status", methods=["GET"])
def api_status():
    """Return live collector, VM connection, and streaming engine telemetry."""
    rt = ensure_runtime_started(start_listeners=True)
    collector: AuditStreamCollector = rt["collector"]
    engine: KoTonInferenceEngine = rt["engine"]
    latest: Optional[Dict[str, object]] = rt.get("latest_tick_result")
    return jsonify(
        {
            "ok": True,
            "collector": {
                "tcp_port": collector.tcp_port,
                "active_vm_clients": collector.active_vm_clients,
                "total_lines_received": collector.total_lines_received,
                "total_edges_emitted": collector.total_edges_emitted,
            },
            "engine": {
                "alpha": round(engine.current_alpha, 4),
                "threshold": round(engine.anomaly_threshold, 4),
                "recon_scale": round(engine.recon_scale, 4),
                "telemetry": latest["telemetry"] if latest else {},
            },
        }
    )


@app.route("/api/snapshot", methods=["GET"])
def api_snapshot():
    """Return full current window snapshot for initial page load or manual refresh."""
    rt = ensure_runtime_started(start_listeners=True)
    engine: KoTonInferenceEngine = rt["engine"]
    latest: Optional[Dict[str, object]] = rt.get("latest_tick_result")
    if latest is None:
        latest = engine.run_streaming_tick()
        rt["latest_tick_result"] = latest
    if latest is None:
        return jsonify({"timestamp": time.time(), "nodes": [], "edges": [], "telemetry": {}, "attack_alert": {}})
    return jsonify(
        {
            "timestamp": latest["timestamp"],
            "nodes": latest["snapshot"]["nodes"],
            "edges": latest["snapshot"]["edges"],
            "telemetry": latest["telemetry"],
            "attack_alert": latest["attack_alert"],
        }
    )


@app.route("/api/threshold", methods=["POST"])
def api_update_threshold():
    """REST endpoint for Zero-Reinference (<1ms) Conformal threshold updates."""
    rt = ensure_runtime_started(start_listeners=True)
    engine: KoTonInferenceEngine = rt["engine"]
    body = request.get_json(silent=True) or {}
    alpha = body.get("alpha")
    threshold = body.get("threshold")
    res = engine.update_threshold_zero_reinference(
        alpha=float(alpha) if alpha is not None else None,
        direct_threshold=float(threshold) if threshold is not None else None,
    )
    socketio.emit("threshold_updated", res)
    return jsonify(res)


@app.route("/api/simulate/live_attack", methods=["POST"])
def api_simulate_live_attack():
    """Inject a realistic multi-stage kernel auditd attack stream into AuditStreamCollector for instant verification."""
    rt = ensure_runtime_started(start_listeners=True)
    collector: AuditStreamCollector = rt["collector"]
    now = time.time()

    sim_lines = [
        f'type=SYSCALL msg=audit({now:.3f}:9101): arch=c000003e syscall=42 success=yes pid=4100 ppid=1 uid=1000 exe="/usr/bin/nmap" comm="nmap"',
        f'type=SOCKADDR msg=audit({now:.3f}:9101): saddr=020000500A23216F0000000000000000',
        f'type=EOE msg=audit({now:.3f}:9101):',
        f'type=SYSCALL msg=audit({now + 0.20:.3f}:9102): arch=c000003e syscall=59 success=yes pid=4120 ppid=4100 uid=33 exe="/bin/sh" comm="sh"',
        f'type=EXECVE msg=audit({now + 0.20:.3f}:9102): argc=3 a0="/bin/sh" a1="-c" a2="/var/www/html/cmd_webshell.sh"',
        f'type=CWD msg=audit({now + 0.20:.3f}:9102): cwd="/var/www/html"',
        f'type=PATH msg=audit({now + 0.20:.3f}:9102): item=0 name="/var/www/html/cmd_webshell.sh" nametype=NORMAL',
        f'type=EOE msg=audit({now + 0.20:.3f}:9102):',
        f'type=SYSCALL msg=audit({now + 0.45:.3f}:9103): arch=c000003e syscall=59 success=yes pid=4155 ppid=4120 uid=0 exe="/usr/bin/sudo" comm="sudo"',
        f'type=EXECVE msg=audit({now + 0.45:.3f}:9103): argc=3 a0="sudo" a1="cat" a2="/etc/shadow"',
        f'type=PATH msg=audit({now + 0.45:.3f}:9103): item=0 name="/etc/shadow" nametype=NORMAL',
        f'type=EOE msg=audit({now + 0.45:.3f}:9103):',
        f'type=SYSCALL msg=audit({now + 0.75:.3f}:9104): arch=c000003e syscall=42 success=yes pid=4180 ppid=4155 uid=0 exe="/usr/bin/curl" comm="curl"',
        f'type=SOCKADDR msg=audit({now + 0.75:.3f}:9104): saddr=0200115CC633642A0000000000000000',
        f'type=PATH msg=audit({now + 0.75:.3f}:9104): item=0 name="/tmp/exfil_customers_secret.tar.gz" nametype=NORMAL',
        f'type=EOE msg=audit({now + 0.75:.3f}:9104):',
    ]
    for ln in sim_lines:
        collector.ingest_raw_line(ln, host_id="ubuntu-vm")
    collector.flush_expired_carryover(force_all=True)
    return jsonify({"ok": True, "injected_lines": len(sim_lines)})


@socketio.on("connect")
def on_socket_connect():
    """Send initial window state (empty on fresh startup) once to newly connected browser tab."""
    rt = ensure_runtime_started(start_listeners=True)
    engine: KoTonInferenceEngine = rt["engine"]
    latest: Optional[Dict[str, object]] = rt.get("latest_tick_result")
    if latest is not None:
        emit(
            "graph_init",
            {
                "timestamp": latest["timestamp"],
                "nodes": latest["snapshot"]["nodes"],
                "edges": latest["snapshot"]["edges"],
                "telemetry": latest["telemetry"],
                "attack_alert": latest["attack_alert"],
            },
        )
    else:
        emit(
            "graph_init",
            {
                "timestamp": time.time(),
                "nodes": [],
                "edges": [],
                "telemetry": {
                    "latency_ms": 0.0,
                    "num_nodes": 0,
                    "num_edges": 0,
                    "alpha": round(engine.current_alpha, 4),
                    "threshold": round(engine.anomaly_threshold, 4),
                    "estimated_fpr": 0.0,
                    "online_cpr_pct": 0.0,
                },
                "attack_alert": {
                    "has_attack": False,
                    "seed_count": 0,
                    "dominant_attack_class": "benign",
                    "subgraph_node_ids": [],
                    "subgraph_edges": [],
                },
            },
        )


@socketio.on("update_threshold")
def on_socket_update_threshold(payload: Dict[str, object]):
    """Handle slider update_threshold event with O(N) Zero-Reinference (<1ms)."""
    rt = ensure_runtime_started(start_listeners=True)
    engine: KoTonInferenceEngine = rt["engine"]
    alpha = payload.get("alpha") if isinstance(payload, dict) else None
    threshold = payload.get("threshold") if isinstance(payload, dict) else None
    res = engine.update_threshold_zero_reinference(
        alpha=float(alpha) if alpha is not None else None,
        direct_threshold=float(threshold) if threshold is not None else None,
    )
    emit("threshold_updated", res, broadcast=True)


if __name__ == "__main__":
    ensure_runtime_started(start_listeners=True)
    port = int(os.environ.get("KOTON_WEB_PORT", "5000"))
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)

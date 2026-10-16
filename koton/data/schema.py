import re
import math
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import torch
from torch_geometric.data import HeteroData

from koton.config import (
    NODE_TYPES,
    CANONICAL_EDGE_TYPES,
    PROCESS_FEAT_DIM,
    FILE_FEAT_DIM,
    SOCKET_FEAT_DIM,
)

# Common sensitive paths and ports for security feature extraction
SENSITIVE_PATHS = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers", "/root",
    "/bin/bash", "/bin/sh", "/usr/bin/python", "/tmp", "/dev/shm",
    "authorized_keys", "id_rsa", ".bash_history"
]

SUSPICIOUS_NAMES = [
    "nc", "ncat", "netcat", "wget", "curl", "bash", "sh", "python",
    "perl", "chmod", "chown", "sudo", "su", "useradd", "iptables",
    "mimikatz", "meterpreter", "exploit", "c2", "revshell"
]

STANDARD_PORTS = {80: 0, 443: 1, 22: 2, 53: 3, 21: 4, 25: 5, 8080: 6, 4444: 7}


@dataclass
class ProcessNode:
    key: str                    # Unique identifier e.g. "proc:1234:bash:1700000000"
    pid: int
    name: str
    cmdline: str
    path: str
    uid: int = 1000
    timestamp: float = 0.0
    label: int = 0              # 0: Benign, 1: Malicious
    is_anomaly: bool = False
    anomaly_score: float = 0.0

    def extract_features(self) -> List[float]:
        """Extract a fixed-size numeric feature vector (length PROCESS_FEAT_DIM=16)."""
        feats = [0.0] * PROCESS_FEAT_DIM
        # 0: Is root (uid == 0)
        feats[0] = 1.0 if self.uid == 0 else 0.0
        # 1: Log-scale PID
        feats[1] = min(math.log(max(self.pid, 1) + 1.0) / 10.0, 1.0)
        # 2: Is in suspicious names list
        name_lower = self.name.lower()
        feats[2] = 1.0 if any(s in name_lower for s in SUSPICIOUS_NAMES) else 0.0
        # 3: Is path located in /tmp or /dev/shm
        path_lower = self.path.lower()
        feats[3] = 1.0 if ("/tmp" in path_lower or "/dev/shm" in path_lower) else 0.0
        # 4: Is hidden file/path (starts with .)
        feats[4] = 1.0 if "/." in path_lower or self.name.startswith(".") else 0.0
        # 5: Length of command line (normalized)
        feats[5] = min(len(self.cmdline) / 256.0, 1.0)
        # 6: Argument count
        feats[6] = min(len(self.cmdline.split()) / 20.0, 1.0)
        # 7-15: Deterministic hash projection of cmdline & path for structural diversity
        h = int(hashlib.md5(f"{self.path}:{self.name}".encode()).hexdigest(), 16)
        for i in range(7, 16):
            feats[i] = ((h >> (i * 3)) & 0xFF) / 255.0
        return feats


@dataclass
class FileNode:
    key: str                    # Unique identifier e.g. "file:/etc/passwd"
    path: str
    extension: str = ""
    timestamp: float = 0.0
    label: int = 0
    is_anomaly: bool = False
    anomaly_score: float = 0.0

    def extract_features(self) -> List[float]:
        """Extract a fixed-size numeric feature vector (length FILE_FEAT_DIM=12)."""
        feats = [0.0] * FILE_FEAT_DIM
        path_lower = self.path.lower()
        # 0: Sensitive system file
        feats[0] = 1.0 if any(sp in path_lower for sp in SENSITIVE_PATHS) else 0.0
        # 1: Temporary location (/tmp, /var/tmp, /dev/shm)
        feats[1] = 1.0 if ("/tmp" in path_lower or "/dev/shm" in path_lower) else 0.0
        # 2: Hidden file
        feats[2] = 1.0 if "/." in path_lower else 0.0
        # 3: Executable extension (.sh, .elf, .bin, .py, .so)
        feats[3] = 1.0 if any(path_lower.endswith(ext) for ext in [".sh", ".elf", ".bin", ".py", ".so", ".pl"]) else 0.0
        # 4: Path depth
        feats[4] = min(path_lower.count("/") / 10.0, 1.0)
        # 5-11: Hash projection of path
        h = int(hashlib.md5(self.path.encode()).hexdigest(), 16)
        for i in range(5, 12):
            feats[i] = ((h >> (i * 3)) & 0xFF) / 255.0
        return feats


@dataclass
class SocketNode:
    key: str                    # Unique identifier e.g. "sock:192.168.1.5:4444"
    ip: str
    port: int
    protocol: str = "tcp"
    timestamp: float = 0.0
    label: int = 0
    is_anomaly: bool = False
    anomaly_score: float = 0.0

    def extract_features(self) -> List[float]:
        """Extract a fixed-size numeric feature vector (length SOCKET_FEAT_DIM=8)."""
        feats = [0.0] * SOCKET_FEAT_DIM
        # 0: Is private / localhost IP
        is_local = self.ip.startswith("127.") or self.ip == "localhost" or self.ip.startswith("192.168.") or self.ip.startswith("10.")
        feats[0] = 0.0 if is_local else 1.0  # 1.0 = External IP
        # 1: Known standard port
        feats[1] = 1.0 if self.port in STANDARD_PORTS else 0.0
        # 2: High ephemeral port (>1024)
        feats[2] = 1.0 if self.port > 1024 else 0.0
        # 3: Well-known reverse shell / C2 ports (4444, 1337, 8888, 9001)
        feats[3] = 1.0 if self.port in [4444, 1337, 8888, 9001, 6667] else 0.0
        # 4: Normalized port number
        feats[4] = min(self.port / 65535.0, 1.0)
        # 5-7: IP address octet hash
        h = int(hashlib.md5(f"{self.ip}:{self.port}".encode()).hexdigest(), 16)
        for i in range(5, 8):
            feats[i] = ((h >> (i * 4)) & 0xFF) / 255.0
        return feats


@dataclass
class ProvenanceEvent:
    timestamp: float
    src_type: str               # "process", "file", "socket"
    src_key: str
    relation: str               # "fork", "exec", "read", "write", "connect"
    dst_type: str               # "process", "file", "socket"
    dst_key: str
    is_attack: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_hetero_data(
    processes: Dict[str, ProcessNode],
    files: Dict[str, FileNode],
    sockets: Dict[str, SocketNode],
    events: List[ProvenanceEvent],
    include_reverse_edges: bool = True
) -> HeteroData:
    """
    Construct a PyTorch Geometric HeteroData graph from nodes and event edges.
    Returns:
        HeteroData with:
          - data['process'].x, data['process'].y, data['process'].keys
          - data['file'].x, data['file'].y, data['file'].keys
          - data['socket'].x, data['socket'].y, data['socket'].keys
          - data[src_type, relation, dst_type].edge_index
    """
    data = HeteroData()

    # Map node keys to integer indices
    proc_keys = sorted(processes.keys())
    proc_map = {k: i for i, k in enumerate(proc_keys)}
    file_keys = sorted(files.keys())
    file_map = {k: i for i, k in enumerate(file_keys)}
    sock_keys = sorted(sockets.keys())
    sock_map = {k: i for i, k in enumerate(sock_keys)}

    # Node Features and Labels
    if proc_keys:
        p_feats = [processes[k].extract_features() for k in proc_keys]
        p_labels = [processes[k].label for k in proc_keys]
        data['process'].x = torch.tensor(p_feats, dtype=torch.float)
        data['process'].y = torch.tensor(p_labels, dtype=torch.long)
    else:
        data['process'].x = torch.empty((0, PROCESS_FEAT_DIM), dtype=torch.float)
        data['process'].y = torch.empty((0,), dtype=torch.long)
    data['process'].node_keys = proc_keys

    if file_keys:
        f_feats = [files[k].extract_features() for k in file_keys]
        f_labels = [files[k].label for k in file_keys]
        data['file'].x = torch.tensor(f_feats, dtype=torch.float)
        data['file'].y = torch.tensor(f_labels, dtype=torch.long)
    else:
        data['file'].x = torch.empty((0, FILE_FEAT_DIM), dtype=torch.float)
        data['file'].y = torch.empty((0,), dtype=torch.long)
    data['file'].node_keys = file_keys

    if sock_keys:
        s_feats = [sockets[k].extract_features() for k in sock_keys]
        s_labels = [sockets[k].label for k in sock_keys]
        data['socket'].x = torch.tensor(s_feats, dtype=torch.float)
        data['socket'].y = torch.tensor(s_labels, dtype=torch.long)
    else:
        data['socket'].x = torch.empty((0, SOCKET_FEAT_DIM), dtype=torch.float)
        data['socket'].y = torch.empty((0,), dtype=torch.long)
    data['socket'].node_keys = sock_keys

    # Group edges by (src_type, relation, dst_type)
    edge_dict: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}

    for src_t, rel, dst_t in CANONICAL_EDGE_TYPES:
        edge_dict[(src_t, rel, dst_t)] = ([], [])

    for ev in events:
        s_t, rel, d_t = ev.src_type, ev.relation, ev.dst_type
        # Lookup source index
        s_idx = None
        if s_t == "process": s_idx = proc_map.get(ev.src_key)
        elif s_t == "file": s_idx = file_map.get(ev.src_key)
        elif s_t == "socket": s_idx = sock_map.get(ev.src_key)

        # Lookup destination index
        d_idx = None
        if d_t == "process": d_idx = proc_map.get(ev.dst_key)
        elif d_t == "file": d_idx = file_map.get(ev.dst_key)
        elif d_t == "socket": d_idx = sock_map.get(ev.dst_key)

        if s_idx is not None and d_idx is not None:
            edge_type = (s_t, rel, d_t)
            if edge_type in edge_dict:
                edge_dict[edge_type][0].append(s_idx)
                edge_dict[edge_type][1].append(d_idx)

            if include_reverse_edges:
                rev_type = (d_t, f"rev_{rel}", s_t)
                if rev_type in edge_dict:
                    edge_dict[rev_type][0].append(d_idx)
                    edge_dict[rev_type][1].append(s_idx)

    # Attach edges to HeteroData
    for edge_type, (src_list, dst_list) in edge_dict.items():
        if src_list:
            data[edge_type].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data

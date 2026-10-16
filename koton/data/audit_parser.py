import re
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple, Any

from koton.data.schema import ProcessNode, FileNode, SocketNode, ProvenanceEvent


def parse_sockaddr(saddr_hex: str) -> Tuple[str, int]:
    """Decodes hex-encoded sockaddr_in structure from auditd log."""
    try:
        raw = bytes.fromhex(saddr_hex)
        if len(raw) >= 8:
            family, port = struct.unpack("!HH", raw[:4])
            if family == socket.AF_INET:
                ip = socket.inet_ntoa(raw[4:8])
                return ip, port
    except Exception:
        pass
    return "0.0.0.0", 0


class LinuxAuditParser:
    """
    Parses raw Linux auditd (/var/log/audit/audit.log) log entries into
    structured Provenance Events and Entity Nodes.
    """

    def __init__(self):
        self.pending_records: Dict[str, Dict[str, Any]] = {}

    def parse_line(self, line: str) -> Optional[ProvenanceEvent]:
        """
        Parses a single line of Linux audit log.
        Example:
        type=SYSCALL msg=audit(1700000000.123:456): arch=c000003e syscall=59 success=yes exit=0 ppid=1024 pid=2048 auid=1000 uid=1000 comm="sh" exe="/bin/sh"
        """
        line = line.strip()
        if not line:
            return None

        # Extract type and msg ID
        type_match = re.search(r"type=([A-Z_]+)", line)
        msg_match = re.search(r"msg=audit\((\d+\.\d+):(\d+)\)", line)

        if not type_match or not msg_match:
            return None

        rec_type = type_match.group(1)
        timestamp = float(msg_match.group(1))
        msg_id = msg_match.group(2)

        # Tokenize key=value pairs (handle both quoted strings and unquoted values)
        kv = {}
        for match in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', line):
            k = match.group(1)
            v = match.group(2) if match.group(2) is not None else match.group(3)
            kv[k] = v

        if rec_type == "SYSCALL":
            syscall_num = kv.get("syscall")
            ppid = int(kv.get("ppid", 1))
            pid = int(kv.get("pid", 0))
            uid = int(kv.get("uid", 1000))
            comm = kv.get("comm", "unknown")
            exe = kv.get("exe", comm)

            # Syscall 42 = connect (x86_64)
            if syscall_num == "42":
                saddr = kv.get("saddr", "")
                ip, port = parse_sockaddr(saddr)
                return ProvenanceEvent(
                    timestamp=timestamp,
                    src_type="process",
                    src_key=f"proc:{pid}:{comm}",
                    relation="connect",
                    dst_type="socket",
                    dst_key=f"sock:{ip}:{port}",
                    is_attack=False,
                    metadata={"pid": pid, "ip": ip, "port": port}
                )

            # Syscall 59 = execve (x86_64)
            elif syscall_num == "59" or (syscall_num is None and comm in ["sh", "bash", "curl", "wget", "python3"]):
                return ProvenanceEvent(
                    timestamp=timestamp,
                    src_type="process",
                    src_key=f"proc:{ppid}:parent",
                    relation="exec",
                    dst_type="process",
                    dst_key=f"proc:{pid}:{comm}",
                    is_attack=False,
                    metadata={"pid": pid, "ppid": ppid, "exe": exe, "comm": comm, "uid": uid}
                )

            # Syscall 257 = openat, 2 = open
            elif syscall_num in ["2", "257"]:
                return ProvenanceEvent(
                    timestamp=timestamp,
                    src_type="process",
                    src_key=f"proc:{pid}:{comm}",
                    relation="read",
                    dst_type="file",
                    dst_key=f"file:{comm}",
                    is_attack=False,
                    metadata={"pid": pid, "comm": comm}
                )

        elif rec_type == "CWD":
            pass
        elif rec_type == "PATH":
            name = kv.get("name", "")
            if name:
                return ProvenanceEvent(
                    timestamp=timestamp,
                    src_type="process",
                    src_key=f"proc:{msg_id}:active",
                    relation="read",
                    dst_type="file",
                    dst_key=f"file:{name}",
                    is_attack=False,
                    metadata={"path": name}
                )

        return None

import binascii
from dataclasses import dataclass, field
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import posixpath
import re
import socket as pysocket
import struct
from typing import Dict, Iterator, List, Optional, Set, Tuple

from src.data.taxonomy import LabelTaxonomy
from src.parser.parser import extract_audit_meta, strip_audit_timestamp

RECORD_TYPE_PATTERN = re.compile(r"^type=([A-Z0-9_]+)\s+")
KV_PAIR_PATTERN = re.compile(r'([a-zA-Z0-9_\-]+)=("[^"]*"|\'[^\']*\'|\S+)')
INNER_MSG_PATTERN = re.compile(r"msg='([^']*)'")
HEX_STRING_PATTERN = re.compile(r"^[0-9A-Fa-f]{4,}$")
APACHE_LOG_PATTERN = re.compile(
    r'^(\S+)\s+\S+\s+\S+\s+\[([^\]]+)\]\s+"([A-Z]+)\s+([^\s"]+)[^"]*"\s+(\d{3})\s+(\S+)\s+"([^"]*)"\s+"([^"]*)"'
)
SYSLOG_TS_PATTERN = re.compile(
    r"^([A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(?:[a-zA-Z0-9._\-]+\s+)?([a-zA-Z0-9_\-]+)(?:\[(\d+)\])?:\s*(.*)$"
)
OPENVPN_TS_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.*)$")

EXEC_SYSCALLS_X86_64 = {59, 322}
FORK_SYSCALLS_X86_64 = {56, 57, 58, 435}
READ_SYSCALLS_X86_64 = {0, 17, 19, 257, 2, 4, 6}
WRITE_SYSCALLS_X86_64 = {1, 18, 20, 82, 83, 84, 87, 263}
CONNECT_SYSCALLS_X86_64 = {41, 42, 43, 44, 45, 46, 47, 49, 288}
MULTILINE_RECORD_TYPES = {"SYSCALL", "EXECVE", "CWD", "PATH", "SOCKADDR", "PROCTITLE", "AVC", "EOE"}


@dataclass
class AuditEvent:
    """Merged representation of a single kernel or user-space auditd event."""

    host_id: str
    serial_id: str
    timestamp: float
    record_types: List[str] = field(default_factory=list)
    syscall: Optional[int] = None
    pid: int = 0
    ppid: int = 0
    uid: int = 0
    exe: str = ""
    comm: str = ""
    cmdline: str = ""
    cwd: str = ""
    paths: List[Tuple[str, str]] = field(default_factory=list)
    sockaddr: Optional[Tuple[str, int]] = None
    unit: str = ""
    op: str = ""
    terminal: str = ""
    acct: str = ""
    line_numbers: List[int] = field(default_factory=list)
    raw_labels: Set[str] = field(default_factory=set)
    has_eoe: bool = False
    sanitized_text: str = ""


@dataclass
class ProvenanceEdgeEvent:
    """Canonical provenance edge connecting two typed host-isolated entities."""

    timestamp: float
    host_id: str
    edge_type: str
    src_type: str
    src_key: Tuple
    dst_type: str
    dst_key: Tuple
    proc_exe: str = ""
    proc_cmdline: str = ""
    proc_uid: int = 0
    dst_proc_exe: str = ""
    dst_proc_cmdline: str = ""
    dst_proc_uid: int = 0
    file_path: str = ""
    sock_ip: str = ""
    sock_port: int = 0
    tactic_id: int = 0
    fine_label: str = "benign"
    is_malicious: bool = False
    cpr_count: int = 1
    delta_t: float = 0.0


def decode_audit_hex(value: str) -> str:
    """Decode hex-encoded auditd argument or proctitle string into printable UTF-8 text."""
    cleaned = value.strip().strip('"').strip("'")
    if len(cleaned) >= 4 and len(cleaned) % 2 == 0 and HEX_STRING_PATTERN.match(cleaned):
        try:
            raw_bytes = binascii.unhexlify(cleaned)
            return raw_bytes.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except (binascii.Error, ValueError):
            return cleaned
    return cleaned


def parse_sockaddr_hex(saddr_hex: str) -> Optional[Tuple[str, int]]:
    """Decode Linux sockaddr_in or sockaddr_in6 hex string into (ip, port)."""
    cleaned = saddr_hex.strip().strip('"')
    if len(cleaned) < 16 or len(cleaned) % 2 != 0 or not HEX_STRING_PATTERN.match(cleaned):
        return None
    try:
        raw = binascii.unhexlify(cleaned)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 8:
        return None
    family = struct.unpack("<H", raw[0:2])[0]
    if family == 2 and len(raw) >= 8:
        port = struct.unpack(">H", raw[2:4])[0]
        ip = pysocket.inet_ntop(pysocket.AF_INET, raw[4:8])
        return ip, port
    if family == 10 and len(raw) >= 24:
        port = struct.unpack(">H", raw[2:4])[0]
        ip = pysocket.inet_ntop(pysocket.AF_INET6, raw[8:24])
        return ip, port
    return None


def normalize_posix_path(cwd: str, raw_path: str) -> str:
    """Combine working directory and path into a clean canonical POSIX path."""
    path_str = raw_path.strip().strip('"').strip("'")
    if not path_str or path_str in {"(null)", "?"}:
        return "/unknown"
    if not path_str.startswith("/"):
        base = cwd.strip().strip('"').strip("'") if cwd else "/"
        if not base.startswith("/"):
            base = "/" + base
        path_str = posixpath.join(base, path_str)
    normalized = posixpath.normpath(path_str)
    return normalized if normalized.startswith("/") else "/" + normalized


def stable_sort_by_timestamp(events: List) -> List:
    """Perform causal order-preserving stable sort over events having a timestamp attribute."""
    unsorted_ts = [ev.timestamp for ev in events]
    return [x for _, _, x in sorted(zip(unsorted_ts, range(len(unsorted_ts)), events))]


class StatefulAuditParser:
    """Streaming multiline auditd parser with chunk carry-over buffering and PID lifecycle tracking."""

    def __init__(
        self,
        chunk_size: int = 8000,
        carryover_sec: float = 2.0,
        include_supplementary_logs: bool = True,
        apache_dirb_sample_stride: int = 20,
        dnsmasq_benign_sample_stride: int = 10,
    ) -> None:
        """Configure chunk streaming, carry-over window, and process version state."""
        self.chunk_size = chunk_size
        self.carryover_sec = carryover_sec
        self.include_supplementary_logs = include_supplementary_logs
        self.apache_dirb_sample_stride = max(1, apache_dirb_sample_stride)
        self.dnsmasq_benign_sample_stride = max(1, dnsmasq_benign_sample_stride)
        self._carryover_buffers: Dict[Tuple[str, str], AuditEvent] = {}
        self._proc_versions: Dict[Tuple[str, int], int] = {}
        self._proc_last_exe: Dict[Tuple[str, int], str] = {}
        self._proc_last_cmd: Dict[Tuple[str, int], str] = {}
        self._proc_last_uid: Dict[Tuple[str, int], int] = {}

    def reset_state(self) -> None:
        """Clear stateful carry-over buffers and per-host PID lifecycle tables."""
        self._carryover_buffers.clear()
        self._proc_versions.clear()
        self._proc_last_exe.clear()
        self._proc_last_cmd.clear()
        self._proc_last_uid.clear()

    def _parse_kv(self, text: str) -> Dict[str, str]:
        """Extract key-value tokens from an auditd record body."""
        result: Dict[str, str] = {}
        for k, v in KV_PAIR_PATTERN.findall(text):
            result[k] = v.strip('"').strip("'")
        inner = INNER_MSG_PATTERN.search(text)
        if inner:
            for k, v in KV_PAIR_PATTERN.findall(inner.group(1)):
                result[k] = v.strip('"').strip("'")
        return result

    def _ingest_audit_line(
        self,
        host_id: str,
        line: str,
        line_no: int,
        line_labels: Optional[List[str]],
    ) -> Optional[AuditEvent]:
        """Merge a single auditd line into the stateful buffer by (host_id, serial_id)."""
        ts, serial_id = extract_audit_meta(line)
        if ts is None or serial_id is None:
            return None

        sanitized_line = strip_audit_timestamp(line.strip())
        type_match = RECORD_TYPE_PATTERN.match(sanitized_line)
        rec_type = type_match.group(1) if type_match else "UNKNOWN"
        key = (host_id, serial_id)

        ev = self._carryover_buffers.get(key)
        if ev is None:
            ev = AuditEvent(host_id=host_id, serial_id=serial_id, timestamp=ts)
            self._carryover_buffers[key] = ev
        else:
            if ts < ev.timestamp:
                ev.timestamp = ts

        ev.record_types.append(rec_type)
        ev.line_numbers.append(line_no)
        if line_labels:
            ev.raw_labels.update(line_labels)
        if not ev.sanitized_text:
            ev.sanitized_text = sanitized_line

        if rec_type == "EOE":
            ev.has_eoe = True
            return self._carryover_buffers.pop(key)

        kv = self._parse_kv(sanitized_line)
        if "pid" in kv and kv["pid"].isdigit():
            ev.pid = int(kv["pid"])
        if "ppid" in kv and kv["ppid"].isdigit():
            ev.ppid = int(kv["ppid"])
        if "uid" in kv and kv["uid"].isdigit():
            ev.uid = int(kv["uid"])
        if "exe" in kv and kv["exe"] not in {"?", "(null)"}:
            ev.exe = kv["exe"]
        if "comm" in kv and kv["comm"] not in {"?", "(null)"}:
            ev.comm = kv["comm"]
        if "cwd" in kv and kv["cwd"] not in {"?", "(null)"}:
            ev.cwd = decode_audit_hex(kv["cwd"])
        if "unit" in kv and kv["unit"] not in {"?", "(null)"}:
            ev.unit = kv["unit"]
        if "op" in kv:
            ev.op = kv["op"]
        if "terminal" in kv and kv["terminal"] not in {"?", "(none)"}:
            ev.terminal = kv["terminal"]
        if "acct" in kv and kv["acct"] != "?":
            ev.acct = kv["acct"]

        if rec_type == "SYSCALL":
            if "syscall" in kv and kv["syscall"].isdigit():
                ev.syscall = int(kv["syscall"])
        elif rec_type == "EXECVE":
            argc = int(kv.get("argc", "0")) if kv.get("argc", "0").isdigit() else 0
            args = [decode_audit_hex(kv[f"a{i}"]) for i in range(min(argc, 32)) if f"a{i}" in kv]
            if args:
                ev.cmdline = " ".join(args)
        elif rec_type == "PROCTITLE":
            if "proctitle" in kv:
                decoded = decode_audit_hex(kv["proctitle"])
                if decoded and not ev.cmdline:
                    ev.cmdline = decoded
        elif rec_type == "CWD":
            if "cwd" in kv:
                ev.cwd = decode_audit_hex(kv["cwd"])
        elif rec_type == "PATH":
            raw_name = kv.get("name", "")
            nametype = kv.get("nametype", "NORMAL")
            if raw_name and raw_name not in {"(null)", "?"}:
                ev.paths.append((decode_audit_hex(raw_name), nametype))
        elif rec_type == "AVC":
            raw_name = kv.get("name", "")
            if raw_name and raw_name not in {"(null)", "?"}:
                ev.paths.append((decode_audit_hex(raw_name), kv.get("operation", "AVC")))
        elif rec_type == "SOCKADDR":
            saddr = kv.get("saddr", "")
            parsed = parse_sockaddr_hex(saddr)
            if parsed:
                ev.sockaddr = parsed
            elif "addr" in kv and kv["addr"] != "?":
                port = int(kv["lport"]) if kv.get("lport", "").isdigit() else 0
                ev.sockaddr = (kv["addr"], port)
        elif rec_type == "USER_CMD":
            if "cmd" in kv:
                ev.cmdline = decode_audit_hex(kv["cmd"])
            if not ev.exe:
                ev.exe = "/usr/bin/sudo"

        if "addr" in kv and kv["addr"] not in {"?", "(null)"} and ev.sockaddr is None:
            addr_val = kv["addr"]
            try:
                ipaddress.ip_address(addr_val)
                default_port = 22 if "sshd" in ev.exe else (143 if "dovecot" in ev.exe else 0)
                ev.sockaddr = (addr_val, default_port)
            except ValueError:
                pass

        if rec_type not in MULTILINE_RECORD_TYPES:
            return self._carryover_buffers.pop(key)
        return None

    def stream_audit_file(
        self,
        audit_path: Path,
        host_id: str,
        line_labels: Dict[int, List[str]],
    ) -> Iterator[List[AuditEvent]]:
        """Stream an audit.log file in bounded chunks with stateful carry-over across chunk boundaries."""
        line_count = 0
        chunk_lines: List[Tuple[int, str]] = []

        with open(audit_path, "r", encoding="utf-8", errors="replace") as fp:
            for raw_line in fp:
                line_count += 1
                chunk_lines.append((line_count, raw_line))
                if len(chunk_lines) >= self.chunk_size:
                    yield self._process_chunk(host_id, chunk_lines, line_labels, is_eof=False)
                    chunk_lines = []

        if chunk_lines or self._carryover_buffers:
            yield self._process_chunk(host_id, chunk_lines, line_labels, is_eof=True)

    def _process_chunk(
        self,
        host_id: str,
        chunk_lines: List[Tuple[int, str]],
        line_labels: Dict[int, List[str]],
        is_eof: bool,
    ) -> List[AuditEvent]:
        """Merge records inside a chunk, retain carry-over candidates, and stable-sort ready events."""
        ready_events: List[AuditEvent] = []
        max_ts = 0.0

        for line_no, raw_line in chunk_lines:
            lbls = line_labels.get(line_no)
            completed = self._ingest_audit_line(host_id, raw_line, line_no, lbls)
            if completed is not None:
                ready_events.append(completed)
                if completed.timestamp > max_ts:
                    max_ts = completed.timestamp

        for buffered_ev in self._carryover_buffers.values():
            if buffered_ev.timestamp > max_ts:
                max_ts = buffered_ev.timestamp

        to_flush_keys: List[Tuple[str, str]] = []
        for buf_key, buf_ev in self._carryover_buffers.items():
            if buf_key[0] != host_id:
                continue
            if is_eof or (max_ts - buf_ev.timestamp) > self.carryover_sec:
                to_flush_keys.append(buf_key)

        for buf_key in to_flush_keys:
            ready_events.append(self._carryover_buffers.pop(buf_key))

        return stable_sort_by_timestamp(ready_events)

    def _get_process_key(
        self,
        host_id: str,
        pid: int,
        exe: str,
        cmdline: str,
        uid: int,
        increment_exec: bool = False,
    ) -> Tuple[str, int, int]:
        """Resolve host-isolated process key (host_id, pid, exec_version) with PID reuse protection."""
        pk = (host_id, pid)
        prev_ver = self._proc_versions.get(pk, 0)
        prev_exe = self._proc_last_exe.get(pk, "")
        if increment_exec or (prev_exe and exe and prev_exe != exe):
            new_ver = prev_ver + 1
        elif prev_ver == 0:
            new_ver = 1
        else:
            new_ver = prev_ver

        self._proc_versions[pk] = new_ver
        if exe:
            self._proc_last_exe[pk] = exe
        if cmdline:
            self._proc_last_cmd[pk] = cmdline
        self._proc_last_uid[pk] = uid
        return (host_id, pid, new_ver)

    def audit_event_to_edges(
        self,
        ev: AuditEvent,
        taxonomy: LabelTaxonomy,
    ) -> List[ProvenanceEdgeEvent]:
        """Convert a merged AuditEvent into one or more typed ProvenanceEdgeEvents (5 canonical edge types)."""
        edges: List[ProvenanceEdgeEvent] = []
        exe = normalize_posix_path("/", ev.exe) if ev.exe else (f"/usr/bin/{ev.comm}" if ev.comm else "/sbin/init")
        cmdline = ev.cmdline if ev.cmdline else (f"{exe} {ev.op} {ev.unit}".strip())
        tactic_id, fine_label, is_mal = taxonomy.map_event_label(
            ev.timestamp,
            ev.raw_labels,
            source_tag=f"{ev.host_id}:audit.log",
        )

        rec_set = set(ev.record_types)
        is_exec = (ev.syscall in EXEC_SYSCALLS_X86_64) or ("EXECVE" in rec_set) or ("USER_CMD" in rec_set)
        is_fork = (ev.syscall in FORK_SYSCALLS_X86_64) or ("LOGIN" in rec_set)

        prev_ver = self._proc_versions.get((ev.host_id, ev.pid), 0)
        src_key = self._get_process_key(ev.host_id, ev.pid, exe, cmdline, ev.uid, increment_exec=is_exec)

        def make_proc_file_edge(edge_type: str, target_path: str) -> ProvenanceEdgeEvent:
            norm_path = normalize_posix_path(ev.cwd, target_path)
            return ProvenanceEdgeEvent(
                timestamp=ev.timestamp,
                host_id=ev.host_id,
                edge_type=edge_type,
                src_type="process",
                src_key=src_key,
                dst_type="file",
                dst_key=(ev.host_id, norm_path),
                proc_exe=exe,
                proc_cmdline=cmdline,
                proc_uid=ev.uid,
                file_path=norm_path,
                tactic_id=tactic_id,
                fine_label=fine_label,
                is_malicious=is_mal,
            )

        if is_fork or (ev.ppid > 0 and prev_ver == 0):
            parent_pid = ev.ppid if ev.ppid > 0 else 1
            parent_exe = self._proc_last_exe.get((ev.host_id, parent_pid), "/lib/systemd/systemd")
            parent_cmd = self._proc_last_cmd.get((ev.host_id, parent_pid), parent_exe)
            parent_uid = self._proc_last_uid.get((ev.host_id, parent_pid), 0)
            parent_key = self._get_process_key(ev.host_id, parent_pid, parent_exe, parent_cmd, parent_uid, False)
            if parent_key != src_key:
                edges.append(
                    ProvenanceEdgeEvent(
                        timestamp=ev.timestamp,
                        host_id=ev.host_id,
                        edge_type="fork",
                        src_type="process",
                        src_key=parent_key,
                        dst_type="process",
                        dst_key=src_key,
                        proc_exe=parent_exe,
                        proc_cmdline=parent_cmd,
                        proc_uid=parent_uid,
                        dst_proc_exe=exe,
                        dst_proc_cmdline=cmdline,
                        dst_proc_uid=ev.uid,
                        tactic_id=tactic_id,
                        fine_label=fine_label,
                        is_malicious=is_mal,
                    )
                )

        if is_exec:
            cmd_tokens = cmdline.split()
            target_bin = cmd_tokens[0] if cmd_tokens and "/" in cmd_tokens[0] else (
                f"/usr/bin/{cmd_tokens[0]}" if cmd_tokens else exe
            )
            edges.append(make_proc_file_edge("execute", target_bin))
            for tok in cmd_tokens[1:]:
                if tok.startswith("/") or tok.startswith("./") or tok.startswith("../"):
                    edges.append(make_proc_file_edge("read", tok))
            if ev.cwd:
                edges.append(make_proc_file_edge("read", ev.cwd))

        if ev.sockaddr is not None:
            ip_str, port_num = ev.sockaddr
            edges.append(
                ProvenanceEdgeEvent(
                    timestamp=ev.timestamp,
                    host_id=ev.host_id,
                    edge_type="connect",
                    src_type="process",
                    src_key=src_key,
                    dst_type="socket",
                    dst_key=(ev.host_id, ip_str, int(port_num)),
                    proc_exe=exe,
                    proc_cmdline=cmdline,
                    proc_uid=ev.uid,
                    sock_ip=ip_str,
                    sock_port=int(port_num),
                    tactic_id=tactic_id,
                    fine_label=fine_label,
                    is_malicious=is_mal,
                )
            )

        if ev.paths:
            for p_str, nametype in ev.paths:
                if ev.syscall in WRITE_SYSCALLS_X86_64 or nametype in {"CREATE", "DELETE"}:
                    rel = "write"
                elif ev.syscall in EXEC_SYSCALLS_X86_64:
                    rel = "execute"
                else:
                    rel = "read"
                edges.append(make_proc_file_edge(rel, p_str))

        if "SERVICE_START" in rec_set or "SERVICE_STOP" in rec_set:
            unit_name = ev.unit or "unknown.service"
            if not unit_name.endswith(".service"):
                unit_name = f"{unit_name}.service"
            edges.append(make_proc_file_edge("execute" if "SERVICE_START" in rec_set else "write", f"/etc/systemd/system/{unit_name}"))

        if not edges:
            if "USER_AUTH" in rec_set or "USER_ACCT" in rec_set:
                edges.append(make_proc_file_edge("read", "/etc/passwd"))
            elif "CRED_ACQ" in rec_set or "CRED_REFR" in rec_set or "CRED_DISP" in rec_set:
                edges.append(make_proc_file_edge("read", "/etc/security/limits.conf"))
            elif "USER_START" in rec_set or "USER_END" in rec_set or "USER_LOGIN" in rec_set:
                tty_path = ev.terminal if ev.terminal.startswith("/") else f"/var/run/utmp"
                edges.append(make_proc_file_edge("write", tty_path))
            else:
                edges.append(make_proc_file_edge("execute", exe))

        return edges

    def _stream_supplementary_host_logs(
        self,
        scenario_dir: Path,
        taxonomy: LabelTaxonomy,
    ) -> List[ProvenanceEdgeEvent]:
        """Parse symmetric AIT-LDS v2.0 host daemon logs to complement -D auditd rules with 6-class ground truth."""
        edges: List[ProvenanceEdgeEvent] = []
        gather_root = scenario_dir / "gather"

        apache_logs = sorted(gather_root.glob("*/logs/apache2/*access.log*"))
        for alog in apache_logs:
            host_id = alog.relative_to(gather_root).parts[0]
            line_labels = taxonomy.load_symmetric_line_labels(alog, scenario_dir)
            apache_key = self._get_process_key(
                host_id, 900001, "/usr/sbin/apache2", "/usr/sbin/apache2 -k start", 33, False
            )
            log_file_path = normalize_posix_path("/var/log/apache2", alog.name)
            with open(alog, "r", encoding="utf-8", errors="replace") as fp:
                for line_no, line in enumerate(fp, 1):
                    raw_lbls = line_labels.get(line_no, [])
                    if "dirb" in raw_lbls and (line_no % self.apache_dirb_sample_stride != 0):
                        continue
                    if not raw_lbls and (line_no % 10 != 0):
                        continue
                    m = APACHE_LOG_PATTERN.match(line)
                    if not m:
                        continue
                    client_ip, ts_str, method, uri, status_code, _, _, ua = m.groups()
                    try:
                        ts = datetime.strptime(ts_str, "%d/%b/%Y:%H:%M:%S %z").timestamp()
                    except ValueError:
                        continue
                    tactic_id, fine_label, is_mal = taxonomy.map_event_label(
                        ts, raw_lbls, source_tag=f"{host_id}:{alog.name}"
                    )
                    clean_uri = uri.split("?", 1)[0]
                    web_path = normalize_posix_path("/var/www/html", clean_uri.lstrip("/"))
                    cmd_str = f"/usr/sbin/apache2 {method} {clean_uri} {status_code} {ua[:40]}"
                    edges.append(
                        ProvenanceEdgeEvent(
                            timestamp=ts,
                            host_id=host_id,
                            edge_type="connect",
                            src_type="process",
                            src_key=apache_key,
                            dst_type="socket",
                            dst_key=(host_id, client_ip, 80),
                            proc_exe="/usr/sbin/apache2",
                            proc_cmdline=cmd_str,
                            proc_uid=33,
                            sock_ip=client_ip,
                            sock_port=80,
                            tactic_id=tactic_id,
                            fine_label=fine_label,
                            is_malicious=is_mal,
                        )
                    )
                    file_edge_type = "write" if method in {"POST", "PUT"} or "upload" in fine_label else "read"
                    edges.append(
                        ProvenanceEdgeEvent(
                            timestamp=ts,
                            host_id=host_id,
                            edge_type=file_edge_type,
                            src_type="process",
                            src_key=apache_key,
                            dst_type="file",
                            dst_key=(host_id, web_path),
                            proc_exe="/usr/sbin/apache2",
                            proc_cmdline=cmd_str,
                            proc_uid=33,
                            file_path=web_path,
                            tactic_id=tactic_id,
                            fine_label=fine_label,
                            is_malicious=is_mal,
                        )
                    )
                    edges.append(
                        ProvenanceEdgeEvent(
                            timestamp=ts,
                            host_id=host_id,
                            edge_type="write",
                            src_type="process",
                            src_key=apache_key,
                            dst_type="file",
                            dst_key=(host_id, log_file_path),
                            proc_exe="/usr/sbin/apache2",
                            proc_cmdline=cmd_str,
                            proc_uid=33,
                            file_path=log_file_path,
                            tactic_id=tactic_id,
                            fine_label=fine_label,
                            is_malicious=is_mal,
                        )
                    )

        dns_logs = sorted(gather_root.glob("*/logs/dnsmasq.log"))
        for dlog in dns_logs:
            host_id = dlog.relative_to(gather_root).parts[0]
            line_labels = taxonomy.load_symmetric_line_labels(dlog, scenario_dir)
            dns_key = self._get_process_key(
                host_id, 900002, "/usr/sbin/dnsmasq", "/usr/sbin/dnsmasq --no-resolv", 0, False
            )
            with open(dlog, "r", encoding="utf-8", errors="replace") as fp:
                for line_no, line in enumerate(fp, 1):
                    raw_lbls = line_labels.get(line_no, [])
                    is_out_of_phase_dnsteal = "dnsteal" in raw_lbls
                    if is_out_of_phase_dnsteal and (line_no % 50 != 1):
                        continue
                    if not raw_lbls and (line_no % self.dnsmasq_benign_sample_stride != 0):
                        continue
                    sm = SYSLOG_TS_PATTERN.match(line)
                    if not sm:
                        continue
                    ts_part, _, _, msg_body = sm.groups()
                    try:
                        ts = datetime.strptime(f"2022 {ts_part}", "%Y %b %d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                    except ValueError:
                        continue
                    tactic_id, fine_label, is_mal = taxonomy.map_event_label(
                        ts, raw_lbls, source_tag=f"{host_id}:dnsmasq.log"
                    )
                    ip_match = re.search(r"(?:from|to|is)\s+(\d+\.\d+\.\d+\.\d+)", msg_body)
                    peer_ip = ip_match.group(1) if ip_match else "127.0.0.53"
                    edges.append(
                        ProvenanceEdgeEvent(
                            timestamp=ts,
                            host_id=host_id,
                            edge_type="connect",
                            src_type="process",
                            src_key=dns_key,
                            dst_type="socket",
                            dst_key=(host_id, peer_ip, 53),
                            proc_exe="/usr/sbin/dnsmasq",
                            proc_cmdline=f"/usr/sbin/dnsmasq {msg_body[:60]}",
                            proc_uid=0,
                            sock_ip=peer_ip,
                            sock_port=53,
                            tactic_id=tactic_id,
                            fine_label=fine_label,
                            is_malicious=is_mal,
                        )
                    )

        cpu_logs = sorted(gather_root.glob("*/logs/logstash/*/*system.cpu.log"))
        for clog in cpu_logs:
            host_id = clog.relative_to(gather_root).parts[0]
            line_labels = taxonomy.load_symmetric_line_labels(clog, scenario_dir)
            proc_key = self._get_process_key(
                host_id, 900003, "/usr/share/logstash/bin/logstash", "logstash system.cpu monitor", 0, False
            )
            with open(clog, "r", encoding="utf-8", errors="replace") as fp:
                for line_no, line in enumerate(fp, 1):
                    raw_lbls = line_labels.get(line_no, [])
                    if not raw_lbls and (line_no % 10 != 0):
                        continue
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                        ts_iso = payload["@timestamp"].replace("Z", "+00:00")
                        ts = datetime.fromisoformat(ts_iso).timestamp()
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue
                    tactic_id, fine_label, is_mal = taxonomy.map_event_label(
                        ts, raw_lbls, source_tag=f"{host_id}:{clog.name}"
                    )
                    cpu_pct = float(payload.get("host", {}).get("cpu", {}).get("pct", 0.0))
                    cmd_desc = f"metricbeat cpu_pct={cpu_pct:.4f}"
                    edges.append(
                        ProvenanceEdgeEvent(
                            timestamp=ts,
                            host_id=host_id,
                            edge_type="read",
                            src_type="process",
                            src_key=proc_key,
                            dst_type="file",
                            dst_key=(host_id, "/proc/stat"),
                            proc_exe="/usr/share/logstash/bin/logstash",
                            proc_cmdline=cmd_desc,
                            proc_uid=0,
                            file_path="/proc/stat",
                            tactic_id=tactic_id,
                            fine_label=fine_label,
                            is_malicious=is_mal,
                        )
                    )

        vpn_logs = sorted(gather_root.glob("*/logs/openvpn.log"))
        for vlog in vpn_logs:
            host_id = vlog.relative_to(gather_root).parts[0]
            line_labels = taxonomy.load_symmetric_line_labels(vlog, scenario_dir)
            if not line_labels:
                continue
            vpn_key = self._get_process_key(
                host_id, 900004, "/usr/sbin/openvpn", "/usr/sbin/openvpn --config server.conf", 0, False
            )
            with open(vlog, "r", encoding="utf-8", errors="replace") as fp:
                for line_no, line in enumerate(fp, 1):
                    raw_lbls = line_labels.get(line_no, [])
                    if not raw_lbls and (line_no % 10 != 0):
                        continue
                    vm = OPENVPN_TS_PATTERN.match(line)
                    if not vm:
                        continue
                    ts_part, body = vm.groups()
                    try:
                        ts = datetime.strptime(ts_part, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                    except ValueError:
                        continue
                    tactic_id, fine_label, is_mal = taxonomy.map_event_label(
                        ts, raw_lbls, source_tag=f"{host_id}:openvpn.log"
                    )
                    ip_m = re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)", body)
                    ip_str = ip_m.group(1) if ip_m else "192.168.255.1"
                    port_num = int(ip_m.group(2)) if ip_m else 1194
                    edges.append(
                        ProvenanceEdgeEvent(
                            timestamp=ts,
                            host_id=host_id,
                            edge_type="connect",
                            src_type="process",
                            src_key=vpn_key,
                            dst_type="socket",
                            dst_key=(host_id, ip_str, port_num),
                            proc_exe="/usr/sbin/openvpn",
                            proc_cmdline=f"/usr/sbin/openvpn {body[:50]}",
                            proc_uid=0,
                            sock_ip=ip_str,
                            sock_port=port_num,
                            tactic_id=tactic_id,
                            fine_label=fine_label,
                            is_malicious=is_mal,
                        )
                    )

        return edges

    def parse_scenario(
        self,
        scenario_dir: str | Path,
        taxonomy: LabelTaxonomy,
    ) -> List[ProvenanceEdgeEvent]:
        """Parse all host auditd streams and optional supplementary telemetry into causally sorted edges."""
        self.reset_state()
        scenario_path = Path(scenario_dir)
        gather_root = scenario_path / "gather"
        audit_files = sorted(gather_root.glob("*/logs/audit/audit.log*"))

        all_edges: List[ProvenanceEdgeEvent] = []
        for audit_file in audit_files:
            if not audit_file.is_file():
                continue
            host_id = audit_file.relative_to(gather_root).parts[0]
            line_labels = taxonomy.load_symmetric_line_labels(audit_file, scenario_path)
            for event_chunk in self.stream_audit_file(audit_file, host_id, line_labels):
                for ev in event_chunk:
                    all_edges.extend(self.audit_event_to_edges(ev, taxonomy))

        if self.include_supplementary_logs:
            all_edges.extend(self._stream_supplementary_host_logs(scenario_path, taxonomy))

        return stable_sort_by_timestamp(all_edges)

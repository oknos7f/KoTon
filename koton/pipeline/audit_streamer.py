import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable
import logging

from koton.data.schema import ProvenanceEvent, ProcessNode, FileNode, SocketNode
from koton.data.audit_parser import LinuxAuditParser
from koton.pipeline.streaming_engine import StreamingProvenanceEngine

logger = logging.getLogger("koton.audit_streamer")


class LinuxAuditWatcher:
    """
    Watches and tails Linux auditd logs (/var/log/audit/audit.log) in real-time.
    Streams syscall events (execve, connect, openat, fork) directly into the
    KoTon Streaming Provenance Engine.
    """

    def __init__(
        self,
        engine: StreamingProvenanceEngine,
        audit_log_path: str = "/var/log/audit/audit.log",
        poll_interval_sec: float = 0.2,
    ):
        self.engine = engine
        self.audit_log_path = Path(os.environ.get("KOTON_AUDIT_LOG_PATH", audit_log_path))
        self.poll_interval = poll_interval_sec
        self.parser = LinuxAuditParser()
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.events_streamed = 0

    def start(self):
        """Starts the background log tailer thread."""
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._tail_loop, daemon=True, name="AuditWatcherThread")
        self.thread.start()
        print(f"[*] LinuxAuditWatcher started (target: {self.audit_log_path})")

    def stop(self):
        """Stops the background log tailer."""
        self.is_running = False
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("[*] LinuxAuditWatcher stopped.")

    def _tail_loop(self):
        """Tails the audit.log file, parsing new records as they arrive."""
        while self.is_running and not self._stop_event.is_set():
            if not self.audit_log_path.exists():
                # Wait until file appears or fallback mode
                time.sleep(2.0)
                continue

            try:
                with open(self.audit_log_path, "r", encoding="utf-8", errors="replace") as f:
                    # Seek to end of file to read only live events
                    f.seek(0, os.SEEK_END)
                    while self.is_running and not self._stop_event.is_set():
                        line = f.readline()
                        if line:
                            event = self.parser.parse_line(line)
                            if event:
                                self._dispatch_event(event)
                                self.events_streamed += 1
                        else:
                            time.sleep(self.poll_interval)
            except Exception as e:
                logger.warning(f"[!] Audit log read exception: {e}")
                time.sleep(1.0)

    def _dispatch_event(self, event: ProvenanceEvent):
        """
        Enriches entities and pushes the provenance event into the sliding window engine.
        """
        # Auto-create source/target nodes if not yet cached
        if event.src_type == "process" and event.src_key not in self.engine.processes:
            pid = event.metadata.get("pid", 1000)
            comm = event.metadata.get("comm", "unknown")
            exe = event.metadata.get("exe", comm)
            uid = event.metadata.get("uid", 1000)
            proc = ProcessNode(
                key=event.src_key,
                pid=pid,
                name=comm,
                cmdline=exe,
                path=exe,
                uid=uid,
                timestamp=event.timestamp,
                label=1 if event.is_attack else 0
            )
            self.engine.add_process(proc)

        if event.dst_type == "process" and event.dst_key not in self.engine.processes:
            pid = event.metadata.get("pid", 1000)
            comm = event.metadata.get("comm", "unknown")
            exe = event.metadata.get("exe", comm)
            uid = event.metadata.get("uid", 1000)
            proc = ProcessNode(
                key=event.dst_key,
                pid=pid,
                name=comm,
                cmdline=exe,
                path=exe,
                uid=uid,
                timestamp=event.timestamp,
                label=1 if event.is_attack else 0
            )
            self.engine.add_process(proc)

        elif event.dst_type == "file" and event.dst_key not in self.engine.files:
            fpath = event.metadata.get("path", event.dst_key.replace("file:", ""))
            fnode = FileNode(
                key=event.dst_key,
                path=fpath,
                extension=Path(fpath).suffix,
                timestamp=event.timestamp,
                label=1 if event.is_attack else 0
            )
            self.engine.add_file(fnode)

        elif event.dst_type == "socket" and event.dst_key not in self.engine.sockets:
            ip = event.metadata.get("ip", "127.0.0.1")
            port = event.metadata.get("port", 80)
            snode = SocketNode(
                key=event.dst_key,
                ip=ip,
                port=port,
                timestamp=event.timestamp,
                label=1 if event.is_attack else 0
            )
            self.engine.add_socket(snode)

        self.engine.push_event(event)

    def replay_log_file(self, file_path: str, speed_multiplier: float = 1.0):
        """
        Replays an existing audit log file for benchmarking and demonstration.
        """
        path = Path(file_path)
        if not path.exists():
            print(f"[!] Replay file not found: {path}")
            return

        print(f"[*] Replaying audit log from {path}...")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                event = self.parser.parse_line(line)
                if event:
                    self._dispatch_event(event)
                    self.events_streamed += 1
                if speed_multiplier > 0:
                    time.sleep(0.01 / speed_multiplier)
        print(f"[+] Replay complete. Streamed {self.events_streamed} events.")

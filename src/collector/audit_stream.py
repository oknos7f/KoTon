import logging
import os
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Callable, List, Optional

from src.data.taxonomy import LabelTaxonomy
from src.parser.streaming_parser import ProvenanceEdgeEvent, StatefulAuditParser

logger = logging.getLogger(__name__)


class AuditStreamCollector:
    """Real-time TCP, Unix domain socket, and replay auditd stream collector backed by StatefulAuditParser."""

    def __init__(
        self,
        taxonomy: LabelTaxonomy,
        tcp_host: str = "0.0.0.0",
        tcp_port: int = 9999,
        unix_socket_path: Optional[str] = "/tmp/koton_audit.sock",
        default_host_id: str = "ubuntu-vm",
        carryover_sec: float = 0.25,
        on_edges_callback: Optional[Callable[[List[ProvenanceEdgeEvent]], None]] = None,
    ) -> None:
        """Initialize network listeners, stateful multiline parser, and thread-safe output queue."""
        self.taxonomy = taxonomy
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.unix_socket_path = unix_socket_path if os.name == "posix" else None
        self.default_host_id = default_host_id
        self.carryover_sec = carryover_sec
        self.on_edges_callback = on_edges_callback

        self.parser = StatefulAuditParser(
            chunk_size=512,
            carryover_sec=carryover_sec,
            include_supplementary_logs=False,
        )
        self._parser_lock = threading.Lock()
        self._edge_queue: queue.Queue[ProvenanceEdgeEvent] = queue.Queue(maxsize=100_000)
        self._raw_log_queue: queue.Queue[ dict ] = queue.Queue(maxsize=2_000)
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._tcp_server: Optional[socket.socket] = None
        self._unix_server: Optional[socket.socket] = None
        self._line_counter = 0
        self.total_lines_received = 0
        self.total_edges_emitted = 0
        self.active_vm_clients = 0

    def start(self) -> None:
        """Start background TCP listener, optional Unix socket listener, and carry-over flusher threads."""
        self._stop_event.clear()

        tcp_thread = threading.Thread(target=self._run_tcp_server, name="AuditTCPListener", daemon=True)
        tcp_thread.start()
        self._threads.append(tcp_thread)

        if self.unix_socket_path:
            unix_thread = threading.Thread(target=self._run_unix_server, name="AuditUnixListener", daemon=True)
            unix_thread.start()
            self._threads.append(unix_thread)

        flush_thread = threading.Thread(target=self._run_periodic_flush, name="AuditCarryoverFlusher", daemon=True)
        flush_thread.start()
        self._threads.append(flush_thread)

    def stop(self) -> None:
        """Signal all listener threads to terminate and close bound sockets."""
        self._stop_event.set()
        for srv in (self._tcp_server, self._unix_server):
            if srv is not None:
                try:
                    srv.close()
                except OSError:
                    pass
        if self.unix_socket_path and os.path.exists(self.unix_socket_path):
            try:
                os.unlink(self.unix_socket_path)
            except OSError:
                pass

    def ingest_raw_line(self, raw_line: str, host_id: Optional[str] = None) -> List[ProvenanceEdgeEvent]:
        """Parse one incoming auditd log line (optionally prefixed with host_id| or host_id\\t) into edges."""
        stripped = raw_line.strip()
        if not stripped:
            return []

        resolved_host = host_id or self.default_host_id
        line_body = stripped
        if "\t" in stripped and not stripped.startswith("type="):
            prefix, rest = stripped.split("\t", 1)
            if rest.startswith("type=") or "msg=audit(" in rest:
                resolved_host = prefix.strip() or resolved_host
                line_body = rest
        elif "|" in stripped and not stripped.startswith("type="):
            prefix, rest = stripped.split("|", 1)
            if rest.startswith("type=") or "msg=audit(" in rest:
                resolved_host = prefix.strip() or resolved_host
                line_body = rest

        emitted: List[ProvenanceEdgeEvent] = []
        with self._parser_lock:
            self._line_counter += 1
            self.total_lines_received += 1
            log_item = {
                "ts": round(time.time(), 3),
                "host": resolved_host,
                "line": line_body,
            }
            try:
                self._raw_log_queue.put_nowait(log_item)
            except queue.Full:
                try:
                    self._raw_log_queue.get_nowait()
                    self._raw_log_queue.put_nowait(log_item)
                except queue.Empty:
                    pass
            completed_ev = self.parser._ingest_audit_line(
                host_id=resolved_host,
                line=line_body,
                line_no=self._line_counter,
                line_labels=None,
            )
            if completed_ev is not None:
                emitted.extend(self.parser.audit_event_to_edges(completed_ev, self.taxonomy))

        if emitted:
            self._publish_edges(emitted)
        return emitted

    def drain_raw_logs(self, max_items: int = 80) -> List[dict]:
        """Drain recent raw auditd log lines for live streaming display on the SOC dashboard."""
        logs: List[dict] = []
        while len(logs) < max_items:
            try:
                logs.append(self._raw_log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def flush_expired_carryover(self, force_all: bool = False) -> List[ProvenanceEdgeEvent]:
        """Flush buffered multiline auditd events exceeding carryover_sec horizon."""
        emitted: List[ProvenanceEdgeEvent] = []
        now_wall = time.time()
        with self._parser_lock:
            if not self.parser._carryover_buffers:
                return []
            max_ts = max(ev.timestamp for ev in self.parser._carryover_buffers.values())
            ready_keys = [
                k
                for k, ev in self.parser._carryover_buffers.items()
                if force_all or (max_ts - ev.timestamp) >= self.carryover_sec or (now_wall - ev.timestamp) >= self.carryover_sec
            ]
            ready_events = [self.parser._carryover_buffers.pop(k) for k in ready_keys]

            ready_events.sort(key=lambda e: e.timestamp)
            for ev in ready_events:
                emitted.extend(self.parser.audit_event_to_edges(ev, self.taxonomy))

        if emitted:
            self._publish_edges(emitted)
        return emitted

    def drain_edges(self, max_items: int = 50_000) -> List[ProvenanceEdgeEvent]:
        """Drain queued ProvenanceEdgeEvents accumulated since the previous micro-tick."""
        self.flush_expired_carryover(force_all=False)
        items: List[ProvenanceEdgeEvent] = []
        while len(items) < max_items:
            try:
                items.append(self._edge_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def replay_audit_file(
        self,
        audit_path: str | Path,
        host_id: str = "fox-replay",
        lines_per_batch: int = 120,
        batch_interval_sec: float = 0.05,
        max_lines: Optional[int] = 1500,
    ) -> int:
        """Stream an existing audit.log file into the live collector pipeline for testing or replay."""
        count = 0
        with open(audit_path, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                if self._stop_event.is_set():
                    break
                self.ingest_raw_line(line, host_id=host_id)
                count += 1
                if max_lines is not None and count >= max_lines:
                    break
                if count % max(1, lines_per_batch) == 0 and batch_interval_sec > 0:
                    time.sleep(batch_interval_sec)
        self.flush_expired_carryover(force_all=True)
        return count

    def _publish_edges(self, edges: List[ProvenanceEdgeEvent]) -> None:
        """Push parsed edges into the internal queue and invoke optional subscriber callback."""
        self.total_edges_emitted += len(edges)
        for edge in edges:
            try:
                self._edge_queue.put_nowait(edge)
            except queue.Full:
                try:
                    self._edge_queue.get_nowait()
                    self._edge_queue.put_nowait(edge)
                except queue.Empty:
                    pass
        if self.on_edges_callback is not None:
            self.on_edges_callback(edges)

    def _run_periodic_flush(self) -> None:
        """Periodically flush carryover buffers during quiet stream intervals."""
        while not self._stop_event.is_set():
            time.sleep(max(0.1, self.carryover_sec))
            self.flush_expired_carryover(force_all=False)

    def _run_tcp_server(self) -> None:
        """Accept concurrent TCP connections from Ubuntu VM auditd forwarders."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((self.tcp_host, self.tcp_port))
            srv.listen(8)
            srv.settimeout(1.0)
            self._tcp_server = srv
            logger.info("AuditStreamCollector TCP listening on %s:%d", self.tcp_host, self.tcp_port)
        except OSError as exc:
            logger.warning("AuditStreamCollector TCP bind failed on %s:%d (%s)", self.tcp_host, self.tcp_port, exc)
            srv.close()
            return

        while not self._stop_event.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client_thread = threading.Thread(
                target=self._handle_client_stream,
                args=(conn, f"vm_{addr[0]}"),
                daemon=True,
            )
            client_thread.start()

    def _run_unix_server(self) -> None:
        """Accept local Unix domain socket connections on POSIX hosts."""
        if not self.unix_socket_path or not hasattr(socket, "AF_UNIX"):
            return
        if os.path.exists(self.unix_socket_path):
            try:
                os.unlink(self.unix_socket_path)
            except OSError:
                return

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(self.unix_socket_path)
            srv.listen(4)
            srv.settimeout(1.0)
            self._unix_server = srv
            logger.info("AuditStreamCollector Unix socket listening on %s", self.unix_socket_path)
        except OSError as exc:
            logger.warning("AuditStreamCollector Unix socket bind failed (%s)", exc)
            srv.close()
            return

        while not self._stop_event.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_client_stream,
                args=(conn, self.default_host_id),
                daemon=True,
            ).start()

    def _handle_client_stream(self, conn: socket.socket, fallback_host: str) -> None:
        """Read newline-delimited auditd records from a connected socket client."""
        self.active_vm_clients += 1
        buf = b""
        try:
            conn.settimeout(2.0)
            while not self._stop_event.is_set():
                try:
                    chunk = conn.recv(16384)
                except socket.timeout:
                    self.flush_expired_carryover(force_all=True)
                    continue
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    self.ingest_raw_line(raw_line.decode("utf-8", errors="replace"), host_id=fallback_host)
            if buf.strip():
                self.ingest_raw_line(buf.decode("utf-8", errors="replace"), host_id=fallback_host)
            self.flush_expired_carryover(force_all=True)
        except OSError:
            pass
        finally:
            self.active_vm_clients = max(0, self.active_vm_clients - 1)
            try:
                conn.close()
            except OSError:
                pass

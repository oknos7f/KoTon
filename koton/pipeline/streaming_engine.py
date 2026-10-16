import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple, Callable
from torch_geometric.data import HeteroData

from koton.config import (
    SLIDING_WINDOW_SEC,
    WINDOW_STEP_SEC,
    MAX_CACHE_ENTITIES,
)
from koton.data.schema import (
    ProcessNode,
    FileNode,
    SocketNode,
    ProvenanceEvent,
    build_hetero_data,
)


class StreamingProvenanceEngine:
    """
    Stateful real-time sliding window provenance accumulator.
    Maintains persistent entity cache across sliding windows to prevent broken attack chains.
    """

    def __init__(
        self,
        window_sec: float = SLIDING_WINDOW_SEC,
        step_sec: float = WINDOW_STEP_SEC,
        max_cache: int = MAX_CACHE_ENTITIES,
    ):
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.max_cache = max_cache

        self.lock = threading.Lock()
        self.event_queue: deque = deque()

        # Stateful Entity Caches
        self.processes: Dict[str, ProcessNode] = {}
        self.files: Dict[str, FileNode] = {}
        self.sockets: Dict[str, SocketNode] = {}

        # Running status
        self.is_running = False
        self.last_emit_time = time.time()

    def add_process(self, proc: ProcessNode):
        with self.lock:
            self.processes[proc.key] = proc
            self._prune_cache()

    def add_file(self, file_node: FileNode):
        with self.lock:
            self.files[file_node.key] = file_node
            self._prune_cache()

    def add_socket(self, sock: SocketNode):
        with self.lock:
            self.sockets[sock.key] = sock
            self._prune_cache()

    def push_event(self, event: ProvenanceEvent):
        """Pushes an incoming audit / provenance event into the sliding window queue."""
        with self.lock:
            self.event_queue.append(event)
            self._prune_old_events(event.timestamp)

    def _prune_old_events(self, current_ts: float):
        cutoff = current_ts - self.window_sec
        while self.event_queue and self.event_queue[0].timestamp < cutoff:
            self.event_queue.popleft()

    def _prune_cache(self):
        """Enforces MAX_CACHE_ENTITIES limit to prevent unbounded memory growth."""
        if len(self.processes) > self.max_cache:
            # Drop oldest processes
            sorted_keys = sorted(self.processes.keys(), key=lambda k: self.processes[k].timestamp)
            for k in sorted_keys[: len(self.processes) - self.max_cache]:
                del self.processes[k]

        if len(self.files) > self.max_cache:
            sorted_keys = sorted(self.files.keys(), key=lambda k: self.files[k].timestamp)
            for k in sorted_keys[: len(self.files) - self.max_cache]:
                del self.files[k]

        if len(self.sockets) > self.max_cache:
            sorted_keys = sorted(self.sockets.keys(), key=lambda k: self.sockets[k].timestamp)
            for k in sorted_keys[: len(self.sockets) - self.max_cache]:
                del self.sockets[k]

    def get_current_window_graph(self) -> Tuple[HeteroData, List[ProvenanceEvent]]:
        """
        Builds a HeteroData snapshot representing the current active sliding window.
        """
        with self.lock:
            active_events = list(self.event_queue)
            # Find all entities involved in active events or recently active
            sub_procs: Dict[str, ProcessNode] = {}
            sub_files: Dict[str, FileNode] = {}
            sub_socks: Dict[str, SocketNode] = {}

            for ev in active_events:
                if ev.src_type == "process" and ev.src_key in self.processes:
                    sub_procs[ev.src_key] = self.processes[ev.src_key]
                elif ev.src_type == "file" and ev.src_key in self.files:
                    sub_files[ev.src_key] = self.files[ev.src_key]
                elif ev.src_type == "socket" and ev.src_key in self.sockets:
                    sub_socks[ev.src_key] = self.sockets[ev.src_key]

                if ev.dst_type == "process" and ev.dst_key in self.processes:
                    sub_procs[ev.dst_key] = self.processes[ev.dst_key]
                elif ev.dst_type == "file" and ev.dst_key in self.files:
                    sub_files[ev.dst_key] = self.files[ev.dst_key]
                elif ev.dst_type == "socket" and ev.dst_key in self.sockets:
                    sub_socks[ev.dst_key] = self.sockets[ev.dst_key]

            # If empty window, include at least the core system baseline
            if not sub_procs and self.processes:
                for k in list(self.processes.keys())[:5]:
                    sub_procs[k] = self.processes[k]

            hdata = build_hetero_data(sub_procs, sub_files, sub_socks, active_events)
            now = time.time()
            hdata.t_start = now - self.window_sec
            hdata.t_end = now
            return hdata, active_events

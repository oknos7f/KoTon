from collections import Counter
import ipaddress
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from gensim.models import FastText
import numpy as np
from sklearn.feature_extraction import FeatureHasher

from src.graph.sanitizer import PathSanitizer


def compute_shannon_entropy(text: str) -> float:
    """Calculate character-level Shannon entropy of a command-line string."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def extract_path_hierarchy_tokens(sanitized_path: str) -> List[str]:
    """Generate hierarchical prefix tokens from a normalized POSIX path for HFH."""
    parts = [p for p in sanitized_path.split("/") if p]
    if not parts:
        return ["/"]
    tokens: List[str] = []
    acc = ""
    for part in parts:
        acc = f"{acc}/{part}"
        tokens.append(acc)
    tokens.append(f"basename:{parts[-1]}")
    return tokens


def extract_ip_subnet_tokens(ip_str: str) -> List[str]:
    """Generate A/B/C/D subnet prefix tokens from an IPv4 or IPv6 address for HFH."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return ["ip:invalid"]

    if isinstance(ip_obj, ipaddress.IPv4Address):
        octets = ip_str.split(".")
        return [
            f"A:{octets[0]}",
            f"B:{octets[0]}.{octets[1]}",
            f"C:{octets[0]}.{octets[1]}.{octets[2]}",
            f"D:{ip_str}",
        ]
    hextets = ip_obj.exploded.split(":")
    return [
        f"v6_16:{hextets[0]}",
        f"v6_32:{':'.join(hextets[:2])}",
        f"v6_48:{':'.join(hextets[:3])}",
        f"v6_64:{':'.join(hextets[:4])}",
    ]


def is_private_rfc1918_ip(ip_str: str) -> bool:
    """Return True if IP belongs to RFC1918 private or loopback internal address space."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return bool(ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local)
    except ValueError:
        return False


class HeterogeneousFeatureEncoder:
    """Type-specific feature encoder producing 68-d process, 36-d file, 20-d socket, and 2-d edge tensors."""

    def __init__(
        self,
        sanitizer: Optional[PathSanitizer] = None,
        proc_exe_dim: int = 16,
        proc_cmd_dim: int = 48,
        file_path_dim: int = 32,
        sock_ip_dim: int = 12,
        sensitive_prefixes: Optional[Sequence[str]] = None,
        seed: int = 42,
    ) -> None:
        """Initialize stateless HFH hashers and training-only FastText embedder parameters."""
        self.sanitizer = sanitizer or PathSanitizer()
        self.proc_exe_dim = proc_exe_dim
        self.proc_cmd_dim = proc_cmd_dim
        self.file_path_dim = file_path_dim
        self.sock_ip_dim = sock_ip_dim
        self.seed = seed
        self.sensitive_prefixes = tuple(
            sensitive_prefixes
            or ("/etc", "/root", "/var/www", "/tmp", "/var/spool/cron", "/usr/bin/sudo", "/bin/su")
        )
        self._exe_hasher = FeatureHasher(n_features=proc_exe_dim, input_type="string", alternate_sign=True)
        self._file_hasher = FeatureHasher(n_features=file_path_dim, input_type="string", alternate_sign=True)
        self._ip_hasher = FeatureHasher(n_features=sock_ip_dim, input_type="string", alternate_sign=True)
        self.cmd_model: Optional[FastText] = None

    def fit_cmdline_embedder(
        self,
        train_cmdlines: Iterable[str],
        min_count: int = 1,
        window: int = 5,
        epochs: int = 10,
    ) -> None:
        """Train field-specific FastText embedding model strictly on training-split process cmdlines."""
        sentences: List[List[str]] = []
        for raw_cmd in train_cmdlines:
            clean_cmd = self.sanitizer.sanitize_text(raw_cmd)
            toks = [t for t in clean_cmd.split() if t]
            if toks:
                sentences.append(toks)
        if not sentences:
            sentences = [["/sbin/init"]]

        self.cmd_model = FastText(
            sentences=sentences,
            vector_size=self.proc_cmd_dim,
            window=window,
            min_count=min_count,
            workers=1,
            seed=self.seed,
            epochs=epochs,
        )

    def save_cmdline_embedder(self, model_path: str | Path) -> None:
        """Persist trained FastText cmdline embedder to artifacts directory."""
        if self.cmd_model is not None:
            path = Path(model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.cmd_model.save(str(path))

    def load_cmdline_embedder(self, model_path: str | Path) -> None:
        """Load pre-trained FastText cmdline embedder from disk."""
        self.cmd_model = FastText.load(str(model_path))

    def _embed_cmdline(self, cmdline: str) -> np.ndarray:
        """Embed sanitized process cmdline tokens into a 48-dim float32 vector."""
        clean_cmd = self.sanitizer.sanitize_text(cmdline)
        toks = [t for t in clean_cmd.split() if t]
        if not toks or self.cmd_model is None:
            return np.zeros(self.proc_cmd_dim, dtype=np.float32)
        vecs = [self.cmd_model.wv[tok] for tok in toks]
        return np.mean(vecs, axis=0).astype(np.float32)

    def encode_process_nodes(
        self,
        records: List[Dict],
    ) -> np.ndarray:
        """Encode process nodes into [N_proc, 68] float32 array using snapshot-local degrees."""
        if not records:
            return np.zeros((0, self.proc_exe_dim + self.proc_cmd_dim + 4), dtype=np.float32)

        exe_tokens = [
            extract_path_hierarchy_tokens(self.sanitizer.sanitize_path(r["exe"])) for r in records
        ]
        exe_hfh = self._exe_hasher.transform(exe_tokens).toarray().astype(np.float32)

        cmd_embs = np.vstack([self._embed_cmdline(r["cmdline"]) for r in records]).astype(np.float32)

        struct_feats = np.zeros((len(records), 4), dtype=np.float32)
        for i, r in enumerate(records):
            clean_cmd = self.sanitizer.sanitize_text(r["cmdline"])
            struct_feats[i, 0] = math.log1p(r["in_deg"])
            struct_feats[i, 1] = math.log1p(r["out_deg"])
            struct_feats[i, 2] = compute_shannon_entropy(clean_cmd)
            struct_feats[i, 3] = 1.0 if int(r["uid"]) == 0 else 0.0

        return np.concatenate([exe_hfh, cmd_embs, struct_feats], axis=1).astype(np.float32)

    def encode_file_nodes(
        self,
        records: List[Dict],
    ) -> np.ndarray:
        """Encode file nodes into [N_file, 36] float32 array using sanitized path HFH and local degrees."""
        if not records:
            return np.zeros((0, self.file_path_dim + 4), dtype=np.float32)

        sanitized_paths = [self.sanitizer.sanitize_path(r["path"]) for r in records]
        path_tokens = [extract_path_hierarchy_tokens(sp) for sp in sanitized_paths]
        path_hfh = self._file_hasher.transform(path_tokens).toarray().astype(np.float32)

        struct_feats = np.zeros((len(records), 4), dtype=np.float32)
        for i, (r, sp) in enumerate(zip(records, sanitized_paths)):
            depth = len([seg for seg in sp.split("/") if seg])
            is_sens = any(sp == pref or sp.startswith(pref + "/") for pref in self.sensitive_prefixes)
            struct_feats[i, 0] = math.log1p(r["in_deg"])
            struct_feats[i, 1] = math.log1p(r["out_deg"])
            struct_feats[i, 2] = float(depth)
            struct_feats[i, 3] = 1.0 if is_sens else 0.0

        return np.concatenate([path_hfh, struct_feats], axis=1).astype(np.float32)

    def encode_socket_nodes(
        self,
        records: List[Dict],
    ) -> np.ndarray:
        """Encode socket nodes into [N_sock, 20] float32 array with subnet HFH, port band, and local stats."""
        if not records:
            return np.zeros((0, self.sock_ip_dim + 8), dtype=np.float32)

        ip_tokens = [extract_ip_subnet_tokens(r["ip"]) for r in records]
        ip_hfh = self._ip_hasher.transform(ip_tokens).toarray().astype(np.float32)

        port_and_struct = np.zeros((len(records), 8), dtype=np.float32)
        for i, r in enumerate(records):
            port = max(0, min(65535, int(r["port"])))
            if port <= 1023:
                port_and_struct[i, 0] = 1.0
            elif port <= 49151:
                port_and_struct[i, 1] = 1.0
            else:
                port_and_struct[i, 2] = 1.0
            port_and_struct[i, 3] = port / 65535.0
            port_and_struct[i, 4] = math.log1p(r["in_deg"])
            port_and_struct[i, 5] = math.log1p(r["out_deg"])
            port_and_struct[i, 6] = 1.0 if is_private_rfc1918_ip(r["ip"]) else 0.0
            port_and_struct[i, 7] = math.log1p(r["conn_freq"])

        return np.concatenate([ip_hfh, port_and_struct], axis=1).astype(np.float32)

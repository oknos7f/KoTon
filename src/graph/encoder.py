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
        self._tok_vec_cache: Dict[str, np.ndarray] = {}
        self._cmd_cache: Dict[str, tuple[np.ndarray, float]] = {}
        self._exe_cache: Dict[str, np.ndarray] = {}
        self._file_cache: Dict[str, tuple[np.ndarray, float, float]] = {}
        self._ip_cache: Dict[str, tuple[np.ndarray, float]] = {}
        self.feature_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    def fit_feature_stats(self, train_graphs: Sequence) -> Dict[str, Dict[str, List[float]]]:
        """Fit per-node-type structural and edge normalization statistics strictly on training graphs."""
        import torch

        # Continuous structural column indices per node type (excluding binary flags & HFH/FastText)
        cont_cols: Dict[str, List[int]] = {
            "process": [self.proc_exe_dim + self.proc_cmd_dim, self.proc_exe_dim + self.proc_cmd_dim + 1, self.proc_exe_dim + self.proc_cmd_dim + 2],
            "file": [self.file_path_dim, self.file_path_dim + 1, self.file_path_dim + 2],
            "socket": [self.sock_ip_dim + 4, self.sock_ip_dim + 5, self.sock_ip_dim + 7],
        }
        stats_np: Dict[str, Dict[str, np.ndarray]] = {}
        stats_serializable: Dict[str, Dict[str, List[float]]] = {}

        for ntype, cols in cont_cols.items():
            chunks = []
            for g in train_graphs:
                x = g[ntype].x
                if x.size(0) > 0:
                    chunks.append(x[:, cols].detach().cpu().numpy())
            if chunks:
                cat = np.concatenate(chunks, axis=0).astype(np.float64)
                mean = np.mean(cat, axis=0).astype(np.float32)
                std = np.maximum(np.std(cat, axis=0), 1e-3).astype(np.float32)
            else:
                mean = np.zeros(len(cols), dtype=np.float32)
                std = np.ones(len(cols), dtype=np.float32)
            stats_np[ntype] = {"cols": np.array(cols, dtype=np.int64), "mean": mean, "std": std}
            stats_serializable[ntype] = {"cols": cols, "mean": mean.tolist(), "std": std.tolist()}

        self.feature_stats = stats_np
        return stats_serializable

    def load_feature_stats(self, stats_dict: Optional[Dict[str, Dict[str, List[float]]]]) -> None:
        """Load pre-fitted train-only normalization statistics."""
        if not stats_dict:
            self.feature_stats = None
            return
        self.feature_stats = {
            ntype: {
                "cols": np.array(entry["cols"], dtype=np.int64),
                "mean": np.array(entry["mean"], dtype=np.float32),
                "std": np.maximum(np.array(entry["std"], dtype=np.float32), 1e-3),
            }
            for ntype, entry in stats_dict.items()
        }

    def export_feature_stats(self) -> Optional[Dict[str, Dict[str, List[float]]]]:
        """Export current normalization statistics as JSON/checkpoint-friendly lists."""
        if not self.feature_stats:
            return None
        return {
            ntype: {
                "cols": entry["cols"].tolist(),
                "mean": entry["mean"].tolist(),
                "std": entry["std"].tolist(),
            }
            for ntype, entry in self.feature_stats.items()
        }

    def apply_feature_stats_to_graph(self, data) -> object:
        """Apply train-only z-score standardization in-place to continuous structural node columns."""
        if not self.feature_stats:
            return data
        import torch

        for ntype, entry in self.feature_stats.items():
            x = data[ntype].x
            if x.size(0) == 0:
                continue
            cols = entry["cols"].tolist()
            mean_t = torch.tensor(entry["mean"], dtype=x.dtype, device=x.device)
            std_t = torch.tensor(entry["std"], dtype=x.dtype, device=x.device)
            x = x.clone()
            x[:, cols] = ((x[:, cols] - mean_t) / std_t).clamp(-10.0, 10.0)
            data[ntype].x = x
        return data

    def _clear_feature_caches(self) -> None:
        """Reset memoized token and path feature caches when FastText weights change."""
        self._tok_vec_cache.clear()
        self._cmd_cache.clear()

    def _warm_vocab_cache(self) -> None:
        """Pre-populate token vector cache from FastText vocabulary vectors to avoid runtime n-gram recomputation."""
        self._clear_feature_caches()
        if self.cmd_model is None:
            return
        wv = self.cmd_model.wv
        if getattr(wv, "vectors", None) is None and hasattr(wv, "adjust_vectors"):
            try:
                wv.adjust_vectors()
            except Exception:
                pass
        mat = getattr(wv, "vectors", None)
        if mat is None:
            mat = getattr(wv, "vectors_vocab", None)
        if mat is not None and hasattr(wv, "key_to_index"):
            for tok, idx in wv.key_to_index.items():
                if idx < len(mat):
                    self._tok_vec_cache[tok] = np.asarray(mat[idx], dtype=np.float32)

    def fit_cmdline_embedder(
        self,
        train_cmdlines: Iterable[str],
        min_count: int = 1,
        window: int = 5,
        epochs: int = 10,
        workers: int = 4,
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
            workers=max(1, int(workers)),
            seed=self.seed,
            epochs=epochs,
        )
        self._warm_vocab_cache()

    def save_cmdline_embedder(self, model_path: str | Path) -> None:
        """Persist trained FastText cmdline embedder to artifacts directory."""
        if self.cmd_model is not None:
            path = Path(model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.cmd_model.save(str(path))

    def load_cmdline_embedder(self, model_path: str | Path) -> None:
        """Load pre-trained FastText cmdline embedder from disk."""
        self.cmd_model = FastText.load(str(model_path))
        self._warm_vocab_cache()

    def _embed_cmdline_and_entropy(self, cmdline: str) -> tuple[np.ndarray, float]:
        """Embed sanitized process cmdline into a 48-dim float32 vector and compute Shannon entropy with caching."""
        cached = self._cmd_cache.get(cmdline)
        if cached is not None:
            return cached

        clean_cmd = self.sanitizer.sanitize_text(cmdline)
        entropy = compute_shannon_entropy(clean_cmd)
        toks = [t for t in clean_cmd.split() if t]
        if not toks or self.cmd_model is None:
            vec = np.zeros(self.proc_cmd_dim, dtype=np.float32)
        else:
            wv = self.cmd_model.wv
            vecs: List[np.ndarray] = []
            for tok in toks:
                tv = self._tok_vec_cache.get(tok)
                if tv is None:
                    tv = np.asarray(wv[tok], dtype=np.float32)
                    if len(self._tok_vec_cache) < 32768:
                        self._tok_vec_cache[tok] = tv
                vecs.append(tv)
            vec = np.mean(vecs, axis=0).astype(np.float32)

        out = (vec, entropy)
        if len(self._cmd_cache) < 16384:
            self._cmd_cache[cmdline] = out
        return out

    def encode_process_nodes(
        self,
        records: List[Dict],
    ) -> np.ndarray:
        """Encode process nodes into [N_proc, 68] float32 array using snapshot-local degrees."""
        if not records:
            return np.zeros((0, self.proc_exe_dim + self.proc_cmd_dim + 4), dtype=np.float32)

        n = len(records)
        out = np.zeros((n, self.proc_exe_dim + self.proc_cmd_dim + 4), dtype=np.float32)
        missing_exes: List[str] = []
        for r in records:
            raw_exe = r["exe"]
            if raw_exe not in self._exe_cache:
                missing_exes.append(raw_exe)
        if missing_exes:
            uniq_exes = list(dict.fromkeys(missing_exes))
            toks_batch = [extract_path_hierarchy_tokens(self.sanitizer.sanitize_path(e)) for e in uniq_exes]
            hfh_batch = self._exe_hasher.transform(toks_batch).toarray().astype(np.float32)
            for e_str, row_vec in zip(uniq_exes, hfh_batch):
                if len(self._exe_cache) < 16384:
                    self._exe_cache[e_str] = row_vec

        d_exe = self.proc_exe_dim
        d_cmd = self.proc_cmd_dim
        for i, r in enumerate(records):
            exe_vec = self._exe_cache.get(r["exe"])
            if exe_vec is None:
                toks = extract_path_hierarchy_tokens(self.sanitizer.sanitize_path(r["exe"]))
                exe_vec = self._exe_hasher.transform([toks]).toarray()[0].astype(np.float32)
            cmd_vec, entropy = self._embed_cmdline_and_entropy(r["cmdline"])
            out[i, :d_exe] = exe_vec
            out[i, d_exe : d_exe + d_cmd] = cmd_vec
            out[i, d_exe + d_cmd] = math.log1p(r["in_deg"])
            out[i, d_exe + d_cmd + 1] = math.log1p(r["out_deg"])
            out[i, d_exe + d_cmd + 2] = entropy
            out[i, d_exe + d_cmd + 3] = 1.0 if int(r["uid"]) == 0 else 0.0

        return out

    def encode_file_nodes(
        self,
        records: List[Dict],
    ) -> np.ndarray:
        """Encode file nodes into [N_file, 36] float32 array using sanitized path HFH and local degrees."""
        if not records:
            return np.zeros((0, self.file_path_dim + 4), dtype=np.float32)

        n = len(records)
        d_fp = self.file_path_dim
        out = np.zeros((n, d_fp + 4), dtype=np.float32)
        missing_paths = [r["path"] for r in records if r["path"] not in self._file_cache]
        if missing_paths:
            uniq_paths = list(dict.fromkeys(missing_paths))
            sanitized_list = [self.sanitizer.sanitize_path(p) for p in uniq_paths]
            toks_batch = [extract_path_hierarchy_tokens(sp) for sp in sanitized_list]
            hfh_batch = self._file_hasher.transform(toks_batch).toarray().astype(np.float32)
            for raw_p, sp, row_vec in zip(uniq_paths, sanitized_list, hfh_batch):
                depth = float(len([seg for seg in sp.split("/") if seg]))
                is_sens = 1.0 if any(sp == pref or sp.startswith(pref + "/") for pref in self.sensitive_prefixes) else 0.0
                if len(self._file_cache) < 32768:
                    self._file_cache[raw_p] = (row_vec, depth, is_sens)

        for i, r in enumerate(records):
            cached = self._file_cache.get(r["path"])
            if cached is None:
                sp = self.sanitizer.sanitize_path(r["path"])
                row_vec = self._file_hasher.transform([extract_path_hierarchy_tokens(sp)]).toarray()[0].astype(np.float32)
                depth = float(len([seg for seg in sp.split("/") if seg]))
                is_sens = 1.0 if any(sp == pref or sp.startswith(pref + "/") for pref in self.sensitive_prefixes) else 0.0
            else:
                row_vec, depth, is_sens = cached
            out[i, :d_fp] = row_vec
            out[i, d_fp] = math.log1p(r["in_deg"])
            out[i, d_fp + 1] = math.log1p(r["out_deg"])
            out[i, d_fp + 2] = depth
            out[i, d_fp + 3] = is_sens

        return out

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

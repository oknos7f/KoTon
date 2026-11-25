import posixpath
import re
from typing import Iterable, List

DEFAULT_SCENARIO_NAMES = (
    "fox",
    "russellmitchell",
    "harrison",
    "santos",
    "shaw",
    "wardbeck",
    "wilson",
    "wheeler",
)

DEFAULT_HOST_NAMES = (
    "intranet_server",
    "intranet-server",
    "internal_share",
    "internal-share",
    "inet-firewall",
    "inet_firewall",
    "inet-dns",
    "inet_dns",
    "cloud_share",
    "cloud-share",
    "webserver",
    "monitoring",
)


class PathSanitizer:
    """Boundary-aware path and command sanitizer preventing scenario or host name leakage."""

    def __init__(
        self,
        scenarios: Iterable[str] = DEFAULT_SCENARIO_NAMES,
        host_names: Iterable[str] = DEFAULT_HOST_NAMES,
    ) -> None:
        """Compile alphabetic-boundary regular expressions for scenario and host tokens."""
        tokens: List[str] = sorted(
            {t.strip() for t in list(scenarios) + list(host_names) if t and t.strip()},
            key=len,
            reverse=True,
        )
        escaped = [re.escape(tok) for tok in tokens]
        pattern_str = r"(?<![a-zA-Z])(?:" + "|".join(escaped) + r")(?![a-zA-Z])"
        self._pattern = re.compile(pattern_str, flags=re.IGNORECASE)
        self._multi_slash = re.compile(r"/{2,}")

    def sanitize_path(self, path: str) -> str:
        """Remove scenario and host identifiers from a file path while preserving binaries like /usr/bin/firefox."""
        if not path:
            return "/unknown"
        cleaned = self._pattern.sub("", path)
        cleaned = self._multi_slash.sub("/", cleaned)
        normed = posixpath.normpath(cleaned)
        return normed if normed.startswith("/") else "/" + normed

    def sanitize_text(self, text: str) -> str:
        """Remove scenario and host identifiers from command-line strings."""
        if not text:
            return ""
        return self._pattern.sub("", text).strip()

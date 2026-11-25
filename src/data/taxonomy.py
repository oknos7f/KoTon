import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml

from src.data.phases import PHASE_TO_TACTIC_ID, ScenarioPhaseMatcher

logger = logging.getLogger(__name__)


class LabelTaxonomy:
    """Two-stage label mapper unifying AIT-LDS v2.0 JSONL labels, phase intervals, and CAM-LDS MITRE taxonomy."""

    def __init__(self, taxonomy_path: str | Path, scenario: str = "fox") -> None:
        """Initialize taxonomy configuration and scenario phase matcher."""
        with open(taxonomy_path, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)

        self.scenario = scenario
        self.phase_matcher = ScenarioPhaseMatcher(scenario)
        self.tactic_classes: Dict[int, str] = {
            int(k): str(v) for k, v in cfg.get("tactic_classes", {}).items()
        }
        ait_cfg = cfg.get("ait_lds", {})
        self.label_priority: List[str] = list(ait_cfg.get("label_priority", []))
        self.fine_to_tactic: Dict[str, int] = {
            str(k): int(v) for k, v in ait_cfg.get("fine_label_to_tactic_id", {}).items()
        }
        cam_cfg = cfg.get("cam_lds", {})
        self.cam_mitre_tactics: Dict[str, int] = {
            str(k).lower(): int(v) for k, v in cam_cfg.get("mitre_tactics", {}).items()
        }
        self.cam_techniques: Dict[str, int] = {
            str(k).upper(): int(v) for k, v in cam_cfg.get("techniques", {}).items()
        }
        self._warned_out_of_phase_keys: Set[Tuple[str, str]] = set()
        self.out_of_phase_fallback_count: int = 0

    @staticmethod
    def load_symmetric_line_labels(gather_file: Path, data_scenario_root: Path) -> Dict[int, List[str]]:
        """Load 1-based line-level malicious labels from the symmetric /labels/ JSONL file if present."""
        gather_root = data_scenario_root / "gather"
        labels_root = data_scenario_root / "labels"
        rel_path = gather_file.relative_to(gather_root)
        label_file = labels_root / rel_path
        if not label_file.is_file():
            return {}

        line_map: Dict[int, List[str]] = {}
        with open(label_file, "r", encoding="utf-8") as fp:
            for raw in fp:
                stripped = raw.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                line_no = int(record["line"])
                labels = [str(lbl) for lbl in record.get("labels", [])]
                if labels:
                    line_map[line_no] = labels
        return line_map

    def resolve_fine_label(self, raw_labels: Iterable[str]) -> Optional[str]:
        """Select a single fine-grained attack label according to configured priority order."""
        label_set = set(raw_labels)
        if not label_set:
            return None
        for priority_lbl in self.label_priority:
            if priority_lbl in label_set:
                return priority_lbl
        for candidate in sorted(label_set):
            if candidate in self.fine_to_tactic:
                return candidate
        return sorted(label_set)[0]

    def map_event_label(
        self,
        timestamp: float,
        raw_labels: Iterable[str],
        source_tag: str = "auditd",
    ) -> Tuple[int, str, bool]:
        """Compute 6-class tactic ID, resolved fine label, and malicious flag via two-stage matching."""
        label_list = list(raw_labels)
        is_malicious_line = len(label_list) > 0
        if not is_malicious_line:
            return PHASE_TO_TACTIC_ID["benign"], "benign", False

        fine_label = self.resolve_fine_label(label_list) or "unknown_attack"
        phase_name = self.phase_matcher.get_attack_phase(timestamp)
        tactic_id = self.phase_matcher.get_tactic_id(timestamp, is_malicious_line=True)

        if phase_name is None or tactic_id == 0:
            self.out_of_phase_fallback_count += 1
            fallback_id = self.fine_to_tactic.get(fine_label, 0)
            if fallback_id == 0:
                for lbl in label_list:
                    if lbl in self.fine_to_tactic:
                        fallback_id = self.fine_to_tactic[lbl]
                        break
            if fallback_id == 0:
                fallback_id = 1
            warn_key = (source_tag, fine_label)
            if warn_key not in self._warned_out_of_phase_keys:
                self._warned_out_of_phase_keys.add(warn_key)
                logger.warning(
                    "Malicious JSONL event outside RAW_PHASES on %s (ts=%.3f, fine_label=%s); applying JSONL tactic_id=%d.",
                    source_tag,
                    timestamp,
                    fine_label,
                    fallback_id,
                )
            return fallback_id, fine_label, True

        return tactic_id, fine_label, True

    def map_cam_lds_label(
        self,
        mitre_tactic: Optional[str] = None,
        mitre_technique: Optional[str] = None,
    ) -> int:
        """Map CAM-LDS MITRE ATT&CK tactic or technique identifiers into the unified 6-class tactic space."""
        if mitre_technique:
            tech_key = mitre_technique.strip().upper()
            if tech_key in self.cam_techniques:
                return self.cam_techniques[tech_key]
        if mitre_tactic:
            tac_key = mitre_tactic.strip().lower().replace(" ", "_").replace("-", "_")
            if tac_key in self.cam_mitre_tactics:
                return self.cam_mitre_tactics[tac_key]
        return 0

    def is_clean_benign_timestamp(self, timestamp: float) -> bool:
        """Check whether a timestamp falls strictly outside all scenario attack intervals."""
        return self.phase_matcher.is_clean_benign_window(timestamp)

    def tactic_name(self, tactic_id: int) -> str:
        """Return canonical 6-class tactic name for a given integer tactic ID."""
        return self.tactic_classes.get(int(tactic_id), "benign")

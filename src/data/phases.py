from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


RAW_PHASES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "fox": {
        "network_scans": ("2022-01-18 11:59:00", "2022-01-18 12:17:00"),
        "service_scans": ("2022-01-18 12:17:00", "2022-01-18 12:17:47"),
        "wpscan": ("2022-01-18 12:17:47", "2022-01-18 12:18:30"),
        "dirb": ("2022-01-18 12:18:30", "2022-01-18 12:38:00"),
        "webshell": ("2022-01-18 12:38:00", "2022-01-18 12:38:30"),
        "cracking": ("2022-01-18 12:38:30", "2022-01-18 13:13:50"),
        "reverse_shell": ("2022-01-18 13:13:50", "2022-01-18 13:14:30"),
        "privilege_escalation": ("2022-01-18 13:14:30", "2022-01-18 13:14:53"),
        "service_stop": ("2022-01-17 09:04:46", "2022-01-17 09:04:48"),
        "dnsteal": ("2022-01-17 09:04:48", "2022-01-17 10:00:00"),
        "false_positive_test": ("2022-01-17 11:00:00", "2022-01-17 16:00:00"),
        "false_positive_same_day": ("2022-01-18 00:00:00", "2022-01-19 00:00:00"),
    },
    # 추후 나머지 7개 시나리오(russellmitchell, harrison 등)를 동일 형식으로 추가
}

PHASE_TO_TACTIC_ID: Dict[str, int] = {
    "benign": 0,
    "network_scans": 1,
    "service_scans": 1,
    "wpscan": 1,
    "dirb": 1,
    "webshell": 2,
    "reverse_shell": 2,
    "cracking": 3,
    "privilege_escalation": 4,
    "service_stop": 5,
    "dnsteal": 5,
}

BENIGN_EVAL_PHASES = {"false_positive_test", "false_positive_same_day"}


class ScenarioPhaseMatcher:
    """AIT-LDS v2.0 시나리오의 공격 단계 및 정상 평가 구간을 UTC Epoch 기반으로 고속 매칭한다."""

    def __init__(self, scenario: str) -> None:
        if scenario not in RAW_PHASES:
            raise ValueError(f"Unsupported scenario: {scenario}")
        self.scenario = scenario
        self.attack_intervals: List[Tuple[str, float, float]] = []
        self.eval_intervals: List[Tuple[str, float, float]] = []

        for name, (start_str, end_str) in RAW_PHASES[scenario].items():
            start_ts = self._to_epoch(start_str)
            end_ts = self._to_epoch(end_str)
            if name in BENIGN_EVAL_PHASES:
                self.eval_intervals.append((name, start_ts, end_ts))
            else:
                self.attack_intervals.append((name, start_ts, end_ts))

    @staticmethod
    def _to_epoch(dt_str: str) -> float:
        """UTC 시간 문자열을 Unix Epoch(float)로 변환한다."""
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

    def get_attack_phase(self, timestamp: float) -> Optional[str]:
        """주어진 auditd 타임스탬프가 속한 공격 단계명을 반환한다. 공격 구간이 아니면 None을 반환한다."""
        for name, start_ts, end_ts in self.attack_intervals:
            if start_ts <= timestamp < end_ts:
                return name
        return None

    def get_tactic_id(self, timestamp: float, is_malicious_line: bool) -> int:
        """JSONL 라인 라벨과 시간 구간을 결합하여 최종 Multi-Class 정수 라벨을 반환한다."""
        if not is_malicious_line:
            return PHASE_TO_TACTIC_ID["benign"]
        phase = self.get_attack_phase(timestamp)
        return PHASE_TO_TACTIC_ID.get(phase, 0) if phase else 0

    def is_clean_benign_window(self, timestamp: float) -> bool:
        """Self-Supervised 정상 학습 및 오탐(FPR) 평가용 순수 정상 구간 여부를 반환한다."""
        if self.get_attack_phase(timestamp) is not None:
            return False
        return True
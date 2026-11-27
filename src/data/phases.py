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
    "russellmitchell": {
        "network_scans": ("2022-01-24 03:01:00", "2022-01-24 03:56:46"),
        "service_scans": ("2022-01-24 03:56:46", "2022-01-24 03:57:26"),
        "dirb": ("2022-01-24 03:57:26", "2022-01-24 03:57:48"),
        "wpscan": ("2022-01-24 03:57:48", "2022-01-24 03:58:20"),
        "webshell": ("2022-01-24 03:58:20", "2022-01-24 03:59:23"),
        "cracking": ("2022-01-24 03:59:23", "2022-01-24 04:36:57"),
        "reverse_shell": ("2022-01-24 04:36:57", "2022-01-24 04:37:39"),
        "privilege_escalation": ("2022-01-24 04:37:39", "2022-01-24 04:38:15"),
        "service_stop": ("2022-01-24 13:50:38", "2022-01-24 13:50:41"),
        "dnsteal": ("2022-01-24 13:50:41", "2022-01-24 14:45:00"),
        "false_positive_test": ("2022-01-22 10:00:00", "2022-01-22 15:00:00"),
        "false_positive_same_day": ("2022-01-24 00:00:00", "2022-01-25 00:00:00"),
    },
    "harrison": {
        "network_scans": ("2022-02-08 07:07:00", "2022-02-08 07:16:19"),
        "service_scans": ("2022-02-08 07:16:19", "2022-02-08 07:28:36"),
        "wpscan": ("2022-02-08 07:28:36", "2022-02-08 07:29:40"),
        "dirb": ("2022-02-08 07:29:40", "2022-02-08 07:55:59"),
        "webshell": ("2022-02-08 07:55:59", "2022-02-08 07:56:38"),
        "cracking": ("2022-02-08 07:56:38", "2022-02-08 08:35:59"),
        "reverse_shell": ("2022-02-08 08:35:59", "2022-02-08 08:36:37"),
        "privilege_escalation": ("2022-02-08 08:36:37", "2022-02-08 08:37:29"),
        "service_stop": ("2022-02-08 09:14:59", "2022-02-08 09:15:02"),
        "dnsteal": ("2022-02-08 09:15:02", "2022-02-08 11:16:00"),
        "false_positive_test": ("2022-02-06 10:00:00", "2022-02-06 15:00:00"),
        "false_positive_same_day": ("2022-02-08 00:00:00", "2022-02-09 00:00:00"),
    },
    "santos": {
        "network_scans": ("2022-01-17 11:15:00", "2022-01-17 11:21:32"),
        "service_scans": ("2022-01-17 11:21:32", "2022-01-17 11:22:02"),
        "dirb": ("2022-01-17 11:22:02", "2022-01-17 11:22:21"),
        "wpscan": ("2022-01-17 11:22:21", "2022-01-17 11:22:57"),
        "webshell": ("2022-01-17 11:22:57", "2022-01-17 11:24:17"),
        "cracking": ("2022-01-17 11:24:17", "2022-01-17 11:57:35"),
        "reverse_shell": ("2022-01-17 11:57:35", "2022-01-17 11:58:17"),
        "privilege_escalation": ("2022-01-17 11:58:17", "2022-01-17 11:59:00"),
        "service_stop": ("2022-01-16 07:16:17", "2022-01-16 07:16:20"),
        "dnsteal": ("2022-01-16 07:16:20", "2022-01-16 08:15:00"),
        "false_positive_test": ("2022-01-16 11:00:00", "2022-01-16 16:00:00"),
        "false_positive_same_day": ("2022-01-17 00:00:00", "2022-01-18 00:00:00"),
    },
    "shaw": {
        "network_scans": ("2022-01-29 14:37:00", "2022-01-29 14:38:37"),
        "service_scans": ("2022-01-29 14:38:37", "2022-01-29 14:38:50"),
        "wpscan": ("2022-01-29 14:38:50", "2022-01-29 14:39:14"),
        "dirb": ("2022-01-29 14:39:14", "2022-01-29 14:39:39"),
        "webshell": ("2022-01-29 14:39:39", "2022-01-29 14:40:02"),
        "cracking": ("2022-01-29 14:40:02", "2022-01-29 15:20:10"),
        "reverse_shell": ("2022-01-29 15:20:10", "2022-01-29 15:20:50"),
        "privilege_escalation": ("2022-01-29 15:20:50", "2022-01-29 15:21:14"),
        "service_stop": ("2022-01-28 21:08:00", "2022-01-28 21:08:03"),
        "dnsteal": ("2022-01-28 21:08:03", "2022-01-28 22:05:00"),
        "false_positive_test": ("2022-01-27 10:00:00", "2022-01-27 15:00:00"),
        "false_positive_same_day": ("2022-01-29 00:00:00", "2022-01-30 00:00:00"),
    },
    "wardbeck": {
        "network_scans": ("2022-01-23 12:10:00", "2022-01-23 12:11:00"),
        "service_scans": ("2022-01-23 12:11:00", "2022-01-23 12:11:27"),
        "wpscan": ("2022-01-23 12:11:27", "2022-01-23 12:11:49"),
        "dirb": ("2022-01-23 12:11:49", "2022-01-23 12:12:17"),
        "webshell": ("2022-01-23 12:12:17", "2022-01-23 12:13:46"),
        "cracking": ("2022-01-23 12:13:46", "2022-01-23 12:54:17"),
        "reverse_shell": ("2022-01-23 12:54:17", "2022-01-23 12:55:00"),
        "privilege_escalation": ("2022-01-23 12:55:00", "2022-01-23 12:55:28"),
        "service_stop": ("2022-01-20 22:12:27", "2022-01-20 22:12:30"),
        "dnsteal": ("2022-01-20 22:12:30", "2022-01-20 23:10:00"),
        "false_positive_test": ("2022-01-21 10:00:00", "2022-01-21 15:00:00"),
        "false_positive_same_day": ("2022-01-23 00:00:00", "2022-01-24 00:00:00"),
    },
    "wilson": {
        "network_scans": ("2022-02-07 10:57:00", "2022-02-07 10:59:13"),
        "service_scans": ("2022-02-07 10:59:13", "2022-02-07 10:59:44"),
        "dirb": ("2022-02-07 10:59:44", "2022-02-07 11:19:53"),
        "wpscan": ("2022-02-07 11:19:53", "2022-02-07 11:20:27"),
        "webshell": ("2022-02-07 11:20:27", "2022-02-07 11:20:40"),
        "cracking": ("2022-02-07 11:20:40", "2022-02-07 11:47:40"),
        "reverse_shell": ("2022-02-07 11:47:40", "2022-02-07 11:48:18"),
        "privilege_escalation": ("2022-02-07 11:48:18", "2022-02-07 11:48:55"),
        "service_stop": ("2022-02-06 10:47:15", "2022-02-06 10:47:18"),
        "dnsteal": ("2022-02-06 10:47:18", "2022-02-06 11:45:00"),
        "false_positive_test": ("2022-02-05 10:00:00", "2022-02-05 15:00:00"),
        "false_positive_same_day": ("2022-02-07 00:00:00", "2022-02-08 00:00:00"),
    },
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

    def is_in_eval_phase(self, start_ts: float, end_ts: float, phase_name: str = "false_positive_test") -> bool:
        """Return True if [start_ts, end_ts) overlaps the specified benign evaluation interval."""
        for name, iv_start, iv_end in self.eval_intervals:
            if name == phase_name and start_ts >= iv_start and end_ts <= iv_end:
                return True
        return False
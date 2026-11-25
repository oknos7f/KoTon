import re
from typing import Optional, Tuple

AUDIT_HEADER_PATTERN = re.compile(r"msg=audit\((\d+\.\d+):(\d+)\):")


def extract_audit_meta(line: str) -> Tuple[Optional[float], Optional[str]]:
    """
    auditd 로그 라인에서 밀리초 정밀도의 Unix timestamp(float)와
    멀티라인 병합용 serial_id(str)를 동시에 추출합니다.
    """
    match = AUDIT_HEADER_PATTERN.search(line)
    if not match:
        return None, None
    return float(match.group(1)), match.group(2)


def strip_audit_timestamp(line: str) -> str:
    """
    Data Leakage 방지를 위해 로그 라인에서 절대 타임스탬프와 serial_id를 제거합니다.
    """
    return AUDIT_HEADER_PATTERN.sub("msg=audit():", line, count=1)
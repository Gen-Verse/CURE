import json
import os
from typing import Any


def get_env_path(name: str) -> str:
    return os.environ.get(name, "").strip()


def get_exp_dir() -> str:
    return get_env_path("CURE_EXP_DIR")


def get_report_dir() -> str:
    report_dir = get_env_path("CURE_REPORT_DIR")
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    return report_dir


def get_raw_report_dir() -> str:
    report_dir = get_report_dir()
    if not report_dir:
        return ""
    raw_dir = os.path.join(report_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    return raw_dir


def get_current_step(default: int = 0) -> int:
    raw = get_env_path("CURE_STEP")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_profile_name(default: str = "default") -> str:
    return get_env_path("CURE_PROFILE") or default


def get_log_path(filename: str) -> str:
    raw_dir = get_raw_report_dir()
    if not raw_dir:
        return ""
    return os.path.join(raw_dir, filename)


def append_jsonl(filename: str, record: dict[str, Any]) -> str:
    path = get_log_path(filename)
    if not path:
        return ""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_json(filename: str, payload: Any) -> str:
    path = get_log_path(filename)
    if not path:
        return ""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path

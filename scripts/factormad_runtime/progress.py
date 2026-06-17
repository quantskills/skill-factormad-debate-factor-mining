"""Small stdout progress helpers for long-running local skills."""

from __future__ import annotations

import os
from typing import Any

PROGRESS_ENV = "PANDA_SKILL_SHOW_PROGRESS"
_FALSE_VALUES = {"false", "0", "no", "n", "off", "disabled"}


def _to_bool(value: Any, *, default: bool = True) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_VALUES
    return bool(value)


def progress_enabled(payload: dict[str, Any], *, default: bool = True) -> bool:
    env_default = _to_bool(os.getenv(PROGRESS_ENV, ""), default=default)
    return _to_bool(payload.get("show_progress", None), default=env_default)


def print_progress(label: str, done: int, total: int, *, enabled: bool = True) -> None:
    if not enabled or total <= 1:
        return
    done = max(0, min(done, total))
    width = 24
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{label}] |{bar}| {done}/{total}", end="", flush=True)
    if done == total:
        print(flush=True)

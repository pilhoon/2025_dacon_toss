from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
import datetime as _dt


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_md_entry(
    log_path: str | Path,
    step: str,
    brief: str,
    bullets: Iterable[str] | None = None,
) -> None:
    """Append a concise markdown log entry.

    - line 1: timestamp | step | brief
    - optional bullets: one per line prefixed by "  - "
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{_ts()} | {step} | {brief}\n")
        if bullets:
            for b in bullets:
                f.write(f"  - {b}\n")
        f.write("\n")



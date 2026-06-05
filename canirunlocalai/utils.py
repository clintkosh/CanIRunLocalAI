from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any


def gb_from_bytes(value: int | float | None) -> float | None:
    if value is None:
        return None
    try:
        if value <= 0:
            return None
        return round(float(value) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return None


def gb_from_mb(value: int | float | str | None) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1024, 2)
    except (TypeError, ValueError):
        return None


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_command(command: list[str], timeout: int = 8) -> tuple[int, str, str]:
    """Run a command safely and return rc/stdout/stderr.

    This project intentionally uses small, read-only system commands. Timeouts
    keep the scanner from hanging on broken vendor utilities, because GPUs are
    apparently tiny dramatic actors.
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"Command not found: {command[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout}s: {' '.join(command)}"
    except Exception as exc:  # pragma: no cover - defensive boundary
        return 1, "", f"Command failed: {exc}"


def parse_json_maybe(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None

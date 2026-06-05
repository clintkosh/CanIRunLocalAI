from __future__ import annotations

import json
import os
import platform
import re
import socket
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError

import psutil

from .utils import command_exists, gb_from_bytes, gb_from_mb, parse_json_maybe, run_command, safe_float


def scan_device(redact: bool = False) -> dict[str, Any]:
    """Return a privacy-conscious hardware and local-AI runtime snapshot."""
    system = _scan_system(redact=redact)
    cpu = _scan_cpu()
    memory = _scan_memory()
    storage = _scan_storage(redact=redact)
    gpus = _scan_gpus()
    runtimes = _scan_runtimes()

    return {
        "metadata": {
            "tool": "CanIRunLocalAI",
            "schema_version": "0.1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "redacted": redact,
        },
        "system": system,
        "cpu": cpu,
        "memory": memory,
        "storage": storage,
        "gpus": gpus,
        "runtimes": runtimes,
        "notes": _make_scan_notes(gpus=gpus, runtimes=runtimes),
    }


def _scan_system(redact: bool) -> dict[str, Any]:
    data = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    if not redact:
        data["hostname"] = socket.gethostname()
        data["user"] = os.environ.get("USERNAME") or os.environ.get("USER")
    return data


def _scan_cpu() -> dict[str, Any]:
    freq = psutil.cpu_freq()
    brand = platform.processor() or platform.uname().processor or "Unknown CPU"
    return {
        "brand": brand.strip() or "Unknown CPU",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_frequency_mhz": round(freq.max, 2) if freq and freq.max else None,
        "current_frequency_mhz": round(freq.current, 2) if freq and freq.current else None,
    }


def _scan_memory() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        swap = psutil.swap_memory()
    return {
        "total_gb": gb_from_bytes(vm.total),
        "available_gb": gb_from_bytes(vm.available),
        "used_percent": vm.percent,
        "swap_total_gb": gb_from_bytes(swap.total),
        "swap_used_percent": swap.percent,
    }


def _scan_storage(redact: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, FileNotFoundError, OSError):
            continue
        mount = part.mountpoint
        device = part.device
        if redact:
            mount = _redact_path(mount)
            device = _redact_path(device)
        rows.append(
            {
                "device": device,
                "mountpoint": mount,
                "fstype": part.fstype,
                "total_gb": gb_from_bytes(usage.total),
                "free_gb": gb_from_bytes(usage.free),
                "used_percent": usage.percent,
            }
        )
    return rows


def _redact_path(path: str) -> str:
    if not path:
        return path
    # Keep only drive/root-level info so people can share reports without
    # accidentally publishing usernames or project folder names.
    if re.match(r"^[A-Za-z]:\\", path):
        return path[:3] + "..."
    p = Path(path)
    if path.startswith("/"):
        return "/..." if len(p.parts) > 1 else path
    return "[redacted]"


def _scan_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    gpus.extend(_scan_nvidia_smi())

    # OS-specific fallback catches integrated GPUs and non-NVIDIA devices.
    system = platform.system().lower()
    if system == "windows":
        gpus.extend(_scan_windows_video_controllers(existing_names={g.get("name") for g in gpus}))
    elif system == "darwin":
        gpus.extend(_scan_macos_gpus(existing_names={g.get("name") for g in gpus}))
    elif system == "linux":
        gpus.extend(_scan_linux_lspci(existing_names={g.get("name") for g in gpus}))

    return _dedupe_gpus(gpus)


def _scan_nvidia_smi() -> list[dict[str, Any]]:
    if not command_exists("nvidia-smi"):
        return []

    rc, query_out, err = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=8,
    )
    if rc != 0 or not query_out:
        return [
            {
                "name": "NVIDIA GPU detected, but nvidia-smi query failed",
                "vendor": "NVIDIA",
                "vram_total_gb": None,
                "vram_free_gb": None,
                "driver_version": None,
                "backend_hint": "CUDA",
                "notes": [err or "nvidia-smi returned no GPU rows"],
            }
        ]

    _, smi_full, _ = run_command(["nvidia-smi"], timeout=8)
    cuda_version = _parse_cuda_version(smi_full)

    gpus: list[dict[str, Any]] = []
    for line in query_out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 4:
            continue
        name, total_mb, free_mb, driver = cells[:4]
        gpus.append(
            {
                "name": name,
                "vendor": "NVIDIA",
                "vram_total_gb": gb_from_mb(total_mb),
                "vram_free_gb": gb_from_mb(free_mb),
                "driver_version": driver,
                "cuda_version_reported_by_nvidia_smi": cuda_version,
                "backend_hint": "CUDA",
                "source": "nvidia-smi",
            }
        )
    return gpus


def _parse_cuda_version(nvidia_smi_text: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_text or "")
    return match.group(1) if match else None


def _scan_windows_video_controllers(existing_names: set[str | None]) -> list[dict[str, Any]]:
    if not command_exists("powershell"):
        return []

    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | ConvertTo-Json -Depth 3"
    )
    rc, out, _ = run_command(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        timeout=10,
    )
    if rc != 0 or not out:
        return []

    raw = parse_json_maybe(out)
    if not raw:
        return []
    rows = raw if isinstance(raw, list) else [raw]

    results: list[dict[str, Any]] = []
    for row in rows:
        name = (row.get("Name") or "Unknown GPU").strip()
        if name in existing_names:
            continue
        vendor = _guess_gpu_vendor(name)
        adapter_ram = row.get("AdapterRAM")
        vram = gb_from_bytes(int(adapter_ram)) if str(adapter_ram).isdigit() else None
        # Win32_VideoController AdapterRAM is often capped/wrong on modern GPUs.
        notes = []
        if vram is not None:
            notes.append("Windows AdapterRAM can be inaccurate on some GPUs; nvidia-smi is preferred for NVIDIA VRAM.")
        results.append(
            {
                "name": name,
                "vendor": vendor,
                "vram_total_gb": vram,
                "vram_free_gb": None,
                "driver_version": row.get("DriverVersion"),
                "backend_hint": _backend_for_vendor(vendor, platform.system()),
                "source": "Win32_VideoController",
                "notes": notes,
            }
        )
    return results


def _scan_macos_gpus(existing_names: set[str | None]) -> list[dict[str, Any]]:
    if not command_exists("system_profiler"):
        return []
    rc, out, _ = run_command(["system_profiler", "SPDisplaysDataType"], timeout=10)
    if rc != 0 or not out:
        return []

    gpus: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            if current:
                gpus.append(current)
            name = stripped.split(":", 1)[1].strip()
            current = {
                "name": name,
                "vendor": _guess_gpu_vendor(name),
                "vram_total_gb": None,
                "vram_free_gb": None,
                "driver_version": None,
                "backend_hint": "Metal",
                "source": "system_profiler",
            }
        elif current and stripped.startswith("VRAM"):
            value = stripped.split(":", 1)[1].strip()
            current["vram_total_gb"] = _parse_vram_string(value)
    if current:
        gpus.append(current)
    return [gpu for gpu in gpus if gpu.get("name") not in existing_names]


def _scan_linux_lspci(existing_names: set[str | None]) -> list[dict[str, Any]]:
    if not command_exists("lspci"):
        return []
    rc, out, _ = run_command(["lspci"], timeout=8)
    if rc != 0 or not out:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        lower = line.lower()
        if "vga compatible controller" not in lower and "3d controller" not in lower and "display controller" not in lower:
            continue
        name = line.split(": ", 1)[-1].strip()
        if name in existing_names:
            continue
        vendor = _guess_gpu_vendor(name)
        gpus.append(
            {
                "name": name,
                "vendor": vendor,
                "vram_total_gb": None,
                "vram_free_gb": None,
                "driver_version": None,
                "backend_hint": _backend_for_vendor(vendor, platform.system()),
                "source": "lspci",
                "notes": ["lspci usually cannot report VRAM. Use vendor tools for precise GPU memory."],
            }
        )
    return gpus


def _parse_vram_string(value: str) -> float | None:
    # Examples: "8 GB", "1536 MB", "Built-In"
    match = re.search(r"([0-9.]+)\s*(GB|MB)", value, flags=re.I)
    if not match:
        return None
    amount = safe_float(match.group(1))
    unit = match.group(2).upper()
    if amount is None:
        return None
    return round(amount if unit == "GB" else amount / 1024, 2)


def _guess_gpu_vendor(name: str | None) -> str:
    lower = (name or "").lower()
    if "nvidia" in lower or "geforce" in lower or "rtx" in lower or "quadro" in lower:
        return "NVIDIA"
    if "amd" in lower or "radeon" in lower or "advanced micro devices" in lower:
        return "AMD"
    if "intel" in lower or "iris" in lower or "uhd" in lower:
        return "Intel"
    if "apple" in lower or "m1" in lower or "m2" in lower or "m3" in lower or "m4" in lower:
        return "Apple"
    return "Unknown"


def _backend_for_vendor(vendor: str, os_name: str) -> str:
    os_lower = os_name.lower()
    if vendor == "NVIDIA":
        return "CUDA"
    if vendor == "AMD":
        return "ROCm or Vulkan"
    if vendor == "Intel":
        return "Vulkan or CPU fallback"
    if vendor == "Apple" or os_lower == "darwin":
        return "Metal"
    return "CPU fallback"


def _dedupe_gpus(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for gpu in gpus:
        key = (gpu.get("name") or "unknown").lower().strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gpu)
    return deduped


def _scan_runtimes() -> dict[str, Any]:
    return {
        "ollama": _scan_ollama(),
        "docker": _scan_docker(),
        "python": {"version": sys.version.split()[0]},
    }


def _scan_ollama() -> dict[str, Any]:
    installed = command_exists("ollama")
    data: dict[str, Any] = {"installed": installed}
    if installed:
        rc, out, err = run_command(["ollama", "--version"], timeout=8)
        data["version_output"] = out if rc == 0 else err
    try:
        with urlopen("http://localhost:11434/api/version", timeout=1.5) as response:
            body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            data["server_running"] = True
            data["api_version"] = parsed.get("version")
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        data["server_running"] = False
    return data


def _scan_docker() -> dict[str, Any]:
    installed = command_exists("docker")
    data: dict[str, Any] = {"installed": installed}
    if installed:
        rc, out, err = run_command(["docker", "--version"], timeout=8)
        data["version_output"] = out if rc == 0 else err
    return data


def _make_scan_notes(gpus: list[dict[str, Any]], runtimes: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if not gpus:
        notes.append("No discrete GPU was detected. CPU-only local models may still work, but expect slower responses.")
    if not runtimes.get("ollama", {}).get("installed"):
        notes.append("Ollama was not detected. Recommendations include Ollama commands, but you need to install it first.")
    if any(g.get("vram_total_gb") is None for g in gpus):
        notes.append("At least one GPU did not report VRAM. Recommendations may be conservative.")
    return notes

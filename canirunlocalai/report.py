from __future__ import annotations

from typing import Any


def make_markdown_report(hardware: dict[str, Any], recommendations: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# CanIRunLocalAI Report")
    lines.append("")
    lines.append("Privacy note: this report is generated locally. Use `--redact` before sharing publicly.")
    lines.append("")

    lines.extend(_hardware_section(hardware))
    lines.extend(_runtime_section(hardware))
    lines.extend(_recommendation_section(recommendations))
    lines.extend(_advice_section(recommendations))
    lines.extend(_notes_section(hardware, recommendations))

    return "\n".join(lines).strip() + "\n"


def _hardware_section(hardware: dict[str, Any]) -> list[str]:
    system = hardware.get("system", {})
    cpu = hardware.get("cpu", {})
    memory = hardware.get("memory", {})
    storage = hardware.get("storage", [])
    gpus = hardware.get("gpus", [])

    lines = ["## Device summary", ""]
    rows = [
        ("OS", f"{system.get('os', 'Unknown')} {system.get('os_release', '')}".strip()),
        ("Machine", system.get("machine", "Unknown")),
        ("CPU", cpu.get("brand", "Unknown")),
        ("CPU cores", f"{cpu.get('physical_cores', 'Unknown')} physical / {cpu.get('logical_cores', 'Unknown')} logical"),
        ("RAM", f"{memory.get('total_gb', 'Unknown')} GB total / {memory.get('available_gb', 'Unknown')} GB available"),
    ]
    lines.extend(_table(["Item", "Value"], rows))
    lines.append("")

    lines.append("### GPUs")
    lines.append("")
    if gpus:
        gpu_rows = []
        for gpu in gpus:
            gpu_rows.append(
                (
                    gpu.get("name", "Unknown"),
                    gpu.get("vendor", "Unknown"),
                    _fmt_gb(gpu.get("vram_total_gb")),
                    gpu.get("backend_hint", "Unknown"),
                    gpu.get("source", "Unknown"),
                )
            )
        lines.extend(_table(["GPU", "Vendor", "VRAM", "Backend hint", "Source"], gpu_rows))
    else:
        lines.append("No GPU detected by the scanner.")
    lines.append("")

    lines.append("### Storage")
    lines.append("")
    if storage:
        storage_rows = []
        for disk in storage[:8]:
            storage_rows.append(
                (
                    disk.get("mountpoint", "Unknown"),
                    _fmt_gb(disk.get("total_gb")),
                    _fmt_gb(disk.get("free_gb")),
                    f"{disk.get('used_percent', 'Unknown')}%",
                )
            )
        lines.extend(_table(["Mount", "Total", "Free", "Used"], storage_rows))
    else:
        lines.append("No storage rows were captured.")
    lines.append("")
    return lines


def _runtime_section(hardware: dict[str, Any]) -> list[str]:
    runtimes = hardware.get("runtimes", {})
    ollama = runtimes.get("ollama", {})
    docker = runtimes.get("docker", {})
    rows = [
        ("Ollama installed", _yes_no(ollama.get("installed"))),
        ("Ollama server running", _yes_no(ollama.get("server_running"))),
        ("Ollama version", ollama.get("api_version") or ollama.get("version_output") or "Unknown"),
        ("Docker installed", _yes_no(docker.get("installed"))),
        ("Docker version", docker.get("version_output") or "Unknown"),
    ]
    return ["## Local AI runtime checks", "", *_table(["Runtime", "Status"], rows), ""]


def _recommendation_section(recommendations: dict[str, Any]) -> list[str]:
    lines = ["## Recommended local models", ""]
    lines.append(f"Use case: `{recommendations.get('use_case', 'chat')}`")
    lines.append("")
    lines.append(f"Device class: **{recommendations.get('device_class', 'Unknown')}**")
    lines.append("")

    rows = []
    for rec in recommendations.get("recommendations", []):
        rows.append(
            (
                rec.get("fit", "Unknown"),
                rec.get("model", "Unknown"),
                f"{rec.get('estimated_q4_gb', '?')} GB",
                ", ".join(rec.get("use_cases", [])[:4]),
                f"`{rec.get('install_command', '')}`",
            )
        )
    if rows:
        lines.extend(_table(["Fit", "Model", "Est. Q4", "Good for", "Try it"], rows))
    else:
        lines.append("No suitable models were found in the current catalog. Try tiny models manually or update the catalog.")
    lines.append("")

    lines.append("### Why these were picked")
    lines.append("")
    for rec in recommendations.get("recommendations", [])[:5]:
        lines.append(f"- **{rec.get('model')}**: {rec.get('reason')}")
    lines.append("")
    return lines


def _advice_section(recommendations: dict[str, Any]) -> list[str]:
    lines = ["## Practical advice", ""]
    for item in recommendations.get("advice", []):
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _notes_section(hardware: dict[str, Any], recommendations: dict[str, Any]) -> list[str]:
    lines = ["## Notes", ""]
    for note in hardware.get("notes", []):
        lines.append(f"- {note}")
    disclaimer = recommendations.get("catalog_disclaimer")
    if disclaimer:
        lines.append(f"- Catalog note: {disclaimer}")
    if len(lines) == 2:
        lines.append("- No additional notes.")
    lines.append("")
    return lines


def _table(headers: list[str], rows: list[tuple[Any, ...]]) -> list[str]:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_escape_cell(str(cell)) for cell in row) + " |")
    return [header, sep, *body]


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _fmt_gb(value: Any) -> str:
    if value is None:
        return "Unknown"
    return f"{value} GB"


def _yes_no(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"

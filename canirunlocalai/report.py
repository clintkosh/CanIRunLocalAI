from __future__ import annotations

from typing import Any


def make_markdown_report(hardware: dict[str, Any], recommendations: dict[str, Any]) -> str:
    lines: list[str] = [
        "# CanIRunLocalAI Report", "",
        "Privacy note: this report is generated locally. Use `--redact` before sharing publicly.", "",
    ]
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
    lines.extend(_table(["Item", "Value"], [
        ("OS", f"{system.get('os', 'Unknown')} {system.get('os_release', '')}".strip()),
        ("Machine", system.get("machine", "Unknown")),
        ("CPU", cpu.get("brand", "Unknown")),
        ("CPU cores", f"{cpu.get('physical_cores', 'Unknown')} physical / {cpu.get('logical_cores', 'Unknown')} logical"),
        ("RAM", f"{memory.get('total_gb', 'Unknown')} GB total / {memory.get('available_gb', 'Unknown')} GB available"),
    ]))
    lines.extend(["", "### GPUs", ""])
    if gpus:
        lines.extend(_table(["GPU", "Vendor", "VRAM", "Backend hint", "Source"], [
            (g.get("name", "Unknown"), g.get("vendor", "Unknown"), _fmt_gb(g.get("vram_total_gb")),
             g.get("backend_hint", "Unknown"), g.get("source", "Unknown")) for g in gpus
        ]))
    else:
        lines.append("No GPU detected by the scanner.")
    lines.extend(["", "### Storage", ""])
    if storage:
        lines.extend(_table(["Mount", "Total", "Free", "Used"], [
            (d.get("mountpoint", "Unknown"), _fmt_gb(d.get("total_gb")), _fmt_gb(d.get("free_gb")),
             f"{d.get('used_percent', 'Unknown')}%") for d in storage[:8]
        ]))
    else:
        lines.append("No storage rows were captured.")
    lines.append("")
    return lines


def _runtime_section(hardware: dict[str, Any]) -> list[str]:
    r = hardware.get("runtimes", {})
    rows = [
        ("Ollama", _runtime_value(r.get("ollama", {}))),
        ("llama.cpp", _runtime_value(r.get("llama_cpp", {}))),
        ("llama-swap", _runtime_value(r.get("llama_swap", {}))),
        ("bitnet.cpp", _runtime_value(r.get("bitnet", {}))),
        ("Docker", _runtime_value(r.get("docker", {}))),
    ]
    lines = ["## Local AI runtime checks", "", *_table(["Runtime", "Status"], rows), ""]
    loaded = r.get("ollama", {}).get("loaded_models") or []
    if loaded:
        headers = sorted({key for row in loaded for key in row})
        lines.extend(["### Ollama loaded models", ""])
        lines.extend(_table(headers, [tuple(row.get(h, "") for h in headers) for row in loaded]))
        lines.append("")
    return lines


def _runtime_value(data: dict[str, Any]) -> str:
    if not data.get("installed"):
        return "Not detected"
    version = data.get("api_version") or data.get("version_output") or data.get("command") or "Detected"
    if data.get("server_running") is True:
        return f"Running — {version}"
    return str(version).replace("\n", " ")[:160]


def _recommendation_section(recommendations: dict[str, Any]) -> list[str]:
    lines = [
        "## Recommended local models", "",
        f"Use case: `{recommendations.get('use_case', 'chat')}`", "",
        f"Objective: `{recommendations.get('objective', 'balanced')}`", "",
        f"Device class: **{recommendations.get('device_class', 'Unknown')}**", "",
    ]
    rows = []
    for rec in recommendations.get("recommendations", []):
        rows.append((
            rec.get("fit", "Unknown"), rec.get("model", "Unknown"),
            f"{rec.get('estimated_q4_gb', '?')} GB", rec.get("recommended_context", "?"),
            " → ".join(rec.get("runtime_priority", [])[:2]),
            f"`{rec.get('install_command', '')}`",
        ))
    if rows:
        lines.extend(_table(["Fit", "Model", "Est. Q4", "Start ctx", "Runtime order", "Try it"], rows))
    else:
        lines.append("No suitable models were found in the current catalog.")
    lines.extend(["", "### Why these were picked", ""])
    for rec in recommendations.get("recommendations", [])[:5]:
        lines.append(f"- **{rec.get('model')}**: {rec.get('reason')}")
    lines.append("")
    return lines


def _advice_section(recommendations: dict[str, Any]) -> list[str]:
    lines = ["## Practical advice", ""]
    lines.extend(f"- {item}" for item in recommendations.get("advice", []))
    lines.append("")
    return lines


def _notes_section(hardware: dict[str, Any], recommendations: dict[str, Any]) -> list[str]:
    lines = ["## Notes", ""]
    lines.extend(f"- {note}" for note in hardware.get("notes", []))
    if recommendations.get("catalog_disclaimer"):
        lines.append(f"- Catalog note: {recommendations['catalog_disclaimer']}")
    if len(lines) == 2:
        lines.append("- No additional notes.")
    lines.append("")
    return lines


def _table(headers: list[str], rows: list[tuple[Any, ...]]) -> list[str]:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    return [header, sep, *[
        "| " + " | ".join(_escape_cell(str(cell)) for cell in row) + " |" for row in rows
    ]]


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _fmt_gb(value: Any) -> str:
    return "Unknown" if value is None else f"{value} GB"

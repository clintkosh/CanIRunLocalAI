from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class FitBand:
    label: str
    score: int
    explanation: str


FIT_GREAT = FitBand("great_fit", 100, "Should run well with GPU acceleration at normal context settings.")
FIT_WORKS = FitBand("works", 80, "Should run, but context length or multitasking may need limits.")
FIT_OFFLOAD = FitBand("slow_or_offload", 55, "May run with CPU/RAM offload, but expect slower responses.")
FIT_CPU = FitBand("cpu_only", 45, "Can run CPU-only, but speed may be poor on larger prompts.")
FIT_TOO_HEAVY = FitBand("not_recommended", 10, "Not recommended for this device without smaller quantization, cloud, or stronger hardware.")


def load_catalog() -> dict[str, Any]:
    catalog_path = files("canirunlocalai").joinpath("models_catalog.json")
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def recommend_models(
    hardware: dict[str, Any],
    use_case: str = "chat",
    limit: int = 8,
    include_not_recommended: bool = False,
) -> dict[str, Any]:
    catalog = load_catalog()
    scored = [score_model(model, hardware, use_case=use_case) for model in catalog["models"]]
    if not include_not_recommended:
        scored = [row for row in scored if row["fit"] != "not_recommended"]

    scored.sort(key=lambda row: (row["score"], row["quality_tier"], -row["estimated_q4_gb"]), reverse=True)

    return {
        "use_case": use_case,
        "catalog_version": catalog.get("catalog_version"),
        "catalog_disclaimer": catalog.get("disclaimer"),
        "device_class": classify_device(hardware),
        "recommendations": scored[:limit],
        "all_considered_count": len(catalog["models"]),
        "advice": make_advice(hardware),
    }


def score_model(model: dict[str, Any], hardware: dict[str, Any], use_case: str = "chat") -> dict[str, Any]:
    ram_total = _num(hardware.get("memory", {}).get("total_gb"), 0)
    ram_available = _num(hardware.get("memory", {}).get("available_gb"), ram_total * 0.6)
    best_gpu = best_gpu_by_vram(hardware)
    vram_total = _num(best_gpu.get("vram_total_gb") if best_gpu else 0, 0)
    vram_free = _num(best_gpu.get("vram_free_gb") if best_gpu else None, vram_total)

    estimated = _num(model.get("estimated_q4_gb"), 999)
    min_ram = _num(model.get("min_ram_gb"), estimated + 4)
    min_vram = _num(model.get("min_vram_gb"), estimated)
    quality = int(model.get("quality_tier", 1))

    use_case_bonus = 15 if use_case in model.get("use_cases", []) else 0
    if use_case == "chat" and "chat" in model.get("use_cases", []):
        use_case_bonus += 5

    fit = _fit_band(
        estimated_q4_gb=estimated,
        min_ram_gb=min_ram,
        min_vram_gb=min_vram,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
    )

    context_warning = _context_warning(estimated, vram_total, ram_total)
    reason = _make_reason(model, fit, estimated, ram_total, vram_total, vram_free, best_gpu, context_warning)
    score = fit.score + use_case_bonus + quality

    return {
        "model": model["id"],
        "family": model.get("family"),
        "fit": fit.label,
        "score": score,
        "quality_tier": quality,
        "estimated_q4_gb": estimated,
        "use_cases": model.get("use_cases", []),
        "reason": reason,
        "install_command": f"ollama run {model['id']}",
        "notes": model.get("notes", ""),
    }


def _fit_band(
    *,
    estimated_q4_gb: float,
    min_ram_gb: float,
    min_vram_gb: float,
    ram_total_gb: float,
    ram_available_gb: float,
    vram_total_gb: float,
    vram_free_gb: float,
) -> FitBand:
    # Give the OS and KV cache breathing room. Nothing says "fun weekend" like
    # a model that technically loads and then turns the PC into soup.
    usable_vram_total = vram_total_gb * 0.82
    usable_vram_free = vram_free_gb * 0.92 if vram_free_gb else usable_vram_total
    usable_ram = max(ram_available_gb * 0.70, ram_total_gb * 0.45)

    if vram_total_gb > 0 and estimated_q4_gb <= usable_vram_total and estimated_q4_gb <= usable_vram_free and vram_total_gb >= min_vram_gb:
        return FIT_GREAT
    if vram_total_gb > 0 and estimated_q4_gb <= vram_total_gb * 0.95 and ram_total_gb >= min_ram_gb:
        return FIT_WORKS
    if ram_total_gb >= min_ram_gb and estimated_q4_gb <= usable_ram:
        return FIT_CPU if vram_total_gb == 0 else FIT_OFFLOAD
    if ram_total_gb >= estimated_q4_gb + 6 and estimated_q4_gb <= ram_total_gb * 0.70:
        return FIT_OFFLOAD
    return FIT_TOO_HEAVY


def best_gpu_by_vram(hardware: dict[str, Any]) -> dict[str, Any] | None:
    gpus = hardware.get("gpus") or []
    if not gpus:
        return None
    return max(gpus, key=lambda g: _num(g.get("vram_total_gb"), 0))


def classify_device(hardware: dict[str, Any]) -> str:
    ram = _num(hardware.get("memory", {}).get("total_gb"), 0)
    gpu = best_gpu_by_vram(hardware)
    vram = _num(gpu.get("vram_total_gb") if gpu else 0, 0)

    if vram >= 48 or ram >= 128:
        return "workstation/server-tier local AI box"
    if vram >= 24:
        return "high-end local AI desktop"
    if vram >= 12:
        return "strong consumer GPU system"
    if vram >= 6:
        return "entry-to-midrange GPU system"
    if ram >= 32:
        return "CPU/offload-capable system"
    if ram >= 16:
        return "basic local AI system"
    return "low-end / tiny-model system"


def make_advice(hardware: dict[str, Any]) -> list[str]:
    advice: list[str] = []
    ram = _num(hardware.get("memory", {}).get("total_gb"), 0)
    gpu = best_gpu_by_vram(hardware)
    vram = _num(gpu.get("vram_total_gb") if gpu else 0, 0)
    vendor = gpu.get("vendor") if gpu else None
    ollama = hardware.get("runtimes", {}).get("ollama", {})

    if not ollama.get("installed"):
        advice.append("Install Ollama before using the generated ollama run commands.")
    if vram == 0:
        advice.append("No reliable VRAM value was found. Prefer 1B-4B models first, then test 7B/8B if RAM allows.")
    elif vram < 6:
        advice.append("Stay near 1B-4B models for the best experience. 7B/8B may need CPU offload and patience.")
    elif vram < 12:
        advice.append("Your likely sweet spot is 4B-8B Q4 models. Keep context modest for smoother performance.")
    elif vram < 24:
        advice.append("Your likely sweet spot is 8B-14B Q4 models. Larger models may run with short context or offload.")
    else:
        advice.append("You can realistically test 27B-32B Q4-class models. Watch context length because KV cache memory still bites.")

    if ram < 16:
        advice.append("System RAM is tight. Close browsers and background apps before running local models.")
    if vendor == "AMD":
        advice.append("AMD acceleration support varies by OS, driver, and backend. Vulkan may be easier than ROCm on some Windows systems.")
    if vendor == "Intel":
        advice.append("Intel integrated GPUs usually mean CPU/Vulkan fallback. Start small and manage expectations, sadly a recurring theme.")
    return advice


def _context_warning(estimated_q4_gb: float, vram_total_gb: float, ram_total_gb: float) -> str | None:
    if vram_total_gb and estimated_q4_gb > vram_total_gb * 0.70:
        return "Large context windows may exceed VRAM."
    if not vram_total_gb and estimated_q4_gb > ram_total_gb * 0.35:
        return "CPU-only use may be slow and memory-sensitive."
    return None


def _make_reason(
    model: dict[str, Any],
    fit: FitBand,
    estimated: float,
    ram_total: float,
    vram_total: float,
    vram_free: float,
    best_gpu: dict[str, Any] | None,
    context_warning: str | None,
) -> str:
    gpu_name = best_gpu.get("name") if best_gpu else "no detected discrete GPU"
    parts = [
        fit.explanation,
        f"Estimated Q4 size is about {estimated:g} GB.",
    ]
    if vram_total:
        parts.append(f"Best detected GPU: {gpu_name} with about {vram_total:g} GB VRAM ({vram_free:g} GB reported free).")
    else:
        parts.append(f"Detected RAM: about {ram_total:g} GB, with no reliable dedicated VRAM value.")
    if context_warning:
        parts.append(context_warning)
    return " ".join(parts)


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

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
VALID_OBJECTIVES = {"balanced", "latency", "quality"}


def load_catalog() -> dict[str, Any]:
    catalog_path = files("canirunlocalai").joinpath("models_catalog.json")
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def recommend_models(hardware: dict[str, Any], use_case: str = "chat", limit: int = 8,
                     include_not_recommended: bool = False, objective: str = "balanced") -> dict[str, Any]:
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of: {', '.join(sorted(VALID_OBJECTIVES))}")
    catalog = load_catalog()
    scored = [score_model(model, hardware, use_case=use_case, objective=objective) for model in catalog["models"]]
    if not include_not_recommended:
        scored = [row for row in scored if row["fit"] != "not_recommended"]
    scored.sort(key=lambda row: (row["score"], row["quality_tier"], -row["estimated_q4_gb"]), reverse=True)
    return {
        "use_case": use_case, "objective": objective,
        "catalog_version": catalog.get("catalog_version"),
        "catalog_disclaimer": catalog.get("disclaimer"),
        "device_class": classify_device(hardware),
        "recommendations": scored[:limit],
        "all_considered_count": len(catalog["models"]),
        "advice": make_advice(hardware, objective=objective),
    }


def score_model(model: dict[str, Any], hardware: dict[str, Any], use_case: str = "chat",
                objective: str = "balanced") -> dict[str, Any]:
    ram_total = _num(hardware.get("memory", {}).get("total_gb"), 0)
    ram_available = _num(hardware.get("memory", {}).get("available_gb"), ram_total * 0.6)
    best_gpu = best_gpu_by_vram(hardware)
    vram_total = _num(best_gpu.get("vram_total_gb") if best_gpu else 0, 0)
    vram_free = _num(best_gpu.get("vram_free_gb") if best_gpu else None, vram_total)

    estimated = _num(model.get("estimated_q4_gb"), 999)
    min_ram = _num(model.get("min_ram_gb"), estimated + 4)
    min_vram = _num(model.get("min_vram_gb"), estimated)
    quality = int(model.get("quality_tier", 1))
    params_b = _num(model.get("params_b"), 1)

    use_case_bonus = 15 if use_case in model.get("use_cases", []) else 0
    if use_case == "chat" and "chat" in model.get("use_cases", []):
        use_case_bonus += 5

    fit = _fit_band(
        estimated_q4_gb=estimated, min_ram_gb=min_ram, min_vram_gb=min_vram,
        ram_total_gb=ram_total, ram_available_gb=ram_available,
        vram_total_gb=vram_total, vram_free_gb=vram_free,
    )
    score = fit.score + use_case_bonus + quality + _objective_bonus(objective, params_b=params_b, quality=quality, fit=fit)
    warning = _context_warning(estimated, vram_total, ram_total)
    return {
        "model": model["id"], "family": model.get("family"), "fit": fit.label, "score": score,
        "quality_tier": quality, "estimated_q4_gb": estimated, "params_b": params_b,
        "use_cases": model.get("use_cases", []),
        "runtime_priority": model.get("runtime_priority", ["llama.cpp", "ollama"]),
        "recommended_context": model.get("recommended_context", 8192),
        "reason": _make_reason(model, fit, estimated, ram_total, vram_total, vram_free, best_gpu, warning, objective),
        "install_command": f"ollama run {model.get('ollama_id') or model['id']}",
        "notes": model.get("notes", ""),
    }


def _objective_bonus(objective: str, *, params_b: float, quality: int, fit: FitBand) -> int:
    if objective == "latency":
        return 38 - min(int(params_b * 2.2), 44) + (12 if fit.label == "great_fit" else 0)
    if objective == "quality":
        return quality * 5
    return quality * 2


def _fit_band(*, estimated_q4_gb: float, min_ram_gb: float, min_vram_gb: float,
              ram_total_gb: float, ram_available_gb: float, vram_total_gb: float,
              vram_free_gb: float) -> FitBand:
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
    return max(gpus, key=lambda g: _num(g.get("vram_total_gb"), 0)) if gpus else None


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


def make_advice(hardware: dict[str, Any], objective: str = "balanced") -> list[str]:
    advice: list[str] = []
    ram = _num(hardware.get("memory", {}).get("total_gb"), 0)
    gpu = best_gpu_by_vram(hardware)
    vram = _num(gpu.get("vram_total_gb") if gpu else 0, 0)
    vendor = gpu.get("vendor") if gpu else None
    runtimes = hardware.get("runtimes", {})
    ollama = runtimes.get("ollama", {})
    llama_cpp = runtimes.get("llama_cpp", {})

    if not ollama.get("installed") and not llama_cpp.get("installed"):
        advice.append("Install Ollama for convenience or llama.cpp for a tunable performance baseline before benchmarking.")
    elif ollama.get("installed") and not llama_cpp.get("installed"):
        advice.append("Add native llama.cpp and benchmark the same model against Ollama before changing your default runtime.")
    if objective == "latency":
        advice.append("Use a warm resident model, disable optional thinking for quick-turn requests, and measure time-to-first-token separately from decode speed.")
    if vram == 0:
        advice.append("No reliable dedicated VRAM was found. For low latency, start at 0.8B-4B and compare CPU vs Vulkan where supported.")
    elif vram < 6:
        advice.append("For low latency, stay near 1B-4B Q4-class models and keep context modest.")
    elif vram < 8:
        advice.append("A 6-8 GB GPU is usually fastest with 2B-4B Q4-class weights fully resident. Treat 8B/9B as a benchmarked quality lane.")
    elif vram < 12:
        advice.append("Your likely sweet spot is 4B-8B Q4 models. Keep context modest for smoother performance.")
    elif vram < 24:
        advice.append("Your likely sweet spot is 8B-14B Q4 models. Larger models may need offload.")
    else:
        advice.append("You can realistically test 27B-32B Q4-class models. Context/KV cache still consumes memory.")
    if ram < 16:
        advice.append("System RAM is tight. Close background apps before benchmarking.")
    if vendor == "AMD":
        advice.append("On Windows AMD systems, A/B test native CPU and Vulkan instead of assuming the iGPU path is faster.")
    if vendor == "Intel":
        advice.append("Intel integrated GPUs often need Vulkan or CPU fallback. Benchmark both.")
    return advice


def _context_warning(estimated_q4_gb: float, vram_total_gb: float, ram_total_gb: float) -> str | None:
    if vram_total_gb and estimated_q4_gb > vram_total_gb * 0.70:
        return "Large context windows may exceed VRAM; start at 4K-8K and expand only when needed."
    if not vram_total_gb and estimated_q4_gb > ram_total_gb * 0.35:
        return "CPU-only use may be slow and memory-sensitive."
    return None


def _make_reason(model: dict[str, Any], fit: FitBand, estimated: float, ram_total: float,
                 vram_total: float, vram_free: float, best_gpu: dict[str, Any] | None,
                 context_warning: str | None, objective: str) -> str:
    gpu_name = best_gpu.get("name") if best_gpu else "no detected discrete GPU"
    parts = [fit.explanation, f"Estimated Q4 weight size is about {estimated:g} GB."]
    if vram_total:
        parts.append(f"Best detected GPU: {gpu_name} with about {vram_total:g} GB VRAM ({vram_free:g} GB reported free).")
    else:
        parts.append(f"Detected RAM: about {ram_total:g} GB, with no reliable dedicated VRAM value.")
    if objective == "latency":
        parts.append("Latency scoring favors smaller fully-resident models over maximum parameter count.")
    if context_warning:
        parts.append(context_warning)
    return " ".join(parts)


def _num(value: Any, default: float) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default

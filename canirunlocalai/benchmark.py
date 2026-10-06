from __future__ import annotations

import json
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def benchmark_runtime(*, runtime: str, model: str, endpoint: str | None = None,
                      prompt: str = "Reply with exactly: local inference benchmark ok",
                      runs: int = 3, max_tokens: int = 96, warmup: bool = True,
                      timeout: float = 120.0, think: bool = False) -> dict[str, Any]:
    runtime = runtime.lower().strip()
    if runtime not in {"ollama", "openai"}:
        raise ValueError("runtime must be 'ollama' or 'openai'")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    endpoint = (endpoint or ("http://127.0.0.1:11434" if runtime == "ollama" else "http://127.0.0.1:8080")).rstrip("/")
    _require_loopback(endpoint)

    runner = _ollama_once if runtime == "ollama" else _openai_once
    if warmup:
        runner(endpoint, model, "Reply with OK.", min(max_tokens, 16), timeout, think)
    samples = []
    for index in range(runs):
        sample = runner(endpoint, model, prompt, max_tokens, timeout, think)
        sample["run"] = index + 1
        samples.append(sample)
    return {
        "runtime": runtime, "endpoint": endpoint, "model": model, "runs": runs,
        "warmup": warmup, "prompt": prompt, "max_tokens": max_tokens,
        "think": think if runtime == "ollama" else None,
        "samples": samples, "summary": summarize_samples(samples),
    }


def _ollama_once(endpoint: str, model: str, prompt: str, max_tokens: int,
                 timeout: float, think: bool) -> dict[str, Any]:
    payload = {
        "model": model, "prompt": prompt, "stream": True, "think": think,
        "keep_alive": -1, "options": {"temperature": 0, "num_predict": max_tokens},
    }
    req = Request(f"{endpoint}/api/generate", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first = None
    final: dict[str, Any] = {}
    chars = 0
    try:
        with urlopen(req, timeout=timeout) as response:
            for raw in response:
                if not raw.strip():
                    continue
                row = json.loads(raw.decode("utf-8", errors="replace"))
                chunk = (row.get("response") or "") + (row.get("thinking") or "")
                if chunk and first is None:
                    first = time.perf_counter()
                chars += len(row.get("response") or "")
                if row.get("done"):
                    final = row
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama benchmark failed: {exc}") from exc
    end = time.perf_counter()
    return _ollama_sample(start, first, end, final, chars)


def _openai_once(endpoint: str, model: str, prompt: str, max_tokens: int,
                 timeout: float, think: bool = False) -> dict[str, Any]:
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = Request(f"{endpoint}/v1/chat/completions", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first = None
    chars = 0
    usage: dict[str, Any] = {}
    try:
        with urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row.get("usage"), dict):
                    usage = row["usage"]
                for choice in row.get("choices") or []:
                    delta = choice.get("delta") or {}
                    chunk = delta.get("content") or delta.get("reasoning_content") or ""
                    if chunk and first is None:
                        first = time.perf_counter()
                    chars += len(chunk)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"OpenAI-compatible benchmark failed: {exc}") from exc
    end = time.perf_counter()
    completion = _number(usage.get("completion_tokens"))
    elapsed = max(end - (first or end), 0.0)
    return {
        "ttft_ms": round(((first or end) - start) * 1000, 2),
        "wall_ms": round((end - start) * 1000, 2),
        "load_ms": None, "server_total_ms": None,
        "prompt_tokens": _integer(_number(usage.get("prompt_tokens"))),
        "output_tokens": _integer(completion), "prompt_tps": None,
        "decode_tps": round(completion / elapsed, 2) if completion and elapsed > 0 else None,
        "output_chars": chars,
    }


def _ollama_sample(start: float, first: float | None, end: float,
                   final: dict[str, Any], output_chars: int | str) -> dict[str, Any]:
    ec, ed = _number(final.get("eval_count")), _number(final.get("eval_duration"))
    pc, pd = _number(final.get("prompt_eval_count")), _number(final.get("prompt_eval_duration"))
    return {
        "ttft_ms": round(((first or end) - start) * 1000, 2),
        "wall_ms": round((end - start) * 1000, 2),
        "load_ms": _ns_ms(final.get("load_duration")),
        "server_total_ms": _ns_ms(final.get("total_duration")),
        "prompt_tokens": _integer(pc), "output_tokens": _integer(ec),
        "prompt_tps": _rate(pc, pd), "decode_tps": _rate(ec, ed),
        "output_chars": len(output_chars) if isinstance(output_chars, str) else output_chars,
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {f"median_{key}": _median(samples, key) for key in
            ("ttft_ms", "wall_ms", "load_ms", "prompt_tps", "decode_tps")}


def _require_loopback(endpoint: str) -> None:
    host = (urlparse(endpoint).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("benchmark endpoint must be localhost/loopback")


def _median(samples: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in samples if row.get(key) is not None]
    return round(statistics.median(vals), 2) if vals else None


def _rate(count: float | None, ns: float | None) -> float | None:
    return round(count / (ns / 1e9), 2) if count and ns and ns > 0 else None


def _ns_ms(value: Any) -> float | None:
    n = _number(value)
    return round(n / 1e6, 2) if n is not None else None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: float | None) -> int | None:
    return int(value) if value is not None else None

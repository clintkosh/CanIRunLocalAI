# CanIRunLocalAI

**CanIRunLocalAI** is a privacy-first local AI scanner and benchmark tool. It answers two separate questions:

1. What models should this hardware be able to run?
2. Which model/runtime combination is actually fastest on this machine?

Version 0.2 adds a latency objective, current small-model candidates, runtime discovery, Ollama model/offload inventory, and real API-path benchmarking for Ollama and OpenAI-compatible local servers such as llama.cpp or llama-swap.

## Fast assessment

```bash
cirla doctor --objective latency --use-case chat --redact
```

The scanner checks CPU, RAM, storage, GPU/VRAM, Ollama, llama.cpp, llama-swap, bitnet.cpp hints, Docker, installed Ollama models, and currently loaded/offloaded Ollama models.

Generate Markdown and JSON reports:

```bash
cirla scan --objective latency --use-case chat --redact
```

## Benchmark the path users actually feel

Ollama, warm-model latency with thinking disabled by default:

```bash
cirla bench --runtime ollama --model qwen3.5:4b --runs 5 --out reports/qwen35-4b-ollama.json
```

A local llama.cpp or llama-swap OpenAI-compatible endpoint:

```bash
cirla bench --runtime openai --endpoint http://127.0.0.1:8080 --model qwen3.5-4b --runs 5 --out reports/qwen35-4b-llamacpp.json
```

For privacy, benchmark endpoints are restricted to localhost/loopback. The benchmark reports median time-to-first-token, wall time, and token rates when exposed by the runtime. Ollama also reports model load time, prompt-evaluation throughput, and decode throughput.

Use `--no-warmup` to measure cold-load behavior. Use `--think` only when reasoning latency is intentionally part of the test.

## Recommendation objectives

- `latency`: rewards smaller fully-resident models and fast first response.
- `balanced`: mixes model quality and hardware fit.
- `quality`: allows larger models to win when the hardware can support them.

The October 2026 catalog starts with Qwen 3.5 and Gemma 4 candidates plus stable baselines. It is deliberately conservative: runtime bundles, multimodal projectors, context length, KV cache, and drivers add memory beyond raw Q4 weights.

## Performance workflow

1. Run `cirla doctor --objective latency`.
2. Confirm Ollama `PROCESSOR` is fully GPU where GPU inference is intended; CPU/GPU splitting can be much slower.
3. Benchmark the same model and comparable context through Ollama and native llama.cpp.
4. Keep the lowest-latency good-enough model warm as the interactive default.
5. Route harder work to a larger quality lane instead of making every request pay the larger-model cost.
6. Re-run after runtime, driver, quant, model, or context changes.

## Install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -e .
```

## Privacy

CanIRunLocalAI runs locally. It does not upload scan data. Use `--redact` before sharing reports.

## Development

```bash
pip install -e . pytest
pytest
```

## Roadmap

- [x] llama.cpp runtime detection and OpenAI-compatible benchmark path
- [x] Ollama model/load inventory
- [x] optional benchmark mode
- [x] latency-vs-quality recommendation objective
- [ ] automatic same-model Ollama vs llama.cpp tournament runner
- [ ] direct `llama-bench` orchestration for GGUF files
- [ ] richer AMD/Intel backend benchmarking
- [ ] signed Windows executable build
- [ ] GitHub Pages estimator and report comparison UI
- [ ] signed/updatable model catalog feed

## Safety and limitations

The recommendation catalog is a starting hypothesis. The benchmark is the authority for the target machine. Thermal throttling, context length, drivers, background GPU use, quantization, and runtime versions can materially change results.

## License

MIT

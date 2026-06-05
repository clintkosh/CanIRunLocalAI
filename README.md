# CanIRunLocalAI

**CanIRunLocalAI** is a privacy-first local hardware scanner that answers the question normal people actually have:

> "What local AI / LLM models can this computer realistically run?"

It scans your device, generates a hardware report, and recommends practical local models for Ollama / llama.cpp-style usage based on RAM, VRAM, GPU vendor, and use case.

No account. No telemetry. No cloud upload. No mystery hardware harvesting. The bar is low, somehow.

## What it does

- Detects OS, CPU, RAM, storage, GPU, VRAM, and driver/runtime hints.
- Checks for Ollama and Docker.
- Recommends local LLMs by realistic fit:
  - `great_fit`
  - `works`
  - `slow_or_offload`
  - `cpu_only`
  - `not_recommended`
- Generates Markdown and JSON reports.
- Supports redacted reports for sharing publicly.
- Includes direct `ollama run ...` commands.

## Current status

Alpha / starter project.

The scanner works as a practical baseline, but model recommendations should stay conservative. Local LLM performance depends on quantization, context length, GPU backend, drivers, OS, background apps, and whether your computer has decided to express itself through thermal throttling.

## Install from source

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -e .
```

## Usage

Generate both Markdown and JSON reports:

```bash
canirunlocalai scan --redact
```

Use the short alias:

```bash
cirla scan --redact
```

Focus recommendations on coding:

```bash
canirunlocalai scan --use-case coding --redact
```

Print a quick terminal report without writing files:

```bash
canirunlocalai doctor --use-case chat --redact
```

List the bundled model catalog:

```bash
canirunlocalai models
```

## Output files

By default, reports are written to:

```text
reports/canirunlocalai-report.md
reports/canirunlocalai-report.json
```

Use a custom folder:

```bash
canirunlocalai scan --out ./my-report --redact
```

## Example recommendation output

```text
Device class: entry-to-midrange GPU system
Top recommendations:
- qwen3:8b [great_fit] :: ollama run qwen3:8b
- deepseek-r1:7b [great_fit] :: ollama run deepseek-r1:7b
- llama3.1:8b [great_fit] :: ollama run llama3.1:8b
- gemma3:4b [great_fit] :: ollama run gemma3:4b
- qwen3:4b [great_fit] :: ollama run qwen3:4b
```

## Privacy model

CanIRunLocalAI runs locally. It does not upload scan data anywhere.

Use this when sharing reports:

```bash
canirunlocalai scan --redact
```

Redaction removes or trims common identifiers such as hostname, username, and detailed local paths.

## Hardware detection notes

### Windows

Windows is the primary target.

The scanner uses:

- Python `platform` and `psutil` for CPU/RAM/storage.
- `nvidia-smi` for NVIDIA GPU memory when available.
- PowerShell `Get-CimInstance Win32_VideoController` as a GPU fallback.

Note: Windows `AdapterRAM` can be inaccurate for some modern GPUs. `nvidia-smi` is preferred for NVIDIA VRAM.

### macOS

Uses `system_profiler SPDisplaysDataType` for GPU info and Python/psutil for general hardware.

Apple Silicon uses unified memory, so model-fit logic is necessarily conservative.

### Linux

Uses `nvidia-smi` if available and `lspci` as a fallback. Many Linux systems need extra tools installed before VRAM can be detected reliably.

## Recommendation logic

The bundled catalog estimates Q4-class memory requirements for common model families. Fit scoring considers:

- total system RAM
- available RAM
- detected VRAM
- GPU vendor/backend hint
- estimated model size
- use case match
- conservative headroom for OS/context/KV cache

The tool intentionally avoids saying "yes" just because a model might technically load. A model that loads but runs like a haunted fax machine is not a good recommendation.

## Bundled seed model families

- Qwen3
- Gemma 3
- DeepSeek-R1 distills
- Llama 3.1 / 3.2
- gpt-oss

The catalog lives in:

```text
canirunlocalai/models_catalog.json
```

Update that file as local model options change.

## GitHub project setup

After creating a new GitHub repo named `CanIRunLocalAI`:

```bash
git init
git add .
git commit -m "Initial CanIRunLocalAI alpha"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CanIRunLocalAI.git
git push -u origin main
```

Then edit `pyproject.toml` and replace `YOUR_USERNAME` with your GitHub username.

## Development

Run tests:

```bash
pip install -e . pytest
pytest
```

Run the CLI locally:

```bash
python -m canirunlocalai doctor --redact
```

## Roadmap

- [ ] Add richer AMD GPU detection.
- [ ] Add Apple unified memory-specific recommendation profiles.
- [ ] Add LM Studio and llama.cpp command output.
- [ ] Add optional benchmark mode.
- [ ] Add signed Windows executable build.
- [ ] Add GitHub Pages manual estimator.
- [ ] Add report import/export UI.
- [ ] Add model catalog updater script.

## Safety and limitations

This is not a benchmark suite. It is a fit advisor.

Real-world performance varies based on:

- quantization
- context length
- model backend
- GPU driver
- thermal limits
- RAM pressure
- disk speed
- running background apps

When in doubt, start with a smaller model and move up.

## License

MIT

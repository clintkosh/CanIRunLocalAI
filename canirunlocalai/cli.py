from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .benchmark import benchmark_runtime
from .recommend import load_catalog, recommend_models
from .report import make_markdown_report
from .scan import scan_device


VALID_USE_CASES = ["chat", "coding", "reasoning", "vision", "rag", "agents", "low_end", "fast", "multilingual"]
VALID_OBJECTIVES = ["balanced", "latency", "quality"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canirunlocalai",
        description="Scan, recommend, and benchmark local AI / LLM runtimes.",
    )
    parser.add_argument("--version", action="version", version=f"CanIRunLocalAI {__version__}")
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="Generate a hardware/runtime scan and model-fit report.")
    scan.add_argument("--out", default="reports")
    scan.add_argument("--format", choices=["all", "json", "md"], default="all")
    scan.add_argument("--redact", action="store_true")
    scan.add_argument("--use-case", choices=VALID_USE_CASES, default="chat")
    scan.add_argument("--objective", choices=VALID_OBJECTIVES, default="balanced")
    scan.add_argument("--limit", type=int, default=8)
    scan.add_argument("--include-not-recommended", action="store_true")
    scan.set_defaults(func=cmd_scan)

    models = sub.add_parser("models", help="List the bundled model catalog.")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    doctor = sub.add_parser("doctor", help="Quick terminal summary without writing files.")
    doctor.add_argument("--redact", action="store_true")
    doctor.add_argument("--use-case", choices=VALID_USE_CASES, default="chat")
    doctor.add_argument("--objective", choices=VALID_OBJECTIVES, default="balanced")
    doctor.add_argument("--limit", type=int, default=5)
    doctor.set_defaults(func=cmd_doctor)

    bench = sub.add_parser("bench", help="Benchmark a local runtime through its real API path.")
    bench.add_argument("--runtime", choices=["ollama", "openai"], required=True,
                       help="Use openai for llama.cpp/llama-swap/OpenAI-compatible local servers.")
    bench.add_argument("--model", required=True)
    bench.add_argument("--endpoint", help="Loopback base URL. Defaults: Ollama 11434, OpenAI-compatible 8080.")
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--max-tokens", type=int, default=96)
    bench.add_argument("--prompt", default="Reply with exactly: local inference benchmark ok")
    bench.add_argument("--timeout", type=float, default=120.0)
    bench.add_argument("--no-warmup", action="store_true")
    bench.add_argument("--think", action="store_true",
                       help="Include Ollama thinking latency. Default is disabled for latency tests.")
    bench.add_argument("--out", help="Optional JSON result path.")
    bench.set_defaults(func=cmd_bench)
    return parser


def _recommend(args: argparse.Namespace, hardware: dict[str, Any]) -> dict[str, Any]:
    return recommend_models(
        hardware,
        use_case=args.use_case,
        limit=max(args.limit, 1),
        include_not_recommended=getattr(args, "include_not_recommended", False),
        objective=args.objective,
    )


def cmd_scan(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hardware = scan_device(redact=args.redact)
    recommendations = _recommend(args, hardware)
    bundle = {"hardware": hardware, "recommendations": recommendations}

    written: list[Path] = []
    if args.format in ("all", "json"):
        path = out_dir / "canirunlocalai-report.json"
        path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        written.append(path)
    if args.format in ("all", "md"):
        path = out_dir / "canirunlocalai-report.md"
        path.write_text(make_markdown_report(hardware, recommendations), encoding="utf-8")
        written.append(path)

    print("CanIRunLocalAI scan complete.")
    for path in written:
        print(f"Wrote: {path.resolve()}")
    _print_top_recommendations(recommendations)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    hardware = scan_device(redact=args.redact)
    print(make_markdown_report(hardware, _recommend(args, hardware)))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    catalog = load_catalog()
    if args.json:
        print(json.dumps(catalog, indent=2))
        return 0
    print(f"Catalog version: {catalog.get('catalog_version')}")
    for model in catalog.get("models", []):
        print(f"- {model['id']:18} {model.get('estimated_q4_gb')} GB est. Q4 | {', '.join(model.get('use_cases', []))}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    result = benchmark_runtime(
        runtime=args.runtime,
        model=args.model,
        endpoint=args.endpoint,
        prompt=args.prompt,
        runs=args.runs,
        max_tokens=args.max_tokens,
        warmup=not args.no_warmup,
        timeout=args.timeout,
        think=args.think,
    )
    summary = result["summary"]
    print(f"Runtime: {result['runtime']}  Model: {result['model']}")
    print(f"Median TTFT: {_fmt(summary.get('median_ttft_ms'), 'ms')}")
    print(f"Median wall: {_fmt(summary.get('median_wall_ms'), 'ms')}")
    print(f"Median prompt: {_fmt(summary.get('median_prompt_tps'), 'tok/s')}")
    print(f"Median decode: {_fmt(summary.get('median_decode_tps'), 'tok/s')}")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote: {path.resolve()}")
    return 0


def _fmt(value: Any, unit: str) -> str:
    return "n/a" if value is None else f"{value} {unit}"


def _print_top_recommendations(recommendations: dict[str, Any]) -> None:
    print("")
    print(f"Device class: {recommendations.get('device_class')}")
    print(f"Objective: {recommendations.get('objective')}")
    print("Top recommendations:")
    for rec in recommendations.get("recommendations", [])[:5]:
        print(f"- {rec['model']} [{rec['fit']}] :: {rec['install_command']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .recommend import load_catalog, recommend_models
from .report import make_markdown_report
from .scan import scan_device


VALID_USE_CASES = ["chat", "coding", "reasoning", "vision", "rag", "agents", "low_end", "fast", "multilingual"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canirunlocalai",
        description="Scan local hardware and recommend realistic local AI / LLM models.",
    )
    parser.add_argument("--version", action="version", version=f"CanIRunLocalAI {__version__}")

    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="Generate a hardware scan and model-fit report.")
    scan.add_argument("--out", default="reports", help="Output directory. Default: reports")
    scan.add_argument("--format", choices=["all", "json", "md"], default="all", help="Report format. Default: all")
    scan.add_argument("--redact", action="store_true", help="Remove hostname/user/path details from the report.")
    scan.add_argument("--use-case", choices=VALID_USE_CASES, default="chat", help="Recommendation focus. Default: chat")
    scan.add_argument("--limit", type=int, default=8, help="Maximum recommendations. Default: 8")
    scan.add_argument("--include-not-recommended", action="store_true", help="Include models that likely do not fit.")
    scan.set_defaults(func=cmd_scan)

    models = sub.add_parser("models", help="List the bundled model catalog.")
    models.add_argument("--json", action="store_true", help="Print raw catalog JSON.")
    models.set_defaults(func=cmd_models)

    doctor = sub.add_parser("doctor", help="Quick terminal summary without writing files.")
    doctor.add_argument("--redact", action="store_true", help="Redact user/device identifiers.")
    doctor.add_argument("--use-case", choices=VALID_USE_CASES, default="chat")
    doctor.add_argument("--limit", type=int, default=5)
    doctor.set_defaults(func=cmd_doctor)

    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hardware = scan_device(redact=args.redact)
    recommendations = recommend_models(
        hardware,
        use_case=args.use_case,
        limit=max(args.limit, 1),
        include_not_recommended=args.include_not_recommended,
    )
    bundle = {"hardware": hardware, "recommendations": recommendations}

    written: list[Path] = []
    if args.format in ("all", "json"):
        json_path = out_dir / "canirunlocalai-report.json"
        json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        written.append(json_path)
    if args.format in ("all", "md"):
        md_path = out_dir / "canirunlocalai-report.md"
        md_path.write_text(make_markdown_report(hardware, recommendations), encoding="utf-8")
        written.append(md_path)

    print("CanIRunLocalAI scan complete.")
    for path in written:
        print(f"Wrote: {path.resolve()}")
    _print_top_recommendations(recommendations)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    hardware = scan_device(redact=args.redact)
    recommendations = recommend_models(hardware, use_case=args.use_case, limit=max(args.limit, 1))
    print(make_markdown_report(hardware, recommendations))
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


def _print_top_recommendations(recommendations: dict[str, Any]) -> None:
    print("")
    print(f"Device class: {recommendations.get('device_class')}")
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


if __name__ == "__main__":
    raise SystemExit(main())

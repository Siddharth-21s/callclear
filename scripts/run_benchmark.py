"""Run the CallClear local benchmark from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmarks.runner import (  # noqa: E402
    DEFAULT_METHODS,
    DEFAULT_SNRS,
    run_local_benchmark,
)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of floating-point SNR values."""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid SNR list: {value!r}. Use comma-separated numbers, "
            "for example: 20,10,5,0"
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError(
            "SNR list cannot be empty. Example: 20,10,5,0"
        )

    return values


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    """Parse a comma-separated list of non-empty strings."""
    values = tuple(item.strip() for item in value.split(",") if item.strip())

    if not values:
        raise argparse.ArgumentTypeError(
            "List cannot be empty. Provide comma-separated values."
        )

    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the CallClear local ASR benchmark.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of clean reference utterances to benchmark (default: 8).",
    )

    parser.add_argument(
        "--model",
        default="base",
        help="Faster-Whisper model name (default: base).",
    )

    parser.add_argument(
        "--snrs",
        type=_parse_csv_floats,
        default=DEFAULT_SNRS,
        help=(
            "Comma-separated SNR values in dB "
            "(default: 20,10,5,0)."
        ),
    )

    parser.add_argument(
        "--methods",
        type=_parse_csv_strings,
        default=DEFAULT_METHODS,
        help=(
            "Comma-separated enhancement methods "
            "(default: none,highpass,spectral_subtraction,wiener,vad)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible degradation (default: 42).",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Benchmark output directory (default: results/).",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and run the benchmark."""
    parser = build_parser()
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be greater than 0")

    args.out = args.out.expanduser()

    print("CallClear benchmark")
    print("-------------------")
    print(f"Results directory: {args.out}")
    print(f"Reference limit:   {args.limit}")
    print(f"Whisper model:     {args.model}")
    print(f"SNRs:              {', '.join(str(snr) for snr in args.snrs)}")
    print(f"Methods:           {', '.join(args.methods)}")
    print(f"Seed:              {args.seed}")
    print()
    print("Required inputs:")
    print("  - cached clean_*.wav files")
    print("  - references.json")
    print()

    run_local_benchmark(
        results_dir=args.out,
        limit=args.limit,
        model_name=args.model,
        snrs=args.snrs,
        methods=args.methods,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
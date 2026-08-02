"""Generate deterministic synthetic telemetry for Day 6 integration tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from dashguard.analysis.telemetry import generate_synthetic_telemetry_csv


def build_parser() -> argparse.ArgumentParser:
    """Create the Day 6 synthetic telemetry command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate simple deterministic synthetic vehicle telemetry for "
            "DashGuard integration testing."
        )
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--duration-sec", default=30.0, type=float)
    parser.add_argument("--frequency-hz", default=10.0, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Generate synthetic telemetry and return a process exit code."""

    try:
        summary = generate_synthetic_telemetry_csv(
            output_csv=args.output_csv,
            duration_sec=args.duration_sec,
            frequency_hz=args.frequency_hz,
            overwrite=args.overwrite,
        )
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        print(
            f"Synthetic telemetry generation failed "
            f"({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and generate synthetic telemetry."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

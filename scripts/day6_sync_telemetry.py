"""Synchronize Day 6 telemetry with Day 5 risk CSV rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from dashguard.analysis.telemetry import sync_risk_telemetry_csv


def build_parser() -> argparse.ArgumentParser:
    """Create the Day 6 risk/telemetry synchronization parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Append nearest telemetry evidence to every Day 5 risk CSV row."
        )
    )
    parser.add_argument("--risk-csv", required=True, type=Path)
    parser.add_argument("--telemetry-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--offset-sec", default=0.0, type=float)
    parser.add_argument("--tolerance-sec", default=0.05, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Synchronize risk and telemetry inputs and return a process exit code."""

    try:
        summary = sync_risk_telemetry_csv(
            risk_csv=args.risk_csv,
            telemetry_csv=args.telemetry_csv,
            output_csv=args.output_csv,
            offset_sec=args.offset_sec,
            tolerance_sec=args.tolerance_sec,
            overwrite=args.overwrite,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"Telemetry synchronization failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and synchronize telemetry."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

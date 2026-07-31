"""Generate Day 5 risk proxy scores and a frame-maximum plot."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from dashguard.analysis.risk_report import generate_video_risk_report


def build_parser() -> argparse.ArgumentParser:
    """Create the Day 5 risk report command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute video risk features and proxy scores from a Day 3 track "
            "CSV, then write the scored CSV and frame-maximum PNG plot."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-plot", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Generate the report and return a process exit code."""

    try:
        summary = generate_video_risk_report(
            input_csv=args.input_csv,
            input_video=args.input_video,
            output_csv=args.output_csv,
            output_plot=args.output_plot,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Risk report failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and generate the Day 5 report."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

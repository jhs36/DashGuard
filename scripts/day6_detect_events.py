"""Generate the Day 6 frame risk timeline and event candidate JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from dashguard.analysis.events import generate_event_outputs


def build_parser() -> argparse.ArgumentParser:
    """Create the Day 6 event detection command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Select one risk row per frame, detect and merge score-based "
            "event candidates, and write a timeline CSV plus event JSON."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output-timeline-csv", required=True, type=Path)
    parser.add_argument("--output-events-json", required=True, type=Path)
    parser.add_argument("--score-threshold", default=0.70, type=float)
    parser.add_argument(
        "--minimum-event-duration-sec",
        default=0.15,
        type=float,
    )
    parser.add_argument("--merge-gap-sec", default=0.10, type=float)
    parser.add_argument("--pre-roll-sec", default=2.0, type=float)
    parser.add_argument("--post-roll-sec", default=1.0, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Generate event outputs and return a process exit code."""

    try:
        summary = generate_event_outputs(
            input_csv=args.input_csv,
            input_video=args.input_video,
            output_timeline_csv=args.output_timeline_csv,
            output_events_json=args.output_events_json,
            score_threshold=args.score_threshold,
            minimum_event_duration_sec=args.minimum_event_duration_sec,
            merge_gap_sec=args.merge_gap_sec,
            pre_roll_sec=args.pre_roll_sec,
            post_roll_sec=args.post_roll_sec,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Event detection failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and generate Day 6 event outputs."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

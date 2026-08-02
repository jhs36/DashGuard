"""Generate validated H.264 clips for existing Day 6 events."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dashguard.video.event_clips import extract_event_clips


def build_parser() -> argparse.ArgumentParser:
    """Create the Day 6 event clip command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Read existing event JSON and create validated, rollback-safe "
            "H.264 MP4 clips plus a manifest."
        )
    )
    parser.add_argument("--events-json", required=True, type=Path)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest-json", required=True, type=Path)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Extract event clips and return a process exit code."""

    try:
        result = extract_event_clips(
            events_json=args.events_json,
            input_video=args.input_video,
            output_dir=args.output_dir,
            output_manifest_json=args.output_manifest_json,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Event clip extraction failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and extract Day 6 event clips."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

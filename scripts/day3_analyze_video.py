"""Create a tracked analysis video, CSV records, and performance metrics."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import cv2

from dashguard.analysis import AnalysisMetrics, run_tracked_video_analysis


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the Day 3 analysis command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Track every selected video frame and export an annotated video, "
            "track CSV, and measured pipeline performance."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument(
        "--trajectory-length",
        type=_positive_int,
        default=30,
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        default=None,
        help="Process at most this many consecutive frames (default: all).",
    )
    return parser


def _print_metrics(metrics: AnalysisMetrics) -> None:
    """Print every metric with stable labels."""

    print("Analysis completed.")
    for field in fields(metrics):
        value = getattr(metrics, field.name)
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"{field.name}: {rendered}")


def run(args: argparse.Namespace) -> int:
    """Run CPU tracking analysis and return a process exit code."""

    try:
        metrics = run_tracked_video_analysis(
            input_path=args.input,
            output_video_path=args.output_video,
            output_csv_path=args.output_csv,
            metrics_path=args.metrics,
            model_path=args.model,
            confidence=args.confidence,
            imgsz=args.imgsz,
            trajectory_length=args.trajectory_length,
            max_frames=args.max_frames,
            device="cpu",
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        csv.Error,
        cv2.error,
    ) as exc:
        print(
            f"Analysis failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    _print_metrics(metrics)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run the analysis."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

"""Sample target-object detections from individual video frames."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from dashguard.detection import TARGET_CLASS_NAMES, UltralyticsDetector
from dashguard.video.reader import VideoReader, VideoReaderError


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the smoke-test command-line parser."""

    parser = argparse.ArgumentParser(
        description="Sample single frames and print target detections."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument("--sample-stride", type=_positive_int, default=210)
    parser.add_argument("--max-samples", type=_positive_int, default=12)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run sampled CPU inference and return a process exit code."""

    try:
        detector = UltralyticsDetector.from_pretrained(
            model_path=args.model,
            confidence=args.confidence,
            imgsz=args.imgsz,
            device="cpu",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"Model loading failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    sample_count = 0
    detection_count = 0
    class_counts: Counter[str] = Counter()

    try:
        with VideoReader(args.input) as reader:
            for frame in reader:
                if frame.frame_index % args.sample_stride != 0:
                    continue

                detections = detector.detect(frame.image)
                sample_count += 1
                detection_count += len(detections)
                class_counts.update(
                    detection.class_name for detection in detections
                )

                print(
                    f"frame_index={frame.frame_index} "
                    f"timestamp_sec={frame.timestamp_sec:.3f} "
                    f"detection_count={len(detections)}"
                )
                for detection in detections:
                    x1, y1, x2, y2 = detection.xyxy
                    print(
                        f"  class_name={detection.class_name} "
                        f"confidence={detection.confidence:.4f} "
                        f"xyxy=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
                    )

                if sample_count >= args.max_samples:
                    break
    except (OSError, RuntimeError, ValueError, VideoReaderError) as exc:
        print(
            f"Video processing failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    counts_text = ", ".join(
        f"{class_name}={class_counts[class_name]}"
        for class_name in sorted(TARGET_CLASS_NAMES)
    )
    print(
        f"summary samples={sample_count} "
        f"target_detections={detection_count}"
    )
    print(f"class_counts {counts_text}")

    return 0 if detection_count > 0 else 2


def main() -> int:
    """Parse command-line arguments and run the smoke test."""

    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

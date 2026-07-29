"""Run consecutive-frame ByteTrack smoke testing on one video."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from dashguard.detection import TARGET_CLASS_NAMES
from dashguard.tracking import UltralyticsByteTracker
from dashguard.video.reader import VideoReader, VideoReaderError


@dataclass
class _TrackHistory:
    """Frame appearances recorded for one track ID."""

    first_frame: int
    last_frame: int
    frame_indices: set[int] = field(default_factory=set)


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the tracking smoke-test command-line parser."""

    parser = argparse.ArgumentParser(
        description="Track target objects in consecutive video frames."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument("--max-frames", type=_positive_int, default=180)
    parser.add_argument("--print-stride", type=_positive_int, default=30)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run CPU ByteTrack inference and return a process exit code."""

    try:
        tracker = UltralyticsByteTracker.from_pretrained(
            model_path=args.model,
            confidence=args.confidence,
            imgsz=args.imgsz,
            device="cpu",
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"Model loading failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    processed_frames = 0
    frames_with_tracks = 0
    total_track_records = 0
    class_counts: Counter[str] = Counter()
    track_histories: dict[int, _TrackHistory] = {}

    try:
        with VideoReader(args.input) as reader:
            for frame in reader:
                tracks = tracker.track(
                    frame.image,
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                )
                processed_frames += 1
                total_track_records += len(tracks)
                if tracks:
                    frames_with_tracks += 1

                class_counts.update(track.class_name for track in tracks)
                for track in tracks:
                    history = track_histories.get(track.track_id)
                    if history is None:
                        history = _TrackHistory(
                            first_frame=track.frame_index,
                            last_frame=track.frame_index,
                        )
                        track_histories[track.track_id] = history
                    history.first_frame = min(
                        history.first_frame,
                        track.frame_index,
                    )
                    history.last_frame = max(
                        history.last_frame,
                        track.frame_index,
                    )
                    history.frame_indices.add(track.frame_index)

                if frame.frame_index % args.print_stride == 0:
                    identities = ", ".join(
                        f"{track.track_id}:{track.class_name}"
                        for track in tracks
                    )
                    print(
                        f"frame_index={frame.frame_index} "
                        f"timestamp_sec={frame.timestamp_sec:.3f} "
                        f"track_count={len(tracks)} "
                        f"tracks=[{identities}]"
                    )

                if processed_frames >= args.max_frames:
                    break
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        VideoReaderError,
    ) as exc:
        print(
            f"Video processing failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"processed_frames={processed_frames}")
    print(f"frames_with_tracks={frames_with_tracks}")
    print(f"total_track_records={total_track_records}")
    print(f"unique_track_ids={len(track_histories)}")
    counts_text = ", ".join(
        f"{class_name}={class_counts[class_name]}"
        for class_name in sorted(TARGET_CLASS_NAMES)
    )
    print(f"class_track_counts {counts_text}")
    for track_id in sorted(track_histories):
        history = track_histories[track_id]
        print(
            f"track_id={track_id} "
            f"first_frame={history.first_frame} "
            f"last_frame={history.last_frame} "
            f"appearance_frames={len(history.frame_indices)}"
        )

    return 0 if track_histories else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run the smoke test."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

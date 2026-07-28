"""Render frame and timestamp overlays onto MP4 debug videos."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dashguard.video.reader import VideoFrame, VideoReader


@dataclass(frozen=True)
class OverlayRenderResult:
    """Summary of a completed debug-overlay render."""

    output_path: Path
    frame_count: int
    elapsed_sec: float
    effective_fps: float


class VideoOverlayError(RuntimeError):
    """Base exception for failures while rendering a video overlay."""


class EmptyVideoError(VideoOverlayError):
    """Raised when the input video contains no decoded frames."""


class VideoWriterOpenError(VideoOverlayError):
    """Raised when OpenCV cannot open the requested output writer."""


class OverlayOutputError(VideoOverlayError):
    """Raised when rendering does not produce a non-empty output file."""


def _draw_overlay(image: np.ndarray, frame: VideoFrame) -> None:
    lines = (
        (f"frame: {frame.frame_index}", (24, 40)),
        (f"time: {frame.timestamp_sec:.3f} sec", (24, 76)),
    )
    for text, origin in lines:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def render_debug_overlay(
    input_path: str | Path,
    output_path: str | Path,
) -> OverlayRenderResult:
    """Render frame indexes and timestamps into an MP4 debug video.

    Args:
        input_path: Video path passed to :class:`VideoReader`.
        output_path: Destination path with an ``.mp4`` extension.

    Returns:
        Rendering statistics and the destination path.

    Raises:
        ValueError: If the destination is not an MP4 path or resolves to the
            same path as the input.
        EmptyVideoError: If the reader does not yield a first frame.
        VideoWriterOpenError: If OpenCV cannot open the output writer.
        OverlayOutputError: If the released writer leaves no non-empty output.
    """

    source = Path(input_path)
    destination = Path(output_path)

    if source.resolve() == destination.resolve():
        raise ValueError(f"Input and output paths must be different: {source}")

    if destination.suffix.lower() != ".mp4":
        raise ValueError(f"Output path must have an .mp4 extension: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    reader = VideoReader(source)
    writer: cv2.VideoWriter | None = None
    frame_count = 0

    try:
        frame = reader.read()
        if frame is None:
            raise EmptyVideoError(f"Input video contains no frames: {source}")

        height, width = frame.image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(destination),
            fourcc,
            reader.fps,
            (width, height),
            True,
        )
        if not writer.isOpened():
            raise VideoWriterOpenError(
                f"OpenCV could not open video writer: {destination}"
            )

        while frame is not None:
            rendered_image = frame.image.copy()
            _draw_overlay(rendered_image, frame)
            writer.write(rendered_image)
            frame_count += 1
            frame = reader.read()
    finally:
        try:
            if writer is not None:
                writer.release()
        finally:
            reader.close()

    elapsed_sec = time.perf_counter() - started_at

    try:
        output_is_valid = destination.is_file() and destination.stat().st_size > 0
    except OSError as exc:
        raise OverlayOutputError(
            f"Could not verify overlay output {destination}: {exc}"
        ) from exc

    if not output_is_valid:
        raise OverlayOutputError(
            f"Overlay output is missing or empty: {destination}"
        )

    effective_fps = frame_count / elapsed_sec if elapsed_sec > 0 else 0.0
    return OverlayRenderResult(
        output_path=destination,
        frame_count=frame_count,
        elapsed_sec=elapsed_sec,
        effective_fps=effective_fps,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render frame and timestamp overlays into an MP4 debug video."
    )
    parser.add_argument("input_path", type=Path, help="input video file")
    parser.add_argument("output_path", type=Path, help="output .mp4 file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return its process exit code."""

    args = _build_parser().parse_args(argv)
    result = render_debug_overlay(args.input_path, args.output_path)
    print(f"frames        : {result.frame_count}")
    print(f"elapsed_sec   : {result.elapsed_sec:.6f}")
    print(f"effective_fps : {result.effective_fps:.2f}")
    print(f"output_path   : {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

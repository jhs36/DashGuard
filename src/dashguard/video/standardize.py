"""Standardize video files by invoking FFmpeg as a subprocess."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path


_VIDEO_FILTER = (
    "scale=w='min(1280,iw)':h='min(720,ih)':"
    "force_original_aspect_ratio=decrease:force_divisible_by=2,fps=30"
)


class VideoStandardizationError(RuntimeError):
    """Base exception for failures while standardizing a video."""


class FFmpegNotFoundError(VideoStandardizationError):
    """Raised when the configured FFmpeg executable cannot be found."""


class FFmpegExecutionError(VideoStandardizationError):
    """Raised when FFmpeg fails or does not create a valid output file."""


def _stderr_detail(stderr: str | None, *, limit: int = 1_000) -> str:
    detail = (stderr or "").strip()
    if not detail:
        return "FFmpeg provided no error details"
    if len(detail) > limit:
        return f"{detail[:limit].rstrip()}..."
    return detail


def standardize_video(
    input_path: str | Path,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Convert a video to the DashGuard MP4 standard with FFmpeg.

    Args:
        input_path: Path to an existing input video file.
        output_path: Destination path, which must have an ``.mp4`` suffix.
        ffmpeg_bin: Executable name or path used to invoke FFmpeg.

    Returns:
        The output path supplied by the caller.

    Raises:
        FileNotFoundError: If ``input_path`` is missing or is not a file.
        ValueError: If the input and output paths are the same or the output
            suffix is not ``.mp4``.
        FFmpegNotFoundError: If ``ffmpeg_bin`` cannot be found.
        FFmpegExecutionError: If FFmpeg cannot run successfully or does not
            create a non-empty output file.
    """

    source = Path(input_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Video input path does not exist or is not a file: {source}"
        )

    if source.resolve() == destination.resolve():
        raise ValueError(
            f"Input and output paths must be different: {source}"
        )

    if destination.suffix.lower() != ".mp4":
        raise ValueError(f"Output path must have an .mp4 extension: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        _VIDEO_FILTER,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "cfr",
        "-movflags",
        "+faststart",
        str(destination),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            f"FFmpeg executable {ffmpeg_bin!r} was not found while processing "
            f"{source}"
        ) from exc
    except OSError as exc:
        raise FFmpegExecutionError(
            f"Could not execute FFmpeg for {source}: {exc}"
        ) from exc

    if completed.returncode != 0:
        detail = _stderr_detail(completed.stderr)
        raise FFmpegExecutionError(f"FFmpeg failed for {source}: {detail}")

    try:
        output_is_valid = destination.is_file() and destination.stat().st_size > 0
    except OSError as exc:
        raise FFmpegExecutionError(
            f"Could not verify FFmpeg output for {source}: {exc}"
        ) from exc

    if not output_is_valid:
        raise FFmpegExecutionError(
            f"FFmpeg reported success for {source}, but the output file is "
            "missing or empty"
        )

    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a video to the DashGuard MP4 standard."
    )
    parser.add_argument("input_path", type=Path, help="input video file")
    parser.add_argument("output_path", type=Path, help="output .mp4 file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return its process exit code."""

    args = _build_parser().parse_args(argv)
    standardize_video(args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

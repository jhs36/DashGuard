"""Video metadata extraction backed by ffprobe JSON output."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing the primary video stream in a media file."""

    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_sec: float
    codec: str


class VideoMetadataError(RuntimeError):
    """Base exception for failures while reading video metadata."""


class FFprobeNotFoundError(VideoMetadataError):
    """Raised when the configured ffprobe executable cannot be found."""


class FFprobeExecutionError(VideoMetadataError):
    """Raised when ffprobe runs but exits with a non-zero status."""


class InvalidFFprobeOutputError(VideoMetadataError):
    """Raised when ffprobe output is malformed or contains invalid metadata."""


class VideoStreamNotFoundError(VideoMetadataError):
    """Raised when ffprobe output does not contain a video stream."""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, (int, str)):
        try:
            parsed = int(value)
        except ValueError:
            return None
    else:
        return None

    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _positive_fps(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None

    try:
        fraction = Fraction(str(value).strip())
        parsed = float(fraction)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None

    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _stderr_detail(stderr: str | None, *, limit: int = 1_000) -> str:
    detail = (stderr or "").strip()
    if not detail:
        return "ffprobe provided no error details"
    if len(detail) > limit:
        return f"{detail[:limit].rstrip()}..."
    return detail


def probe_video_metadata(
    video_path: str | Path,
    ffprobe_bin: str = "ffprobe",
) -> VideoMetadata:
    """Run ffprobe and return validated metadata for the first video stream.

    Args:
        video_path: Path to the local video file to inspect.
        ffprobe_bin: Executable name or path used to invoke ffprobe.

    Returns:
        Validated metadata from the first stream whose codec type is ``video``.

    Raises:
        FileNotFoundError: If ``video_path`` does not identify an existing file.
        FFprobeNotFoundError: If ``ffprobe_bin`` cannot be found.
        FFprobeExecutionError: If ffprobe exits with a non-zero status.
        InvalidFFprobeOutputError: If JSON or required metadata is invalid.
        VideoStreamNotFoundError: If no video stream is present.
    """

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video path does not exist or is not a file: {path}")

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        (
            "stream=codec_type,codec_name,width,height,r_frame_rate,"
            "avg_frame_rate,nb_frames,duration:format=duration"
        ),
        "-of",
        "json",
        str(path),
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
        raise FFprobeNotFoundError(
            f"ffprobe executable {ffprobe_bin!r} was not found while probing {path}"
        ) from exc
    except OSError as exc:
        raise FFprobeExecutionError(
            f"Could not execute ffprobe for {path}: {exc}"
        ) from exc

    if completed.returncode != 0:
        detail = _stderr_detail(completed.stderr)
        raise FFprobeExecutionError(f"ffprobe failed for {path}: {detail}")

    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidFFprobeOutputError(
            f"ffprobe returned invalid JSON for {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise InvalidFFprobeOutputError(
            f"ffprobe JSON root must be an object for {path}"
        )

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise InvalidFFprobeOutputError(
            f"ffprobe JSON is missing a valid 'streams' array for {path}"
        )

    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if video_stream is None:
        raise VideoStreamNotFoundError(f"No video stream found in {path}")

    width = _positive_int(video_stream.get("width"))
    if width is None:
        raise InvalidFFprobeOutputError(
            f"Video stream has an invalid width for {path}"
        )

    height = _positive_int(video_stream.get("height"))
    if height is None:
        raise InvalidFFprobeOutputError(
            f"Video stream has an invalid height for {path}"
        )

    fps = _positive_fps(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _positive_fps(video_stream.get("r_frame_rate"))
    if fps is None:
        raise InvalidFFprobeOutputError(
            f"Video stream has no valid positive FPS for {path}"
        )

    format_section = payload.get("format")
    format_duration = (
        _positive_float(format_section.get("duration"))
        if isinstance(format_section, dict)
        else None
    )
    duration = format_duration or _positive_float(video_stream.get("duration"))
    if duration is None:
        raise InvalidFFprobeOutputError(
            f"ffprobe returned no valid positive duration for {path}"
        )

    codec_value = video_stream.get("codec_name")
    codec = codec_value.strip() if isinstance(codec_value, str) else ""
    if not codec:
        raise InvalidFFprobeOutputError(
            f"Video stream has an invalid codec name for {path}"
        )

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=_positive_int(video_stream.get("nb_frames")),
        duration_sec=duration,
        codec=codec,
    )

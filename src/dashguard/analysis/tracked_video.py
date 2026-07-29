"""Tracked-video rendering, CSV export, transcoding, and metrics."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from dashguard.tracking import Track, UltralyticsByteTracker
from dashguard.video.metadata import VideoMetadata, probe_video_metadata
from dashguard.video.reader import VideoFrame, VideoReader


CSV_HEADER = (
    "frame_index",
    "timestamp_sec",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
)


@dataclass(frozen=True)
class AnalysisMetrics:
    """Performance and output counts for one tracked-video analysis."""

    input_path: str
    output_video_path: str
    output_csv_path: str
    processed_frames: int
    frames_with_tracks: int
    total_track_records: int
    unique_track_ids: int
    input_fps: float
    model_load_sec: float
    tracking_render_sec: float
    transcode_sec: float
    end_to_end_sec: float
    tracking_render_fps: float
    end_to_end_fps: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-safe dictionary in dataclass field order."""

        result = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        for name, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"Analysis metric {name!r} must be finite; received {value!r}"
                )
        return result


class Tracker(Protocol):
    """Interface required from a consecutive-frame tracker."""

    def track(
        self,
        image: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Track]:
        """Return tracks for one BGR frame."""


class VideoWriter(Protocol):
    """Subset of the OpenCV writer API used by this module."""

    def isOpened(self) -> bool:
        """Return whether the writer initialized successfully."""

    def write(self, image: np.ndarray) -> Any:
        """Write one BGR frame."""

    def release(self) -> Any:
        """Finalize and close the writer."""


WriterFactory = Callable[[Path, float, tuple[int, int]], VideoWriter]
FFmpegRunner = Callable[..., subprocess.CompletedProcess[str]]
TrackerFactory = Callable[..., Tracker]


def bbox_center(
    xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return the midpoint of an ``(x1, y1, x2, y2)`` box."""

    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def clip_point(
    point: tuple[float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    """Round and clamp a point to valid image coordinates."""

    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be greater than zero")
    x, y = point
    return (
        min(max(int(round(x)), 0), width - 1),
        min(max(int(round(y)), 0), height - 1),
    )


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track ID."""

    value = int(track_id)
    return (
        48 + (value * 67) % 208,
        48 + (value * 131) % 208,
        48 + (value * 197) % 208,
    )


class TrackRenderer:
    """Draw plain tracks and bounded recent trajectories on BGR frames."""

    def __init__(self, trajectory_length: int = 30) -> None:
        if trajectory_length <= 0:
            raise ValueError("trajectory_length must be greater than zero")
        self._trajectory_length = trajectory_length
        self._trajectories: dict[int, deque[tuple[int, int]]] = {}

    def trajectory_points(self, track_id: int) -> tuple[tuple[int, int], ...]:
        """Return the retained points for one track ID."""

        return tuple(self._trajectories.get(track_id, ()))

    def render(
        self,
        image: np.ndarray,
        tracks: Sequence[Track],
        frame_index: int,
        timestamp_sec: float,
    ) -> np.ndarray:
        """Return a copy of ``image`` annotated from ``tracks`` only."""

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Tracked-video frames must be NumPy BGR images")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Tracked-video frames must not be empty")

        rendered = image.copy()
        for track in tracks:
            color = track_color(track.track_id)
            x1, y1 = clip_point(
                (track.xyxy[0], track.xyxy[1]),
                width,
                height,
            )
            x2, y2 = clip_point(
                (track.xyxy[2], track.xyxy[3]),
                width,
                height,
            )
            center = clip_point(bbox_center(track.xyxy), width, height)

            trajectory = self._trajectories.setdefault(
                track.track_id,
                deque(maxlen=self._trajectory_length),
            )
            trajectory.append(center)
            if len(trajectory) >= 2:
                points = np.asarray(trajectory, dtype=np.int32).reshape(
                    (-1, 1, 2)
                )
                cv2.polylines(
                    rendered,
                    [points],
                    isClosed=False,
                    color=color,
                    thickness=2,
                )

            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            cv2.circle(rendered, center, 3, color, -1)
            label = (
                f"ID {track.track_id} {track.class_name} "
                f"{track.confidence:.2f}"
            )
            label_origin = clip_point(
                (x1, y1 - 8 if y1 >= 18 else y1 + 16),
                width,
                height,
            )
            cv2.putText(
                rendered,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        overlay_lines = (
            f"frame index: {frame_index}",
            f"timestamp sec: {timestamp_sec:.6f}",
            f"current tracks: {len(tracks)}",
        )
        for line_index, text in enumerate(overlay_lines):
            origin = clip_point(
                (10, 18 + line_index * 18),
                width,
                height,
            )
            cv2.putText(
                rendered,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return rendered


def write_track_rows(
    csv_writer: Any,
    tracks: Sequence[Track],
) -> None:
    """Write track rows in tracker order with six-decimal float fields."""

    for track in tracks:
        x1, y1, x2, y2 = track.xyxy
        csv_writer.writerow(
            (
                track.frame_index,
                f"{track.timestamp_sec:.6f}",
                track.track_id,
                track.class_name,
                f"{x1:.6f}",
                f"{y1:.6f}",
                f"{x2:.6f}",
                f"{y2:.6f}",
                f"{track.confidence:.6f}",
            )
        )


def _create_video_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, frame_size)


def transcode_to_h264(
    intermediate_path: str | Path,
    output_path: str | Path,
    *,
    runner: FFmpegRunner | None = None,
) -> None:
    """Transcode an intermediate MP4 to H.264 and remove it on success."""

    intermediate = Path(intermediate_path)
    output = Path(output_path)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(intermediate),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output),
    ]
    execute = subprocess.run if runner is None else runner
    try:
        completed = execute(
            command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not execute ffmpeg: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        if not detail:
            detail = "ffmpeg provided no error details"
        raise RuntimeError(
            f"ffmpeg failed with exit code {completed.returncode}: {detail}"
        )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(
            f"ffmpeg did not create a non-empty output video: {output}"
        )

    intermediate.unlink()


def write_metrics_json(
    metrics: AnalysisMetrics,
    metrics_path: str | Path,
) -> None:
    """Write metrics as an indented UTF-8 JSON object."""

    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(
            metrics.to_dict(),
            metrics_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        metrics_file.write("\n")


def _elapsed(start: float, end: float) -> float:
    elapsed = float(end - start)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError(
            f"Measured elapsed time must be finite and non-negative: {elapsed!r}"
        )
    return elapsed


def _processing_fps(processed_frames: int, elapsed_sec: float) -> float:
    if processed_frames <= 0 or elapsed_sec <= 0:
        return 0.0
    return processed_frames / elapsed_sec


def _validate_metadata(metadata: VideoMetadata) -> None:
    if metadata.width <= 0 or metadata.height <= 0:
        raise ValueError("Video metadata dimensions must be greater than zero")
    if not math.isfinite(metadata.fps) or metadata.fps <= 0:
        raise ValueError("Video metadata FPS must be finite and positive")


def _load_tracker(
    *,
    model_path: str | Path,
    confidence: float,
    imgsz: int,
    device: str,
    tracker_factory: TrackerFactory | None,
) -> Tracker:
    factory: TrackerFactory = (
        UltralyticsByteTracker.from_pretrained
        if tracker_factory is None
        else tracker_factory
    )
    return factory(
        model_path=model_path,
        confidence=confidence,
        imgsz=imgsz,
        device=device,
    )


def run_tracked_video_analysis(
    input_path: str | Path,
    output_video_path: str | Path,
    output_csv_path: str | Path,
    metrics_path: str | Path,
    *,
    model_path: str | Path = "yolo11n.pt",
    confidence: float = 0.25,
    imgsz: int = 640,
    trajectory_length: int = 30,
    max_frames: int | None = None,
    tracker: Tracker | None = None,
    frame_iterable: Iterable[VideoFrame] | None = None,
    video_metadata: VideoMetadata | None = None,
    metadata_loader: Callable[[str | Path], VideoMetadata] | None = None,
    writer: VideoWriter | None = None,
    writer_factory: WriterFactory | None = None,
    ffmpeg_runner: FFmpegRunner | None = None,
    timer: Callable[[], float] = time.perf_counter,
    tracker_factory: TrackerFactory | None = None,
    device: str = "cpu",
) -> AnalysisMetrics:
    """Analyze all selected frames and create video, CSV, and metrics outputs."""

    if trajectory_length <= 0:
        raise ValueError("trajectory_length must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided")
    if tracker is not None and tracker_factory is not None:
        raise ValueError("tracker and tracker_factory cannot both be provided")
    if writer is not None and writer_factory is not None:
        raise ValueError("writer and writer_factory cannot both be provided")

    input_file = Path(input_path)
    output_video = Path(output_video_path)
    output_csv = Path(output_csv_path)
    metrics_file = Path(metrics_path)
    intermediate = output_video.with_name(
        f"{output_video.stem}.intermediate.mp4"
    )

    for parent in (
        output_video.parent,
        output_csv.parent,
        metrics_file.parent,
    ):
        parent.mkdir(parents=True, exist_ok=True)

    if video_metadata is None:
        load_metadata = (
            probe_video_metadata
            if metadata_loader is None
            else metadata_loader
        )
        video_metadata = load_metadata(input_file)
    _validate_metadata(video_metadata)

    if tracker is None:
        model_load_start = timer()
        tracker = _load_tracker(
            model_path=model_path,
            confidence=confidence,
            imgsz=imgsz,
            device=device,
            tracker_factory=tracker_factory,
        )
        model_load_sec = _elapsed(model_load_start, timer())
    else:
        model_load_sec = 0.0

    renderer = TrackRenderer(trajectory_length)
    processed_frames = 0
    frames_with_tracks = 0
    total_track_records = 0
    unique_track_ids: set[int] = set()

    tracking_start = timer()
    active_writer = (
        writer
        if writer is not None
        else (
            _create_video_writer
            if writer_factory is None
            else writer_factory
        )(
            intermediate,
            video_metadata.fps,
            (video_metadata.width, video_metadata.height),
        )
    )
    try:
        if not active_writer.isOpened():
            raise RuntimeError(
                f"OpenCV VideoWriter could not open: {intermediate}"
            )

        with output_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)

            with ExitStack() as stack:
                frames = frame_iterable
                if frames is None:
                    frames = stack.enter_context(VideoReader(input_file))

                for frame in frames:
                    tracks = tracker.track(
                        frame.image,
                        frame.frame_index,
                        frame.timestamp_sec,
                    )
                    write_track_rows(csv_writer, tracks)
                    rendered = renderer.render(
                        frame.image,
                        tracks,
                        frame.frame_index,
                        frame.timestamp_sec,
                    )
                    active_writer.write(rendered)

                    processed_frames += 1
                    total_track_records += len(tracks)
                    if tracks:
                        frames_with_tracks += 1
                    unique_track_ids.update(
                        track.track_id for track in tracks
                    )

                    if (
                        max_frames is not None
                        and processed_frames >= max_frames
                    ):
                        break
    finally:
        active_writer.release()

    tracking_render_sec = _elapsed(tracking_start, timer())
    transcode_start = timer()
    transcode_to_h264(
        intermediate,
        output_video,
        runner=ffmpeg_runner,
    )
    transcode_sec = _elapsed(transcode_start, timer())
    end_to_end_sec = tracking_render_sec + transcode_sec

    metrics = AnalysisMetrics(
        input_path=str(input_file),
        output_video_path=str(output_video),
        output_csv_path=str(output_csv),
        processed_frames=processed_frames,
        frames_with_tracks=frames_with_tracks,
        total_track_records=total_track_records,
        unique_track_ids=len(unique_track_ids),
        input_fps=video_metadata.fps,
        model_load_sec=model_load_sec,
        tracking_render_sec=tracking_render_sec,
        transcode_sec=transcode_sec,
        end_to_end_sec=end_to_end_sec,
        tracking_render_fps=_processing_fps(
            processed_frames,
            tracking_render_sec,
        ),
        end_to_end_fps=_processing_fps(
            processed_frames,
            end_to_end_sec,
        ),
    )
    write_metrics_json(metrics, metrics_file)
    return metrics

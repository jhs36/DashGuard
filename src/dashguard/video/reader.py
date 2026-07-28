"""Sequential video frame reading backed by OpenCV."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    """A decoded video frame with its sequential index and timestamp."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray


class VideoReaderError(RuntimeError):
    """Base exception for failures while reading video frames."""


class VideoOpenError(VideoReaderError):
    """Raised when OpenCV cannot open a video file."""


class InvalidVideoFPSError(VideoReaderError):
    """Raised when OpenCV reports an invalid video frame rate."""


class VideoReaderClosedError(VideoReaderError):
    """Raised when a read is attempted after the reader is closed."""


class VideoReader(Iterator[VideoFrame]):
    """Read decoded video frames sequentially with fixed-rate timestamps."""

    def __init__(self, video_path: str | Path) -> None:
        """Open an existing video file and validate its frame rate.

        Args:
            video_path: Path to the video file to read.

        Raises:
            FileNotFoundError: If ``video_path`` is missing or is not a file.
            VideoOpenError: If OpenCV cannot open the file.
            InvalidVideoFPSError: If OpenCV reports a non-positive or
                non-finite frame rate, or a value that cannot be converted to
                a number.
        """

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Video path does not exist or is not a file: {path}"
            )

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(f"OpenCV could not open video file: {path}")

        raw_fps = capture.get(cv2.CAP_PROP_FPS)
        try:
            fps = float(raw_fps)
        except (TypeError, ValueError, OverflowError) as exc:
            capture.release()
            raise InvalidVideoFPSError(
                f"OpenCV reported an invalid FPS for {path}: {raw_fps!r}"
            ) from exc

        if not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise InvalidVideoFPSError(
                f"OpenCV reported an invalid FPS for {path}: {raw_fps!r}"
            )

        self._video_path = path
        self._fps = fps
        self._capture = capture
        self._next_frame_index = 0
        self._closed = False

    @property
    def video_path(self) -> Path:
        """Return the video file path supplied when the reader was created."""

        return self._video_path

    @property
    def fps(self) -> float:
        """Return the validated frames-per-second value reported by OpenCV."""

        return self._fps

    def read(self) -> VideoFrame | None:
        """Read and return the next frame, or ``None`` at end of stream.

        Raises:
            VideoReaderClosedError: If the reader has already been closed.
        """

        if self._closed:
            raise VideoReaderClosedError(
                f"Cannot read from closed video reader: {self._video_path}"
            )

        success, image = self._capture.read()
        if not success:
            return None

        frame_index = self._next_frame_index
        self._next_frame_index += 1
        return VideoFrame(
            frame_index=frame_index,
            timestamp_sec=frame_index / self._fps,
            image=image,
        )

    def close(self) -> None:
        """Release the OpenCV capture once; repeated calls are safe."""

        if self._closed:
            return

        self._closed = True
        self._capture.release()

    def __iter__(self) -> VideoReader:
        """Return this reader as its own frame iterator."""

        return self

    def __next__(self) -> VideoFrame:
        """Return the next frame or raise ``StopIteration`` at end of stream."""

        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self) -> VideoReader:
        """Return this reader when entering a context manager."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the reader when leaving a context manager."""

        self.close()

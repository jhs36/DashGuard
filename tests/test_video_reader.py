from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dashguard.video.reader import (
    InvalidVideoFPSError,
    VideoFrame,
    VideoOpenError,
    VideoReader,
    VideoReaderClosedError,
)


def _video_file(tmp_path: Path) -> Path:
    video_path = tmp_path / "sample video.mp4"
    video_path.touch()
    return video_path


def _capture(
    *frames: np.ndarray,
    fps: object = 30.0,
) -> MagicMock:
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.return_value = fps
    capture.read.side_effect = [
        *((True, frame) for frame in frames),
        (False, None),
    ]
    return capture


def test_video_frame_is_a_frozen_dataclass() -> None:
    frame = VideoFrame(
        frame_index=0,
        timestamp_sec=0.0,
        image=np.zeros((1, 1, 3), dtype=np.uint8),
    )

    with pytest.raises(FrozenInstanceError):
        frame.frame_index = 1  # type: ignore[misc]


def test_constructor_passes_input_path_to_opencv(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ) as video_capture:
        reader = VideoReader(video_path)

    video_capture.assert_called_once_with(str(video_path))
    assert reader.video_path == video_path
    assert reader.fps == 30.0


def test_missing_video_raises_before_opencv_is_called(tmp_path: Path) -> None:
    video_path = tmp_path / "missing video.mp4"

    with patch("dashguard.video.reader.cv2.VideoCapture") as video_capture:
        with pytest.raises(FileNotFoundError, match="missing video\\.mp4"):
            VideoReader(video_path)

    video_capture.assert_not_called()


def test_unopenable_video_releases_capture_and_raises(
    tmp_path: Path,
) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()
    capture.isOpened.return_value = False

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        with pytest.raises(VideoOpenError, match="sample video\\.mp4"):
            VideoReader(video_path)

    capture.release.assert_called_once_with()


def test_zero_fps_releases_capture_and_raises(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture(fps=0)

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        with pytest.raises(InvalidVideoFPSError, match="sample video\\.mp4"):
            VideoReader(video_path)

    capture.release.assert_called_once_with()


def test_nan_fps_raises_invalid_fps_error(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture(fps=float("nan"))

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        with pytest.raises(InvalidVideoFPSError):
            VideoReader(video_path)

    capture.release.assert_called_once_with()


def test_first_frame_index_is_zero(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture(np.zeros((1, 1), dtype=np.uint8))

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        frame = VideoReader(video_path).read()

    assert frame is not None
    assert frame.frame_index == 0


def test_consecutive_frame_indexes_are_sequential(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    frames = [np.full((1, 1), value, dtype=np.uint8) for value in range(3)]
    capture = _capture(*frames)

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        reader = VideoReader(video_path)
        indexes = [reader.read().frame_index for _ in frames]  # type: ignore[union-attr]

    assert indexes == [0, 1, 2]


def test_timestamps_use_frame_index_and_fps(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    frames = [np.full((1, 1), value, dtype=np.uint8) for value in range(3)]
    capture = _capture(*frames, fps=30.0)

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        reader = VideoReader(video_path)
        timestamps = [
            reader.read().timestamp_sec  # type: ignore[union-attr]
            for _ in frames
        ]

    assert timestamps == pytest.approx([0.0, 1 / 30, 2 / 30])


def test_frame_image_is_the_array_returned_by_opencv(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    image = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    capture = _capture(image)

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        frame = VideoReader(video_path).read()

    assert frame is not None
    assert frame.image is image


def test_read_returns_none_at_end_of_stream(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        reader = VideoReader(video_path)

    assert reader.read() is None


def test_iterator_raises_stop_iteration_at_end_of_stream(
    tmp_path: Path,
) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        reader = VideoReader(video_path)

    assert iter(reader) is reader
    with pytest.raises(StopIteration):
        next(reader)


def test_close_releases_capture(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        reader = VideoReader(video_path)

    reader.close()

    capture.release.assert_called_once_with()


def test_close_is_idempotent(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        reader = VideoReader(video_path)

    reader.close()
    reader.close()
    reader.close()

    capture.release.assert_called_once_with()


def test_context_manager_releases_capture_on_exit(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        with VideoReader(video_path) as reader:
            assert reader.video_path == video_path

    capture.release.assert_called_once_with()


def test_read_after_close_raises_closed_error(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    capture = _capture()

    with patch(
        "dashguard.video.reader.cv2.VideoCapture",
        return_value=capture,
    ):
        reader = VideoReader(video_path)

    reader.close()

    with pytest.raises(VideoReaderClosedError, match="sample video\\.mp4"):
        reader.read()
    capture.read.assert_not_called()

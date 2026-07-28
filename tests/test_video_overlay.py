from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import cv2
import numpy as np
import pytest

from dashguard.video.overlay import (
    EmptyVideoError,
    OverlayOutputError,
    OverlayRenderResult,
    VideoWriterOpenError,
    main,
    render_debug_overlay,
)
from dashguard.video.reader import VideoFrame


def _frame(
    frame_index: int,
    timestamp_sec: float,
    *,
    height: int = 96,
    width: int = 160,
    value: int = 0,
) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        image=np.full((height, width, 3), value, dtype=np.uint8),
    )


def _reader(*frames: VideoFrame, fps: float = 24.0) -> MagicMock:
    reader = MagicMock()
    reader.fps = fps
    reader.read.side_effect = [*frames, None]
    return reader


def _writer_creating(output_path: Path) -> MagicMock:
    writer = MagicMock()
    writer.isOpened.return_value = True
    writer.release.side_effect = lambda: output_path.write_bytes(b"debug mp4")
    return writer


def test_overlay_render_result_is_a_frozen_dataclass(tmp_path: Path) -> None:
    result = OverlayRenderResult(
        output_path=tmp_path / "output.mp4",
        frame_count=3,
        elapsed_sec=0.5,
        effective_fps=6.0,
    )

    with pytest.raises(FrozenInstanceError):
        result.frame_count = 4  # type: ignore[misc]


def test_non_mp4_output_raises_value_error(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.avi"

    with (
        patch("dashguard.video.overlay.VideoReader") as video_reader,
        patch("dashguard.video.overlay.cv2.VideoWriter") as video_writer,
    ):
        with pytest.raises(ValueError, match="\\.mp4"):
            render_debug_overlay(input_path, output_path)

    video_reader.assert_not_called()
    video_writer.assert_not_called()


def test_same_input_and_output_path_raises_value_error(tmp_path: Path) -> None:
    video_path = tmp_path / "same.mp4"

    with (
        patch("dashguard.video.overlay.VideoReader") as video_reader,
        patch("dashguard.video.overlay.cv2.VideoWriter") as video_writer,
    ):
        with pytest.raises(ValueError, match="different"):
            render_debug_overlay(video_path, video_path)

    video_reader.assert_not_called()
    video_writer.assert_not_called()


def test_creates_missing_output_parent_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "new" / "nested" / "output.mp4"
    reader = _reader()

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter") as video_writer,
    ):
        with pytest.raises(EmptyVideoError):
            render_debug_overlay(input_path, output_path)

    assert output_path.parent.is_dir()
    video_writer.assert_not_called()


def test_video_reader_receives_input_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input video.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader()

    with patch(
        "dashguard.video.overlay.VideoReader",
        return_value=reader,
    ) as video_reader:
        with pytest.raises(EmptyVideoError):
            render_debug_overlay(input_path, output_path)

    video_reader.assert_called_once_with(input_path)


def test_empty_video_raises_and_closes_reader(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader()

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter") as video_writer,
        patch("dashguard.video.overlay.cv2.VideoWriter_fourcc") as fourcc,
    ):
        with pytest.raises(EmptyVideoError, match="no frames"):
            render_debug_overlay(input_path, output_path)

    reader.close.assert_called_once_with()
    fourcc.assert_not_called()
    video_writer.assert_not_called()


def test_writer_uses_mp4v_reader_fps_and_first_frame_size(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader(_frame(0, 0.0, height=72, width=128), fps=29.97)
    writer = _writer_creating(output_path)

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter",
            return_value=writer,
        ) as video_writer,
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=4321,
        ) as fourcc,
    ):
        render_debug_overlay(input_path, output_path)

    fourcc.assert_called_once_with(*"mp4v")
    video_writer.assert_called_once_with(
        str(output_path),
        4321,
        29.97,
        (128, 72),
        True,
    )


def test_unopenable_writer_raises_and_releases_resources(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader(_frame(0, 0.0))
    writer = MagicMock()
    writer.isOpened.return_value = False

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
    ):
        with pytest.raises(VideoWriterOpenError, match="output\\.mp4"):
            render_debug_overlay(input_path, output_path)

    reader.close.assert_called_once_with()
    writer.release.assert_called_once_with()
    writer.write.assert_not_called()


def test_render_failure_still_releases_writer_and_closes_reader(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader(_frame(0, 0.0))
    writer = MagicMock()
    writer.isOpened.return_value = True
    writer.write.side_effect = RuntimeError("write failed")

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
    ):
        with pytest.raises(RuntimeError, match="write failed"):
            render_debug_overlay(input_path, output_path)

    reader.close.assert_called_once_with()
    writer.release.assert_called_once_with()


def test_writes_first_and_all_remaining_frames_without_mutating_originals(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    frames = [_frame(index, index / 10, value=index) for index in range(3)]
    originals = [frame.image.copy() for frame in frames]
    reader = _reader(*frames)
    writer = _writer_creating(output_path)

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
    ):
        result = render_debug_overlay(input_path, output_path)

    assert writer.write.call_count == 3
    assert result.frame_count == 3
    for frame, original, write_call in zip(
        frames,
        originals,
        writer.write.call_args_list,
        strict=True,
    ):
        rendered_image = write_call.args[0]
        assert rendered_image is not frame.image
        assert np.array_equal(frame.image, original)


def test_draws_frame_and_time_text_with_black_outline_and_white_body(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    frame = _frame(7, 1.23456)
    reader = _reader(frame)
    writer = _writer_creating(output_path)

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
        patch(
            "dashguard.video.overlay.cv2.putText",
            wraps=cv2.putText,
        ) as put_text,
    ):
        render_debug_overlay(input_path, output_path)

    put_text.assert_has_calls(
        [
            call(
                ANY,
                "frame: 7",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            ),
            call(
                ANY,
                "frame: 7",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            ),
            call(
                ANY,
                "time: 1.235 sec",
                (24, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            ),
            call(
                ANY,
                "time: 1.235 sec",
                (24, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            ),
        ]
    )
    assert put_text.call_count == 4


def test_reports_exact_elapsed_time_and_effective_fps(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader(_frame(0, 0.0), _frame(1, 0.04))
    writer = _writer_creating(output_path)

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
        patch(
            "dashguard.video.overlay.time.perf_counter",
            side_effect=[10.0, 10.5],
        ),
    ):
        result = render_debug_overlay(input_path, output_path)

    assert result.elapsed_sec == pytest.approx(0.5)
    assert result.effective_fps == pytest.approx(4.0)


def test_zero_elapsed_time_produces_zero_effective_fps(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader(_frame(0, 0.0))
    writer = _writer_creating(output_path)

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
        patch(
            "dashguard.video.overlay.time.perf_counter",
            side_effect=[5.0, 5.0],
        ),
    ):
        result = render_debug_overlay(input_path, output_path)

    assert result.elapsed_sec == 0.0
    assert result.effective_fps == 0.0


def test_success_releases_writer_closes_reader_and_returns_result(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader(_frame(0, 0.0))
    writer = _writer_creating(output_path)

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
    ):
        result = render_debug_overlay(input_path, output_path)

    assert isinstance(result, OverlayRenderResult)
    assert result.output_path == output_path
    assert result.frame_count == 1
    reader.close.assert_called_once_with()
    writer.release.assert_called_once_with()


@pytest.mark.parametrize("create_empty_output", [False, True])
def test_missing_or_empty_output_raises_overlay_output_error(
    tmp_path: Path,
    create_empty_output: bool,
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    reader = _reader(_frame(0, 0.0))
    writer = MagicMock()
    writer.isOpened.return_value = True
    if create_empty_output:
        writer.release.side_effect = output_path.touch

    with (
        patch("dashguard.video.overlay.VideoReader", return_value=reader),
        patch("dashguard.video.overlay.cv2.VideoWriter", return_value=writer),
        patch(
            "dashguard.video.overlay.cv2.VideoWriter_fourcc",
            return_value=1234,
        ),
    ):
        with pytest.raises(OverlayOutputError, match="missing or empty"):
            render_debug_overlay(input_path, output_path)

    reader.close.assert_called_once_with()
    writer.release.assert_called_once_with()


def test_cli_prints_summary_and_returns_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"
    result = OverlayRenderResult(
        output_path=output_path,
        frame_count=42,
        elapsed_sec=1.25,
        effective_fps=33.6,
    )

    with patch(
        "dashguard.video.overlay.render_debug_overlay",
        return_value=result,
    ) as render:
        exit_code = main([str(input_path), str(output_path)])

    assert exit_code == 0
    render.assert_called_once_with(input_path, output_path)
    assert capsys.readouterr().out == (
        "frames        : 42\n"
        "elapsed_sec   : 1.250000\n"
        "effective_fps : 33.60\n"
        f"output_path   : {output_path}\n"
    )

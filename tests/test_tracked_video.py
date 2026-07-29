from __future__ import annotations

import builtins
import csv
import io
import json
import subprocess
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from dashguard.analysis import AnalysisMetrics, run_tracked_video_analysis
from dashguard.analysis.tracked_video import (
    CSV_HEADER,
    TrackRenderer,
    bbox_center,
    clip_point,
    transcode_to_h264,
    write_track_rows,
)
from dashguard.tracking import Track
from dashguard.video.metadata import VideoMetadata
from dashguard.video.reader import VideoFrame


METRIC_FIELDS = [
    "input_path",
    "output_video_path",
    "output_csv_path",
    "processed_frames",
    "frames_with_tracks",
    "total_track_records",
    "unique_track_ids",
    "input_fps",
    "model_load_sec",
    "tracking_render_sec",
    "transcode_sec",
    "end_to_end_sec",
    "tracking_render_fps",
    "end_to_end_fps",
]


class FakeTracker:
    def __init__(self, tracks_by_frame: dict[int, list[Track]]) -> None:
        self.tracks_by_frame = tracks_by_frame
        self.calls: list[tuple[np.ndarray, int, float]] = []

    def track(
        self,
        image: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Track]:
        self.calls.append((image, frame_index, timestamp_sec))
        return self.tracks_by_frame.get(frame_index, [])


class FakeWriter:
    def __init__(
        self,
        *,
        opened: bool = True,
        intermediate_path: Path | None = None,
    ) -> None:
        self.opened = opened
        self.intermediate_path = intermediate_path
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, image: np.ndarray) -> None:
        self.frames.append(image)

    def release(self) -> None:
        self.released = True
        if self.intermediate_path is not None:
            self.intermediate_path.write_bytes(b"mp4v")


def _track(
    track_id: int,
    frame_index: int,
    timestamp_sec: float,
    *,
    xyxy: tuple[float, float, float, float] = (1.0, 2.0, 8.0, 9.0),
    confidence: float = 0.75,
    class_name: str = "car",
) -> Track:
    return Track(
        track_id=track_id,
        xyxy=xyxy,
        confidence=confidence,
        class_name=class_name,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
    )


def _frames() -> list[VideoFrame]:
    return [
        VideoFrame(
            frame_index=index,
            timestamp_sec=index / 25.0,
            image=np.zeros((20, 30, 3), dtype=np.uint8),
        )
        for index in range(3)
    ]


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        width=30,
        height=20,
        fps=25.0,
        frame_count=3,
        duration_sec=0.12,
        codec="h264",
    )


def _success_runner(
    command: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    Path(command[-1]).write_bytes(b"h264")
    return subprocess.CompletedProcess(
        args=command,
        returncode=0,
        stdout="",
        stderr="",
    )


def test_analysis_metrics_is_frozen_dataclass_with_exact_fields() -> None:
    metrics = AnalysisMetrics(
        "in.mp4",
        "out.mp4",
        "out.csv",
        1,
        1,
        1,
        1,
        30.0,
        0.1,
        0.2,
        0.3,
        0.5,
        5.0,
        2.0,
    )

    assert is_dataclass(AnalysisMetrics)
    assert [field.name for field in fields(AnalysisMetrics)] == METRIC_FIELDS
    assert list(metrics.to_dict()) == METRIC_FIELDS
    with pytest.raises(FrozenInstanceError):
        metrics.processed_frames = 2  # type: ignore[misc]


def test_analysis_metrics_rejects_non_finite_json_value() -> None:
    metrics = AnalysisMetrics(
        "in.mp4",
        "out.mp4",
        "out.csv",
        0,
        0,
        0,
        0,
        float("nan"),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    with pytest.raises(ValueError, match="must be finite"):
        metrics.to_dict()


def test_csv_header_has_exact_required_order() -> None:
    assert ",".join(CSV_HEADER) == (
        "frame_index,timestamp_sec,track_id,class_name,"
        "x1,y1,x2,y2,confidence"
    )


def test_csv_track_values_use_six_decimal_places() -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(CSV_HEADER)
    write_track_rows(
        writer,
        [
            _track(
                7,
                12,
                1.23456789,
                xyxy=(-1.2, 2.3456789, 30.0, 40.9999999),
                confidence=0.87654321,
                class_name="truck",
            )
        ],
    )

    rows = list(csv.reader(io.StringIO(stream.getvalue())))
    assert rows[1] == [
        "12",
        "1.234568",
        "7",
        "truck",
        "-1.200000",
        "2.345679",
        "30.000000",
        "41.000000",
        "0.876543",
    ]


def test_csv_with_no_tracks_contains_header_only() -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(CSV_HEADER)

    write_track_rows(writer, [])

    assert list(csv.reader(io.StringIO(stream.getvalue()))) == [
        list(CSV_HEADER)
    ]


def test_bbox_center_calculation() -> None:
    assert bbox_center((-2.0, 4.0, 8.0, 14.0)) == (3.0, 9.0)


def test_clip_point_limits_coordinates_to_image_boundaries() -> None:
    assert clip_point((-100.0, 200.0), width=30, height=20) == (0, 19)
    assert clip_point((29.4, 0.4), width=30, height=20) == (29, 0)


def test_renderer_passes_only_bounded_drawing_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangle = Mock()
    circle = Mock()
    put_text = Mock()
    monkeypatch.setattr(
        "dashguard.analysis.tracked_video.cv2.rectangle",
        rectangle,
    )
    monkeypatch.setattr(
        "dashguard.analysis.tracked_video.cv2.circle",
        circle,
    )
    monkeypatch.setattr(
        "dashguard.analysis.tracked_video.cv2.putText",
        put_text,
    )
    image = np.zeros((10, 15, 3), dtype=np.uint8)

    TrackRenderer().render(
        image,
        [
            _track(
                1,
                0,
                0.0,
                xyxy=(-20.0, -30.0, 99.0, 88.0),
            )
        ],
        frame_index=0,
        timestamp_sec=0.0,
    )

    _, corner1, corner2, _, _ = rectangle.call_args.args
    _, center, _, _, _ = circle.call_args.args
    origins = [call.args[2] for call in put_text.call_args_list]
    for x, y in [corner1, corner2, center, *origins]:
        assert 0 <= x < 15
        assert 0 <= y < 10


@pytest.mark.parametrize("trajectory_length", [0, -1])
def test_trajectory_length_must_be_positive(trajectory_length: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        TrackRenderer(trajectory_length)


def test_trajectory_does_not_exceed_configured_length() -> None:
    renderer = TrackRenderer(trajectory_length=2)
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    for frame_index in range(5):
        renderer.render(
            image,
            [
                _track(
                    4,
                    frame_index,
                    frame_index / 25.0,
                    xyxy=(
                        float(frame_index),
                        0.0,
                        float(frame_index + 2),
                        2.0,
                    ),
                )
            ],
            frame_index=frame_index,
            timestamp_sec=frame_index / 25.0,
        )

    assert renderer.trajectory_points(4) == ((4, 1), (5, 1))


def test_pipeline_calls_tracker_once_per_consecutive_frame_and_counts_metrics(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mp4"
    output_video = tmp_path / "nested" / "analysis.mp4"
    output_csv = tmp_path / "tables" / "tracks.csv"
    metrics_path = tmp_path / "metrics" / "metrics.json"
    frames = _frames()
    tracker = FakeTracker(
        {
            0: [_track(3, 0, 0.0)],
            2: [
                _track(3, 2, 0.08),
                _track(9, 2, 0.08, class_name="bus"),
            ],
        }
    )
    writer_holder: dict[str, FakeWriter] = {}

    def writer_factory(
        path: Path,
        fps: float,
        size: tuple[int, int],
    ) -> FakeWriter:
        assert path == output_video.with_name("analysis.intermediate.mp4")
        assert fps == 25.0
        assert size == (30, 20)
        fake_writer = FakeWriter(intermediate_path=path)
        writer_holder["writer"] = fake_writer
        return fake_writer

    times = iter((10.0, 12.0, 20.0, 24.0))
    metrics = run_tracked_video_analysis(
        input_path,
        output_video,
        output_csv,
        metrics_path,
        tracker=tracker,
        frame_iterable=frames,
        video_metadata=_metadata(),
        writer_factory=writer_factory,
        ffmpeg_runner=_success_runner,
        timer=lambda: next(times),
    )

    assert [(index, timestamp) for _, index, timestamp in tracker.calls] == [
        (0, 0.0),
        (1, 0.04),
        (2, 0.08),
    ]
    assert [image for image, _, _ in tracker.calls] == [
        frame.image for frame in frames
    ]
    assert writer_holder["writer"].released
    assert len(writer_holder["writer"].frames) == 3
    assert metrics.processed_frames == 3
    assert metrics.frames_with_tracks == 2
    assert metrics.total_track_records == 3
    assert metrics.unique_track_ids == 2
    assert metrics.tracking_render_sec == 2.0
    assert metrics.transcode_sec == 4.0
    assert metrics.end_to_end_sec == 6.0
    assert metrics.tracking_render_fps == 1.5
    assert metrics.end_to_end_fps == 0.5
    assert metrics.model_load_sec == 0.0
    assert output_video.read_bytes() == b"h264"
    assert not output_video.with_name("analysis.intermediate.mp4").exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == (
        metrics.to_dict()
    )


def test_pipeline_no_tracks_still_creates_csv_header(
    tmp_path: Path,
) -> None:
    output_video = tmp_path / "out.mp4"
    output_csv = tmp_path / "tracks.csv"
    intermediate = tmp_path / "out.intermediate.mp4"
    writer = FakeWriter(intermediate_path=intermediate)
    times = iter((0.0, 1.0, 2.0, 3.0))

    run_tracked_video_analysis(
        tmp_path / "input.mp4",
        output_video,
        output_csv,
        tmp_path / "metrics.json",
        tracker=FakeTracker({}),
        frame_iterable=_frames(),
        video_metadata=_metadata(),
        writer=writer,
        ffmpeg_runner=_success_runner,
        timer=lambda: next(times),
    )

    assert output_csv.read_text(encoding="utf-8").splitlines() == [
        ",".join(CSV_HEADER)
    ]


def test_writer_open_failure_releases_writer_and_skips_ffmpeg(
    tmp_path: Path,
) -> None:
    writer = FakeWriter(opened=False)
    runner = Mock()
    times = iter((1.0,))

    with pytest.raises(RuntimeError, match="VideoWriter could not open"):
        run_tracked_video_analysis(
            tmp_path / "input.mp4",
            tmp_path / "out.mp4",
            tmp_path / "tracks.csv",
            tmp_path / "metrics.json",
            tracker=FakeTracker({}),
            frame_iterable=[],
            video_metadata=_metadata(),
            writer=writer,
            ffmpeg_runner=runner,
            timer=lambda: next(times),
        )

    assert writer.released
    runner.assert_not_called()


def test_ffmpeg_failure_includes_stderr_and_preserves_intermediate(
    tmp_path: Path,
) -> None:
    intermediate = tmp_path / "out.intermediate.mp4"
    intermediate.write_bytes(b"diagnostic")

    def failed_runner(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=7,
            stdout="",
            stderr="encoder exploded",
        )

    with pytest.raises(RuntimeError, match="encoder exploded"):
        transcode_to_h264(
            intermediate,
            tmp_path / "out.mp4",
            runner=failed_runner,
        )

    assert intermediate.read_bytes() == b"diagnostic"


def test_ffmpeg_success_uses_required_options_and_deletes_intermediate(
    tmp_path: Path,
) -> None:
    intermediate = tmp_path / "out.intermediate.mp4"
    output = tmp_path / "out.mp4"
    intermediate.write_bytes(b"mp4v")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        output.write_bytes(b"h264")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    transcode_to_h264(intermediate, output, runner=runner)

    command, kwargs = calls[0]
    assert command == [
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
    assert kwargs == {
        "shell": False,
        "check": False,
        "capture_output": True,
        "text": True,
    }
    assert not intermediate.exists()
    assert output.read_bytes() == b"h264"


def test_injected_pipeline_never_imports_ultralytics_or_runs_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise AssertionError("ultralytics must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        "dashguard.analysis.tracked_video.subprocess.run",
        Mock(side_effect=AssertionError("real subprocess must not run")),
    )
    intermediate = tmp_path / "out.intermediate.mp4"
    times = iter((0.0, 1.0, 2.0, 3.0))

    run_tracked_video_analysis(
        tmp_path / "input.mp4",
        tmp_path / "out.mp4",
        tmp_path / "tracks.csv",
        tmp_path / "metrics.json",
        tracker=FakeTracker({}),
        frame_iterable=[],
        video_metadata=_metadata(),
        writer=FakeWriter(intermediate_path=intermediate),
        ffmpeg_runner=_success_runner,
        timer=lambda: next(times),
    )

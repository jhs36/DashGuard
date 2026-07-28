from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest


from dashguard.video.metadata import (
    FFprobeExecutionError,
    FFprobeNotFoundError,
    InvalidFFprobeOutputError,
    VideoMetadata,
    VideoStreamNotFoundError,
    probe_video_metadata,
)


def _video_file(tmp_path: Path) -> Path:
    video_path = tmp_path / "sample video.mp4"
    video_path.touch()
    return video_path


def _ffprobe_result(payload: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout=json.dumps(payload),
        stderr="",
    )


def _payload(
    *,
    avg_frame_rate: str = "30/1",
    r_frame_rate: str = "30/1",
    include_nb_frames: bool = True,
) -> dict[str, object]:
    video_stream: dict[str, object] = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": avg_frame_rate,
        "r_frame_rate": r_frame_rate,
        "duration": "9.5",
    }
    if include_nb_frames:
        video_stream["nb_frames"] = "300"

    return {
        "streams": [
            {"codec_type": "audio", "codec_name": "aac"},
            video_stream,
        ],
        "format": {"duration": "10.0"},
    }


def test_probe_converts_normal_json_to_video_metadata(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)

    with patch(
        "dashguard.video.metadata.subprocess.run",
        return_value=_ffprobe_result(_payload()),
    ):
        metadata = probe_video_metadata(video_path)

    assert metadata == VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=300,
        duration_sec=10.0,
        codec="h264",
    )


def test_probe_converts_fractional_fps(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)

    with patch(
        "dashguard.video.metadata.subprocess.run",
        return_value=_ffprobe_result(_payload(avg_frame_rate="30000/1001")),
    ):
        metadata = probe_video_metadata(video_path)

    assert metadata.fps == pytest.approx(30000 / 1001)


def test_probe_falls_back_to_r_frame_rate_for_zero_average(
    tmp_path: Path,
) -> None:
    video_path = _video_file(tmp_path)

    with patch(
        "dashguard.video.metadata.subprocess.run",
        return_value=_ffprobe_result(
            _payload(avg_frame_rate="0/0", r_frame_rate="24000/1001")
        ),
    ):
        metadata = probe_video_metadata(video_path)

    assert metadata.fps == pytest.approx(24000 / 1001)


def test_probe_leaves_frame_count_none_when_nb_frames_is_missing(
    tmp_path: Path,
) -> None:
    video_path = _video_file(tmp_path)

    with patch(
        "dashguard.video.metadata.subprocess.run",
        return_value=_ffprobe_result(_payload(include_nb_frames=False)),
    ):
        metadata = probe_video_metadata(video_path)

    assert metadata.frame_count is None


def test_missing_input_raises_before_ffprobe_runs(tmp_path: Path) -> None:
    video_path = tmp_path / "missing.mp4"

    with patch("dashguard.video.metadata.subprocess.run") as run:
        with pytest.raises(FileNotFoundError, match="missing\\.mp4"):
            probe_video_metadata(video_path)

    run.assert_not_called()


def test_missing_ffprobe_executable_has_specific_error(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)

    with patch(
        "dashguard.video.metadata.subprocess.run",
        side_effect=FileNotFoundError("ffprobe"),
    ):
        with pytest.raises(FFprobeNotFoundError, match="sample video\\.mp4"):
            probe_video_metadata(video_path)


def test_nonzero_ffprobe_exit_includes_stderr_and_path(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    result = subprocess.CompletedProcess[str](
        args=["ffprobe"],
        returncode=1,
        stdout="",
        stderr="Invalid data found when processing input",
    )

    with patch("dashguard.video.metadata.subprocess.run", return_value=result):
        with pytest.raises(FFprobeExecutionError) as exc_info:
            probe_video_metadata(video_path)

    message = str(exc_info.value)
    assert "sample video.mp4" in message
    assert "Invalid data found" in message


def test_invalid_json_raises_invalid_output_error(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    result = subprocess.CompletedProcess[str](
        args=["ffprobe"],
        returncode=0,
        stdout="{not json",
        stderr="",
    )

    with patch("dashguard.video.metadata.subprocess.run", return_value=result):
        with pytest.raises(InvalidFFprobeOutputError, match="invalid JSON"):
            probe_video_metadata(video_path)


def test_absent_video_stream_raises_specific_error(tmp_path: Path) -> None:
    video_path = _video_file(tmp_path)
    payload = {
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
        "format": {"duration": "10.0"},
    }

    with patch(
        "dashguard.video.metadata.subprocess.run",
        return_value=_ffprobe_result(payload),
    ):
        with pytest.raises(VideoStreamNotFoundError, match="sample video\\.mp4"):
            probe_video_metadata(video_path)

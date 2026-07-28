from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dashguard.video.standardize import (
    FFmpegExecutionError,
    FFmpegNotFoundError,
    main,
    standardize_video,
)


def _input_file(tmp_path: Path) -> Path:
    input_path = tmp_path / "input video.mov"
    input_path.write_bytes(b"input")
    return input_path


def _successful_run_creating(output_path: Path):
    def run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        output_path.write_bytes(b"standardized video")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    return run


def test_success_returns_output_path_and_builds_required_command(
    tmp_path: Path,
) -> None:
    input_path = _input_file(tmp_path)
    output_path = tmp_path / "standard.mp4"

    with patch(
        "dashguard.video.standardize.subprocess.run",
        side_effect=_successful_run_creating(output_path),
    ) as run:
        result = standardize_video(input_path, output_path)

    assert result == output_path
    command = run.call_args.args[0]
    assert command[0] == "ffmpeg"
    assert "-y" in command
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-preset") + 1] == "medium"
    assert command[command.index("-crf") + 1] == "23"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert "-an" in command
    assert "-sn" in command
    assert "-dn" in command

    video_filter = command[command.index("-vf") + 1]
    assert "min(1280,iw)" in video_filter
    assert "min(720,ih)" in video_filter
    assert "force_original_aspect_ratio=decrease" in video_filter
    assert "force_divisible_by=2" in video_filter
    assert "fps=30" in video_filter

    assert run.call_args.kwargs["shell"] is False


def test_creates_missing_output_parent_directory(tmp_path: Path) -> None:
    input_path = _input_file(tmp_path)
    output_path = tmp_path / "new" / "nested" / "standard.mp4"

    with patch(
        "dashguard.video.standardize.subprocess.run",
        side_effect=_successful_run_creating(output_path),
    ):
        standardize_video(input_path, output_path)

    assert output_path.parent.is_dir()


def test_missing_input_raises_before_ffmpeg_runs(tmp_path: Path) -> None:
    input_path = tmp_path / "missing.mov"
    output_path = tmp_path / "standard.mp4"

    with patch("dashguard.video.standardize.subprocess.run") as run:
        with pytest.raises(FileNotFoundError, match="missing\\.mov"):
            standardize_video(input_path, output_path)

    run.assert_not_called()


def test_same_input_and_output_path_raises_value_error(tmp_path: Path) -> None:
    input_path = _input_file(tmp_path)

    with patch("dashguard.video.standardize.subprocess.run") as run:
        with pytest.raises(ValueError, match="different"):
            standardize_video(input_path, input_path)

    run.assert_not_called()


def test_non_mp4_output_raises_value_error(tmp_path: Path) -> None:
    input_path = _input_file(tmp_path)
    output_path = tmp_path / "standard.avi"

    with patch("dashguard.video.standardize.subprocess.run") as run:
        with pytest.raises(ValueError, match="\\.mp4"):
            standardize_video(input_path, output_path)

    run.assert_not_called()


def test_missing_ffmpeg_executable_raises_specific_error(tmp_path: Path) -> None:
    input_path = _input_file(tmp_path)
    output_path = tmp_path / "standard.mp4"

    with patch(
        "dashguard.video.standardize.subprocess.run",
        side_effect=FileNotFoundError("ffmpeg"),
    ):
        with pytest.raises(FFmpegNotFoundError, match="input video\\.mov"):
            standardize_video(input_path, output_path)


def test_nonzero_ffmpeg_exit_includes_stderr_and_input_path(
    tmp_path: Path,
) -> None:
    input_path = _input_file(tmp_path)
    output_path = tmp_path / "standard.mp4"
    completed = subprocess.CompletedProcess[str](
        args=["ffmpeg"],
        returncode=1,
        stdout="",
        stderr="Invalid data found when processing input",
    )

    with patch(
        "dashguard.video.standardize.subprocess.run",
        return_value=completed,
    ):
        with pytest.raises(FFmpegExecutionError) as exc_info:
            standardize_video(input_path, output_path)

    message = str(exc_info.value)
    assert "input video.mov" in message
    assert "Invalid data found" in message


@pytest.mark.parametrize("create_empty_output", [False, True])
def test_success_without_nonempty_output_raises_execution_error(
    tmp_path: Path,
    create_empty_output: bool,
) -> None:
    input_path = _input_file(tmp_path)
    output_path = tmp_path / "standard.mp4"

    def successful_run_without_valid_output(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if create_empty_output:
            output_path.touch()
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    with patch(
        "dashguard.video.standardize.subprocess.run",
        side_effect=successful_run_without_valid_output,
    ):
        with pytest.raises(FFmpegExecutionError, match="missing or empty"):
            standardize_video(input_path, output_path)


def test_cli_parses_arguments_and_exits_successfully(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mp4"

    with patch(
        "dashguard.video.standardize.standardize_video",
        return_value=output_path,
    ) as standardize:
        exit_code = main([str(input_path), str(output_path)])

    assert exit_code == 0
    standardize.assert_called_once_with(input_path, output_path)

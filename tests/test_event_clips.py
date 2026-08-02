from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from dashguard.video.event_clips import (
    ClipManifestRecord,
    ClipValidationError,
    EventClipPipelineResult,
    EventJSONValidationError,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    build_clip_manifest,
    build_ffmpeg_command,
    extract_event_clips,
    generate_event_clip,
    load_event_json,
    validate_generated_clip,
    write_clip_manifest,
)
from dashguard.video.metadata import VideoMetadata


def _event(
    event_id: str = "event_001",
    start: float = 1.0,
    end: float = 2.0,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "clip_start_sec": start,
        "clip_end_sec": end,
    }


def _event_document(
    events: object | None = None,
    *,
    video_duration_sec: float = 10.0,
    source_video: str = "provenance/source.mp4",
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source_video": source_video,
        "video_duration_sec": video_duration_sec,
        "events": [_event()] if events is None else events,
    }


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _metadata(
    *,
    duration: float = 1.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    frame_count: int | None = 30,
    codec: str = "h264",
) -> VideoMetadata:
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        codec=codec,
    )


def _completed(
    command: list[str],
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, "", stderr)


def _successful_runner(
    command: list[str],
    **_: object,
) -> subprocess.CompletedProcess[str]:
    Path(command[-1]).write_bytes(b"encoded-clip")
    return _completed(command)


def _prepare_pipeline(
    tmp_path: Path,
    *,
    events: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    events_path = tmp_path / "events.json"
    input_video = tmp_path / "input.mp4"
    output_dir = tmp_path / "clips"
    manifest_path = tmp_path / "manifest.json"
    input_video.write_bytes(b"source-video")
    _write_json(
        events_path,
        _event_document(
            events
            if events is not None
            else [_event("event_001", 1.0, 2.0)]
        ),
    )
    return events_path, input_video, output_dir, manifest_path


def _pipeline_metadata_loader(
    input_video: Path,
    durations: list[float] | None = None,
):
    pending = list(durations or [1.0])

    def load(path: str | Path, ffprobe_bin: str) -> VideoMetadata:
        assert ffprobe_bin == "probe-custom"
        if Path(path) == input_video:
            return _metadata(duration=10.0, frame_count=300)
        return _metadata(duration=pending.pop(0))

    return load


def test_load_event_json_accepts_valid_document_and_preserves_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.json"
    _write_json(
        path,
        _event_document(
            [
                _event("event_002", 3.0, 4.5),
                _event("event_001", 1.0, 2.0),
            ],
            source_video="different/from/cli.mp4",
        ),
    )

    document = load_event_json(
        path,
        actual_video_duration_sec=10.0,
    )

    assert document.source_video == "different/from/cli.mp4"
    assert [event.event_id for event in document.events] == [
        "event_002",
        "event_001",
    ]
    assert document.events[0].requested_duration_sec == 1.5


@pytest.mark.parametrize(
    "missing_field",
    ["schema_version", "source_video", "video_duration_sec", "events"],
)
def test_load_event_json_rejects_missing_top_level_fields(
    tmp_path: Path,
    missing_field: str,
) -> None:
    path = tmp_path / "events.json"
    document = _event_document()
    del document[missing_field]
    _write_json(path, document)

    with pytest.raises(EventJSONValidationError, match=missing_field):
        load_event_json(path, actual_video_duration_sec=10.0)


def test_load_event_json_requires_events_list(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_json(path, _event_document({"event_id": "event_001"}))

    with pytest.raises(EventJSONValidationError, match="events must be a list"):
        load_event_json(path, actual_video_duration_sec=10.0)


def test_load_event_json_rejects_empty_events(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_json(path, _event_document([]))

    with pytest.raises(EventJSONValidationError, match="must not be empty"):
        load_event_json(path, actual_video_duration_sec=10.0)


@pytest.mark.parametrize(
    "missing_field",
    ["event_id", "clip_start_sec", "clip_end_sec"],
)
def test_load_event_json_rejects_missing_event_fields(
    tmp_path: Path,
    missing_field: str,
) -> None:
    path = tmp_path / "events.json"
    event = _event()
    del event[missing_field]
    _write_json(path, _event_document([event]))

    with pytest.raises(EventJSONValidationError, match=missing_field):
        load_event_json(path, actual_video_duration_sec=10.0)


def test_load_event_json_rejects_duplicate_event_ids(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_json(
        path,
        _event_document([_event(), _event("event_001", 3.0, 4.0)]),
    )

    with pytest.raises(EventJSONValidationError, match="Duplicate event_id"):
        load_event_json(path, actual_video_duration_sec=10.0)


@pytest.mark.parametrize("event_id", ["", "   "])
def test_load_event_json_rejects_empty_event_id(
    tmp_path: Path,
    event_id: str,
) -> None:
    path = tmp_path / "events.json"
    _write_json(path, _event_document([_event(event_id)]))

    with pytest.raises(EventJSONValidationError, match="non-empty string"):
        load_event_json(path, actual_video_duration_sec=10.0)


@pytest.mark.parametrize(
    "event_id",
    [
        "../escape",
        "folder/event",
        "folder\\event",
        "bad\x01event",
        "bad:name",
        "CON",
        "trailing.",
        " padded",
    ],
)
def test_load_event_json_rejects_unsafe_event_id(
    tmp_path: Path,
    event_id: str,
) -> None:
    path = tmp_path / "events.json"
    _write_json(path, _event_document([_event(event_id)]))

    with pytest.raises(EventJSONValidationError, match="safe filename"):
        load_event_json(path, actual_video_duration_sec=10.0)


@pytest.mark.parametrize(
    ("field", "constant"),
    [
        ("clip_start_sec", "NaN"),
        ("clip_start_sec", "Infinity"),
        ("clip_end_sec", "-Infinity"),
    ],
)
def test_load_event_json_rejects_nonfinite_event_range(
    tmp_path: Path,
    field: str,
    constant: str,
) -> None:
    path = tmp_path / "events.json"
    text = json.dumps(_event_document(), allow_nan=False)
    old_value = '"clip_start_sec": 1.0' if field == "clip_start_sec" else '"clip_end_sec": 2.0'
    path.write_text(
        text.replace(old_value, f'"{field}": {constant}'),
        encoding="utf-8",
    )

    with pytest.raises(EventJSONValidationError, match="non-finite"):
        load_event_json(path, actual_video_duration_sec=10.0)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-0.1, 1.0, "non-negative"),
        (1.0, 1.0, "less than"),
        (2.0, 1.0, "less than"),
        (9.0, 10.1, "exceeds"),
    ],
)
def test_load_event_json_rejects_invalid_clip_ranges(
    tmp_path: Path,
    start: float,
    end: float,
    message: str,
) -> None:
    path = tmp_path / "events.json"
    _write_json(path, _event_document([_event(start=start, end=end)]))

    with pytest.raises(EventJSONValidationError, match=message):
        load_event_json(path, actual_video_duration_sec=10.0)


def test_load_event_json_rejects_material_video_duration_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.json"
    _write_json(path, _event_document(video_duration_sec=11.0))

    with pytest.raises(EventJSONValidationError, match="does not match"):
        load_event_json(
            path,
            actual_video_duration_sec=10.0,
            duration_tolerance_sec=0.1,
        )


def test_ffmpeg_command_uses_accurate_seek_and_h264_reencoding() -> None:
    command = build_ffmpeg_command(
        "input.mp4",
        "output.mp4",
        clip_start_sec=1.25,
        requested_duration_sec=2.5,
        ffmpeg_bin="ffmpeg-custom",
    )

    assert command[0] == "ffmpeg-custom"
    assert command[command.index("-i") + 1] == "input.mp4"
    assert command.index("-ss") > command.index("-i")
    assert command[command.index("-ss") + 1] == "1.25"
    assert command[command.index("-t") + 1] == "2.5"
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "-an" in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert "copy" not in command
    assert "-c" not in command
    assert "-y" not in command
    assert "-n" not in command
    assert "-nostdin" in command


def test_generate_event_clip_runs_without_shell_and_creates_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    captured: dict[str, object] = {}

    def runner(command: list[str], **kwargs: object):
        captured["command"] = command
        captured["kwargs"] = kwargs
        Path(command[-1]).write_bytes(b"clip")
        return _completed(command)

    result = generate_event_clip(
        source,
        output,
        clip_start_sec=1.0,
        requested_duration_sec=2.0,
        runner=runner,
    )

    assert result == output
    assert captured["kwargs"] == {
        "capture_output": True,
        "text": True,
        "check": False,
        "shell": False,
    }


def test_generate_event_clip_reports_missing_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")

    def missing(*_: object, **__: object):
        raise FileNotFoundError("missing")

    with pytest.raises(FFmpegNotFoundError, match="missing-ffmpeg"):
        generate_event_clip(
            source,
            tmp_path / "output.mp4",
            clip_start_sec=0.0,
            requested_duration_sec=1.0,
            ffmpeg_bin="missing-ffmpeg",
            runner=missing,
        )


def test_generate_event_clip_preserves_ffmpeg_stderr(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")

    def failing(command: list[str], **_: object):
        return _completed(command, 7, "encoder exploded")

    with pytest.raises(FFmpegExecutionError) as exc_info:
        generate_event_clip(
            source,
            tmp_path / "output.mp4",
            clip_start_sec=0.0,
            requested_duration_sec=1.0,
            runner=failing,
        )

    assert "exit code 7" in str(exc_info.value)
    assert "encoder exploded" in str(exc_info.value)


@pytest.mark.parametrize("empty_output", [False, True])
def test_generate_event_clip_rejects_missing_or_empty_output(
    tmp_path: Path,
    empty_output: bool,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")

    def fake_success(command: list[str], **_: object):
        if empty_output:
            Path(command[-1]).touch()
        return _completed(command)

    with pytest.raises(FFmpegExecutionError, match="missing or empty"):
        generate_event_clip(
            source,
            tmp_path / "output.mp4",
            clip_start_sec=0.0,
            requested_duration_sec=1.0,
            runner=fake_success,
        )


def test_validate_generated_clip_wraps_metadata_failure(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    def fail_metadata(_: str | Path, __: str) -> VideoMetadata:
        raise RuntimeError("bad ffprobe payload")

    with pytest.raises(ClipValidationError, match="bad ffprobe payload"):
        validate_generated_clip(
            clip,
            requested_duration_sec=1.0,
            metadata_loader=fail_metadata,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(width=0),
        _metadata(height=0),
        _metadata(fps=0.0),
        _metadata(duration=0.0),
        _metadata(codec=""),
        _metadata(frame_count=0),
    ],
)
def test_validate_generated_clip_rejects_invalid_metadata(
    tmp_path: Path,
    metadata: VideoMetadata,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    with pytest.raises(ClipValidationError):
        validate_generated_clip(
            clip,
            requested_duration_sec=1.0,
            metadata_loader=lambda *_: metadata,
        )


def test_validate_generated_clip_includes_duration_tolerance_boundary(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    validated = validate_generated_clip(
        clip,
        requested_duration_sec=1.0,
        metadata_loader=lambda *_: _metadata(duration=1.1, fps=30.0),
    )

    assert validated.duration_difference_sec == pytest.approx(0.1)


def test_validate_generated_clip_rejects_duration_beyond_tolerance(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    with pytest.raises(ClipValidationError, match="beyond tolerance"):
        validate_generated_clip(
            clip,
            requested_duration_sec=1.0,
            metadata_loader=lambda *_: _metadata(duration=1.101, fps=30.0),
        )


@pytest.mark.parametrize("existing_kind", ["clip", "manifest"])
def test_pipeline_refuses_existing_outputs_before_ffmpeg(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    events, video, output_dir, manifest = _prepare_pipeline(tmp_path)
    output_dir.mkdir()
    existing = output_dir / "event_001.mp4" if existing_kind == "clip" else manifest
    existing.write_bytes(b"old")
    runner_calls = 0

    def runner(command: list[str], **_: object):
        nonlocal runner_calls
        runner_calls += 1
        return _completed(command)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        extract_event_clips(
            events_json=events,
            input_video=video,
            output_dir=output_dir,
            output_manifest_json=manifest,
            ffprobe_bin="probe-custom",
            runner=runner,
            metadata_loader=_pipeline_metadata_loader(video),
        )

    assert runner_calls == 0
    assert existing.read_bytes() == b"old"


def test_second_event_failure_leaves_no_partial_final_outputs(
    tmp_path: Path,
) -> None:
    event_records = [
        _event("event_001", 1.0, 2.0),
        _event("event_002", 3.0, 4.0),
    ]
    events, video, output_dir, manifest = _prepare_pipeline(
        tmp_path,
        events=event_records,
    )
    calls = 0

    def runner(command: list[str], **_: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            Path(command[-1]).write_bytes(b"first")
            return _completed(command)
        return _completed(command, 1, "second clip failed")

    with pytest.raises(FFmpegExecutionError, match="second clip failed"):
        extract_event_clips(
            events_json=events,
            input_video=video,
            output_dir=output_dir,
            output_manifest_json=manifest,
            ffprobe_bin="probe-custom",
            runner=runner,
            metadata_loader=_pipeline_metadata_loader(video, [1.0]),
        )

    assert not (output_dir / "event_001.mp4").exists()
    assert not (output_dir / "event_002.mp4").exists()
    assert not manifest.exists()
    assert list(output_dir.iterdir()) == []


def test_metadata_failure_cleans_temporary_clip(tmp_path: Path) -> None:
    events, video, output_dir, manifest = _prepare_pipeline(tmp_path)

    def metadata_loader(path: str | Path, _: str) -> VideoMetadata:
        if Path(path) == video:
            return _metadata(duration=10.0)
        raise RuntimeError("probe failure")

    with pytest.raises(ClipValidationError, match="probe failure"):
        extract_event_clips(
            events_json=events,
            input_video=video,
            output_dir=output_dir,
            output_manifest_json=manifest,
            runner=_successful_runner,
            metadata_loader=metadata_loader,
        )

    assert list(output_dir.iterdir()) == []
    assert not manifest.exists()


def test_overwrite_replaces_existing_outputs_after_validation(
    tmp_path: Path,
) -> None:
    events, video, output_dir, manifest = _prepare_pipeline(tmp_path)
    output_dir.mkdir()
    clip = output_dir / "event_001.mp4"
    clip.write_bytes(b"old-clip")
    manifest.write_text("old-manifest", encoding="utf-8")

    result = extract_event_clips(
        events_json=events,
        input_video=video,
        output_dir=output_dir,
        output_manifest_json=manifest,
        ffprobe_bin="probe-custom",
        overwrite=True,
        runner=_successful_runner,
        metadata_loader=_pipeline_metadata_loader(video),
    )

    assert result.clip_count == 1
    assert clip.read_bytes() == b"encoded-clip"
    assert json.loads(manifest.read_text(encoding="utf-8"))["summary"][
        "clip_count"
    ] == 1
    assert not list(tmp_path.rglob("*.backup"))


def test_install_failure_restores_all_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_records = [
        _event("event_001", 1.0, 2.0),
        _event("event_002", 3.0, 4.0),
    ]
    events, video, output_dir, manifest = _prepare_pipeline(
        tmp_path,
        events=event_records,
    )
    output_dir.mkdir()
    first = output_dir / "event_001.mp4"
    second = output_dir / "event_002.mp4"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    manifest.write_text("old-manifest", encoding="utf-8")
    original_replace = Path.replace

    def fail_manifest_install(self: Path, target: Path) -> Path:
        if Path(target) == manifest and not self.name.endswith(".backup"):
            raise OSError("simulated manifest install failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_install)

    with pytest.raises(OSError, match="simulated manifest install failure"):
        extract_event_clips(
            events_json=events,
            input_video=video,
            output_dir=output_dir,
            output_manifest_json=manifest,
            ffprobe_bin="probe-custom",
            overwrite=True,
            runner=_successful_runner,
            metadata_loader=_pipeline_metadata_loader(video, [1.0, 1.0]),
        )

    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert manifest.read_text(encoding="utf-8") == "old-manifest"
    assert not list(tmp_path.rglob("*.backup"))


def test_pipeline_manifest_fields_order_filenames_and_event_order(
    tmp_path: Path,
) -> None:
    event_records = [
        _event("z_event", 1.0, 2.0),
        _event("a_event", 3.0, 5.0),
    ]
    events, video, output_dir, manifest = _prepare_pipeline(
        tmp_path,
        events=event_records,
    )

    result = extract_event_clips(
        events_json=events,
        input_video=video,
        output_dir=output_dir,
        output_manifest_json=manifest,
        ffprobe_bin="probe-custom",
        runner=_successful_runner,
        metadata_loader=_pipeline_metadata_loader(video, [1.0, 2.0]),
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))

    assert list(document) == [
        "schema_version",
        "source_events_json",
        "source_video",
        "output_directory",
        "summary",
        "clips",
    ]
    assert list(document["summary"]) == [
        "event_count",
        "clip_count",
        "total_requested_duration_sec",
        "total_actual_duration_sec",
    ]
    assert list(document["clips"][0]) == [
        "event_id",
        "output_path",
        "clip_start_sec",
        "clip_end_sec",
        "requested_duration_sec",
        "actual_duration_sec",
        "duration_difference_sec",
        "width",
        "height",
        "fps",
        "frame_count",
        "codec",
        "file_size_bytes",
    ]
    assert [clip["event_id"] for clip in document["clips"]] == [
        "z_event",
        "a_event",
    ]
    assert [Path(clip.output_path).name for clip in result.clips] == [
        "z_event.mp4",
        "a_event.mp4",
    ]
    assert document["summary"] == {
        "event_count": 2,
        "clip_count": 2,
        "total_requested_duration_sec": 3.0,
        "total_actual_duration_sec": 3.0,
    }


def test_manifest_rejects_nan_and_infinity(tmp_path: Path) -> None:
    record = ClipManifestRecord(
        event_id="event_001",
        output_path="event_001.mp4",
        clip_start_sec=0.0,
        clip_end_sec=1.0,
        requested_duration_sec=1.0,
        actual_duration_sec=float("nan"),
        duration_difference_sec=float("inf"),
        width=1280,
        height=720,
        fps=30.0,
        frame_count=30,
        codec="h264",
        file_size_bytes=100,
    )

    with pytest.raises(ValueError, match="non-finite"):
        build_clip_manifest(
            source_events_json="events.json",
            source_video="input.mp4",
            output_directory="clips",
            clips=[record],
        )

    with pytest.raises(ValueError, match="non-finite"):
        write_clip_manifest({"bad": float("nan")}, tmp_path / "manifest.json")


def test_pipeline_accepts_absolute_temporary_paths(tmp_path: Path) -> None:
    events, video, output_dir, manifest = _prepare_pipeline(tmp_path)

    result = extract_event_clips(
        events_json=events.resolve(),
        input_video=video.resolve(),
        output_dir=output_dir.resolve(),
        output_manifest_json=manifest.resolve(),
        ffprobe_bin="probe-custom",
        runner=_successful_runner,
        metadata_loader=_pipeline_metadata_loader(video.resolve()),
    )

    assert Path(result.clips[0].output_path).is_absolute()
    assert Path(result.manifest_path).is_file()


def test_cli_passes_all_arguments_and_prints_success_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import day6_extract_event_clips

    captured: dict[str, object] = {}
    clip = ClipManifestRecord(
        event_id="event_001",
        output_path=str(tmp_path / "clips" / "event_001.mp4"),
        clip_start_sec=1.0,
        clip_end_sec=2.0,
        requested_duration_sec=1.0,
        actual_duration_sec=1.0,
        duration_difference_sec=0.0,
        width=1280,
        height=720,
        fps=30.0,
        frame_count=30,
        codec="h264",
        file_size_bytes=123,
    )

    def fake_extract(**kwargs: object) -> EventClipPipelineResult:
        captured.update(kwargs)
        return EventClipPipelineResult(
            events_json_path=str(kwargs["events_json"]),
            input_video_path=str(kwargs["input_video"]),
            output_directory=str(kwargs["output_dir"]),
            manifest_path=str(kwargs["output_manifest_json"]),
            event_count=1,
            clip_count=1,
            clips=(clip,),
        )

    monkeypatch.setattr(day6_extract_event_clips, "extract_event_clips", fake_extract)
    paths = [
        tmp_path / "events.json",
        tmp_path / "input.mp4",
        tmp_path / "clips",
        tmp_path / "manifest.json",
    ]
    exit_code = day6_extract_event_clips.main(
        [
            "--events-json",
            str(paths[0]),
            "--input-video",
            str(paths[1]),
            "--output-dir",
            str(paths[2]),
            "--output-manifest-json",
            str(paths[3]),
            "--ffmpeg-bin",
            "ffmpeg-custom",
            "--ffprobe-bin",
            "ffprobe-custom",
            "--overwrite",
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert captured == {
        "events_json": paths[0],
        "input_video": paths[1],
        "output_dir": paths[2],
        "output_manifest_json": paths[3],
        "ffmpeg_bin": "ffmpeg-custom",
        "ffprobe_bin": "ffprobe-custom",
        "overwrite": True,
    }
    assert stdout["clip_count"] == 1
    assert stdout["clips"][0]["event_id"] == "event_001"
    assert stdout["clips"][0]["codec"] == "h264"


def test_cli_reports_failure_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import day6_extract_event_clips

    def fail_extract(**_: object) -> EventClipPipelineResult:
        raise ValueError("bad event JSON")

    monkeypatch.setattr(day6_extract_event_clips, "extract_event_clips", fail_extract)
    exit_code = day6_extract_event_clips.main(
        [
            "--events-json",
            "events.json",
            "--input-video",
            "input.mp4",
            "--output-dir",
            "clips",
            "--output-manifest-json",
            "manifest.json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ValueError" in captured.err
    assert "bad event JSON" in captured.err


def test_new_files_do_not_contain_absolute_project_path() -> None:
    repository = Path(__file__).resolve().parents[1]
    paths = [
        repository / "src" / "dashguard" / "video" / "event_clips.py",
        repository / "scripts" / "day6_extract_event_clips.py",
        Path(__file__),
    ]

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "E:\\Projects\\DashGuard" not in source

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dashguard.demo import workflow
from dashguard.video.metadata import VideoMetadata


def _metadata(
    *,
    duration_sec: float = 1.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    frame_count: int | None = 30,
) -> VideoMetadata:
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        codec="h264",
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "input.mp4"
    model = tmp_path / "yolo11n.pt"
    video.write_bytes(b"input-video")
    model.write_bytes(b"model")
    return video, model


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    event_count: int = 0,
    matched_rows: int = 1,
    metadata: VideoMetadata | None = None,
    fail_stage: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    video_metadata = metadata or _metadata()

    def probe(path: str | Path) -> VideoMetadata:
        calls.append(("probe", {"path": Path(path)}))
        return video_metadata

    def standardize(**kwargs: Any) -> Path:
        calls.append(("standardize", kwargs))
        if fail_stage == "standardize":
            raise RuntimeError("standardize failed")
        destination = Path(kwargs["output_path"])
        destination.write_bytes(b"standardized-video")
        return destination

    def track(**kwargs: Any) -> SimpleNamespace:
        calls.append(("tracking", kwargs))
        if fail_stage == "tracking":
            raise RuntimeError("tracking failed")
        Path(kwargs["output_video_path"]).write_bytes(b"annotated-video")
        Path(kwargs["output_csv_path"]).write_text(
            "frame_index,timestamp_sec,track_id\n0,0,1\n", encoding="utf-8"
        )
        Path(kwargs["metrics_path"]).write_text(
            '{"processed_frames": 30}\n', encoding="utf-8"
        )
        return SimpleNamespace(processed_frames=30)

    def risk(**kwargs: Any) -> SimpleNamespace:
        calls.append(("risk", kwargs))
        if fail_stage == "risk":
            raise RuntimeError("risk failed")
        Path(kwargs["output_csv"]).write_text(
            "timestamp_sec,total_score\n0,0.8\n", encoding="utf-8"
        )
        Path(kwargs["output_plot"]).write_bytes(b"risk-plot")
        return SimpleNamespace(output_rows=1)

    def synthetic(**kwargs: Any) -> SimpleNamespace:
        calls.append(("synthetic", kwargs))
        if fail_stage == "synthetic":
            raise RuntimeError("synthetic failed")
        Path(kwargs["output_csv"]).write_text(
            "timestamp_sec,speed_kmh,accel_longitudinal,accel_lateral,brake,"
            "telemetry_source\n0,30,0,0,0,synthetic\n",
            encoding="utf-8",
        )
        return SimpleNamespace(rows=1)

    def sync(**kwargs: Any) -> SimpleNamespace:
        calls.append(("sync", kwargs))
        if fail_stage == "sync":
            raise RuntimeError("sync failed")
        Path(kwargs["output_csv"]).write_text(
            "timestamp_sec,speed_kmh,accel_longitudinal,accel_lateral,brake,"
            "telemetry_matched,hard_brake\n"
            f"0,30,0,0,0,{'True' if matched_rows else 'False'},False\n",
            encoding="utf-8",
        )
        return SimpleNamespace(matched_rows=matched_rows)

    def plot(
        synchronized_csv_path: str | Path,
        output_plot_path: str | Path,
        *,
        telemetry_source: str,
    ) -> Path:
        calls.append(
            (
                "plot",
                {
                    "synchronized_csv_path": Path(synchronized_csv_path),
                    "output_plot_path": Path(output_plot_path),
                    "telemetry_source": telemetry_source,
                },
            )
        )
        if fail_stage == "plot":
            raise RuntimeError("plot failed")
        output = Path(output_plot_path)
        output.write_bytes(b"telemetry-plot")
        return output

    def events(**kwargs: Any) -> SimpleNamespace:
        calls.append(("events", kwargs))
        if fail_stage == "events":
            raise RuntimeError("events failed")
        Path(kwargs["output_timeline_csv"]).write_text(
            "frame_index,event_candidate\n0,True\n", encoding="utf-8"
        )
        event_rows = [
            {
                "event_id": f"event_{index:03d}",
                "clip_start_sec": 0.0,
                "clip_end_sec": 0.5,
            }
            for index in range(1, event_count + 1)
        ]
        Path(kwargs["output_events_json"]).write_text(
            json.dumps(
                {"summary": {"event_count": event_count}, "events": event_rows}
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(event_count=event_count)

    def clips(**kwargs: Any) -> SimpleNamespace:
        calls.append(("clips", kwargs))
        if fail_stage == "clips":
            raise RuntimeError("clips failed")
        output_dir = Path(kwargs["output_dir"])
        records = []
        for index in range(1, event_count + 1):
            path = output_dir / f"event_{index:03d}.mp4"
            path.write_bytes(f"clip-{index}".encode())
            records.append(SimpleNamespace(output_path=str(path)))
        Path(kwargs["output_manifest_json"]).write_text(
            json.dumps({"summary": {"clip_count": event_count}}),
            encoding="utf-8",
        )
        return SimpleNamespace(clips=tuple(records), clip_count=event_count)

    monkeypatch.setattr(workflow, "probe_video_metadata", probe)
    monkeypatch.setattr(workflow, "standardize_video", standardize)
    monkeypatch.setattr(workflow, "run_tracked_video_analysis", track)
    monkeypatch.setattr(workflow, "generate_video_risk_report", risk)
    monkeypatch.setattr(workflow, "generate_synthetic_telemetry_csv", synthetic)
    monkeypatch.setattr(workflow, "sync_risk_telemetry_csv", sync)
    monkeypatch.setattr(workflow, "create_telemetry_plot", plot)
    monkeypatch.setattr(workflow, "generate_event_outputs", events)
    monkeypatch.setattr(workflow, "extract_event_clips", clips)
    return calls


def _run(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    **kwargs: Any,
) -> workflow.DemoRunResult:
    video, model = inputs
    arguments: dict[str, Any] = {
        "video_path": video,
        "model_path": model,
        "output_root": tmp_path / "runs",
        "run_id": "unit_run",
    }
    arguments.update(kwargs)
    return workflow.run_dashguard_demo(**arguments)


def test_generates_unique_run_workspaces(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    first = _run(tmp_path, inputs, run_id=None)
    second = _run(tmp_path, inputs, run_id=None)

    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir
    assert first.run_dir.is_dir()
    assert second.run_dir.is_dir()


def test_accepts_explicit_safe_run_id(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    result = _run(tmp_path, inputs, run_id="day8-safe_20260804")

    assert result.run_id == "day8-safe_20260804"
    assert result.run_dir.name == "day8-safe_20260804"


def test_duplicate_run_id_is_rejected_without_changing_existing_run(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch)
    result = _run(tmp_path, inputs)
    before = result.run_summary_path.read_bytes()
    calls.clear()

    with pytest.raises(FileExistsError, match="already exists"):
        _run(tmp_path, inputs)

    assert calls == []
    assert result.run_summary_path.read_bytes() == before


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "folder/run", "folder\\run", "..", "CON", "bad:name", ""],
)
def test_rejects_unsafe_run_ids_before_workspace_creation(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    run_id: str,
) -> None:
    video, model = inputs
    root = tmp_path / "runs"

    with pytest.raises(ValueError, match="run_id"):
        workflow.run_dashguard_demo(
            video_path=video,
            model_path=model,
            output_root=root,
            run_id=run_id,
        )

    assert not root.exists()


def test_missing_video_is_rejected_without_workspace(
    tmp_path: Path,
    inputs: tuple[Path, Path],
) -> None:
    _, model = inputs
    root = tmp_path / "runs"

    with pytest.raises(FileNotFoundError, match="Video path"):
        workflow.run_dashguard_demo(
            video_path=tmp_path / "missing.mp4",
            model_path=model,
            output_root=root,
        )

    assert not root.exists()


def test_unsupported_video_extension_is_rejected_without_workspace(
    tmp_path: Path,
    inputs: tuple[Path, Path],
) -> None:
    _, model = inputs
    video = tmp_path / "input.txt"
    video.write_bytes(b"video")
    root = tmp_path / "runs"

    with pytest.raises(ValueError, match="Unsupported video extension"):
        workflow.run_dashguard_demo(
            video_path=video,
            model_path=model,
            output_root=root,
        )

    assert not root.exists()


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(duration_sec=0.0),
        _metadata(duration_sec=float("nan")),
        _metadata(width=0),
        _metadata(height=0),
        _metadata(fps=0.0),
    ],
)
def test_invalid_metadata_is_rejected_without_workspace(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    metadata: VideoMetadata,
) -> None:
    video, model = inputs
    root = tmp_path / "runs"
    monkeypatch.setattr(workflow, "probe_video_metadata", lambda _: metadata)

    with pytest.raises(ValueError):
        workflow.run_dashguard_demo(
            video_path=video,
            model_path=model,
            output_root=root,
        )

    assert not root.exists()


def test_over_30_second_video_is_rejected_without_workspace(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video, model = inputs
    root = tmp_path / "runs"
    monkeypatch.setattr(
        workflow, "probe_video_metadata", lambda _: _metadata(duration_sec=30.001)
    )

    with pytest.raises(ValueError, match="exceeds"):
        workflow.run_dashguard_demo(
            video_path=video,
            model_path=model,
            output_root=root,
        )

    assert not root.exists()


def test_missing_model_is_rejected_without_workspace(
    tmp_path: Path,
    inputs: tuple[Path, Path],
) -> None:
    video, _ = inputs
    root = tmp_path / "runs"

    with pytest.raises(FileNotFoundError, match="Model path"):
        workflow.run_dashguard_demo(
            video_path=video,
            model_path=tmp_path / "missing.pt",
            output_root=root,
        )

    assert not root.exists()


@pytest.mark.parametrize("name", ["missing.csv", "telemetry.txt"])
def test_invalid_telemetry_path_or_extension_is_rejected(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    video, model = inputs
    telemetry = tmp_path / name
    if telemetry.suffix == ".txt":
        telemetry.write_text("data", encoding="utf-8")
    root = tmp_path / "runs"
    monkeypatch.setattr(workflow, "probe_video_metadata", lambda _: _metadata())

    with pytest.raises((FileNotFoundError, ValueError)):
        workflow.run_dashguard_demo(
            video_path=video,
            telemetry_csv_path=telemetry,
            model_path=model,
            output_root=root,
        )

    assert not root.exists()


def test_public_library_call_order(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch, event_count=1)
    _run(tmp_path, inputs)

    assert [name for name, _ in calls] == [
        "probe",
        "standardize",
        "probe",
        "tracking",
        "risk",
        "synthetic",
        "sync",
        "plot",
        "events",
        "clips",
    ]


def test_backend_and_default_parameters_are_fixed_to_pytorch_cpu(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch)
    result = _run(tmp_path, inputs)
    arguments = dict(calls)["tracking"]

    assert result.backend == "pytorch_cpu"
    assert arguments["confidence"] == 0.25
    assert arguments["imgsz"] == 640
    assert arguments["trajectory_length"] == 30
    assert arguments["max_frames"] == 30
    assert arguments["device"] == "cpu"


def test_provided_telemetry_is_copied_and_not_regenerated(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch)
    telemetry = tmp_path / "sensor.csv"
    telemetry.write_bytes(b"sensor-input")

    result = _run(tmp_path, inputs, telemetry_csv_path=telemetry)

    assert result.telemetry_source == "provided"
    assert result.telemetry_csv_path.name == "telemetry_input.csv"
    assert result.telemetry_csv_path.read_bytes() == b"sensor-input"
    assert "synthetic" not in [name for name, _ in calls]
    assert telemetry.read_bytes() == b"sensor-input"


def test_missing_telemetry_uses_standardized_duration_and_default_frequency(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch, metadata=_metadata(duration_sec=2.5))
    result = _run(tmp_path, inputs)
    arguments = dict(calls)["synthetic"]

    assert result.telemetry_source == "synthetic"
    assert result.telemetry_csv_path.name == "synthetic_telemetry.csv"
    assert arguments["duration_sec"] == 2.5
    assert arguments["frequency_hz"] == 10.0
    assert arguments["overwrite"] is False


def test_all_unmatched_telemetry_is_a_success_with_warning(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch, matched_rows=0)
    result = _run(tmp_path, inputs)

    assert result.status == workflow.SUCCESS_STATUS
    assert any("No telemetry rows matched" in warning for warning in result.warnings)


def test_zero_events_skips_clip_extraction_and_has_no_fake_manifest(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch, event_count=0)
    result = _run(tmp_path, inputs)

    assert "clips" not in [name for name, _ in calls]
    assert result.event_count == 0
    assert result.clip_manifest_path is None
    assert result.clip_paths == ()
    assert result.published_clip_paths == ()
    assert list((result.run_dir / "clips").iterdir()) == []


def test_events_are_extracted_returned_and_published(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch, event_count=2)
    result = _run(tmp_path, inputs)

    assert "clips" in [name for name, _ in calls]
    assert result.event_count == 2
    assert result.clip_manifest_path == result.run_dir / "clips/clip_manifest.json"
    assert [path.name for path in result.clip_paths] == [
        "event_001.mp4",
        "event_002.mp4",
    ]
    assert all(path.is_file() for path in result.published_clip_paths)


def test_published_artifacts_are_distinct_content_identical_copies(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch, event_count=1)
    result = _run(tmp_path, inputs)
    pairs = [
        (result.annotated_video_path, result.published_annotated_video_path),
        (result.risk_plot_path, result.published_risk_plot_path),
        (result.telemetry_plot_path, result.published_telemetry_plot_path),
        (result.clip_paths[0], result.published_clip_paths[0]),
    ]

    for primary, published in pairs:
        assert primary != published
        assert primary.read_bytes() == published.read_bytes()

    summary = json.loads(result.run_summary_path.read_text(encoding="utf-8"))
    integrity = summary["published_integrity"]["annotated_video"]
    assert integrity["primary_sha256_before"] == integrity["primary_sha256_after"]
    assert integrity["primary_sha256_after"] == integrity["published_sha256"]


def test_progress_callback_is_coarse_ordered_and_bounded(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    progress: list[tuple[str, float]] = []
    _run(tmp_path, inputs, progress_callback=lambda stage, value: progress.append((stage, value)))

    assert [stage for stage, _ in progress] == [
        "validating",
        "standardizing",
        "tracking",
        "risk",
        "telemetry",
        "events",
        "clips",
        "publishing",
        "completed",
    ]
    assert all(0.0 <= value <= 1.0 for _, value in progress)
    assert [value for _, value in progress] == sorted(value for _, value in progress)


def test_middle_stage_exception_stops_later_stages_and_is_reraised(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch, fail_stage="risk")

    with pytest.raises(RuntimeError, match="risk failed"):
        _run(tmp_path, inputs)

    assert [name for name, _ in calls][-1] == "risk"
    assert "sync" not in [name for name, _ in calls]


def test_failure_after_workspace_creation_writes_failed_summary(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch, fail_stage="risk")

    with pytest.raises(RuntimeError):
        _run(tmp_path, inputs)

    summary_path = tmp_path / "runs/unit_run/run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failed_stage"] == "risk"
    assert summary["error_type"] == "RuntimeError"
    assert summary["error_message"] == "risk failed"
    assert "analysis/annotated_video.mp4" in summary["artifacts"]


def test_success_summary_contains_relative_artifacts_and_contract_fields(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    result = _run(tmp_path, inputs)
    summary = json.loads(result.run_summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "succeeded"
    assert summary["backend"] == "pytorch_cpu"
    assert summary["telemetry_source"] == "synthetic"
    assert summary["event_count"] == 0
    assert summary["artifacts"]["annotated_video"] == "analysis/annotated_video.mp4"
    assert summary["artifacts"]["clip_manifest"] is None
    assert summary["total_processing_sec"] >= 0.0
    assert summary["limitations"]


def test_atomic_summary_replace_failure_leaves_no_partial_final_summary(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    original_replace = workflow.os.replace

    def fail_summary_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == "run_summary.json":
            raise OSError("summary install failed")
        original_replace(source, destination)

    monkeypatch.setattr(workflow.os, "replace", fail_summary_replace)

    with pytest.raises(OSError, match="summary install failed"):
        _run(tmp_path, inputs)

    run_dir = tmp_path / "runs/unit_run"
    assert not (run_dir / "run_summary.json").exists()
    assert not list(run_dir.glob(".run_summary.json.*.tmp"))


def test_two_explicit_runs_never_share_output_paths(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    first = _run(tmp_path, inputs, run_id="run_one")
    second = _run(tmp_path, inputs, run_id="run_two")

    assert first.annotated_video_path != second.annotated_video_path
    assert first.published_annotated_video_path != second.published_annotated_video_path
    assert first.run_summary_path.is_file()
    assert second.run_summary_path.is_file()


def test_primary_sha256_matches_published_sha256(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    result = _run(tmp_path, inputs)

    primary = hashlib.sha256(result.annotated_video_path.read_bytes()).hexdigest()
    published = hashlib.sha256(
        result.published_annotated_video_path.read_bytes()
    ).hexdigest()
    assert primary == published

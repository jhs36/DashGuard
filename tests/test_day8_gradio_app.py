from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import gradio as gr
import pandas as pd
import pytest

from dashguard.demo import gradio_app
from dashguard.demo.workflow import DemoRunResult


_STAGES = (
    "validating",
    "standardizing",
    "tracking",
    "risk",
    "telemetry",
    "events",
    "clips",
    "publishing",
    "completed",
)


def _write(path: Path, content: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _event(index: int) -> dict[str, object]:
    return {
        "event_id": f"event_{index:03d}",
        "start_sec": float(index),
        "end_sec": float(index) + 0.5,
        "duration_sec": 0.5,
        "max_total_score": 0.8,
        "primary_track_id": index,
        "primary_class": "car",
        "hard_brake": index % 2 == 0,
        "telemetry_source": "synthetic",
        "clip_start_sec": max(0.0, float(index) - 1.0),
        "clip_end_sec": float(index) + 1.0,
        "not_for_display": "internal",
    }


def _result(
    output_root: Path,
    run_id: str,
    *,
    event_count: int = 1,
    model_path: Path = Path("yolo11n.pt"),
) -> DemoRunResult:
    run_dir = output_root / run_id
    annotated = _write(run_dir / "analysis/annotated_video.mp4", b"video")
    risk = _write(run_dir / "risk/risk_proxy.png", b"risk")
    telemetry_plot = _write(
        run_dir / "telemetry/telemetry_plot.png", b"telemetry"
    )
    published_annotated = _write(
        run_dir / "published/annotated_video.mp4", b"video"
    )
    published_risk = _write(run_dir / "published/risk_proxy.png", b"risk")
    published_telemetry = _write(
        run_dir / "published/telemetry_plot.png", b"telemetry"
    )

    events = [_event(index) for index in range(1, event_count + 1)]
    events_json = run_dir / "events/events.json"
    events_json.parent.mkdir(parents=True, exist_ok=True)
    events_json.write_text(
        json.dumps({"schema_version": "1.0", "events": events}),
        encoding="utf-8",
    )

    primary_clips = tuple(
        _write(run_dir / f"clips/event_{index:03d}.mp4", b"clip")
        for index in range(1, event_count + 1)
    )
    published_clips = tuple(
        _write(run_dir / f"published/event_{index:03d}.mp4", b"clip")
        for index in range(1, event_count + 1)
    )

    return DemoRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status="succeeded",
        backend="pytorch_cpu",
        model_path=model_path,
        telemetry_source="synthetic",
        input_duration_sec=12.5,
        total_processing_sec=4.25,
        standardized_video_path=run_dir / "input/standardized_input.mp4",
        annotated_video_path=annotated,
        published_annotated_video_path=published_annotated,
        tracks_csv_path=run_dir / "analysis/tracks.csv",
        metrics_json_path=run_dir / "analysis/metrics.json",
        risk_csv_path=run_dir / "risk/risk_scores.csv",
        risk_plot_path=risk,
        published_risk_plot_path=published_risk,
        telemetry_csv_path=run_dir / "input/synthetic_telemetry.csv",
        synchronized_csv_path=run_dir / "telemetry/risk_telemetry_sync.csv",
        telemetry_plot_path=telemetry_plot,
        published_telemetry_plot_path=published_telemetry,
        timeline_csv_path=run_dir / "events/frame_timeline.csv",
        events_json_path=events_json,
        event_count=event_count,
        clip_manifest_path=(
            run_dir / "clips/clip_manifest.json" if event_count else None
        ),
        clip_paths=primary_clips,
        published_clip_paths=published_clips,
        warnings=("detector warning",),
        limitations=(
            "The risk proxy is not an estimate of real distance, TTC, or crash probability.",
            "Synthetic telemetry is illustrative.",
            "The PyTorch CPU backend does not guarantee real-time 30 FPS processing.",
        ),
        run_summary_path=run_dir / "run_summary.json",
    )


class _ProgressRecorder:
    def __init__(self) -> None:
        self.updates: list[tuple[float, str | None]] = []

    def __call__(self, value: float, *, desc: str | None = None) -> None:
        self.updates.append((value, desc))


def _install_success_workflow(
    monkeypatch: pytest.MonkeyPatch,
    output_root: Path,
    *,
    event_count: int = 1,
    mutate_result: Any = None,
) -> dict[str, Any]:
    call: dict[str, Any] = {}

    def fake_workflow(**kwargs: Any) -> DemoRunResult:
        call.update(kwargs)
        for index, stage in enumerate(_STAGES):
            kwargs["progress_callback"](stage, float(index) / 8.0)
        result = _result(
            output_root,
            kwargs["run_id"],
            event_count=event_count,
            model_path=Path(kwargs["model_path"]),
        )
        return mutate_result(result) if mutate_result is not None else result

    monkeypatch.setattr(gradio_app, "run_dashguard_demo", fake_workflow)
    return call


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    telemetry_path: str | None = None,
    event_count: int = 1,
    mutate_result: Any = None,
) -> tuple[tuple[Any, ...], dict[str, Any], _ProgressRecorder]:
    output_root = tmp_path / "runs"
    calls = _install_success_workflow(
        monkeypatch,
        output_root,
        event_count=event_count,
        mutate_result=mutate_result,
    )
    progress = _ProgressRecorder()
    callback = gradio_app._analysis_callback(
        output_root=output_root,
        model_path=tmp_path / "yolo11n.pt",
    )
    returned = callback("uploaded.mp4", telemetry_path, progress)
    return returned, calls, progress


def test_create_gradio_app_returns_blocks() -> None:
    assert isinstance(gradio_app.create_gradio_app(), gr.Blocks)


def test_components_and_single_callback_binding_are_created() -> None:
    app = gradio_app.create_gradio_app()
    config = app.config
    component_types = [component["type"] for component in config["components"]]
    labels = {
        component["props"].get("label") for component in config["components"]
    }

    assert len(config["dependencies"]) == 1
    assert config["dependencies"][0]["backend_fn"] is True
    callback = next(iter(app.fns.values()))
    assert callback.concurrency_limit == 1
    assert callback.concurrency_id == "dashguard-analysis"
    assert component_types.count("video") == 3
    assert {"상태 요약", "risk proxy", "telemetry", "event 후보"} <= labels
    assert "전체 event clips" in labels


def test_video_is_upload_only_and_telemetry_is_optional_csv_filepath() -> None:
    app = gradio_app.create_gradio_app()
    components = app.config["components"]
    video = next(
        component
        for component in components
        if component["props"].get("label", "").startswith("입력 영상")
    )
    telemetry = next(
        component
        for component in components
        if component["props"].get("label") == "telemetry CSV (선택)"
    )

    assert video["props"]["sources"] == ["upload"]
    assert video["props"].get("format") is None
    assert telemetry["props"]["file_types"] == [".csv"]
    assert telemetry["props"]["type"] == "filepath"


def test_no_backend_selector_or_setting_slider_is_exposed() -> None:
    app = gradio_app.create_gradio_app()
    types = {component["type"] for component in app.config["components"]}
    text = json.dumps(app.config, ensure_ascii=False).lower()

    assert "dropdown" not in types
    assert "slider" not in types
    assert "onnx" not in text
    assert "directml" not in text


def test_missing_video_is_rejected_before_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(**_: Any) -> None:
        pytest.fail("workflow must not run without a video")

    monkeypatch.setattr(gradio_app, "run_dashguard_demo", fail_if_called)
    callback = gradio_app._analysis_callback(
        output_root=tmp_path / "runs", model_path=tmp_path / "yolo11n.pt"
    )

    with pytest.raises(gr.Error, match="영상을 업로드"):
        callback(None, None, _ProgressRecorder())


@pytest.mark.parametrize("telemetry_path", [None, "uploaded-telemetry.csv"])
def test_workflow_receives_telemetry_filepath_or_none_and_fixed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry_path: str | None,
) -> None:
    _, call, _ = _invoke(
        tmp_path, monkeypatch, telemetry_path=telemetry_path
    )

    assert call["video_path"] == "uploaded.mp4"
    assert call["telemetry_csv_path"] == telemetry_path
    assert call["output_root"] == tmp_path / "runs"
    assert call["model_path"] == tmp_path / "yolo11n.pt"
    assert "backend" not in call
    assert "max_duration_sec" not in call


def test_progress_stages_use_descriptions_and_bounded_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, progress = _invoke(tmp_path, monkeypatch)

    assert len(progress.updates) == len(_STAGES)
    assert all(0.0 <= value <= 1.0 for value, _ in progress.updates)
    assert [value for value, _ in progress.updates] == sorted(
        value for value, _ in progress.updates
    )
    assert [description for _, description in progress.updates] == [
        "입력 확인",
        "영상 표준화",
        "객체 탐지·추적",
        "risk proxy 계산",
        "telemetry 처리",
        "event 후보 생성",
        "event clip 생성",
        "브라우저용 결과 준비",
        "완료",
    ]


@pytest.mark.parametrize(
    ("primary_attribute", "published_attribute", "output_index"),
    [
        ("annotated_video_path", "published_annotated_video_path", 1),
        ("risk_plot_path", "published_risk_plot_path", 2),
        ("telemetry_plot_path", "published_telemetry_plot_path", 3),
    ],
)
def test_primary_video_and_pngs_are_not_returned_and_published_files_are(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_attribute: str,
    published_attribute: str,
    output_index: int,
) -> None:
    captured: dict[str, DemoRunResult] = {}

    def capture(result: DemoRunResult) -> DemoRunResult:
        captured["result"] = result
        return result

    returned, _, _ = _invoke(tmp_path, monkeypatch, mutate_result=capture)
    result = captured["result"]

    assert returned[output_index] == str(
        Path(getattr(result, published_attribute)).resolve()
    )
    assert returned[output_index] != str(
        Path(getattr(result, primary_attribute)).resolve()
    )


def test_only_published_clips_are_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, DemoRunResult] = {}

    def capture(result: DemoRunResult) -> DemoRunResult:
        captured["result"] = result
        return result

    returned, _, _ = _invoke(tmp_path, monkeypatch, mutate_result=capture)
    result = captured["result"]
    published = str(result.published_clip_paths[0].resolve())
    primary = str(result.clip_paths[0].resolve())

    assert returned[5] == published
    assert returned[6] == [published]
    assert primary != returned[5]
    assert primary not in returned[6]


def test_published_path_outside_current_published_directory_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = _write(tmp_path / "outside.mp4")

    def move_published(result: DemoRunResult) -> DemoRunResult:
        return replace(result, published_annotated_video_path=outside)

    with pytest.raises(gr.Error, match="결과 파일 검증"):
        _invoke(tmp_path, monkeypatch, mutate_result=move_published)


def test_a_different_or_prior_run_is_never_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def switch_run(result: DemoRunResult) -> DemoRunResult:
        return _result(tmp_path / "runs", "prior_run")

    with pytest.raises(gr.Error, match="결과 파일 검증"):
        _invoke(tmp_path, monkeypatch, mutate_result=switch_run)


def test_events_json_is_read_into_dataframe_using_actual_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returned, _, _ = _invoke(tmp_path, monkeypatch)
    table = returned[4]

    assert isinstance(table, pd.DataFrame)
    assert table.loc[0, "event_id"] == "event_001"
    assert table.loc[0, "primary_class"] == "car"
    assert "start_sec" in table.columns
    assert "not_for_display" not in table.columns


def test_zero_events_has_fixed_empty_table_no_representative_and_empty_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returned, _, _ = _invoke(tmp_path, monkeypatch, event_count=0)

    assert list(returned[4].columns) == list(gradio_app._EVENT_COLUMNS)
    assert returned[4].empty
    assert returned[5] is None
    assert returned[6] == []
    assert "event candidate 없음" in returned[0]


def test_nonempty_events_return_first_published_clip_in_events_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returned, _, _ = _invoke(tmp_path, monkeypatch, event_count=2)

    assert Path(returned[5]).name == "event_001.mp4"
    assert [Path(path).name for path in returned[6]] == [
        "event_001.mp4",
        "event_002.mp4",
    ]
    assert "첫 번째 clip" in returned[0]


def test_status_contains_backend_model_duration_source_time_count_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = _invoke(tmp_path, monkeypatch)[0][0]

    assert "PyTorch CPU" in status
    assert "pytorch_cpu" in status
    assert "yolo11n.pt" in status
    assert "12.50초" in status
    assert "synthetic" in status
    assert "event count" in status and "1개" in status
    assert "4.25초" in status
    assert "risk proxy" in status
    assert "TTC" in status
    assert "30 FPS" in status


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ValueError("Video duration exceeds 30 seconds"), "최대 30초"),
        (RuntimeError("ffmpeg failed"), "분석을 완료하지 못했습니다"),
    ],
)
def test_workflow_value_and_runtime_errors_become_gradio_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    def fail(**_: Any) -> None:
        raise error

    monkeypatch.setattr(gradio_app, "run_dashguard_demo", fail)
    callback = gradio_app._analysis_callback(
        output_root=tmp_path / "runs", model_path=tmp_path / "yolo11n.pt"
    )

    with pytest.raises(gr.Error, match=message):
        callback("uploaded.mp4", None, _ProgressRecorder())


def test_unexpected_error_is_logged_and_never_returns_success_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(**_: Any) -> None:
        raise KeyError("secret-internal-detail")

    monkeypatch.setattr(gradio_app, "run_dashguard_demo", fail)
    callback = gradio_app._analysis_callback(
        output_root=tmp_path / "runs", model_path=tmp_path / "yolo11n.pt"
    )

    with pytest.raises(gr.Error, match="예상하지 못한 오류") as raised:
        callback("uploaded.mp4", None, _ProgressRecorder())

    assert "secret-internal-detail" not in str(raised.value)
    assert "secret-internal-detail" in caplog.text


def test_queue_accepts_default_concurrency_one_without_launching() -> None:
    app = gradio_app.create_gradio_app()

    queued = app.queue(default_concurrency_limit=1)

    assert queued is app
    assert app.enable_queue is True


def _load_launch_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts/day8_gradio_demo.py"
    spec = importlib.util.spec_from_file_location("day8_gradio_demo", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_fixes_queue_localhost_share_and_upload_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_launch_script()
    recorded: dict[str, Any] = {}

    class FakeApp:
        def queue(self, **kwargs: Any) -> "FakeApp":
            recorded["queue"] = kwargs
            return self

        def launch(self, **kwargs: Any) -> None:
            recorded["launch"] = kwargs

    def fake_create(**kwargs: Any) -> FakeApp:
        recorded["create"] = kwargs
        return FakeApp()

    monkeypatch.setattr(script, "create_gradio_app", fake_create)
    output_root = tmp_path / "runs"
    script.main(
        [
            "--output-root",
            str(output_root),
            "--model-path",
            str(tmp_path / "yolo11n.pt"),
            "--port",
            "8765",
        ]
    )

    assert recorded["queue"] == {"default_concurrency_limit": 1}
    assert recorded["launch"]["server_name"] == "127.0.0.1"
    assert recorded["launch"]["server_port"] == 8765
    assert recorded["launch"]["share"] is False
    assert recorded["launch"]["max_file_size"] == "1gb"
    assert recorded["launch"]["allowed_paths"] == [str(output_root.resolve())]

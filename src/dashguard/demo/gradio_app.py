"""Thin local Gradio UI for the validated Day 8 demo workflow."""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from dashguard.demo.workflow import BACKEND, SUCCESS_STATUS, DemoRunResult
from dashguard.demo.workflow import run_dashguard_demo


_LOGGER = logging.getLogger(__name__)

_PROGRESS_STAGES: dict[str, tuple[float, str]] = {
    "validating": (0.00, "입력 확인"),
    "standardizing": (0.10, "영상 표준화"),
    "tracking": (0.25, "객체 탐지·추적"),
    "risk": (0.45, "risk proxy 계산"),
    "telemetry": (0.58, "telemetry 처리"),
    "events": (0.70, "event 후보 생성"),
    "clips": (0.80, "event clip 생성"),
    "publishing": (0.90, "브라우저용 결과 준비"),
    "completed": (1.00, "완료"),
}

# These names come from the Day 8 events.json schema.  For non-empty input,
# only keys actually present in the document are selected.
_EVENT_COLUMNS = (
    "event_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "max_total_score",
    "primary_track_id",
    "primary_class",
    "hard_brake",
    "telemetry_source",
    "clip_start_sec",
    "clip_end_sec",
)

_INTRODUCTION = """
# DashGuard 로컬 영상 분석 데모

- **PyTorch CPU / `yolo11n.pt` 고정**, 입력 영상은 최대 **30초**입니다.
- telemetry CSV는 선택 입력이며, 미입력 시 deterministic synthetic telemetry를 사용합니다.
- risk proxy는 실제 거리, TTC, 사고·충돌 확률이 아닙니다.
- 실제 안전성 또는 30 FPS 실시간 처리를 보장하지 않습니다.
""".strip()


class _ResultValidationError(RuntimeError):
    """Raised when workflow output is unsafe or inconsistent for presentation."""


def _make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"gradio_{timestamp}_{uuid.uuid4().hex}"


def _resolved_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise _ResultValidationError(f"{label} is missing or is not a file")
    try:
        return candidate.resolve(strict=True)
    except OSError as error:
        raise _ResultValidationError(f"{label} could not be resolved") from error


def _require_beneath(path: Path, directory: Path, *, label: str) -> None:
    try:
        path.relative_to(directory)
    except ValueError as error:
        raise _ResultValidationError(
            f"{label} is outside the current run directory"
        ) from error


def _validate_pair(
    *,
    primary_path: str | Path,
    published_path: str | Path,
    run_dir: Path,
    published_dir: Path,
    label: str,
) -> Path:
    primary = _resolved_file(primary_path, label=f"primary {label}")
    published = _resolved_file(published_path, label=f"published {label}")
    _require_beneath(primary, run_dir, label=f"primary {label}")
    _require_beneath(published, published_dir, label=f"published {label}")
    if primary == published:
        raise _ResultValidationError(
            f"primary and published {label} paths must be different"
        )
    return published


def _read_events(events_json_path: Path) -> tuple[pd.DataFrame, int]:
    try:
        with events_json_path.open("r", encoding="utf-8") as input_file:
            document = json.load(input_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _ResultValidationError("events.json could not be read") from error

    if not isinstance(document, dict) or not isinstance(document.get("events"), list):
        raise _ResultValidationError("events.json must contain an events array")
    events = document["events"]
    if not events:
        return pd.DataFrame(columns=list(_EVENT_COLUMNS)), 0
    if not all(isinstance(event, dict) for event in events):
        raise _ResultValidationError("events.json contains an invalid event record")

    columns = [
        column
        for column in _EVENT_COLUMNS
        if any(column in event for event in events)
    ]
    return pd.DataFrame.from_records(events).loc[:, columns], len(events)


def _validate_result(
    result: DemoRunResult,
    *,
    expected_output_root: Path,
    expected_run_id: str,
) -> tuple[str, str, str, pd.DataFrame, str | None, list[str]]:
    if result.status != SUCCESS_STATUS:
        raise _ResultValidationError("workflow did not return a successful result")
    if result.backend != BACKEND:
        raise _ResultValidationError("workflow returned an unexpected backend")
    if result.run_id != expected_run_id:
        raise _ResultValidationError("workflow returned a different run")

    try:
        output_root = expected_output_root.resolve(strict=True)
        run_dir = Path(result.run_dir).resolve(strict=True)
        published_dir = (run_dir / "published").resolve(strict=True)
    except OSError as error:
        raise _ResultValidationError("current run directory is missing") from error
    if not run_dir.is_dir() or run_dir.parent != output_root:
        raise _ResultValidationError("workflow returned a run outside output_root")
    if run_dir.name != expected_run_id or not published_dir.is_dir():
        raise _ResultValidationError("workflow returned an inconsistent run directory")

    annotated_video = _validate_pair(
        primary_path=result.annotated_video_path,
        published_path=result.published_annotated_video_path,
        run_dir=run_dir,
        published_dir=published_dir,
        label="annotated video",
    )
    risk_plot = _validate_pair(
        primary_path=result.risk_plot_path,
        published_path=result.published_risk_plot_path,
        run_dir=run_dir,
        published_dir=published_dir,
        label="risk plot",
    )
    telemetry_plot = _validate_pair(
        primary_path=result.telemetry_plot_path,
        published_path=result.published_telemetry_plot_path,
        run_dir=run_dir,
        published_dir=published_dir,
        label="telemetry plot",
    )

    primary_clips = tuple(result.clip_paths)
    published_clips = tuple(result.published_clip_paths)
    if len(primary_clips) != len(published_clips):
        raise _ResultValidationError("primary and published clip counts differ")
    safe_clips = [
        _validate_pair(
            primary_path=primary,
            published_path=published,
            run_dir=run_dir,
            published_dir=published_dir,
            label=f"event clip {index}",
        )
        for index, (primary, published) in enumerate(
            zip(primary_clips, published_clips, strict=True), start=1
        )
    ]

    events_json = _resolved_file(result.events_json_path, label="events.json")
    expected_events_json = (run_dir / "events" / "events.json").resolve()
    if events_json != expected_events_json:
        raise _ResultValidationError("events.json is not from the current run")
    event_table, event_count = _read_events(events_json)
    if event_count != result.event_count:
        raise _ResultValidationError("events.json event count differs from the result")
    if event_count != len(safe_clips):
        raise _ResultValidationError("event and published clip counts differ")

    return (
        str(annotated_video),
        str(risk_plot),
        str(telemetry_plot),
        event_table,
        str(safe_clips[0]) if safe_clips else None,
        [str(path) for path in safe_clips],
    )


def _markdown_text(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("`", "\\`")


def _status_markdown(result: DemoRunResult) -> str:
    event_text = (
        "event candidate 없음"
        if result.event_count == 0
        else f"{result.event_count}개 (대표 영상은 events 순서상 첫 번째 clip)"
    )
    telemetry_text = (
        "synthetic (deterministic, illustrative)"
        if result.telemetry_source == "synthetic"
        else "provided"
    )
    warnings = list(result.warnings)
    limitations = list(result.limitations)
    warning_lines = "\n".join(
        f"- {_markdown_text(warning)}" for warning in warnings
    ) or "- 없음"
    limitation_lines = "\n".join(
        f"- {_markdown_text(limitation)}" for limitation in limitations
    ) or "- risk proxy는 실제 거리, TTC, 사고·충돌 확률이 아닙니다."

    return f"""
## ✅ 분석 성공

- **run ID:** `{_markdown_text(result.run_id)}`
- **backend:** PyTorch CPU (`{_markdown_text(result.backend)}`)
- **model:** `{_markdown_text(Path(result.model_path).name)}`
- **입력 길이:** {result.input_duration_sec:.2f}초
- **telemetry source:** {telemetry_text}
- **event count:** {event_text}
- **전체 처리시간:** {result.total_processing_sec:.2f}초

**결과 적용 범위:** 현재 업로드 영상의 로컬 데모 분석 결과이며 실제 안전성 판단에 사용할 수 없습니다.

**warnings**
{warning_lines}

**limitations**
{limitation_lines}
""".strip()


def _failure_summary_context(output_root: Path, run_id: str) -> str | None:
    """Read only this callback's failed summary; never search prior runs."""

    summary_path = output_root / run_id / "run_summary.json"
    if not summary_path.is_file():
        return None
    try:
        document = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(document, dict)
        or document.get("status") != "failed"
        or document.get("run_id") != run_id
    ):
        return None
    failed_stage = document.get("failed_stage")
    return failed_stage if isinstance(failed_stage, str) else None


def _safe_error_message(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, FileNotFoundError):
        if "model" in message:
            return "분석 모델 파일을 찾을 수 없습니다."
        if "telemetry" in message:
            return "telemetry CSV 파일을 찾을 수 없습니다."
        return "업로드한 영상 파일을 찾을 수 없습니다. 다시 업로드해 주세요."
    if isinstance(error, ValueError) and not isinstance(error, gr.Error):
        if "duration" in message and ("exceed" in message or "30" in message):
            return "영상은 최대 30초까지 분석할 수 있습니다."
        if "unsupported video" in message or "extension" in message:
            return "지원하지 않는 영상 컨테이너입니다."
        if "telemetry" in message or "csv" in message:
            return "telemetry CSV 형식이 올바르지 않습니다."
        return "입력 또는 분석 결과 형식이 올바르지 않습니다."
    if isinstance(error, _ResultValidationError):
        return "브라우저용 결과 파일 검증에 실패했습니다."
    if isinstance(error, RuntimeError):
        return "영상 분석을 완료하지 못했습니다."
    return "예상하지 못한 오류로 분석을 완료하지 못했습니다."


def _analysis_callback(
    *,
    output_root: Path,
    model_path: Path,
) -> Callable[..., tuple[Any, ...]]:
    resolved_output_root = output_root.resolve()

    def analyze_video(
        video_path: str | None,
        telemetry_csv_path: str | None,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[Any, ...]:
        if video_path is None:
            raise gr.Error(
                "분석할 영상을 업로드하세요.", print_exception=False
            )

        run_id = _make_run_id()
        current_stage = "validating"

        def report_progress(stage: str, workflow_value: float) -> None:
            nonlocal current_stage
            current_stage = stage
            if stage in _PROGRESS_STAGES:
                value, description = _PROGRESS_STAGES[stage]
            else:
                try:
                    value = min(1.0, max(0.0, float(workflow_value)))
                except (TypeError, ValueError, OverflowError):
                    value = 0.0
                if not math.isfinite(value):
                    value = 0.0
                description = "분석 진행"
            progress(value, desc=description)

        try:
            result = run_dashguard_demo(
                video_path=video_path,
                telemetry_csv_path=telemetry_csv_path,
                output_root=output_root,
                model_path=model_path,
                run_id=run_id,
                progress_callback=report_progress,
            )
            current_stage = "presenting"
            (
                published_video,
                published_risk,
                published_telemetry,
                event_table,
                representative_clip,
                published_clips,
            ) = _validate_result(
                result,
                expected_output_root=resolved_output_root,
                expected_run_id=run_id,
            )
            return (
                _status_markdown(result),
                published_video,
                published_risk,
                published_telemetry,
                event_table,
                representative_clip,
                published_clips,
            )
        except gr.Error:
            raise
        except Exception as error:
            failed_stage = _failure_summary_context(resolved_output_root, run_id)
            diagnostic_stage = failed_stage or current_stage
            if not isinstance(error, (ValueError, FileNotFoundError)):
                _LOGGER.exception(
                    "DashGuard Gradio run %s failed during %s",
                    run_id,
                    diagnostic_stage,
                )
            safe_message = _safe_error_message(error)
            context = f" (run ID: {run_id}, 단계: {diagnostic_stage})"
            raise gr.Error(
                safe_message + context,
                print_exception=False,
            ) from None

    return analyze_video


def create_gradio_app(
    *,
    output_root: str | Path = Path("outputs/day8/runs"),
    model_path: str | Path = Path("yolo11n.pt"),
) -> gr.Blocks:
    """Create the local DashGuard UI without launching a server."""

    output_root_path = Path(output_root)
    model_path_value = Path(model_path)
    analyze_video = _analysis_callback(
        output_root=output_root_path,
        model_path=model_path_value,
    )

    with gr.Blocks(
        title="DashGuard 로컬 영상 분석 데모",
        analytics_enabled=False,
    ) as app:
        gr.Markdown(_INTRODUCTION)
        with gr.Row():
            video_input = gr.Video(
                label="입력 영상 (업로드 전용, 최대 30초)",
                sources=["upload"],
                format=None,
            )
            telemetry_input = gr.File(
                label="telemetry CSV (선택)",
                file_types=[".csv"],
                type="filepath",
            )
        run_button = gr.Button("분석 실행", variant="primary")

        status_output = gr.Markdown(label="상태 요약")
        with gr.Row():
            video_output = gr.Video(label="분석 영상", interactive=False)
            representative_clip_output = gr.Video(
                label="첫 번째 event clip (events 순서)", interactive=False
            )
        with gr.Row():
            risk_output = gr.Image(label="risk proxy", interactive=False)
            telemetry_output = gr.Image(label="telemetry", interactive=False)
        events_output = gr.Dataframe(
            value=pd.DataFrame(columns=list(_EVENT_COLUMNS)),
            label="event 후보",
            interactive=False,
            type="pandas",
        )
        clips_output = gr.File(
            label="전체 event clips",
            file_count="multiple",
            type="filepath",
            interactive=False,
        )

        run_button.click(
            fn=analyze_video,
            inputs=[video_input, telemetry_input],
            outputs=[
                status_output,
                video_output,
                risk_output,
                telemetry_output,
                events_output,
                representative_clip_output,
                clips_output,
            ],
            concurrency_limit=1,
            concurrency_id="dashguard-analysis",
            api_name=False,
        )

    return app


__all__ = ["create_gradio_app"]

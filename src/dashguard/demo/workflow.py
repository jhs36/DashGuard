"""Gradio-independent Day 8 PyTorch CPU demo workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashguard.analysis.events import generate_event_outputs
from dashguard.analysis.risk_report import generate_video_risk_report
from dashguard.analysis.telemetry import (
    generate_synthetic_telemetry_csv,
    sync_risk_telemetry_csv,
)
from dashguard.analysis.tracked_video import run_tracked_video_analysis
from dashguard.demo.presentation import create_telemetry_plot
from dashguard.video.event_clips import extract_event_clips
from dashguard.video.metadata import VideoMetadata, probe_video_metadata
from dashguard.video.standardize import standardize_video


SUCCESS_STATUS = "succeeded"
FAILED_STATUS = "failed"
BACKEND = "pytorch_cpu"
HARD_MAX_DURATION_SEC = 30.0
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
)

LIMITATIONS = (
    "The risk proxy is not an estimate of real distance, TTC, or crash probability.",
    "Synthetic telemetry is illustrative and is not evidence of real sensor performance.",
    "Detector false positives, false negatives, and track ID fragmentation are possible.",
    "The PyTorch CPU backend does not guarantee real-time 30 FPS processing.",
)

_RUN_ID_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?\Z"
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


@dataclass(frozen=True)
class DemoRunResult:
    """Immutable artifact map for one successful demo run."""

    run_id: str
    run_dir: Path
    status: str
    backend: str
    model_path: Path
    telemetry_source: str
    input_duration_sec: float
    total_processing_sec: float
    standardized_video_path: Path
    annotated_video_path: Path
    published_annotated_video_path: Path
    tracks_csv_path: Path
    metrics_json_path: Path
    risk_csv_path: Path
    risk_plot_path: Path
    published_risk_plot_path: Path
    telemetry_csv_path: Path
    synchronized_csv_path: Path
    telemetry_plot_path: Path
    published_telemetry_plot_path: Path
    timeline_csv_path: Path
    events_json_path: Path
    event_count: int
    clip_manifest_path: Path | None
    clip_paths: tuple[Path, ...]
    published_clip_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    run_summary_path: Path


@dataclass(frozen=True)
class _PreflightResult:
    video_path: Path
    telemetry_csv_path: Path | None
    output_root: Path
    model_path: Path
    run_id: str
    input_metadata: VideoMetadata


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_finite_float(value: object, *, setting: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{setting} must be a finite number greater than zero")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{setting} must be a finite number greater than zero"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{setting} must be a finite number greater than zero")
    return parsed


def _validate_run_id(run_id: str | None) -> str:
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"run_{timestamp}_{uuid.uuid4().hex}"
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty filesystem-safe string")
    candidate_path = Path(run_id)
    if (
        candidate_path.is_absolute()
        or "/" in run_id
        or "\\" in run_id
        or ".." in run_id
        or _RUN_ID_PATTERN.fullmatch(run_id) is None
    ):
        raise ValueError(f"run_id is not filesystem-safe: {run_id!r}")
    device_name = run_id.split(".", 1)[0].upper()
    if device_name in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"run_id uses a Windows reserved name: {run_id!r}")
    return run_id


def _validate_metadata(metadata: VideoMetadata, *, path: Path) -> None:
    width = getattr(metadata, "width", None)
    height = getattr(metadata, "height", None)
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError(f"Video metadata has invalid dimensions for {path}")
    _positive_finite_float(getattr(metadata, "fps", None), setting="video FPS")
    _positive_finite_float(
        getattr(metadata, "duration_sec", None), setting="video duration"
    )


def _preflight(
    *,
    video_path: str | Path,
    telemetry_csv_path: str | Path | None,
    output_root: str | Path,
    model_path: str | Path,
    max_duration_sec: float,
    run_id: str | None,
) -> _PreflightResult:
    duration_limit = _positive_finite_float(
        max_duration_sec, setting="max_duration_sec"
    )
    if duration_limit > HARD_MAX_DURATION_SEC:
        raise ValueError(
            f"max_duration_sec cannot exceed {HARD_MAX_DURATION_SEC:g} seconds"
        )

    safe_run_id = _validate_run_id(run_id)
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video path does not exist or is not a file: {video}")
    if video.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video extension {video.suffix!r}; supported: {supported}"
        )

    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Model path does not exist or is not a file: {model}")

    telemetry: Path | None = None
    if telemetry_csv_path is not None:
        telemetry = Path(telemetry_csv_path)
        if not telemetry.is_file():
            raise FileNotFoundError(
                "Telemetry CSV path does not exist or is not a file: "
                f"{telemetry}"
            )
        if telemetry.suffix.lower() != ".csv":
            raise ValueError(f"Telemetry input must have a .csv extension: {telemetry}")

    root = Path(output_root)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Output root is not a directory: {root}")
    if (root / safe_run_id).exists():
        raise FileExistsError(f"Demo run already exists: {root / safe_run_id}")

    metadata = probe_video_metadata(video)
    _validate_metadata(metadata, path=video)
    if metadata.duration_sec > duration_limit:
        raise ValueError(
            f"Video duration {metadata.duration_sec:.6g}s exceeds the "
            f"{duration_limit:.6g}s limit"
        )

    return _PreflightResult(
        video_path=video,
        telemetry_csv_path=telemetry,
        output_root=root,
        model_path=model,
        run_id=safe_run_id,
        input_metadata=metadata,
    )


def _copy_file_exclusive(source: Path, destination: Path) -> Path:
    """Copy a regular file while refusing to replace an existing destination."""

    if not source.is_file():
        raise FileNotFoundError(f"Copy source does not exist or is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as source_file, destination.open("xb") as output_file:
            shutil.copyfileobj(source_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        shutil.copystat(source, destination)
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_copy(source: Path, destination: Path) -> dict[str, str]:
    primary_before = _sha256(source)
    _copy_file_exclusive(source, destination)
    primary_after = _sha256(source)
    published = _sha256(destination)
    if primary_before != primary_after or primary_before != published:
        raise RuntimeError(
            "Published artifact integrity verification failed for " f"{source}"
        )
    return {
        "primary_path": str(source),
        "published_path": str(destination),
        "primary_sha256_before": primary_before,
        "primary_sha256_after": primary_after,
        "published_sha256": published,
    }


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Write one UTF-8 JSON document without exposing a partial final file."""

    if path.exists():
        raise FileExistsError(f"JSON output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(
                document,
                output,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            raise FileExistsError(f"JSON output appeared while writing: {path}")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _metadata_document(metadata: VideoMetadata) -> dict[str, Any]:
    return {
        "width": metadata.width,
        "height": metadata.height,
        "fps": metadata.fps,
        "frame_count": metadata.frame_count,
        "duration_sec": metadata.duration_sec,
        "codec": metadata.codec,
    }


def _summary_value(value: object) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _summary_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _relative(run_dir: Path, path: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def _read_event_count(events_json_path: Path) -> int:
    with events_json_path.open("r", encoding="utf-8") as input_file:
        document = json.load(input_file)
    if not isinstance(document, dict) or not isinstance(document.get("events"), list):
        raise ValueError("events.json must contain an 'events' array")
    return len(document["events"])


def _existing_artifacts(run_dir: Path, *, summary_path: Path) -> list[str]:
    return sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path != summary_path
        and not path.name.startswith(f".{summary_path.name}.")
    )


def run_dashguard_demo(
    *,
    video_path: str | Path,
    telemetry_csv_path: str | Path | None = None,
    output_root: str | Path = Path("outputs/day8/runs"),
    model_path: str | Path = Path("yolo11n.pt"),
    max_duration_sec: float = HARD_MAX_DURATION_SEC,
    run_id: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> DemoRunResult:
    """Run the Day 2/3/5/6 libraries once in an isolated Day 8 workspace."""

    started_at = _utc_now()
    overall_started = time.perf_counter()
    stage_started = overall_started
    current_stage = "validating"
    stage_durations: dict[str, float] = {}

    def notify(stage: str, progress: float) -> None:
        if progress_callback is not None:
            progress_callback(stage, progress)

    def enter_stage(stage: str, progress: float) -> None:
        nonlocal current_stage, stage_started
        now = time.perf_counter()
        stage_durations[current_stage] = (
            stage_durations.get(current_stage, 0.0) + now - stage_started
        )
        current_stage = stage
        stage_started = now
        notify(stage, progress)

    notify("validating", 0.0)
    preflight = _preflight(
        video_path=video_path,
        telemetry_csv_path=telemetry_csv_path,
        output_root=output_root,
        model_path=model_path,
        max_duration_sec=max_duration_sec,
        run_id=run_id,
    )

    preflight.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = preflight.output_root / preflight.run_id
    run_dir.mkdir(exist_ok=False)

    input_dir = run_dir / "input"
    analysis_dir = run_dir / "analysis"
    risk_dir = run_dir / "risk"
    telemetry_dir = run_dir / "telemetry"
    events_dir = run_dir / "events"
    clips_dir = run_dir / "clips"
    published_dir = run_dir / "published"
    run_summary_path = run_dir / "run_summary.json"

    standardized_video_path = input_dir / "standardized_input.mp4"
    annotated_video_path = analysis_dir / "annotated_video.mp4"
    tracks_csv_path = analysis_dir / "tracks.csv"
    metrics_json_path = analysis_dir / "metrics.json"
    risk_csv_path = risk_dir / "risk_scores.csv"
    risk_plot_path = risk_dir / "risk_proxy.png"
    synchronized_csv_path = telemetry_dir / "risk_telemetry_sync.csv"
    telemetry_plot_path = telemetry_dir / "telemetry_plot.png"
    timeline_csv_path = events_dir / "frame_timeline.csv"
    events_json_path = events_dir / "events.json"
    clip_manifest_path = clips_dir / "clip_manifest.json"
    published_annotated_video_path = published_dir / "annotated_video.mp4"
    published_risk_plot_path = published_dir / "risk_proxy.png"
    published_telemetry_plot_path = published_dir / "telemetry_plot.png"

    telemetry_source = (
        "provided" if preflight.telemetry_csv_path is not None else "synthetic"
    )
    warnings: list[str] = []
    clip_paths: tuple[Path, ...] = ()
    published_clip_paths: tuple[Path, ...] = ()
    actual_clip_manifest_path: Path | None = None
    stage_results: dict[str, Any] = {}
    published_integrity: dict[str, Any] = {}

    try:
        for directory in (
            input_dir,
            analysis_dir,
            risk_dir,
            telemetry_dir,
            events_dir,
            clips_dir,
            published_dir,
        ):
            directory.mkdir(exist_ok=False)

        enter_stage("standardizing", 0.10)
        standardize_video(
            input_path=preflight.video_path,
            output_path=standardized_video_path,
        )
        standardized_metadata = probe_video_metadata(standardized_video_path)
        _validate_metadata(standardized_metadata, path=standardized_video_path)

        frame_count = standardized_metadata.frame_count
        max_frames = (
            frame_count
            if isinstance(frame_count, int)
            and not isinstance(frame_count, bool)
            and frame_count > 0
            else None
        )

        enter_stage("tracking", 0.25)
        tracking_result = run_tracked_video_analysis(
            input_path=standardized_video_path,
            output_video_path=annotated_video_path,
            output_csv_path=tracks_csv_path,
            metrics_path=metrics_json_path,
            model_path=preflight.model_path,
            confidence=0.25,
            imgsz=640,
            trajectory_length=30,
            max_frames=max_frames,
            device="cpu",
        )
        stage_results["tracking"] = _summary_value(tracking_result)

        enter_stage("risk", 0.45)
        risk_result = generate_video_risk_report(
            input_csv=tracks_csv_path,
            input_video=standardized_video_path,
            output_csv=risk_csv_path,
            output_plot=risk_plot_path,
            overwrite=False,
        )
        stage_results["risk"] = _summary_value(risk_result)

        enter_stage("telemetry", 0.58)
        if preflight.telemetry_csv_path is None:
            telemetry_csv_path_in_run = input_dir / "synthetic_telemetry.csv"
            telemetry_result = generate_synthetic_telemetry_csv(
                output_csv=telemetry_csv_path_in_run,
                duration_sec=standardized_metadata.duration_sec,
                frequency_hz=10.0,
                overwrite=False,
            )
            stage_results["telemetry_generation"] = _summary_value(
                telemetry_result
            )
        else:
            telemetry_csv_path_in_run = input_dir / "telemetry_input.csv"
            _copy_file_exclusive(
                preflight.telemetry_csv_path, telemetry_csv_path_in_run
            )

        sync_result = sync_risk_telemetry_csv(
            risk_csv=risk_csv_path,
            telemetry_csv=telemetry_csv_path_in_run,
            output_csv=synchronized_csv_path,
            offset_sec=0.0,
            tolerance_sec=0.05,
            overwrite=False,
        )
        stage_results["telemetry_sync"] = _summary_value(sync_result)
        if getattr(sync_result, "matched_rows", None) == 0:
            warnings.append(
                "No telemetry rows matched risk timestamps within 0.05 seconds."
            )
        create_telemetry_plot(
            synchronized_csv_path,
            telemetry_plot_path,
            telemetry_source=telemetry_source,
        )

        enter_stage("events", 0.70)
        event_result = generate_event_outputs(
            input_csv=synchronized_csv_path,
            input_video=standardized_video_path,
            output_timeline_csv=timeline_csv_path,
            output_events_json=events_json_path,
            score_threshold=0.70,
            minimum_event_duration_sec=0.15,
            merge_gap_sec=0.10,
            pre_roll_sec=2.0,
            post_roll_sec=1.0,
            overwrite=False,
        )
        stage_results["events"] = _summary_value(event_result)
        event_count = _read_event_count(events_json_path)

        enter_stage("clips", 0.80)
        if event_count > 0:
            clip_result = extract_event_clips(
                events_json=events_json_path,
                input_video=standardized_video_path,
                output_dir=clips_dir,
                output_manifest_json=clip_manifest_path,
                overwrite=False,
            )
            stage_results["clips"] = _summary_value(clip_result)
            clip_paths = tuple(
                Path(record.output_path) for record in clip_result.clips
            )
            if len(clip_paths) != event_count:
                raise RuntimeError(
                    "Event clip count does not match events.json: "
                    f"events={event_count}, clips={len(clip_paths)}"
                )
            actual_clip_manifest_path = clip_manifest_path

        enter_stage("publishing", 0.90)
        for label, source, destination in (
            (
                "annotated_video",
                annotated_video_path,
                published_annotated_video_path,
            ),
            ("risk_plot", risk_plot_path, published_risk_plot_path),
            (
                "telemetry_plot",
                telemetry_plot_path,
                published_telemetry_plot_path,
            ),
        ):
            published_integrity[label] = _publish_copy(source, destination)

        published_clips: list[Path] = []
        clip_integrity: list[dict[str, str]] = []
        for clip_path in clip_paths:
            published_clip = published_dir / clip_path.name
            clip_integrity.append(_publish_copy(clip_path, published_clip))
            published_clips.append(published_clip)
        published_clip_paths = tuple(published_clips)
        published_integrity["event_clips"] = clip_integrity

        enter_stage("completed", 1.0)
        finished_counter = time.perf_counter()
        stage_durations[current_stage] = (
            stage_durations.get(current_stage, 0.0)
            + finished_counter
            - stage_started
        )
        stage_started = finished_counter
        finished_at = _utc_now()
        total_processing_sec = finished_counter - overall_started

        artifact_paths: dict[str, Any] = {
            "standardized_video": _relative(run_dir, standardized_video_path),
            "annotated_video": _relative(run_dir, annotated_video_path),
            "published_annotated_video": _relative(
                run_dir, published_annotated_video_path
            ),
            "tracks_csv": _relative(run_dir, tracks_csv_path),
            "metrics_json": _relative(run_dir, metrics_json_path),
            "risk_csv": _relative(run_dir, risk_csv_path),
            "risk_plot": _relative(run_dir, risk_plot_path),
            "published_risk_plot": _relative(
                run_dir, published_risk_plot_path
            ),
            "telemetry_csv": _relative(run_dir, telemetry_csv_path_in_run),
            "synchronized_csv": _relative(run_dir, synchronized_csv_path),
            "telemetry_plot": _relative(run_dir, telemetry_plot_path),
            "published_telemetry_plot": _relative(
                run_dir, published_telemetry_plot_path
            ),
            "timeline_csv": _relative(run_dir, timeline_csv_path),
            "events_json": _relative(run_dir, events_json_path),
            "clip_manifest": (
                _relative(run_dir, actual_clip_manifest_path)
                if actual_clip_manifest_path is not None
                else None
            ),
            "event_clips": [_relative(run_dir, path) for path in clip_paths],
            "published_event_clips": [
                _relative(run_dir, path) for path in published_clip_paths
            ],
        }
        summary = {
            "status": SUCCESS_STATUS,
            "run_id": preflight.run_id,
            "backend": BACKEND,
            "model": str(preflight.model_path),
            "input": {
                "video_path": str(preflight.video_path),
                "metadata": _metadata_document(preflight.input_metadata),
                "standardized_metadata": _metadata_document(
                    standardized_metadata
                ),
            },
            "telemetry_source": telemetry_source,
            "started_at": started_at,
            "finished_at": finished_at,
            "stage_durations_sec": stage_durations,
            "total_processing_sec": total_processing_sec,
            "artifacts": artifact_paths,
            "event_count": event_count,
            "stage_results": stage_results,
            "published_integrity": published_integrity,
            "warnings": warnings,
            "limitations": list(LIMITATIONS),
        }
        _write_json_atomic(run_summary_path, summary)

        return DemoRunResult(
            run_id=preflight.run_id,
            run_dir=run_dir,
            status=SUCCESS_STATUS,
            backend=BACKEND,
            model_path=preflight.model_path,
            telemetry_source=telemetry_source,
            input_duration_sec=preflight.input_metadata.duration_sec,
            total_processing_sec=total_processing_sec,
            standardized_video_path=standardized_video_path,
            annotated_video_path=annotated_video_path,
            published_annotated_video_path=published_annotated_video_path,
            tracks_csv_path=tracks_csv_path,
            metrics_json_path=metrics_json_path,
            risk_csv_path=risk_csv_path,
            risk_plot_path=risk_plot_path,
            published_risk_plot_path=published_risk_plot_path,
            telemetry_csv_path=telemetry_csv_path_in_run,
            synchronized_csv_path=synchronized_csv_path,
            telemetry_plot_path=telemetry_plot_path,
            published_telemetry_plot_path=published_telemetry_plot_path,
            timeline_csv_path=timeline_csv_path,
            events_json_path=events_json_path,
            event_count=event_count,
            clip_manifest_path=actual_clip_manifest_path,
            clip_paths=clip_paths,
            published_clip_paths=published_clip_paths,
            warnings=tuple(warnings),
            limitations=LIMITATIONS,
            run_summary_path=run_summary_path,
        )
    except Exception as error:
        failure_counter = time.perf_counter()
        stage_durations[current_stage] = (
            stage_durations.get(current_stage, 0.0)
            + failure_counter
            - stage_started
        )
        failure_summary = {
            "status": FAILED_STATUS,
            "run_id": preflight.run_id,
            "backend": BACKEND,
            "model": str(preflight.model_path),
            "telemetry_source": telemetry_source,
            "failed_stage": current_stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "stage_durations_sec": stage_durations,
            "total_processing_sec": failure_counter - overall_started,
            "artifacts": _existing_artifacts(
                run_dir, summary_path=run_summary_path
            ),
            "warnings": warnings,
            "limitations": list(LIMITATIONS),
        }
        try:
            _write_json_atomic(run_summary_path, failure_summary)
        except Exception:
            pass
        raise


__all__ = [
    "BACKEND",
    "DemoRunResult",
    "FAILED_STATUS",
    "HARD_MAX_DURATION_SEC",
    "LIMITATIONS",
    "SUCCESS_STATUS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "run_dashguard_demo",
]

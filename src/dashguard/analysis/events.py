"""Build frame-level risk timelines and deterministic event candidates."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashguard.video.metadata import VideoMetadata, probe_video_metadata


REQUIRED_EVENT_COLUMNS = (
    "frame_index",
    "timestamp_sec",
    "track_id",
    "class_name",
    "total_score",
    "telemetry_matched",
    "telemetry_source",
    "hard_brake",
)

TIMELINE_ADDED_COLUMNS = (
    "source_row_index",
    "event_candidate",
)

TRIGGER_REASON = "total_score_gte_threshold"
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EventDetectionConfig:
    """Experimental event detection settings used by the Day 6 pipeline."""

    score_threshold: float = 0.70
    minimum_event_duration_sec: float = 0.15
    merge_gap_sec: float = 0.10
    pre_roll_sec: float = 2.0
    post_roll_sec: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "score_threshold",
            _finite_float(self.score_threshold, setting="score_threshold"),
        )
        for setting in (
            "minimum_event_duration_sec",
            "merge_gap_sec",
            "pre_roll_sec",
            "post_roll_sec",
        ):
            object.__setattr__(
                self,
                setting,
                _nonnegative_finite_float(
                    getattr(self, setting),
                    setting=setting,
                ),
            )


@dataclass(frozen=True)
class CandidateRun:
    """A strict or merged run of candidate frames."""

    start_frame: int
    end_frame: int
    start_timestamp_sec: float
    last_candidate_timestamp_sec: float
    candidate_frame_count: int


@dataclass(frozen=True)
class EventDetectionResult:
    """In-memory timeline, runs, and JSON-ready event document."""

    timeline: pd.DataFrame
    strict_runs: tuple[CandidateRun, ...]
    merged_runs: tuple[CandidateRun, ...]
    event_document: dict[str, Any]


@dataclass(frozen=True)
class EventOutputSummary:
    """Serializable summary returned after both output files are installed."""

    input_rows: int
    timeline_rows: int
    candidate_frames: int
    strict_run_count: int
    merged_run_count_before_duration_filter: int
    event_count: int
    event_ids: tuple[str, ...]
    output_timeline_path: str
    output_json_path: str
    config: EventDetectionConfig
    video_duration_sec: float


def _finite_float(value: object, *, setting: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{setting} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{setting} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{setting} must be finite")
    return result


def _nonnegative_finite_float(value: object, *, setting: str) -> float:
    result = _finite_float(value, setting=setting)
    if result < 0.0:
        raise ValueError(f"{setting} must be non-negative")
    return result


def _positive_finite_float(value: object, *, setting: str) -> float:
    result = _finite_float(value, setting=setting)
    if result <= 0.0:
        raise ValueError(f"{setting} must be greater than zero")
    return result


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"column '{column}' must be numeric")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"column '{column}' must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"column '{column}' must contain only finite values")
    return values


def _integral_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric_values(frame, column)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"column '{column}' must contain only integer values")
    return values


def _validate_input_rows(
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")

    missing = [
        column for column in REQUIRED_EVENT_COLUMNS if column not in rows.columns
    ]
    if missing:
        raise ValueError("input is missing required columns: " + ", ".join(missing))

    conflicts = [
        column for column in TIMELINE_ADDED_COLUMNS if column in rows.columns
    ]
    if conflicts:
        raise ValueError(
            "input already contains timeline output columns: "
            + ", ".join(conflicts)
        )

    frame_indices = _integral_values(rows, "frame_index")
    if (frame_indices < 0.0).any():
        raise ValueError("column 'frame_index' must be non-negative")

    timestamps = _numeric_values(rows, "timestamp_sec")
    if (timestamps < 0.0).any():
        raise ValueError("column 'timestamp_sec' must be non-negative")

    track_ids = _integral_values(rows, "track_id")
    total_scores = _numeric_values(rows, "total_score")

    validation = pd.DataFrame(
        {
            "frame_index": frame_indices,
            "timestamp_sec": timestamps,
        }
    )
    inconsistent = validation.groupby("frame_index", sort=False)[
        "timestamp_sec"
    ].nunique()
    inconsistent_frames = inconsistent[inconsistent > 1].index.tolist()
    if inconsistent_frames:
        formatted = ", ".join(str(int(value)) for value in inconsistent_frames)
        raise ValueError(
            "each frame_index must have exactly one timestamp_sec; "
            f"conflicting frame_index values: {formatted}"
        )

    return frame_indices, timestamps, track_ids, total_scores


def build_frame_risk_timeline(
    rows: pd.DataFrame,
    *,
    score_threshold: float = 0.70,
) -> pd.DataFrame:
    """Select one deterministic representative row for every input frame."""

    threshold = _finite_float(score_threshold, setting="score_threshold")
    frame_indices, timestamps, track_ids, total_scores = _validate_input_rows(
        rows
    )
    source_positions = np.arange(len(rows), dtype=np.int64)
    ranking = pd.DataFrame(
        {
            "frame_index": frame_indices,
            "total_score": total_scores,
            "timestamp_sec": timestamps,
            "track_id": track_ids,
            "source_row_index": source_positions,
        }
    ).sort_values(
        [
            "frame_index",
            "total_score",
            "timestamp_sec",
            "track_id",
            "source_row_index",
        ],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    selected = ranking.drop_duplicates("frame_index", keep="first")
    selected_positions = selected["source_row_index"].to_numpy(dtype=np.int64)

    timeline = rows.iloc[selected_positions].copy(deep=True).reset_index(drop=True)
    timeline["source_row_index"] = selected_positions
    timeline["event_candidate"] = (
        total_scores[selected_positions] >= threshold
    )

    selected_frames = frame_indices[selected_positions]
    if len(selected_frames) > 1 and not (np.diff(selected_frames) > 0.0).all():
        raise ValueError(
            "frame_index must be strictly increasing after timeline generation"
        )
    return timeline


def estimate_frame_interval_sec(timeline: pd.DataFrame) -> float:
    """Return the median positive difference between adjacent timestamps."""

    if "timestamp_sec" not in timeline.columns:
        raise ValueError("timeline is missing required column: timestamp_sec")
    timestamps = _numeric_values(timeline, "timestamp_sec")
    positive_differences = np.diff(timestamps)
    positive_differences = positive_differences[positive_differences > 0.0]
    if positive_differences.size == 0:
        raise ValueError(
            "cannot determine frame interval: no positive adjacent "
            "timestamp_sec difference exists"
        )
    interval = float(np.median(positive_differences))
    return _positive_finite_float(interval, setting="frame_interval_sec")


def _bool_value(
    value: object,
    *,
    column: str,
    missing_value: bool | None,
) -> bool | None:
    try:
        if pd.isna(value):
            return missing_value
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    elif isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(numeric)
    raise ValueError(
        f"column '{column}' must contain boolean, 0/1, or missing values"
    )


def _candidate_mask(timeline: pd.DataFrame) -> np.ndarray:
    if "event_candidate" not in timeline.columns:
        raise ValueError("timeline is missing required column: event_candidate")
    return np.asarray(
        [
            bool(
                _bool_value(
                    value,
                    column="event_candidate",
                    missing_value=False,
                )
            )
            for value in timeline["event_candidate"]
        ],
        dtype=bool,
    )


def build_strict_candidate_runs(
    timeline: pd.DataFrame,
) -> tuple[CandidateRun, ...]:
    """Group candidate rows whose integer frame indices differ by exactly one."""

    frame_indices = _integral_values(timeline, "frame_index")
    timestamps = _numeric_values(timeline, "timestamp_sec")
    candidate_positions = np.flatnonzero(_candidate_mask(timeline))
    if candidate_positions.size == 0:
        return ()

    candidate_frames = frame_indices[candidate_positions]
    if len(frame_indices) > 1 and not (np.diff(frame_indices) > 0.0).all():
        raise ValueError("timeline frame_index must be strictly increasing")

    split_points = np.flatnonzero(np.diff(candidate_frames) != 1.0) + 1
    position_groups = np.split(candidate_positions, split_points)
    runs = []
    for positions in position_groups:
        first_position = int(positions[0])
        last_position = int(positions[-1])
        runs.append(
            CandidateRun(
                start_frame=int(frame_indices[first_position]),
                end_frame=int(frame_indices[last_position]),
                start_timestamp_sec=float(timestamps[first_position]),
                last_candidate_timestamp_sec=float(timestamps[last_position]),
                candidate_frame_count=len(positions),
            )
        )
    return tuple(runs)


def merge_candidate_runs(
    runs: Sequence[CandidateRun],
    *,
    frame_interval_sec: float,
    merge_gap_sec: float = 0.10,
) -> tuple[CandidateRun, ...]:
    """Merge chronologically close strict runs without counting gap frames."""

    frame_interval = _positive_finite_float(
        frame_interval_sec,
        setting="frame_interval_sec",
    )
    merge_gap = _nonnegative_finite_float(
        merge_gap_sec,
        setting="merge_gap_sec",
    )
    if not runs:
        return ()

    merged: list[CandidateRun] = [runs[0]]
    for next_run in runs[1:]:
        previous = merged[-1]
        gap_sec = (
            next_run.start_timestamp_sec
            - previous.last_candidate_timestamp_sec
            - frame_interval
        )
        within_boundary = gap_sec <= merge_gap or math.isclose(
            gap_sec,
            merge_gap,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        if within_boundary:
            merged[-1] = CandidateRun(
                start_frame=previous.start_frame,
                end_frame=next_run.end_frame,
                start_timestamp_sec=previous.start_timestamp_sec,
                last_candidate_timestamp_sec=(
                    next_run.last_candidate_timestamp_sec
                ),
                candidate_frame_count=(
                    previous.candidate_frame_count
                    + next_run.candidate_frame_count
                ),
            )
        else:
            merged.append(next_run)
    return tuple(merged)


def _telemetry_evidence(event_timeline: pd.DataFrame) -> dict[str, Any]:
    matched = np.asarray(
        [
            bool(
                _bool_value(
                    value,
                    column="telemetry_matched",
                    missing_value=False,
                )
            )
            for value in event_timeline["telemetry_matched"]
        ],
        dtype=bool,
    )
    hard_brake_values = [
        _bool_value(
            value,
            column="hard_brake",
            missing_value=None,
        )
        for value in event_timeline["hard_brake"]
    ]
    if any(value is True for value in hard_brake_values):
        hard_brake: bool | None = True
    elif matched.any():
        hard_brake = False
    else:
        hard_brake = None

    sources: set[str] = set()
    for value in event_timeline["telemetry_source"]:
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        source = str(value).strip()
        if source:
            sources.add(source)
    telemetry_source = (
        next(iter(sources))
        if len(sources) == 1
        else "mixed" if len(sources) > 1 else None
    )

    matched_count = int(matched.sum())
    return {
        "hard_brake": hard_brake,
        "telemetry_source": telemetry_source,
        "telemetry_matched_frame_count": matched_count,
        "telemetry_unmatched_frame_count": len(event_timeline) - matched_count,
    }


def _primary_values(candidate_rows: pd.DataFrame) -> tuple[int, str | None, float]:
    frame_indices = _integral_values(candidate_rows, "frame_index")
    timestamps = _numeric_values(candidate_rows, "timestamp_sec")
    track_ids = _integral_values(candidate_rows, "track_id")
    total_scores = _numeric_values(candidate_rows, "total_score")
    source_rows = _integral_values(candidate_rows, "source_row_index")
    ranking = pd.DataFrame(
        {
            "position": np.arange(len(candidate_rows), dtype=np.int64),
            "total_score": total_scores,
            "timestamp_sec": timestamps,
            "frame_index": frame_indices,
            "track_id": track_ids,
            "source_row_index": source_rows,
        }
    ).sort_values(
        [
            "total_score",
            "timestamp_sec",
            "frame_index",
            "track_id",
            "source_row_index",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    primary_position = int(ranking.iloc[0]["position"])
    primary = candidate_rows.iloc[primary_position]
    class_value = primary["class_name"]
    try:
        primary_class = None if pd.isna(class_value) else str(class_value)
    except (TypeError, ValueError):
        primary_class = str(class_value)
    return (
        int(track_ids[primary_position]),
        primary_class,
        float(total_scores[primary_position]),
    )


def build_event_records(
    timeline: pd.DataFrame,
    merged_runs: Sequence[CandidateRun],
    *,
    video_duration_sec: float,
    frame_interval_sec: float,
    config: EventDetectionConfig = EventDetectionConfig(),
) -> list[dict[str, Any]]:
    """Apply duration filtering and summarize deterministic event records."""

    video_duration = _positive_finite_float(
        video_duration_sec,
        setting="video_duration_sec",
    )
    frame_interval = _positive_finite_float(
        frame_interval_sec,
        setting="frame_interval_sec",
    )
    if not isinstance(config, EventDetectionConfig):
        raise TypeError("config must be an EventDetectionConfig")

    frame_indices = _integral_values(timeline, "frame_index")
    candidate_mask = _candidate_mask(timeline)
    events: list[dict[str, Any]] = []
    for run in merged_runs:
        start_sec = float(run.start_timestamp_sec)
        end_sec = float(
            min(
                run.last_candidate_timestamp_sec + frame_interval,
                video_duration,
            )
        )
        duration_sec = float(end_sec - start_sec)
        if duration_sec < config.minimum_event_duration_sec and not math.isclose(
            duration_sec,
            config.minimum_event_duration_sec,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            continue
        if end_sec <= start_sec:
            raise ValueError(
                "event duration must be positive after video duration clamp: "
                f"start_sec={start_sec}, end_sec={end_sec}"
            )

        event_range = (
            (frame_indices >= run.start_frame)
            & (frame_indices <= run.end_frame)
        )
        event_timeline = timeline.loc[event_range]
        candidate_rows = timeline.loc[event_range & candidate_mask]
        if candidate_rows.empty:
            raise RuntimeError("merged candidate run contains no candidate rows")
        primary_track_id, primary_class, maximum_score = _primary_values(
            candidate_rows
        )
        evidence = _telemetry_evidence(event_timeline)

        clip_start_sec = float(max(0.0, start_sec - config.pre_roll_sec))
        clip_end_sec = float(
            min(video_duration, end_sec + config.post_roll_sec)
        )
        if not clip_start_sec < clip_end_sec:
            raise ValueError(
                "clip range must be non-empty after video duration clamp: "
                f"clip_start_sec={clip_start_sec}, "
                f"clip_end_sec={clip_end_sec}"
            )

        event_id = f"event_{len(events) + 1:03d}"
        events.append(
            {
                "event_id": event_id,
                "start_frame": int(run.start_frame),
                "end_frame": int(run.end_frame),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "clip_start_sec": clip_start_sec,
                "clip_end_sec": clip_end_sec,
                "candidate_frame_count": int(run.candidate_frame_count),
                "max_total_score": maximum_score,
                "primary_track_id": primary_track_id,
                "primary_class": primary_class,
                "hard_brake": evidence["hard_brake"],
                "telemetry_source": evidence["telemetry_source"],
                "telemetry_matched_frame_count": evidence[
                    "telemetry_matched_frame_count"
                ],
                "telemetry_unmatched_frame_count": evidence[
                    "telemetry_unmatched_frame_count"
                ],
                "trigger_reasons": [TRIGGER_REASON],
            }
        )
    return events


def _assert_standard_json(value: object, *, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {location} must be a string")
            _assert_standard_json(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_standard_json(item, location=f"{location}[{index}]")
        return
    raise TypeError(
        f"unsupported JSON value at {location}: {type(value).__name__}"
    )


def detect_events(
    rows: pd.DataFrame,
    *,
    video_duration_sec: float,
    config: EventDetectionConfig = EventDetectionConfig(),
    source_sync_csv: str | Path | None = None,
    source_video: str | Path | None = None,
) -> EventDetectionResult:
    """Build a representative timeline and JSON-ready event candidates."""

    if not isinstance(config, EventDetectionConfig):
        raise TypeError("config must be an EventDetectionConfig")
    video_duration = _positive_finite_float(
        video_duration_sec,
        setting="video_duration_sec",
    )
    timeline = build_frame_risk_timeline(
        rows,
        score_threshold=config.score_threshold,
    )
    frame_interval = estimate_frame_interval_sec(timeline)
    strict_runs = build_strict_candidate_runs(timeline)
    merged_runs = merge_candidate_runs(
        strict_runs,
        frame_interval_sec=frame_interval,
        merge_gap_sec=config.merge_gap_sec,
    )
    events = build_event_records(
        timeline,
        merged_runs,
        video_duration_sec=video_duration,
        frame_interval_sec=frame_interval,
        config=config,
    )
    event_document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_sync_csv": (
            str(source_sync_csv) if source_sync_csv is not None else None
        ),
        "source_video": str(source_video) if source_video is not None else None,
        "video_duration_sec": video_duration,
        "frame_interval_sec": frame_interval,
        "config": asdict(config),
        "summary": {
            "input_rows": len(rows),
            "timeline_rows": len(timeline),
            "candidate_frames": int(_candidate_mask(timeline).sum()),
            "strict_run_count": len(strict_runs),
            "merged_run_count_before_duration_filter": len(merged_runs),
            "event_count": len(events),
        },
        "events": events,
    }
    _assert_standard_json(event_document)
    return EventDetectionResult(
        timeline=timeline,
        strict_runs=strict_runs,
        merged_runs=merged_runs,
        event_document=event_document,
    )


def _require_input_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file: {path}"
        )


def _check_output_paths(
    output_timeline: Path,
    output_json: Path,
    *,
    input_csv: Path,
    input_video: Path,
    overwrite: bool,
) -> None:
    resolved_outputs = (output_timeline.resolve(), output_json.resolve())
    if resolved_outputs[0] == resolved_outputs[1]:
        raise ValueError("Output timeline CSV and event JSON paths must differ")
    for output in resolved_outputs:
        if output in {input_csv.resolve(), input_video.resolve()}:
            raise ValueError("Output paths must not replace an input file")

    for label, path in (
        ("Output timeline CSV", output_timeline),
        ("Output event JSON", output_json),
    ):
        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"{label} path is not a file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{label} already exists; pass overwrite=True to replace it: "
                f"{path}"
            )


def _new_temporary_path(target: Path, *, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=target.parent,
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def _replace_output_pair(
    pairs: Sequence[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for _, target in pairs:
            if target.exists():
                if not overwrite:
                    raise FileExistsError(
                        "Output appeared while generating events; refusing "
                        f"to replace it: {target}"
                    )
                backup = _new_temporary_path(target, suffix=".backup")
                try:
                    target.replace(backup)
                except Exception:
                    if backup.is_file():
                        backup.unlink()
                    raise
                backups[target] = backup

        for temporary, target in pairs:
            temporary.replace(target)
            installed.append(target)
    except Exception:
        for target in reversed(installed):
            if target.is_file():
                target.unlink()
        for target, backup in reversed(tuple(backups.items())):
            if backup.is_file():
                backup.replace(target)
        raise
    else:
        for backup in backups.values():
            try:
                if backup.is_file():
                    backup.unlink()
            except OSError:
                pass


def _write_json(document: Mapping[str, Any], path: Path) -> None:
    _assert_standard_json(document)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(
            document,
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        output.write("\n")


def generate_event_outputs(
    *,
    input_csv: str | Path,
    input_video: str | Path,
    output_timeline_csv: str | Path,
    output_events_json: str | Path,
    score_threshold: float = 0.70,
    minimum_event_duration_sec: float = 0.15,
    merge_gap_sec: float = 0.10,
    pre_roll_sec: float = 2.0,
    post_roll_sec: float = 1.0,
    metadata_loader: Callable[[str | Path], VideoMetadata] = probe_video_metadata,
    overwrite: bool = False,
) -> EventOutputSummary:
    """Generate the timeline CSV and event JSON as a rollback-safe pair."""

    input_csv_path = Path(input_csv)
    input_video_path = Path(input_video)
    output_timeline_path = Path(output_timeline_csv)
    output_json_path = Path(output_events_json)
    _require_input_file(input_csv_path, label="Input sync CSV")
    _require_input_file(input_video_path, label="Input video")
    _check_output_paths(
        output_timeline_path,
        output_json_path,
        input_csv=input_csv_path,
        input_video=input_video_path,
        overwrite=overwrite,
    )

    config = EventDetectionConfig(
        score_threshold=score_threshold,
        minimum_event_duration_sec=minimum_event_duration_sec,
        merge_gap_sec=merge_gap_sec,
        pre_roll_sec=pre_roll_sec,
        post_roll_sec=post_roll_sec,
    )
    metadata = metadata_loader(input_video_path)
    video_duration = _positive_finite_float(
        metadata.duration_sec,
        setting="video_duration_sec",
    )
    rows = pd.read_csv(input_csv_path)
    result = detect_events(
        rows,
        video_duration_sec=video_duration,
        config=config,
        source_sync_csv=input_csv_path,
        source_video=input_video_path,
    )

    output_timeline_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_timeline: Path | None = None
    temporary_json: Path | None = None
    try:
        temporary_timeline = _new_temporary_path(
            output_timeline_path,
            suffix=".csv",
        )
        temporary_json = _new_temporary_path(
            output_json_path,
            suffix=".json",
        )
        result.timeline.to_csv(
            temporary_timeline,
            index=False,
            float_format="%.17g",
        )
        _write_json(result.event_document, temporary_json)
        _replace_output_pair(
            (
                (temporary_timeline, output_timeline_path),
                (temporary_json, output_json_path),
            ),
            overwrite=overwrite,
        )
        temporary_timeline = None
        temporary_json = None
    finally:
        for temporary in (temporary_timeline, temporary_json):
            if temporary is not None and temporary.is_file():
                temporary.unlink()

    document_summary = result.event_document["summary"]
    return EventOutputSummary(
        input_rows=int(document_summary["input_rows"]),
        timeline_rows=int(document_summary["timeline_rows"]),
        candidate_frames=int(document_summary["candidate_frames"]),
        strict_run_count=int(document_summary["strict_run_count"]),
        merged_run_count_before_duration_filter=int(
            document_summary["merged_run_count_before_duration_filter"]
        ),
        event_count=int(document_summary["event_count"]),
        event_ids=tuple(event["event_id"] for event in result.event_document["events"]),
        output_timeline_path=str(output_timeline_path),
        output_json_path=str(output_json_path),
        config=config,
        video_duration_sec=video_duration,
    )


__all__ = [
    "CandidateRun",
    "EventDetectionConfig",
    "EventDetectionResult",
    "EventOutputSummary",
    "REQUIRED_EVENT_COLUMNS",
    "SCHEMA_VERSION",
    "TIMELINE_ADDED_COLUMNS",
    "TRIGGER_REASON",
    "build_event_records",
    "build_frame_risk_timeline",
    "build_strict_candidate_runs",
    "detect_events",
    "estimate_frame_interval_sec",
    "generate_event_outputs",
    "merge_candidate_runs",
]

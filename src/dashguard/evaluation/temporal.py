"""Temporal event-label loading, matching, metrics, and report output."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
EVENT_TYPE = "visual_rapid_approach"
EVENT_DEFINITION = (
    "전방 객체가 자차 진행 영역에서 지속적으로 가까워지거나 화면상 크기가 "
    "명확히 증가하여 사람이 검토할 필요가 있다고 판단할 수 있는 영상 구간"
)
INTERVAL_SEMANTICS = "[start_sec, end_sec)"
PRIMARY_IOU_THRESHOLD = 0.30
SENSITIVITY_IOU_THRESHOLD = 0.50
MINIMUM_LABEL_DURATION_SEC = 0.30
SCOPE_DURATION_ABS_TOLERANCE_SEC = 0.001
EVALUATION_NAME = "single-source small-sample pilot evaluation"
PROVENANCE_STATUS = "documented_in_project_state_but_source_url_not_archived"

LABEL_CSV_COLUMNS = (
    "schema_version",
    "video_id",
    "source_video_sha256",
    "label_id",
    "event_type",
    "start_sec",
    "end_sec",
    "reviewer_id",
    "review_confidence",
    "notes",
)
LABEL_SCOPE_CSV_COLUMNS = (
    "schema_version",
    "video_id",
    "source_video_sha256",
    "video_duration_sec",
    "independent_source_id",
    "reviewer_id",
    "review_status",
)
REVIEW_CONFIDENCES = frozenset({"high", "medium", "low"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class EvaluationValidationError(ValueError):
    """Raised when an evaluation input violates the public contract."""


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{field} must be numeric and finite")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationValidationError(f"{field} must be numeric and finite")
    return result


def _csv_finite_number(value: object, *, field: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationValidationError(f"{field} must be numeric and finite")
    try:
        result = float(value)
    except ValueError as exc:
        raise EvaluationValidationError(
            f"{field} must be numeric and finite"
        ) from exc
    if not math.isfinite(result):
        raise EvaluationValidationError(f"{field} must be numeric and finite")
    return result


def _video_duration(value: object, *, field: str = "video_duration_sec") -> float:
    duration = _finite_number(value, field=field)
    if duration <= 0.0:
        raise EvaluationValidationError(f"{field} must be greater than zero")
    return duration


def _threshold(value: object, *, field: str) -> float:
    threshold = _finite_number(value, field=field)
    if not 0.0 <= threshold <= 1.0:
        raise EvaluationValidationError(f"{field} must be between 0 and 1")
    return threshold


def _validate_csv_row_values(
    row: Mapping[str | None, Any],
    *,
    columns: Sequence[str],
    prefix: str,
) -> None:
    if None in row:
        raise EvaluationValidationError(
            f"{prefix} contains values beyond the required columns"
        )
    missing = [field for field in columns if row.get(field) is None]
    if missing:
        raise EvaluationValidationError(
            f"{prefix} is missing required values: {missing}"
        )
    invalid_types = [
        field for field in columns if not isinstance(row.get(field), str)
    ]
    if invalid_types:
        raise EvaluationValidationError(
            f"{prefix} contains non-text CSV values: {invalid_types}"
        )


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    """An immutable half-open temporal event interval."""

    video_id: str
    event_type: str
    event_id: str
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "video_id",
            _nonempty_string(self.video_id, field="video_id"),
        )
        object.__setattr__(
            self,
            "event_id",
            _nonempty_string(self.event_id, field="event_id"),
        )
        if self.event_type != EVENT_TYPE:
            raise EvaluationValidationError(
                f"event_type must be {EVENT_TYPE!r}"
            )
        start = _finite_number(self.start_sec, field="start_sec")
        end = _finite_number(self.end_sec, field="end_sec")
        if start < 0.0:
            raise EvaluationValidationError("start_sec must be non-negative")
        if end <= start:
            raise EvaluationValidationError(
                "end_sec must be greater than start_sec"
            )
        object.__setattr__(self, "start_sec", start)
        object.__setattr__(self, "end_sec", end)

    @property
    def duration_sec(self) -> float:
        """Return the interval duration in seconds."""

        return self.end_sec - self.start_sec


@dataclass(frozen=True, slots=True)
class EventLabel(TemporalEvent):
    """A validated human label and its review metadata."""

    source_video_sha256: str
    reviewer_id: str
    review_confidence: str
    notes: str

    @property
    def label_id(self) -> str:
        return self.event_id


@dataclass(frozen=True, slots=True)
class EventPrediction(TemporalEvent):
    """A Day 6/8 prediction adapted to the external event type."""

    source_video: str

    @property
    def prediction_id(self) -> str:
        return self.event_id


@dataclass(frozen=True, slots=True)
class LabelScopeEntry:
    """One reviewed video in the human-label evaluation scope."""

    schema_version: str
    video_id: str
    source_video_sha256: str
    video_duration_sec: float
    independent_source_id: str
    reviewer_id: str
    review_status: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise EvaluationValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}"
            )
        object.__setattr__(
            self,
            "video_id",
            _nonempty_string(self.video_id, field="video_id"),
        )
        source_sha = _nonempty_string(
            self.source_video_sha256,
            field="source_video_sha256",
        )
        if _SHA256_PATTERN.fullmatch(source_sha) is None:
            raise EvaluationValidationError(
                "source_video_sha256 must be 64 lowercase hex characters"
            )
        object.__setattr__(self, "source_video_sha256", source_sha)
        object.__setattr__(
            self,
            "video_duration_sec",
            _video_duration(self.video_duration_sec),
        )
        object.__setattr__(
            self,
            "independent_source_id",
            _nonempty_string(
                self.independent_source_id,
                field="independent_source_id",
            ),
        )
        object.__setattr__(
            self,
            "reviewer_id",
            _nonempty_string(self.reviewer_id, field="reviewer_id"),
        )
        if self.review_status != "completed":
            raise EvaluationValidationError(
                "review_status must be 'completed'"
            )


@dataclass(frozen=True, slots=True)
class LabelScopeDataset:
    """Validated reviewed-video scope independent of positive labels."""

    entries: tuple[LabelScopeEntry, ...]

    @property
    def reviewed_video_ids(self) -> tuple[str, ...]:
        return tuple(entry.video_id for entry in self.entries)

    @property
    def source_sha256_by_video(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (entry.video_id, entry.source_video_sha256)
            for entry in self.entries
        )

    @property
    def video_durations_sec_by_video(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (entry.video_id, entry.video_duration_sec)
            for entry in self.entries
        )

    @property
    def independent_source_id_by_video(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (entry.video_id, entry.independent_source_id)
            for entry in self.entries
        )

    @property
    def unique_independent_source_count(self) -> int:
        return len({entry.independent_source_id for entry in self.entries})

    def source_sha256(self, video_id: str) -> str | None:
        return dict(self.source_sha256_by_video).get(video_id)

    def video_duration_sec(self, video_id: str) -> float | None:
        return dict(self.video_durations_sec_by_video).get(video_id)

    def independent_source_id(self, video_id: str) -> str | None:
        return dict(self.independent_source_id_by_video).get(video_id)


@dataclass(frozen=True, slots=True)
class LabelDataset:
    """Validated labels plus source hashes keyed by video."""

    events: tuple[EventLabel, ...]
    source_sha256_by_video: tuple[tuple[str, str], ...]

    def source_sha256(self, video_id: str) -> str | None:
        return dict(self.source_sha256_by_video).get(video_id)


@dataclass(frozen=True, slots=True)
class PredictionDataset:
    """Predictions and document-level provenance for one video."""

    video_id: str
    source_video: str
    video_duration_sec: float
    events: tuple[EventPrediction, ...]
    events_json_path: str


@dataclass(frozen=True, slots=True)
class TemporalMatch:
    """One selected label-prediction pair."""

    label: TemporalEvent
    prediction: TemporalEvent
    intersection_sec: float
    temporal_iou: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.label.video_id,
            "event_type": self.label.event_type,
            "label_id": self.label.event_id,
            "prediction_id": self.prediction.event_id,
            "label_start_sec": self.label.start_sec,
            "label_end_sec": self.label.end_sec,
            "prediction_start_sec": self.prediction.start_sec,
            "prediction_end_sec": self.prediction.end_sec,
            "intersection_sec": self.intersection_sec,
            "temporal_iou": self.temporal_iou,
        }


@dataclass(frozen=True, slots=True)
class TemporalMatchingResult:
    """Deterministic one-to-one temporal matching result."""

    threshold: float
    matched_pairs: tuple[TemporalMatch, ...]
    unmatched_labels: tuple[TemporalEvent, ...]
    unmatched_predictions: tuple[TemporalEvent, ...]


@dataclass(frozen=True, slots=True)
class TemporalEvaluationResult:
    """Counts and metrics at one IoU threshold."""

    threshold: float
    label_count: int
    prediction_count: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    empty_agreement: bool
    matched_pairs: tuple[TemporalMatch, ...]
    unmatched_labels: tuple[TemporalEvent, ...]
    unmatched_predictions: tuple[TemporalEvent, ...]

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "label_count": self.label_count,
            "prediction_count": self.prediction_count,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "empty_agreement": self.empty_agreement,
        }


def _validate_interval_against_duration(
    event: TemporalEvent,
    video_duration_sec: float,
    *,
    role: str,
) -> None:
    if event.end_sec > video_duration_sec:
        raise EvaluationValidationError(
            f"{role} {event.event_id!r} end_sec exceeds video_duration_sec"
        )


def _reject_json_constant(value: str) -> None:
    raise EvaluationValidationError(f"non-finite JSON number is not allowed: {value}")


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise EvaluationValidationError(
            f"invalid JSON in {path}: {exc.msg}"
        ) from exc


def load_event_label_scope(path: str | Path) -> LabelScopeDataset:
    """Load the fixed reviewed-video scope CSV for human labels."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise EvaluationValidationError("label scope CSV must contain a header")
        if tuple(reader.fieldnames) != LABEL_SCOPE_CSV_COLUMNS:
            raise EvaluationValidationError(
                "label scope CSV columns must exactly match the required ordered "
                f"schema; expected {list(LABEL_SCOPE_CSV_COLUMNS)}, "
                f"got {reader.fieldnames}"
            )

        entries: list[LabelScopeEntry] = []
        seen_video_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            prefix = f"row {row_number}"
            _validate_csv_row_values(
                row,
                columns=LABEL_SCOPE_CSV_COLUMNS,
                prefix=prefix,
            )
            schema_version = row["schema_version"].strip()
            if schema_version != SCHEMA_VERSION:
                raise EvaluationValidationError(
                    f"{prefix} schema_version must be {SCHEMA_VERSION!r}"
                )
            video_id = _nonempty_string(
                row["video_id"], field=f"{prefix} video_id"
            )
            if video_id in seen_video_ids:
                raise EvaluationValidationError(
                    f"duplicate video_id in label scope: {video_id!r}"
                )
            source_sha = row["source_video_sha256"].strip()
            if _SHA256_PATTERN.fullmatch(source_sha) is None:
                raise EvaluationValidationError(
                    f"{prefix} source_video_sha256 must be 64 lowercase hex "
                    "characters"
                )
            duration = _csv_finite_number(
                row["video_duration_sec"],
                field=f"{prefix} video_duration_sec",
            )
            if duration <= 0.0:
                raise EvaluationValidationError(
                    f"{prefix} video_duration_sec must be greater than zero"
                )
            independent_source_id = _nonempty_string(
                row["independent_source_id"],
                field=f"{prefix} independent_source_id",
            )
            reviewer_id = _nonempty_string(
                row["reviewer_id"],
                field=f"{prefix} reviewer_id",
            )
            review_status = row["review_status"].strip()
            if review_status != "completed":
                raise EvaluationValidationError(
                    f"{prefix} review_status must be 'completed'"
                )
            entries.append(
                LabelScopeEntry(
                    schema_version=schema_version,
                    video_id=video_id,
                    source_video_sha256=source_sha,
                    video_duration_sec=duration,
                    independent_source_id=independent_source_id,
                    reviewer_id=reviewer_id,
                    review_status=review_status,
                )
            )
            seen_video_ids.add(video_id)

    if not entries:
        raise EvaluationValidationError(
            "label scope CSV must contain at least one entry"
        )
    entries.sort(key=lambda entry: entry.video_id)
    return LabelScopeDataset(entries=tuple(entries))


def load_event_labels(
    path: str | Path,
    *,
    video_durations_sec: Mapping[str, float],
) -> LabelDataset:
    """Load and validate the fixed Day 9 human-label CSV schema."""

    csv_path = Path(path)
    durations = {
        _nonempty_string(video_id, field="video_durations_sec key"): _video_duration(
            duration,
            field=f"video_durations_sec[{video_id!r}]",
        )
        for video_id, duration in video_durations_sec.items()
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise EvaluationValidationError("label CSV must contain a header")
        if tuple(reader.fieldnames) != LABEL_CSV_COLUMNS:
            raise EvaluationValidationError(
                "label CSV columns must exactly match the required ordered schema; "
                f"expected {list(LABEL_CSV_COLUMNS)}, got {reader.fieldnames}"
            )

        labels: list[EventLabel] = []
        seen_ids: set[tuple[str, str]] = set()
        sha_by_video: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            prefix = f"row {row_number}"
            _validate_csv_row_values(
                row,
                columns=LABEL_CSV_COLUMNS,
                prefix=prefix,
            )
            if row["schema_version"].strip() != SCHEMA_VERSION:
                raise EvaluationValidationError(
                    f"{prefix} schema_version must be {SCHEMA_VERSION!r}"
                )
            video_id = _nonempty_string(
                row["video_id"], field=f"{prefix} video_id"
            )
            if video_id not in durations:
                raise EvaluationValidationError(
                    f"{prefix} video_id has no declared video duration: {video_id!r}"
                )
            source_sha = row["source_video_sha256"].strip()
            if _SHA256_PATTERN.fullmatch(source_sha) is None:
                raise EvaluationValidationError(
                    f"{prefix} source_video_sha256 must be 64 lowercase hex characters"
                )
            previous_sha = sha_by_video.setdefault(video_id, source_sha)
            if previous_sha != source_sha:
                raise EvaluationValidationError(
                    f"all labels for video {video_id!r} must use the same source SHA-256"
                )
            label_id = _nonempty_string(
                row["label_id"], field=f"{prefix} label_id"
            )
            identity = (video_id, label_id)
            if identity in seen_ids:
                raise EvaluationValidationError(
                    f"duplicate label_id within video {video_id!r}: {label_id!r}"
                )
            event_type = row["event_type"].strip()
            start = _csv_finite_number(
                row["start_sec"], field=f"{prefix} start_sec"
            )
            end = _csv_finite_number(
                row["end_sec"], field=f"{prefix} end_sec"
            )
            reviewer_id = _nonempty_string(
                row["reviewer_id"], field=f"{prefix} reviewer_id"
            )
            confidence = row["review_confidence"].strip()
            if confidence not in REVIEW_CONFIDENCES:
                raise EvaluationValidationError(
                    f"{prefix} review_confidence must be high, medium, or low"
                )
            notes = row["notes"]
            label = EventLabel(
                video_id=video_id,
                event_type=event_type,
                event_id=label_id,
                start_sec=start,
                end_sec=end,
                source_video_sha256=source_sha,
                reviewer_id=reviewer_id,
                review_confidence=confidence,
                notes=notes,
            )
            if label.duration_sec < MINIMUM_LABEL_DURATION_SEC and not math.isclose(
                label.duration_sec,
                MINIMUM_LABEL_DURATION_SEC,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise EvaluationValidationError(
                    f"{prefix} label duration must be at least "
                    f"{MINIMUM_LABEL_DURATION_SEC:.2f} seconds"
                )
            _validate_interval_against_duration(
                label,
                durations[video_id],
                role="label",
            )
            seen_ids.add(identity)
            labels.append(label)

    labels.sort(key=_event_sort_key)
    return LabelDataset(
        events=tuple(labels),
        source_sha256_by_video=tuple(sorted(sha_by_video.items())),
    )


def load_event_predictions(
    path: str | Path,
    *,
    video_id: str,
) -> PredictionDataset:
    """Adapt Day 6/8 events.json intervals into evaluation predictions."""

    events_path = Path(path)
    stable_video_id = _nonempty_string(video_id, field="video_id")
    payload = _load_json(events_path)
    if not isinstance(payload, Mapping):
        raise EvaluationValidationError("events JSON root must be an object")
    required = ("source_video", "video_duration_sec", "events")
    missing = [field for field in required if field not in payload]
    if missing:
        raise EvaluationValidationError(
            f"events JSON is missing required fields: {missing}"
        )
    source_video = _nonempty_string(payload["source_video"], field="source_video")
    duration = _video_duration(payload["video_duration_sec"])
    raw_events = payload["events"]
    if not isinstance(raw_events, list):
        raise EvaluationValidationError("events must be an array")

    predictions: list[EventPrediction] = []
    seen_ids: set[str] = set()
    for index, raw_event in enumerate(raw_events):
        prefix = f"events[{index}]"
        if not isinstance(raw_event, Mapping):
            raise EvaluationValidationError(f"{prefix} must be an object")
        event_missing = [
            field
            for field in ("event_id", "start_sec", "end_sec")
            if field not in raw_event
        ]
        if event_missing:
            raise EvaluationValidationError(
                f"{prefix} is missing required fields: {event_missing}"
            )
        prediction_id = _nonempty_string(
            raw_event["event_id"], field=f"{prefix}.event_id"
        )
        if prediction_id in seen_ids:
            raise EvaluationValidationError(
                f"duplicate prediction event_id: {prediction_id!r}"
            )
        prediction = EventPrediction(
            video_id=stable_video_id,
            event_type=EVENT_TYPE,
            event_id=prediction_id,
            start_sec=_finite_number(
                raw_event["start_sec"], field=f"{prefix}.start_sec"
            ),
            end_sec=_finite_number(
                raw_event["end_sec"], field=f"{prefix}.end_sec"
            ),
            source_video=source_video,
        )
        _validate_interval_against_duration(
            prediction,
            duration,
            role="prediction",
        )
        seen_ids.add(prediction_id)
        predictions.append(prediction)

    predictions.sort(key=_event_sort_key)
    return PredictionDataset(
        video_id=stable_video_id,
        source_video=source_video,
        video_duration_sec=duration,
        events=tuple(predictions),
        events_json_path=str(events_path),
    )


def _intersection_duration(first: TemporalEvent, second: TemporalEvent) -> float:
    return max(
        0.0,
        min(first.end_sec, second.end_sec)
        - max(first.start_sec, second.start_sec),
    )


def temporal_iou(first: TemporalEvent, second: TemporalEvent) -> float:
    """Calculate temporal IoU for two half-open intervals."""

    intersection = _intersection_duration(first, second)
    union = first.duration_sec + second.duration_sec - intersection
    return intersection / union


def _event_identity(event: TemporalEvent) -> tuple[str, str, str]:
    return (event.video_id, event.event_type, event.event_id)


def _event_sort_key(event: TemporalEvent) -> tuple[Any, ...]:
    return (
        event.video_id,
        event.event_type,
        event.start_sec,
        event.end_sec,
        event.event_id,
    )


def _validate_unique_events(
    events: Sequence[TemporalEvent],
    *,
    role: str,
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        if not isinstance(event, TemporalEvent):
            raise TypeError(f"{role} entries must be TemporalEvent instances")
        identity = _event_identity(event)
        if identity in seen:
            raise EvaluationValidationError(
                f"duplicate {role} ID within video: {event.event_id!r}"
            )
        seen.add(identity)


def match_temporal_events(
    labels: Sequence[TemporalEvent],
    predictions: Sequence[TemporalEvent],
    *,
    threshold: float,
) -> TemporalMatchingResult:
    """Greedily select deterministic one-to-one pairs at an IoU threshold."""

    selected_threshold = _threshold(threshold, field="threshold")
    _validate_unique_events(labels, role="label")
    _validate_unique_events(predictions, role="prediction")
    candidates: list[tuple[tuple[Any, ...], TemporalMatch]] = []
    for label in labels:
        for prediction in predictions:
            if (
                label.video_id != prediction.video_id
                or label.event_type != prediction.event_type
            ):
                continue
            intersection = _intersection_duration(label, prediction)
            iou = temporal_iou(label, prediction)
            if iou < selected_threshold:
                continue
            sort_key = (
                -iou,
                -intersection,
                abs(prediction.start_sec - label.start_sec),
                abs(prediction.end_sec - label.end_sec),
                label.event_id,
                prediction.event_id,
                label.video_id,
                label.event_type,
            )
            candidates.append(
                (
                    sort_key,
                    TemporalMatch(
                        label=label,
                        prediction=prediction,
                        intersection_sec=intersection,
                        temporal_iou=iou,
                    ),
                )
            )
    candidates.sort(key=lambda item: item[0])

    matched: list[TemporalMatch] = []
    used_labels: set[tuple[str, str, str]] = set()
    used_predictions: set[tuple[str, str, str]] = set()
    for _, candidate in candidates:
        label_identity = _event_identity(candidate.label)
        prediction_identity = _event_identity(candidate.prediction)
        if label_identity in used_labels or prediction_identity in used_predictions:
            continue
        used_labels.add(label_identity)
        used_predictions.add(prediction_identity)
        matched.append(candidate)

    unmatched_labels = tuple(
        sorted(
            (label for label in labels if _event_identity(label) not in used_labels),
            key=_event_sort_key,
        )
    )
    unmatched_predictions = tuple(
        sorted(
            (
                prediction
                for prediction in predictions
                if _event_identity(prediction) not in used_predictions
            ),
            key=_event_sort_key,
        )
    )
    return TemporalMatchingResult(
        threshold=selected_threshold,
        matched_pairs=tuple(matched),
        unmatched_labels=unmatched_labels,
        unmatched_predictions=unmatched_predictions,
    )


def evaluate_temporal_events(
    labels: Sequence[TemporalEvent],
    predictions: Sequence[TemporalEvent],
    *,
    threshold: float,
) -> TemporalEvaluationResult:
    """Calculate TP, FP, FN, precision, and recall at one threshold."""

    matching = match_temporal_events(labels, predictions, threshold=threshold)
    tp = len(matching.matched_pairs)
    fp = len(matching.unmatched_predictions)
    fn = len(matching.unmatched_labels)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    return TemporalEvaluationResult(
        threshold=matching.threshold,
        label_count=len(labels),
        prediction_count=len(predictions),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=(tp / precision_denominator if precision_denominator else None),
        recall=(tp / recall_denominator if recall_denominator else None),
        empty_agreement=not labels and not predictions,
        matched_pairs=matching.matched_pairs,
        unmatched_labels=matching.unmatched_labels,
        unmatched_predictions=matching.unmatched_predictions,
    )


def aggregate_micro_results(
    results: Sequence[TemporalEvaluationResult],
) -> dict[str, Any]:
    """Aggregate counts first, then calculate micro precision and recall."""

    thresholds = {result.threshold for result in results}
    if len(thresholds) > 1:
        raise EvaluationValidationError(
            "micro aggregation requires results from one threshold"
        )
    tp = sum(result.tp for result in results)
    fp = sum(result.fp for result in results)
    fn = sum(result.fn for result in results)
    label_count = sum(result.label_count for result in results)
    prediction_count = sum(result.prediction_count for result in results)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    return {
        "threshold": next(iter(thresholds), None),
        "video_count": len(results),
        "label_count": label_count,
        "prediction_count": prediction_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / precision_denominator if precision_denominator else None,
        "recall": tp / recall_denominator if recall_denominator else None,
        "empty_agreement": label_count == 0 and prediction_count == 0,
    }


def _unmatched_record(event: TemporalEvent, *, id_field: str) -> dict[str, Any]:
    return {
        "video_id": event.video_id,
        "event_type": event.event_type,
        id_field: event.event_id,
        "start_sec": event.start_sec,
        "end_sec": event.end_sec,
    }


def _prediction_datasets_by_video(
    predictions: Sequence[PredictionDataset],
) -> dict[str, PredictionDataset]:
    prediction_by_video: dict[str, PredictionDataset] = {}
    for dataset in predictions:
        if not isinstance(dataset, PredictionDataset):
            raise TypeError("predictions must contain PredictionDataset instances")
        if dataset.video_id in prediction_by_video:
            raise EvaluationValidationError(
                f"duplicate prediction video_id: {dataset.video_id!r}"
            )
        prediction_by_video[dataset.video_id] = dataset
    return prediction_by_video


def validate_label_scope_predictions(
    label_scope: LabelScopeDataset,
    predictions: Sequence[PredictionDataset],
) -> None:
    """Require exactly one duration-compatible prediction per scoped video."""

    if not isinstance(label_scope, LabelScopeDataset):
        raise TypeError("label_scope must be a LabelScopeDataset")
    prediction_by_video = _prediction_datasets_by_video(predictions)
    scope_video_ids = set(label_scope.reviewed_video_ids)
    prediction_video_ids = set(prediction_by_video)
    if scope_video_ids != prediction_video_ids:
        raise EvaluationValidationError(
            "label scope and prediction video IDs must match exactly; "
            f"missing predictions={sorted(scope_video_ids - prediction_video_ids)}, "
            f"unexpected predictions={sorted(prediction_video_ids - scope_video_ids)}"
        )
    scope_durations = dict(label_scope.video_durations_sec_by_video)
    for video_id in sorted(scope_video_ids):
        scope_duration = scope_durations[video_id]
        prediction_duration = prediction_by_video[video_id].video_duration_sec
        if not math.isclose(
            prediction_duration,
            scope_duration,
            rel_tol=0.0,
            abs_tol=SCOPE_DURATION_ABS_TOLERANCE_SEC,
        ):
            raise EvaluationValidationError(
                f"prediction duration does not match label scope for {video_id!r}: "
                f"scope={scope_duration}, prediction={prediction_duration}, "
                f"absolute tolerance={SCOPE_DURATION_ABS_TOLERANCE_SEC}"
            )


def _validate_labels_against_scope(
    labels: LabelDataset,
    label_scope: LabelScopeDataset,
) -> None:
    scope_sha_by_video = dict(label_scope.source_sha256_by_video)
    for label in labels.events:
        scope_sha = scope_sha_by_video.get(label.video_id)
        if scope_sha is None:
            raise EvaluationValidationError(
                f"positive label video_id is outside label scope: {label.video_id!r}"
            )
        if label.source_video_sha256 != scope_sha:
            raise EvaluationValidationError(
                f"positive label SHA-256 does not match label scope for "
                f"{label.video_id!r}"
            )


def build_evaluation_report(
    labels: LabelDataset,
    predictions: Sequence[PredictionDataset],
    *,
    label_scope: LabelScopeDataset,
    primary_threshold: float = PRIMARY_IOU_THRESHOLD,
    sensitivity_threshold: float = SENSITIVITY_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Build the complete JSON-serializable Day 9 evaluation report."""

    primary = _threshold(primary_threshold, field="primary_threshold")
    sensitivity = _threshold(
        sensitivity_threshold,
        field="sensitivity_threshold",
    )
    validate_label_scope_predictions(label_scope, predictions)
    _validate_labels_against_scope(labels, label_scope)
    prediction_by_video = _prediction_datasets_by_video(predictions)
    scope_sha_by_video = dict(label_scope.source_sha256_by_video)
    scope_duration_by_video = dict(label_scope.video_durations_sec_by_video)

    threshold_values = {"primary": primary, "sensitivity": sensitivity}
    results_by_threshold: dict[str, list[TemporalEvaluationResult]] = {
        name: [] for name in threshold_values
    }
    matched_pairs: dict[str, list[dict[str, Any]]] = {
        name: [] for name in threshold_values
    }
    unmatched_labels: dict[str, list[dict[str, Any]]] = {
        name: [] for name in threshold_values
    }
    unmatched_predictions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in threshold_values
    }
    per_video: list[dict[str, Any]] = []
    for video_id in sorted(prediction_by_video):
        dataset = prediction_by_video[video_id]
        video_labels = tuple(
            label for label in labels.events if label.video_id == video_id
        )
        video_results: dict[str, dict[str, Any]] = {}
        for threshold_name, threshold_value in threshold_values.items():
            result = evaluate_temporal_events(
                video_labels,
                dataset.events,
                threshold=threshold_value,
            )
            results_by_threshold[threshold_name].append(result)
            video_results[threshold_name] = result.metrics_dict()
            matched_pairs[threshold_name].extend(
                match.to_dict() for match in result.matched_pairs
            )
            unmatched_labels[threshold_name].extend(
                _unmatched_record(event, id_field="label_id")
                for event in result.unmatched_labels
            )
            unmatched_predictions[threshold_name].extend(
                _unmatched_record(event, id_field="prediction_id")
                for event in result.unmatched_predictions
            )
        per_video.append(
            {
                "video_id": video_id,
                "source_video": dataset.source_video,
                "source_video_sha256": scope_sha_by_video[video_id],
                "video_duration_sec": scope_duration_by_video[video_id],
                "events_json": dataset.events_json_path,
                "results": video_results,
            }
        )

    micro = {
        name: aggregate_micro_results(results_by_threshold[name])
        for name in threshold_values
    }
    source_count = label_scope.unique_independent_source_count
    evaluation_name = (
        EVALUATION_NAME
        if source_count == 1
        else "multi-source small-sample pilot evaluation"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_name": evaluation_name,
        "evaluation_scope": {
            "independent_source_count": source_count,
            "segment_count": len(label_scope.entries),
            "reviewed_video_ids": list(label_scope.reviewed_video_ids),
            "reviewed_source_sha256_by_video": dict(
                label_scope.source_sha256_by_video
            ),
            "independence_note": (
                "동일 independent_source_id의 segment는 독립 원본 수를 "
                "늘리지 않는다."
            ),
        },
        "event_definition": {
            "event_type": EVENT_TYPE,
            "definition": EVENT_DEFINITION,
            "minimum_human_label_duration_sec": MINIMUM_LABEL_DURATION_SEC,
        },
        "interval_semantics": INTERVAL_SEMANTICS,
        "thresholds": threshold_values,
        "per_video": per_video,
        "micro": micro,
        "matched_pairs": matched_pairs,
        "unmatched_labels": unmatched_labels,
        "unmatched_predictions": unmatched_predictions,
        "provenance_status": PROVENANCE_STATUS,
        "limitations": [
            (
                "한 독립 원본을 사용하는 small-sample pilot 평가이다."
                if source_count == 1
                else f"{source_count}개 독립 "
                "원본을 사용하는 small-sample pilot 평가이다."
            ),
            "실제 거리 또는 TTC 평가는 아니다.",
            "사고·충돌 확률 평가가 아니다.",
            "안전성 검증이 아니다.",
            "현재 event score threshold 0.70을 최적화하거나 검증하지 않는다.",
            "synthetic telemetry 성능 평가가 아니다.",
            "IDF1 또는 HOTA 평가는 아니다.",
        ],
    }


def _assert_json_serializable(report: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError(
            f"evaluation report is not standard JSON serializable: {exc}"
        ) from exc


def write_evaluation_report(
    report: Mapping[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a report, refusing replacement unless authorized."""

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"evaluation report already exists; pass overwrite=True: {destination}"
        )
    serialized = _assert_json_serializable(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(
                "evaluation report appeared during generation; refusing to replace: "
                f"{destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "EVENT_DEFINITION",
    "EVENT_TYPE",
    "EVALUATION_NAME",
    "EvaluationValidationError",
    "EventLabel",
    "EventPrediction",
    "INTERVAL_SEMANTICS",
    "LABEL_CSV_COLUMNS",
    "LABEL_SCOPE_CSV_COLUMNS",
    "LabelDataset",
    "LabelScopeDataset",
    "LabelScopeEntry",
    "MINIMUM_LABEL_DURATION_SEC",
    "PRIMARY_IOU_THRESHOLD",
    "PROVENANCE_STATUS",
    "PredictionDataset",
    "SCHEMA_VERSION",
    "SCOPE_DURATION_ABS_TOLERANCE_SEC",
    "SENSITIVITY_IOU_THRESHOLD",
    "TemporalEvaluationResult",
    "TemporalEvent",
    "TemporalMatch",
    "TemporalMatchingResult",
    "aggregate_micro_results",
    "build_evaluation_report",
    "evaluate_temporal_events",
    "load_event_label_scope",
    "load_event_labels",
    "load_event_predictions",
    "match_temporal_events",
    "temporal_iou",
    "validate_label_scope_predictions",
    "write_evaluation_report",
]

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import dashguard.evaluation.temporal as temporal_module
from dashguard.evaluation import (
    EVENT_TYPE,
    LABEL_CSV_COLUMNS,
    LABEL_SCOPE_CSV_COLUMNS,
    EvaluationValidationError,
    LabelDataset,
    LabelScopeDataset,
    TemporalEvent,
    aggregate_micro_results,
    build_evaluation_report,
    evaluate_temporal_events,
    load_event_label_scope,
    load_event_labels,
    load_event_predictions,
    match_temporal_events,
    temporal_iou,
    write_evaluation_report,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _event(
    event_id: str,
    start: float,
    end: float,
    *,
    video_id: str = "video-1",
) -> TemporalEvent:
    return TemporalEvent(video_id, EVENT_TYPE, event_id, start, end)


def _label_row(**changes: str) -> dict[str, str]:
    row = {
        "schema_version": "1.0",
        "video_id": "video-1",
        "source_video_sha256": SHA_A,
        "label_id": "L1",
        "event_type": EVENT_TYPE,
        "start_sec": "1.0",
        "end_sec": "1.3",
        "reviewer_id": "reviewer-1",
        "review_confidence": "high",
        "notes": "",
    }
    row.update(changes)
    return row


def _write_labels(
    path: Path,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] = LABEL_CSV_COLUMNS,
    encoding: str = "utf-8",
) -> None:
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _scope_row(**changes: str) -> dict[str, str]:
    row = {
        "schema_version": "1.0",
        "video_id": "video-1",
        "source_video_sha256": SHA_A,
        "video_duration_sec": "10.0",
        "independent_source_id": "source-1",
        "reviewer_id": "reviewer-1",
        "review_status": "completed",
    }
    row.update(changes)
    return row


def _write_scope(
    path: Path,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] = LABEL_SCOPE_CSV_COLUMNS,
    encoding: str = "utf-8",
) -> None:
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_document(
    events: list[dict[str, Any]],
    *,
    duration: Any = 10.0,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source_video": "source.mp4",
        "video_duration_sec": duration,
        "events": events,
    }


def _write_prediction_json(
    path: Path,
    events: list[dict[str, Any]],
    *,
    duration: Any = 10.0,
) -> None:
    path.write_text(
        json.dumps(_prediction_document(events, duration=duration)),
        encoding="utf-8",
    )


def _prediction_event(
    event_id: str,
    start: Any = 1.0,
    end: Any = 2.0,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "start_sec": start,
        "end_sec": end,
        **extra,
    }


def _load_prediction_datasets(
    directory: Path,
    specifications: list[tuple[str, float]],
) -> tuple[Any, ...]:
    datasets = []
    for video_id, duration in specifications:
        path = directory / f"{video_id}.json"
        _write_prediction_json(path, [], duration=duration)
        datasets.append(load_event_predictions(path, video_id=video_id))
    return tuple(datasets)


def test_identical_interval_has_iou_one() -> None:
    assert temporal_iou(_event("L1", 1.0, 2.0), _event("P1", 1.0, 2.0)) == 1.0


def test_partial_overlap_iou() -> None:
    assert temporal_iou(
        _event("L1", 10.0, 12.0),
        _event("P1", 10.5, 11.5),
    ) == pytest.approx(0.5)


def test_touching_intervals_have_zero_iou() -> None:
    assert temporal_iou(_event("L1", 1.0, 2.0), _event("P1", 2.0, 3.0)) == 0.0


def test_disjoint_intervals_have_zero_iou() -> None:
    assert temporal_iou(_event("L1", 1.0, 2.0), _event("P1", 4.0, 5.0)) == 0.0


def test_label_duration_exactly_point_three_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row(start_sec="1.0", end_sec="1.3")])

    dataset = load_event_labels(path, video_durations_sec={"video-1": 5.0})

    assert len(dataset.events) == 1
    assert dataset.events[0].duration_sec == pytest.approx(0.3)


def test_label_duration_below_point_three_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row(start_sec="1.0", end_sec="1.299")])

    with pytest.raises(EvaluationValidationError, match="at least 0.30"):
        load_event_labels(path, video_durations_sec={"video-1": 5.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_time_is_rejected(value: float) -> None:
    with pytest.raises(EvaluationValidationError, match="numeric and finite"):
        _event("L1", value, 2.0)


def test_negative_start_is_rejected() -> None:
    with pytest.raises(EvaluationValidationError, match="non-negative"):
        _event("L1", -0.1, 1.0)


@pytest.mark.parametrize("end", [1.0, 0.5])
def test_end_not_greater_than_start_is_rejected(end: float) -> None:
    with pytest.raises(EvaluationValidationError, match="greater than start_sec"):
        _event("L1", 1.0, end)


def test_interval_exceeding_video_duration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row(start_sec="4.8", end_sec="5.1")])

    with pytest.raises(EvaluationValidationError, match="exceeds video_duration"):
        load_event_labels(path, video_durations_sec={"video-1": 5.0})


def test_duplicate_label_id_within_video_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(
        path,
        [
            _label_row(label_id="L1", start_sec="1.0", end_sec="1.3"),
            _label_row(label_id="L1", start_sec="2.0", end_sec="2.3"),
        ],
    )

    with pytest.raises(EvaluationValidationError, match="duplicate label_id"):
        load_event_labels(path, video_durations_sec={"video-1": 5.0})


def test_duplicate_prediction_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_prediction_json(
        path,
        [_prediction_event("P1"), _prediction_event("P1", 2.0, 3.0)],
    )

    with pytest.raises(EvaluationValidationError, match="duplicate prediction"):
        load_event_predictions(path, video_id="video-1")


@pytest.mark.parametrize("sha", ["A" * 64, "a" * 63, "g" * 64])
def test_invalid_source_sha256_is_rejected(tmp_path: Path, sha: str) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row(source_video_sha256=sha)])

    with pytest.raises(EvaluationValidationError, match="64 lowercase hex"):
        load_event_labels(path, video_durations_sec={"video-1": 5.0})


def test_unexpected_csv_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    columns = (*LABEL_CSV_COLUMNS, "prediction_id")
    _write_labels(
        path,
        [{**_label_row(), "prediction_id": "P1"}],
        columns=columns,
    )

    with pytest.raises(EvaluationValidationError, match="exactly match"):
        load_event_labels(path, video_durations_sec={"video-1": 5.0})


def test_header_only_label_csv_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path, [])

    dataset = load_event_labels(path, video_durations_sec={"video-1": 5.0})

    assert dataset == LabelDataset(events=(), source_sha256_by_video=())


def test_one_prediction_overlapping_multiple_labels_is_only_one_tp() -> None:
    labels = [_event("L1", 0.0, 2.0), _event("L2", 0.5, 2.0)]
    predictions = [_event("P1", 0.0, 2.0)]

    result = evaluate_temporal_events(labels, predictions, threshold=0.3)

    assert (result.tp, result.fp, result.fn) == (1, 0, 1)
    assert result.matched_pairs[0].label.event_id == "L1"


def test_multiple_predictions_overlapping_one_label_are_only_one_tp() -> None:
    labels = [_event("L1", 0.0, 2.0)]
    predictions = [_event("P1", 0.0, 2.0), _event("P2", 0.5, 2.0)]

    result = evaluate_temporal_events(labels, predictions, threshold=0.3)

    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.matched_pairs[0].prediction.event_id == "P1"


def test_equal_iou_tie_break_is_deterministic() -> None:
    labels = [_event("L2", 1.0, 2.0), _event("L1", 1.0, 2.0)]
    predictions = [_event("P2", 1.0, 2.0), _event("P1", 1.0, 2.0)]

    matching = match_temporal_events(labels, predictions, threshold=0.3)

    assert [
        (pair.label.event_id, pair.prediction.event_id)
        for pair in matching.matched_pairs
    ] == [("L1", "P1"), ("L2", "P2")]


def test_input_order_does_not_change_matching() -> None:
    labels = [_event("L1", 0.0, 2.0), _event("L2", 3.0, 5.0)]
    predictions = [_event("P1", 0.5, 2.0), _event("P2", 3.0, 4.5)]

    first = match_temporal_events(labels, predictions, threshold=0.3)
    second = match_temporal_events(
        list(reversed(labels)),
        list(reversed(predictions)),
        threshold=0.3,
    )

    assert first == second


def test_iou_equal_to_threshold_is_matched() -> None:
    result = evaluate_temporal_events(
        [_event("L1", 0.0, 1.0)],
        [_event("P1", 0.0, 0.3)],
        threshold=0.3,
    )

    assert result.tp == 1


def test_empty_labels_and_predictions_contract() -> None:
    result = evaluate_temporal_events([], [], threshold=0.3)

    assert (result.tp, result.fp, result.fn) == (0, 0, 0)
    assert result.precision is None
    assert result.recall is None
    assert result.empty_agreement is True


def test_empty_labels_with_predictions_contract() -> None:
    result = evaluate_temporal_events(
        [],
        [_event("P1", 0.0, 1.0)],
        threshold=0.3,
    )

    assert (result.tp, result.fp, result.fn) == (0, 1, 0)
    assert result.precision == 0.0
    assert result.recall is None
    assert result.empty_agreement is False


def test_labels_with_empty_predictions_contract() -> None:
    result = evaluate_temporal_events(
        [_event("L1", 0.0, 1.0)],
        [],
        threshold=0.3,
    )

    assert (result.tp, result.fp, result.fn) == (0, 0, 1)
    assert result.precision is None
    assert result.recall == 0.0
    assert result.empty_agreement is False


def _manual_example() -> tuple[list[TemporalEvent], list[TemporalEvent]]:
    labels = [_event("L1", 10.0, 12.0), _event("L2", 20.0, 22.0)]
    predictions = [
        _event("P1", 10.5, 11.5),
        _event("P2", 21.0, 23.0),
        _event("P3", 5.0, 6.0),
    ]
    return labels, predictions


def test_manual_example_primary_point_three() -> None:
    labels, predictions = _manual_example()
    result = evaluate_temporal_events(labels, predictions, threshold=0.3)

    assert temporal_iou(labels[0], predictions[0]) == pytest.approx(0.5)
    assert temporal_iou(labels[1], predictions[1]) == pytest.approx(1 / 3)
    assert (result.tp, result.fp, result.fn) == (2, 1, 0)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(1.0)


def test_manual_example_sensitivity_point_five() -> None:
    labels, predictions = _manual_example()
    result = evaluate_temporal_events(labels, predictions, threshold=0.5)

    assert (result.tp, result.fp, result.fn) == (1, 2, 1)
    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == pytest.approx(0.5)


def test_multiple_video_micro_aggregates_counts_before_metrics() -> None:
    video_one = evaluate_temporal_events(
        [_event("L1", 0.0, 1.0, video_id="v1")],
        [_event("P1", 0.0, 1.0, video_id="v1")],
        threshold=0.3,
    )
    video_two = evaluate_temporal_events(
        [_event("L1", 0.0, 1.0, video_id="v2")],
        [_event("P1", 2.0, 3.0, video_id="v2")],
        threshold=0.3,
    )

    micro = aggregate_micro_results([video_one, video_two])

    assert (micro["tp"], micro["fp"], micro["fn"]) == (1, 1, 1)
    assert micro["precision"] == pytest.approx(0.5)
    assert micro["recall"] == pytest.approx(0.5)


def test_day8_events_json_adapter_uses_event_interval(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_prediction_json(path, [_prediction_event("event_001", 1.25, 2.75)])

    dataset = load_event_predictions(path, video_id="segment-1")

    assert dataset.video_id == "segment-1"
    assert dataset.source_video == "source.mp4"
    assert dataset.video_duration_sec == 10.0
    assert dataset.events[0].event_type == EVENT_TYPE
    assert (dataset.events[0].start_sec, dataset.events[0].end_sec) == (1.25, 2.75)


def test_prediction_shorter_than_point_three_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_prediction_json(path, [_prediction_event("P1", 1.0, 1.1)])

    dataset = load_event_predictions(path, video_id="video-1")

    assert dataset.events[0].duration_sec == pytest.approx(0.1)


def test_adapter_does_not_use_clip_interval(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_prediction_json(
        path,
        [
            _prediction_event(
                "P1",
                2.0,
                3.0,
                clip_start_sec=0.0,
                clip_end_sec=9.0,
            )
        ],
    )

    prediction = load_event_predictions(path, video_id="video-1").events[0]

    assert (prediction.start_sec, prediction.end_sec) == (2.0, 3.0)


def test_hard_brake_and_max_score_do_not_affect_matching(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    _write_prediction_json(
        path,
        [
            _prediction_event("P_a", 1.0, 2.0, hard_brake=False, max_total_score=0.1),
            _prediction_event("P_b", 1.0, 2.0, hard_brake=True, max_total_score=1.0),
        ],
    )
    predictions = load_event_predictions(path, video_id="video-1").events

    result = evaluate_temporal_events(
        [_event("L1", 1.0, 2.0)],
        predictions,
        threshold=0.3,
    )

    assert result.matched_pairs[0].prediction.event_id == "P_a"


def test_output_overwrite_is_protected_and_none_serializes_as_null(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    path.write_text("old report", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_evaluation_report({"precision": None}, path)

    assert path.read_text(encoding="utf-8") == "old report"
    write_evaluation_report({"precision": None}, path, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["precision"] is None
    assert '"precision": null' in path.read_text(encoding="utf-8")


def test_atomic_output_failure_preserves_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    path.write_text("old report", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(temporal_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        write_evaluation_report({"precision": 1.0}, path, overwrite=True)

    assert path.read_text(encoding="utf-8") == "old report"
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_report_separates_primary_and_sensitivity_and_has_required_sections(
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.csv"
    scope_path = tmp_path / "scope.csv"
    events_path = tmp_path / "events.json"
    _write_labels(
        labels_path,
        [_label_row(start_sec="1.0", end_sec="2.0")],
    )
    _write_prediction_json(
        events_path,
        [_prediction_event("P1", 1.0, 1.5)],
    )
    _write_scope(scope_path, [_scope_row()])
    prediction_dataset = load_event_predictions(events_path, video_id="video-1")
    scope_dataset = load_event_label_scope(scope_path)
    label_dataset = load_event_labels(
        labels_path,
        video_durations_sec=dict(scope_dataset.video_durations_sec_by_video),
    )

    report = build_evaluation_report(
        label_dataset,
        [prediction_dataset],
        label_scope=scope_dataset,
    )

    assert {
        "schema_version",
        "evaluation_name",
        "evaluation_scope",
        "event_definition",
        "interval_semantics",
        "thresholds",
        "per_video",
        "micro",
        "matched_pairs",
        "unmatched_labels",
        "unmatched_predictions",
        "limitations",
    } <= report.keys()
    assert report["evaluation_scope"] == {
        "independent_source_count": 1,
        "segment_count": 1,
        "reviewed_video_ids": ["video-1"],
        "reviewed_source_sha256_by_video": {"video-1": SHA_A},
        "independence_note": (
            "동일 independent_source_id의 segment는 독립 원본 수를 늘리지 않는다."
        ),
    }
    assert report["evaluation_name"] == (
        "single-source small-sample pilot evaluation"
    )
    assert report["evaluation_scope"]["independent_source_count"] == 1
    assert report["micro"]["primary"]["tp"] == 1
    assert report["micro"]["sensitivity"]["tp"] == 1
    pair = report["matched_pairs"]["primary"][0]
    assert {
        "video_id",
        "label_id",
        "prediction_id",
        "label_start_sec",
        "label_end_sec",
        "prediction_start_sec",
        "prediction_end_sec",
        "intersection_sec",
        "temporal_iou",
    } <= pair.keys()


def test_cli_evaluates_explicit_video_and_writes_report(tmp_path: Path) -> None:
    from scripts import day9_evaluate_events

    labels_path = tmp_path / "labels.csv"
    scope_path = tmp_path / "scope.csv"
    events_path = tmp_path / "events.json"
    output_path = tmp_path / "report.json"
    _write_labels(labels_path, [_label_row(start_sec="1.0", end_sec="2.0")])
    _write_scope(scope_path, [_scope_row()])
    _write_prediction_json(events_path, [_prediction_event("P1", 1.0, 2.0)])

    exit_code = day9_evaluate_events.main(
        [
            "--labels-csv",
            str(labels_path),
            "--label-scope-csv",
            str(scope_path),
            "--prediction",
            "video-1",
            str(events_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["micro"]["primary"]["tp"] == 1


def test_cli_evaluates_manifest_with_relative_events_path(tmp_path: Path) -> None:
    from scripts import day9_evaluate_events

    labels_path = tmp_path / "labels.csv"
    scope_path = tmp_path / "scope.csv"
    events_path = tmp_path / "inputs" / "events.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "report.json"
    events_path.parent.mkdir()
    _write_labels(labels_path, [])
    _write_scope(scope_path, [_scope_row()])
    _write_prediction_json(events_path, [])
    manifest_path.write_text(
        json.dumps(
            {
                "predictions": [
                    {"video_id": "video-1", "events_json": "inputs/events.json"}
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = day9_evaluate_events.main(
        [
            "--labels-csv",
            str(labels_path),
            "--label-scope-csv",
            str(scope_path),
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["micro"]["primary"]["empty_agreement"] is True


def test_truncated_label_row_is_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        ",".join(LABEL_CSV_COLUMNS)
        + "\n1.0,video-1,"
        + SHA_A
        + ",L1,"
        + EVENT_TYPE
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationValidationError, match="missing required values"):
        load_event_labels(path, video_durations_sec={"video-1": 10.0})


def test_label_row_missing_final_notes_is_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    values = list(_label_row().values())[:-1]
    path.write_text(
        ",".join(LABEL_CSV_COLUMNS) + "\n" + ",".join(values) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationValidationError, match="missing required values"):
        load_event_labels(path, video_durations_sec={"video-1": 10.0})


def test_label_row_with_extra_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        ",".join(LABEL_CSV_COLUMNS)
        + "\n"
        + ",".join(_label_row().values())
        + ",unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationValidationError, match="beyond the required"):
        load_event_labels(path, video_durations_sec={"video-1": 10.0})


def test_utf8_bom_label_csv_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row()], encoding="utf-8-sig")

    dataset = load_event_labels(path, video_durations_sec={"video-1": 10.0})

    assert len(dataset.events) == 1


def test_utf8_bom_scope_csv_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "scope.csv"
    _write_scope(path, [_scope_row()], encoding="utf-8-sig")

    dataset = load_event_label_scope(path)

    assert dataset.reviewed_video_ids == ("video-1",)
    assert dataset.unique_independent_source_count == 1


def test_header_only_scope_csv_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scope.csv"
    _write_scope(path, [])

    with pytest.raises(EvaluationValidationError, match="at least one entry"):
        load_event_label_scope(path)


def test_duplicate_scope_video_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scope.csv"
    _write_scope(path, [_scope_row(), _scope_row()])

    with pytest.raises(EvaluationValidationError, match="duplicate video_id"):
        load_event_label_scope(path)


@pytest.mark.parametrize(
    ("scope_rows", "prediction_specs"),
    [
        (
            [
                _scope_row(),
                _scope_row(video_id="video-2", source_video_sha256=SHA_B),
            ],
            [("video-1", 10.0)],
        ),
        (
            [_scope_row()],
            [("video-1", 10.0), ("video-2", 10.0)],
        ),
    ],
)
def test_scope_and_prediction_video_set_mismatch_is_rejected(
    tmp_path: Path,
    scope_rows: list[dict[str, str]],
    prediction_specs: list[tuple[str, float]],
) -> None:
    scope_path = tmp_path / "scope.csv"
    labels_path = tmp_path / "labels.csv"
    _write_scope(scope_path, scope_rows)
    scope = load_event_label_scope(scope_path)
    _write_labels(labels_path, [])
    labels = load_event_labels(
        labels_path,
        video_durations_sec=dict(scope.video_durations_sec_by_video),
    )
    predictions = _load_prediction_datasets(tmp_path, prediction_specs)

    with pytest.raises(EvaluationValidationError, match="video IDs must match"):
        build_evaluation_report(labels, predictions, label_scope=scope)


def test_scope_and_prediction_duration_mismatch_is_rejected(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.csv"
    labels_path = tmp_path / "labels.csv"
    _write_scope(scope_path, [_scope_row(video_duration_sec="10.0")])
    scope = load_event_label_scope(scope_path)
    _write_labels(labels_path, [])
    labels = load_event_labels(
        labels_path,
        video_durations_sec=dict(scope.video_durations_sec_by_video),
    )
    predictions = _load_prediction_datasets(tmp_path, [("video-1", 10.01)])

    with pytest.raises(EvaluationValidationError, match="duration does not match"):
        build_evaluation_report(labels, predictions, label_scope=scope)


def test_positive_label_sha_must_match_scope_sha(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.csv"
    labels_path = tmp_path / "labels.csv"
    _write_scope(scope_path, [_scope_row(source_video_sha256=SHA_A)])
    scope = load_event_label_scope(scope_path)
    _write_labels(labels_path, [_label_row(source_video_sha256=SHA_B)])
    labels = load_event_labels(
        labels_path,
        video_durations_sec=dict(scope.video_durations_sec_by_video),
    )
    predictions = _load_prediction_datasets(tmp_path, [("video-1", 10.0)])

    with pytest.raises(EvaluationValidationError, match="SHA-256 does not match"):
        build_evaluation_report(labels, predictions, label_scope=scope)


def test_header_only_labels_with_three_video_scope_are_valid(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.csv"
    labels_path = tmp_path / "labels.csv"
    _write_scope(
        scope_path,
        [
            _scope_row(),
            _scope_row(
                video_id="video-2",
                source_video_sha256=SHA_B,
                video_duration_sec="20.0",
            ),
            _scope_row(
                video_id="video-3",
                source_video_sha256=SHA_C,
                video_duration_sec="30.0",
            ),
        ],
    )
    scope = load_event_label_scope(scope_path)
    _write_labels(labels_path, [])
    labels = load_event_labels(
        labels_path,
        video_durations_sec=dict(scope.video_durations_sec_by_video),
    )
    predictions = _load_prediction_datasets(
        tmp_path,
        [("video-1", 10.0), ("video-2", 20.0), ("video-3", 30.0)],
    )

    report = build_evaluation_report(labels, predictions, label_scope=scope)

    assert report["evaluation_scope"]["segment_count"] == 3
    assert report["evaluation_scope"]["reviewed_video_ids"] == [
        "video-1",
        "video-2",
        "video-3",
    ]
    assert report["micro"]["primary"]["label_count"] == 0


def test_report_independent_source_count_comes_from_scope(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.csv"
    labels_path = tmp_path / "labels.csv"
    _write_scope(
        scope_path,
        [
            _scope_row(independent_source_id="source-A"),
            _scope_row(
                video_id="video-2",
                source_video_sha256=SHA_B,
                independent_source_id="source-B",
            ),
        ],
    )
    scope = load_event_label_scope(scope_path)
    _write_labels(labels_path, [])
    labels = load_event_labels(
        labels_path,
        video_durations_sec=dict(scope.video_durations_sec_by_video),
    )
    predictions = _load_prediction_datasets(
        tmp_path,
        [("video-1", 10.0), ("video-2", 10.0)],
    )

    report = build_evaluation_report(labels, predictions, label_scope=scope)

    assert scope.unique_independent_source_count == 2
    assert report["evaluation_name"] == (
        "multi-source small-sample pilot evaluation"
    )
    assert report["evaluation_scope"]["independent_source_count"] == 2


def test_cli_refuses_to_replace_label_scope_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import day9_evaluate_events

    labels_path = tmp_path / "labels.csv"
    scope_path = tmp_path / "scope.csv"
    events_path = tmp_path / "events.json"
    _write_labels(labels_path, [])
    _write_scope(scope_path, [_scope_row()])
    original_scope = scope_path.read_bytes()
    _write_prediction_json(events_path, [])

    exit_code = day9_evaluate_events.main(
        [
            "--labels-csv",
            str(labels_path),
            "--label-scope-csv",
            str(scope_path),
            "--prediction",
            "video-1",
            str(events_path),
            "--output-json",
            str(scope_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "must not replace an input file" in captured.err
    assert "Traceback" not in captured.err
    assert scope_path.read_bytes() == original_scope

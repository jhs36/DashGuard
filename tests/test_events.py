from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dashguard.analysis import events as events_module
from dashguard.analysis.events import (
    CandidateRun,
    EventDetectionConfig,
    EventOutputSummary,
    build_event_records,
    build_frame_risk_timeline,
    build_strict_candidate_runs,
    detect_events,
    estimate_frame_interval_sec,
    generate_event_outputs,
    merge_candidate_runs,
)


BASE_COLUMNS = [
    "frame_index",
    "timestamp_sec",
    "track_id",
    "class_name",
    "total_score",
    "telemetry_matched",
    "telemetry_source",
    "hard_brake",
    "original_payload",
]


def _row(
    frame_index: object,
    timestamp_sec: object,
    *,
    track_id: object = 1,
    class_name: object = "car",
    total_score: object = 0.1,
    telemetry_matched: object = True,
    telemetry_source: object = "sensor-a",
    hard_brake: object = False,
    original_payload: object = "kept",
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "track_id": track_id,
        "class_name": class_name,
        "total_score": total_score,
        "telemetry_matched": telemetry_matched,
        "telemetry_source": telemetry_source,
        "hard_brake": hard_brake,
        "original_payload": original_payload,
    }


def _rows(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=BASE_COLUMNS)


def _timeline(
    candidates: list[bool],
    *,
    timestamps: list[float] | None = None,
    tracks: list[int] | None = None,
    classes: list[str] | None = None,
    scores: list[float] | None = None,
    matched: list[object] | None = None,
    sources: list[object] | None = None,
    brakes: list[object] | None = None,
) -> pd.DataFrame:
    size = len(candidates)
    timestamps = timestamps or [index * 0.1 for index in range(size)]
    tracks = tracks or [index + 1 for index in range(size)]
    classes = classes or ["car"] * size
    scores = scores or [0.8 if value else 0.1 for value in candidates]
    matched = matched or [True] * size
    sources = sources or ["sensor-a"] * size
    brakes = brakes or [False] * size
    frame = _rows(
        *[
            _row(
                index,
                timestamps[index],
                track_id=tracks[index],
                class_name=classes[index],
                total_score=scores[index],
                telemetry_matched=matched[index],
                telemetry_source=sources[index],
                hard_brake=brakes[index],
                original_payload=f"row-{index}",
            )
            for index in range(size)
        ]
    )
    frame["source_row_index"] = np.arange(size, dtype=np.int64)
    frame["event_candidate"] = candidates
    return frame


def _metadata(duration_sec: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(duration_sec=duration_sec)


def _write_input_csv(path: Path) -> None:
    _rows(
        _row(0, 0.0, total_score=0.8),
        _row(1, 0.1, total_score=0.8),
    ).to_csv(path, index=False)


def test_representative_timeline_has_one_row_per_frame_and_score_wins() -> None:
    source = _rows(
        _row(1, 0.1, track_id=9, total_score=0.2),
        _row(0, 0.0, track_id=8, total_score=0.4),
        _row(1, 0.1, track_id=7, total_score=0.9),
        _row(0, 0.0, track_id=6, total_score=0.3),
    )

    timeline = build_frame_risk_timeline(source)

    assert timeline["frame_index"].tolist() == [0, 1]
    assert timeline["track_id"].tolist() == [8, 7]


def test_valid_equal_frame_timestamps_continue_to_track_id_tie_break() -> None:
    source = _rows(
        _row(0, 0.25, track_id=8, total_score=0.7),
        _row(0, 0.25, track_id=3, total_score=0.7),
    )

    timeline = build_frame_risk_timeline(source)

    assert timeline.loc[0, "timestamp_sec"] == 0.25
    assert timeline.loc[0, "track_id"] == 3
    assert timeline.loc[0, "source_row_index"] == 1


def test_source_row_breaks_a_complete_representative_tie() -> None:
    source = _rows(
        _row(0, 0.0, track_id=1, total_score=0.8, original_payload="first"),
        _row(0, 0.0, track_id=1, total_score=0.8, original_payload="second"),
    )

    timeline = build_frame_risk_timeline(source)

    assert timeline.loc[0, "original_payload"] == "first"
    assert timeline.loc[0, "source_row_index"] == 0


def test_timeline_does_not_mutate_input_and_appends_columns_in_order() -> None:
    source = _rows(
        _row(1, 0.1, total_score=0.8),
        _row(0, 0.0, total_score=0.2),
    )
    original = source.copy(deep=True)
    original_columns = list(source.columns)

    timeline = build_frame_risk_timeline(source)

    assert_frame_equal(source, original)
    assert list(source.columns) == original_columns
    assert list(timeline.columns) == original_columns + [
        "source_row_index",
        "event_candidate",
    ]


def test_score_threshold_boundary_is_inclusive() -> None:
    source = _rows(
        _row(0, 0.0, total_score=0.7),
        _row(1, 0.1, total_score=np.nextafter(0.7, 0.0)),
    )

    timeline = build_frame_risk_timeline(source, score_threshold=0.7)

    assert timeline["event_candidate"].tolist() == [True, False]


def test_strict_runs_split_nonconsecutive_and_join_consecutive_frames() -> None:
    timeline = _timeline([True, True, False, True, False, True, True])

    runs = build_strict_candidate_runs(timeline)

    assert [(run.start_frame, run.end_frame) for run in runs] == [
        (0, 1),
        (3, 3),
        (5, 6),
    ]
    assert [run.candidate_frame_count for run in runs] == [2, 1, 2]


def test_merge_gap_boundary_is_inclusive_despite_float_error() -> None:
    runs = (
        CandidateRun(0, 0, 0.0, 0.0, 1),
        CandidateRun(2, 2, 0.2 + 1e-15, 0.2 + 1e-15, 1),
    )

    merged = merge_candidate_runs(
        runs,
        frame_interval_sec=0.1,
        merge_gap_sec=0.1,
    )

    assert len(merged) == 1
    assert merged[0].candidate_frame_count == 2


def test_merge_gap_above_boundary_stays_separate() -> None:
    runs = (
        CandidateRun(0, 0, 0.0, 0.0, 1),
        CandidateRun(2, 2, 0.2001, 0.2001, 1),
    )

    merged = merge_candidate_runs(
        runs,
        frame_interval_sec=0.1,
        merge_gap_sec=0.1,
    )

    assert len(merged) == 2


def test_merge_does_not_count_non_candidate_gap_frames() -> None:
    timeline = _timeline([True, False, True])
    strict = build_strict_candidate_runs(timeline)

    merged = merge_candidate_runs(
        strict,
        frame_interval_sec=0.1,
        merge_gap_sec=0.1,
    )

    assert len(merged) == 1
    assert merged[0].candidate_frame_count == 2


def test_minimum_duration_boundary_is_inclusive() -> None:
    timeline = _timeline([True, True])
    events = build_event_records(
        timeline,
        build_strict_candidate_runs(timeline),
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(minimum_event_duration_sec=0.2),
    )

    assert len(events) == 1
    assert events[0]["duration_sec"] == pytest.approx(0.2)


def test_short_run_is_removed() -> None:
    timeline = _timeline([True, False])
    events = build_event_records(
        timeline,
        build_strict_candidate_runs(timeline),
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(minimum_event_duration_sec=0.15),
    )

    assert events == []


def test_minimum_duration_is_applied_after_run_merge() -> None:
    source = _rows(
        _row(0, 0.0, total_score=0.8),
        _row(1, 0.1, total_score=0.1),
        _row(2, 0.2, total_score=0.8),
        _row(3, 0.3, total_score=0.1),
    )
    result = detect_events(
        source,
        video_duration_sec=1.0,
        config=EventDetectionConfig(
            minimum_event_duration_sec=0.25,
            merge_gap_sec=0.1,
        ),
    )

    assert len(result.strict_runs) == 2
    assert len(result.merged_runs) == 1
    assert result.event_document["summary"]["event_count"] == 1
    assert result.event_document["events"][0]["candidate_frame_count"] == 2


def test_frame_interval_is_median_of_positive_adjacent_differences() -> None:
    timeline = _timeline(
        [False, False, False, False],
        timestamps=[0.0, 0.1, 0.4, 0.6],
    )

    assert estimate_frame_interval_sec(timeline) == pytest.approx(0.2)


def test_frame_interval_requires_a_positive_difference() -> None:
    timeline = _timeline([False, False], timestamps=[0.0, 0.0])

    with pytest.raises(ValueError, match="no positive adjacent"):
        estimate_frame_interval_sec(timeline)


def test_primary_track_and_class_come_from_deterministic_maximum_row() -> None:
    timeline = _timeline(
        [True, True, True],
        tracks=[30, 20, 10],
        classes=["truck", "bus", "car"],
        scores=[0.8, 0.95, 0.95],
    )
    run = CandidateRun(0, 2, 0.0, 0.2, 3)

    event = build_event_records(
        timeline,
        [run],
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(minimum_event_duration_sec=0.0),
    )[0]

    assert event["primary_track_id"] == 20
    assert event["primary_class"] == "bus"
    assert event["max_total_score"] == 0.95


@pytest.mark.parametrize(
    ("matched", "brakes", "expected"),
    [
        ([True, True], [False, True], True),
        ([True, False], [False, pd.NA], False),
        ([False, False], [pd.NA, pd.NA], None),
    ],
)
def test_hard_brake_three_state_evidence(
    matched: list[object],
    brakes: list[object],
    expected: bool | None,
) -> None:
    timeline = _timeline([True, True], matched=matched, brakes=brakes)

    event = build_event_records(
        timeline,
        [CandidateRun(0, 1, 0.0, 0.1, 2)],
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(minimum_event_duration_sec=0.0),
    )[0]

    assert event["hard_brake"] is expected
    assert event["telemetry_matched_frame_count"] == sum(bool(x) for x in matched)
    assert event["telemetry_unmatched_frame_count"] == 2 - sum(
        bool(x) for x in matched
    )


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (["sensor-a", "sensor-a", ""], "sensor-a"),
        (["sensor-a", "sensor-b", None], "mixed"),
        ([None, pd.NA, "  "], None),
    ],
)
def test_telemetry_source_summary(
    sources: list[object],
    expected: str | None,
) -> None:
    timeline = _timeline([True, True, True], sources=sources)

    event = build_event_records(
        timeline,
        [CandidateRun(0, 2, 0.0, 0.2, 3)],
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(minimum_event_duration_sec=0.0),
    )[0]

    assert event["telemetry_source"] == expected


def test_event_telemetry_counts_include_non_candidate_merge_gap_frame() -> None:
    timeline = _timeline(
        [True, False, True],
        matched=[True, False, True],
        brakes=[False, pd.NA, False],
    )
    strict = build_strict_candidate_runs(timeline)
    merged = merge_candidate_runs(
        strict,
        frame_interval_sec=0.1,
        merge_gap_sec=0.1,
    )

    event = build_event_records(
        timeline,
        merged,
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(minimum_event_duration_sec=0.0),
    )[0]

    assert event["candidate_frame_count"] == 2
    assert event["telemetry_matched_frame_count"] == 2
    assert event["telemetry_unmatched_frame_count"] == 1


def test_clip_ranges_clamp_at_start_and_end_boundaries() -> None:
    start_timeline = _timeline([True, True])
    start_event = build_event_records(
        start_timeline,
        [CandidateRun(0, 1, 0.0, 0.1, 2)],
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(
            minimum_event_duration_sec=0.0,
            pre_roll_sec=2.0,
            post_roll_sec=0.0,
        ),
    )[0]
    end_timeline = _timeline([True, True], timestamps=[0.8, 0.9])
    end_event = build_event_records(
        end_timeline,
        [CandidateRun(0, 1, 0.8, 0.9, 2)],
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(
            minimum_event_duration_sec=0.0,
            pre_roll_sec=0.0,
            post_roll_sec=2.0,
        ),
    )[0]

    assert start_event["clip_start_sec"] == 0.0
    assert start_event["clip_start_sec"] < start_event["clip_end_sec"]
    assert end_event["end_sec"] == 1.0
    assert end_event["clip_end_sec"] == 1.0
    assert end_event["clip_start_sec"] < end_event["clip_end_sec"]


def test_event_ids_are_assigned_after_duration_filter_in_time_order() -> None:
    timeline = _timeline([True, False, True, True, False, True, True])
    runs = build_strict_candidate_runs(timeline)

    records = build_event_records(
        timeline,
        runs,
        video_duration_sec=1.0,
        frame_interval_sec=0.1,
        config=EventDetectionConfig(minimum_event_duration_sec=0.15),
    )

    assert [event["event_id"] for event in records] == ["event_001", "event_002"]
    assert [event["start_frame"] for event in records] == [2, 5]


def test_event_document_is_strict_json_without_nan_or_infinity() -> None:
    source = _rows(
        _row(0, 0.0, total_score=0.8, telemetry_matched=False, hard_brake=pd.NA),
        _row(1, 0.1, total_score=0.8, telemetry_matched=False, hard_brake=pd.NA),
    )

    result = detect_events(
        source,
        video_duration_sec=1.0,
        config=EventDetectionConfig(minimum_event_duration_sec=0.0),
    )
    serialized = json.dumps(result.event_document, allow_nan=False)

    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert result.event_document["events"][0]["trigger_reasons"] == [
        "total_score_gte_threshold"
    ]


def test_missing_required_columns_are_named_in_error() -> None:
    source = _rows(_row(0, 0.0)).drop(columns=["track_id", "hard_brake"])

    with pytest.raises(ValueError) as exc_info:
        build_frame_risk_timeline(source)

    assert "track_id" in str(exc_info.value)
    assert "hard_brake" in str(exc_info.value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-number", "numeric"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        (-1, "non-negative"),
        (1.5, "integer"),
    ],
)
def test_invalid_frame_index_is_rejected(value: object, message: str) -> None:
    source = _rows(_row(value, 0.0))

    with pytest.raises(ValueError, match=message):
        build_frame_risk_timeline(source)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("timestamp_sec", -0.1, "non-negative"),
        ("timestamp_sec", np.nan, "finite"),
        ("track_id", 1.2, "integer"),
        ("track_id", np.inf, "finite"),
        ("total_score", np.nan, "finite"),
    ],
)
def test_other_required_numeric_values_are_validated(
    column: str,
    value: object,
    message: str,
) -> None:
    data = _row(0, 0.0)
    data[column] = value

    with pytest.raises(ValueError, match=message):
        build_frame_risk_timeline(_rows(data))


def test_different_timestamps_within_one_frame_are_rejected() -> None:
    source = _rows(
        _row(7, 0.1, track_id=1),
        _row(7, 0.2, track_id=2),
    )

    with pytest.raises(ValueError, match="conflicting frame_index values: 7"):
        build_frame_risk_timeline(source)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("score_threshold", np.nan),
        ("score_threshold", np.inf),
        ("minimum_event_duration_sec", -0.1),
        ("merge_gap_sec", -0.1),
        ("pre_roll_sec", -0.1),
        ("post_roll_sec", -0.1),
    ],
)
def test_invalid_config_is_rejected(setting: str, value: float) -> None:
    arguments = {setting: value}

    with pytest.raises(ValueError, match=setting):
        EventDetectionConfig(**arguments)


@pytest.mark.parametrize("duration", [0.0, -1.0, np.nan, np.inf])
def test_invalid_video_duration_is_rejected(duration: float) -> None:
    source = _rows(_row(0, 0.0), _row(1, 0.1))

    with pytest.raises(ValueError, match="video_duration_sec"):
        detect_events(source, video_duration_sec=duration)


def test_existing_outputs_are_refused_by_default(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    input_video = tmp_path / "video.mp4"
    timeline = tmp_path / "timeline.csv"
    event_json = tmp_path / "events.json"
    _write_input_csv(input_csv)
    input_video.touch()
    timeline.write_text("old timeline", encoding="utf-8")
    event_json.write_text("old json", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_event_outputs(
            input_csv=input_csv,
            input_video=input_video,
            output_timeline_csv=timeline,
            output_events_json=event_json,
            metadata_loader=lambda _: _metadata(),
        )

    assert timeline.read_text(encoding="utf-8") == "old timeline"
    assert event_json.read_text(encoding="utf-8") == "old json"


def test_overwrite_atomically_replaces_both_outputs(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    input_video = tmp_path / "video.mp4"
    timeline = tmp_path / "nested" / "timeline.csv"
    event_json = tmp_path / "other" / "events.json"
    _write_input_csv(input_csv)
    input_video.touch()
    timeline.parent.mkdir()
    event_json.parent.mkdir()
    timeline.write_text("old timeline", encoding="utf-8")
    event_json.write_text("old json", encoding="utf-8")

    summary = generate_event_outputs(
        input_csv=input_csv,
        input_video=input_video,
        output_timeline_csv=timeline,
        output_events_json=event_json,
        minimum_event_duration_sec=0.0,
        metadata_loader=lambda _: _metadata(),
        overwrite=True,
    )

    written_timeline = pd.read_csv(timeline)
    written_document = json.loads(event_json.read_text(encoding="utf-8"))
    assert written_timeline.shape == (2, len(BASE_COLUMNS) + 2)
    assert written_document["summary"]["event_count"] == 1
    assert summary.event_ids == ("event_001",)


def test_pair_install_failure_restores_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    input_video = tmp_path / "video.mp4"
    timeline = tmp_path / "timeline.csv"
    event_json = tmp_path / "events.json"
    _write_input_csv(input_csv)
    input_video.touch()
    timeline.write_text("old timeline", encoding="utf-8")
    event_json.write_text("old json", encoding="utf-8")
    original_replace = Path.replace

    def fail_json_install(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if target_path == event_json and not self.name.endswith(".backup"):
            raise OSError("simulated JSON install failure")
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", fail_json_install)

    with pytest.raises(OSError, match="simulated JSON install failure"):
        generate_event_outputs(
            input_csv=input_csv,
            input_video=input_video,
            output_timeline_csv=timeline,
            output_events_json=event_json,
            minimum_event_duration_sec=0.0,
            metadata_loader=lambda _: _metadata(),
            overwrite=True,
        )

    assert timeline.read_text(encoding="utf-8") == "old timeline"
    assert event_json.read_text(encoding="utf-8") == "old json"
    assert not list(tmp_path.glob(".*.backup"))


def test_absolute_temporary_paths_work_without_hardcoded_locations(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "input.csv"
    input_video = tmp_path / "video.mp4"
    timeline = tmp_path / "deep" / "timeline.csv"
    event_json = tmp_path / "different" / "events.json"
    _write_input_csv(input_csv)
    input_video.touch()

    summary = generate_event_outputs(
        input_csv=input_csv.resolve(),
        input_video=input_video.resolve(),
        output_timeline_csv=timeline.resolve(),
        output_events_json=event_json.resolve(),
        minimum_event_duration_sec=0.0,
        metadata_loader=lambda _: _metadata(),
    )

    assert timeline.is_file()
    assert event_json.is_file()
    assert Path(summary.output_timeline_path).is_absolute()
    assert Path(summary.output_json_path).is_absolute()


def test_cli_passes_all_arguments_and_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import day6_detect_events

    captured: dict[str, object] = {}

    def fake_generate(**kwargs: object) -> EventOutputSummary:
        captured.update(kwargs)
        return EventOutputSummary(
            input_rows=2,
            timeline_rows=2,
            candidate_frames=1,
            strict_run_count=1,
            merged_run_count_before_duration_filter=1,
            event_count=1,
            event_ids=("event_001",),
            output_timeline_path=str(kwargs["output_timeline_csv"]),
            output_json_path=str(kwargs["output_events_json"]),
            config=EventDetectionConfig(
                score_threshold=0.75,
                minimum_event_duration_sec=0.2,
                merge_gap_sec=0.12,
                pre_roll_sec=1.5,
                post_roll_sec=0.5,
            ),
            video_duration_sec=30.0,
        )

    monkeypatch.setattr(day6_detect_events, "generate_event_outputs", fake_generate)
    paths = [tmp_path / name for name in ("in.csv", "in.mp4", "out.csv", "out.json")]

    exit_code = day6_detect_events.main(
        [
            "--input-csv",
            str(paths[0]),
            "--input-video",
            str(paths[1]),
            "--output-timeline-csv",
            str(paths[2]),
            "--output-events-json",
            str(paths[3]),
            "--score-threshold",
            "0.75",
            "--minimum-event-duration-sec",
            "0.2",
            "--merge-gap-sec",
            "0.12",
            "--pre-roll-sec",
            "1.5",
            "--post-roll-sec",
            "0.5",
            "--overwrite",
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert captured == {
        "input_csv": paths[0],
        "input_video": paths[1],
        "output_timeline_csv": paths[2],
        "output_events_json": paths[3],
        "score_threshold": 0.75,
        "minimum_event_duration_sec": 0.2,
        "merge_gap_sec": 0.12,
        "pre_roll_sec": 1.5,
        "post_roll_sec": 0.5,
        "overwrite": True,
    }
    assert stdout["event_ids"] == ["event_001"]
    assert stdout["config"] == asdict(EventDetectionConfig(
        score_threshold=0.75,
        minimum_event_duration_sec=0.2,
        merge_gap_sec=0.12,
        pre_roll_sec=1.5,
        post_roll_sec=0.5,
    ))


def test_cli_reports_failure_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import day6_detect_events

    def fail_generate(**_: object) -> EventOutputSummary:
        raise ValueError("bad configuration")

    monkeypatch.setattr(day6_detect_events, "generate_event_outputs", fail_generate)
    exit_code = day6_detect_events.main(
        [
            "--input-csv",
            "in.csv",
            "--input-video",
            "in.mp4",
            "--output-timeline-csv",
            "out.csv",
            "--output-events-json",
            "out.json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ValueError" in captured.err
    assert "bad configuration" in captured.err

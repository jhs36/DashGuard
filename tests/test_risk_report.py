from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from dashguard.analysis.risk_features import FEATURE_COLUMNS
from dashguard.analysis.risk_report import (
    RiskReportSummary,
    generate_video_risk_report,
)
from dashguard.analysis.risk_score import SCORE_COLUMNS
from dashguard.video.metadata import VideoMetadata
from scripts import day5_score_tracks


TRACK_COLUMNS = (
    "frame_index",
    "timestamp_sec",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
)

CONTRIBUTION_COLUMNS = (
    "area_contribution",
    "growth_contribution",
    "center_contribution",
    "roi_contribution",
)


def _metadata(width: int = 100, height: int = 100) -> VideoMetadata:
    return VideoMetadata(
        width=width,
        height=height,
        fps=30.0,
        frame_count=120,
        duration_sec=4.0,
        codec="h264",
    )


def _row(
    frame_index: int,
    timestamp_sec: float,
    track_id: int,
    *,
    box: tuple[float, float, float, float] = (40.0, 30.0, 60.0, 60.0),
    class_name: str = "car",
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "track_id": track_id,
        "class_name": class_name,
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "confidence": confidence,
    }


def _tracks(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _representative_tracks() -> pd.DataFrame:
    return _tracks(
        [
            _row(4, 0.12, 8, box=(35.0, 20.0, 65.0, 60.0)),
            _row(1, 0.03, 8, box=(42.0, 35.0, 58.0, 55.0)),
            _row(2, 0.06, 8, box=(40.0, 32.0, 60.0, 57.0)),
            _row(7, 0.21, 3, box=(10.0, 10.0, 20.0, 20.0)),
            _row(3, 0.09, 8, box=(38.0, 28.0, 62.0, 58.0)),
        ]
    )


def _write_inputs(
    tmp_path: Path,
    tracks: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    input_csv = tmp_path / "tracks.csv"
    input_video = tmp_path / "reference.mp4"
    (tracks if tracks is not None else _representative_tracks()).to_csv(
        input_csv,
        index=False,
    )
    input_video.write_bytes(b"metadata loader injection avoids decoding")
    return input_csv, input_video


def _generate(
    tmp_path: Path,
    tracks: pd.DataFrame | None = None,
    **overrides: object,
) -> tuple[RiskReportSummary, Path, Path]:
    input_csv, input_video = _write_inputs(tmp_path, tracks)
    output_csv = tmp_path / "report" / "risk.csv"
    output_plot = tmp_path / "plot" / "risk.png"
    arguments: dict[str, object] = {
        "input_csv": input_csv,
        "input_video": input_video,
        "output_csv": output_csv,
        "output_plot": output_plot,
        "metadata_loader": lambda _: _metadata(),
    }
    arguments.update(overrides)
    summary = generate_video_risk_report(**arguments)
    return summary, output_csv, output_plot


def _summary() -> RiskReportSummary:
    return RiskReportSummary(
        input_csv_path="입력.csv",
        input_video_path="영상.mp4",
        output_csv_path="출력.csv",
        output_plot_path="그래프.png",
        input_rows=1,
        output_rows=1,
        unique_track_ids=1,
        frame_width=100,
        frame_height=100,
        growth_valid_rows=0,
        max_score=0.5,
        max_score_track_id=7,
        max_score_frame_index=3,
        max_score_timestamp_sec=0.1,
    )


def test_summary_is_frozen_dataclass_with_public_fields() -> None:
    summary = _summary()

    assert is_dataclass(RiskReportSummary)
    assert [field.name for field in fields(RiskReportSummary)] == [
        "input_csv_path",
        "input_video_path",
        "output_csv_path",
        "output_plot_path",
        "input_rows",
        "output_rows",
        "unique_track_ids",
        "frame_width",
        "frame_height",
        "growth_valid_rows",
        "max_score",
        "max_score_track_id",
        "max_score_frame_index",
        "max_score_timestamp_sec",
    ]
    with pytest.raises(FrozenInstanceError):
        summary.input_rows = 2  # type: ignore[misc]


def test_metadata_dimensions_are_used_for_default_feature_config(
    tmp_path: Path,
) -> None:
    tracks = _tracks(
        [_row(0, 0.0, 1, box=(0.0, 0.0, 100.0, 50.0))]
    )
    input_csv, input_video = _write_inputs(tmp_path, tracks)
    loader = Mock(return_value=_metadata(width=200, height=100))
    output_csv = tmp_path / "risk.csv"

    summary = generate_video_risk_report(
        input_csv=input_csv,
        input_video=input_video,
        output_csv=output_csv,
        output_plot=tmp_path / "risk.png",
        metadata_loader=loader,
    )
    result = pd.read_csv(output_csv)

    loader.assert_called_once_with(input_video)
    assert summary.frame_width == 200
    assert summary.frame_height == 100
    assert result.loc[0, "area_ratio"] == pytest.approx(0.25)


def test_missing_input_csv_is_rejected(tmp_path: Path) -> None:
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"video")

    with pytest.raises(FileNotFoundError, match="Input CSV"):
        generate_video_risk_report(
            input_csv=tmp_path / "missing.csv",
            input_video=video,
            output_csv=tmp_path / "out.csv",
            output_plot=tmp_path / "out.png",
            metadata_loader=lambda _: _metadata(),
        )


def test_missing_input_video_is_rejected(tmp_path: Path) -> None:
    input_csv = tmp_path / "tracks.csv"
    _tracks([]).to_csv(input_csv, index=False)

    with pytest.raises(FileNotFoundError, match="Input video"):
        generate_video_risk_report(
            input_csv=input_csv,
            input_video=tmp_path / "missing.mp4",
            output_csv=tmp_path / "out.csv",
            output_plot=tmp_path / "out.png",
            metadata_loader=lambda _: _metadata(),
        )


def test_output_parent_directories_are_created_and_png_is_nonempty(
    tmp_path: Path,
) -> None:
    _, output_csv, output_plot = _generate(tmp_path)

    assert output_csv.is_file()
    assert output_plot.is_file()
    assert output_plot.stat().st_size > 0
    assert output_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_existing_csv_is_rejected_before_loading_metadata(
    tmp_path: Path,
) -> None:
    input_csv, input_video = _write_inputs(tmp_path)
    output_csv = tmp_path / "risk.csv"
    output_csv.write_bytes(b"original csv")
    loader = Mock(side_effect=AssertionError("metadata must not be loaded"))

    with pytest.raises(FileExistsError, match="Output CSV"):
        generate_video_risk_report(
            input_csv=input_csv,
            input_video=input_video,
            output_csv=output_csv,
            output_plot=tmp_path / "risk.png",
            metadata_loader=loader,
        )

    assert output_csv.read_bytes() == b"original csv"
    assert not (tmp_path / "risk.png").exists()
    loader.assert_not_called()


def test_existing_plot_is_rejected_before_loading_metadata(
    tmp_path: Path,
) -> None:
    input_csv, input_video = _write_inputs(tmp_path)
    output_plot = tmp_path / "risk.png"
    output_plot.write_bytes(b"original plot")
    loader = Mock(side_effect=AssertionError("metadata must not be loaded"))

    with pytest.raises(FileExistsError, match="Output plot"):
        generate_video_risk_report(
            input_csv=input_csv,
            input_video=input_video,
            output_csv=tmp_path / "risk.csv",
            output_plot=output_plot,
            metadata_loader=loader,
        )

    assert not (tmp_path / "risk.csv").exists()
    assert output_plot.read_bytes() == b"original plot"
    loader.assert_not_called()


def test_overwrite_refusal_changes_neither_existing_output(
    tmp_path: Path,
) -> None:
    input_csv, input_video = _write_inputs(tmp_path)
    output_csv = tmp_path / "risk.csv"
    output_plot = tmp_path / "risk.png"
    output_csv.write_bytes(b"keep csv")
    output_plot.write_bytes(b"keep plot")

    with pytest.raises(FileExistsError):
        generate_video_risk_report(
            input_csv=input_csv,
            input_video=input_video,
            output_csv=output_csv,
            output_plot=output_plot,
            metadata_loader=lambda _: _metadata(),
        )

    assert output_csv.read_bytes() == b"keep csv"
    assert output_plot.read_bytes() == b"keep plot"


def test_overwrite_true_replaces_both_outputs(tmp_path: Path) -> None:
    input_csv, input_video = _write_inputs(tmp_path)
    output_csv = tmp_path / "risk.csv"
    output_plot = tmp_path / "risk.png"
    output_csv.write_bytes(b"old csv")
    output_plot.write_bytes(b"old plot")

    generate_video_risk_report(
        input_csv=input_csv,
        input_video=input_video,
        output_csv=output_csv,
        output_plot=output_plot,
        metadata_loader=lambda _: _metadata(),
        overwrite=True,
    )

    assert output_csv.read_bytes() != b"old csv"
    assert output_plot.read_bytes() != b"old plot"
    assert output_plot.read_bytes().startswith(b"\x89PNG")


def test_overwrite_generation_failure_changes_neither_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv, input_video = _write_inputs(tmp_path)
    output_csv = tmp_path / "risk.csv"
    output_plot = tmp_path / "risk.png"
    output_csv.write_bytes(b"keep csv")
    output_plot.write_bytes(b"keep plot")
    monkeypatch.setattr(
        "dashguard.analysis.risk_report._write_frame_maximum_plot",
        Mock(side_effect=RuntimeError("plot failed")),
    )

    with pytest.raises(RuntimeError, match="plot failed"):
        generate_video_risk_report(
            input_csv=input_csv,
            input_video=input_video,
            output_csv=output_csv,
            output_plot=output_plot,
            metadata_loader=lambda _: _metadata(),
            overwrite=True,
        )

    assert output_csv.read_bytes() == b"keep csv"
    assert output_plot.read_bytes() == b"keep plot"


def test_rows_order_columns_and_contribution_sum_are_preserved(
    tmp_path: Path,
) -> None:
    tracks = _representative_tracks()
    summary, output_csv, _ = _generate(tmp_path, tracks)
    result = pd.read_csv(output_csv)

    assert summary.input_rows == len(tracks)
    assert summary.output_rows == len(tracks)
    assert len(result) == len(tracks)
    assert result["frame_index"].tolist() == tracks["frame_index"].tolist()
    assert result["track_id"].tolist() == tracks["track_id"].tolist()
    assert list(result.columns) == [
        *TRACK_COLUMNS,
        *FEATURE_COLUMNS,
        *SCORE_COLUMNS,
    ]
    expected_total = result.loc[:, CONTRIBUTION_COLUMNS].sum(axis=1)
    np.testing.assert_allclose(result["total_score"], expected_total)
    assert "threshold" not in result.columns
    assert "telemetry" not in result.columns


def test_empty_csv_writes_full_schema_empty_plot_and_none_maximum(
    tmp_path: Path,
) -> None:
    summary, output_csv, output_plot = _generate(tmp_path, _tracks([]))
    result = pd.read_csv(output_csv)

    assert result.empty
    assert list(result.columns) == [
        *TRACK_COLUMNS,
        *FEATURE_COLUMNS,
        *SCORE_COLUMNS,
    ]
    assert summary.input_rows == 0
    assert summary.output_rows == 0
    assert summary.unique_track_ids == 0
    assert summary.growth_valid_rows == 0
    assert summary.max_score is None
    assert summary.max_score_track_id is None
    assert summary.max_score_frame_index is None
    assert summary.max_score_timestamp_sec is None
    assert output_plot.stat().st_size > 0


def test_missing_required_headers_preserves_feature_schema_error(
    tmp_path: Path,
) -> None:
    tracks = _tracks([]).drop(columns=["track_id", "confidence"])
    input_csv, input_video = _write_inputs(tmp_path, tracks)

    with pytest.raises(ValueError) as error:
        generate_video_risk_report(
            input_csv=input_csv,
            input_video=input_video,
            output_csv=tmp_path / "risk.csv",
            output_plot=tmp_path / "risk.png",
            metadata_loader=lambda _: _metadata(),
        )

    assert "tracks is missing required columns" in str(error.value)
    assert "track_id" in str(error.value)
    assert "confidence" in str(error.value)


def test_maximum_score_tie_breaks_by_time_frame_then_track(
    tmp_path: Path,
) -> None:
    tracks = _tracks(
        [
            _row(1, 1.0, 1),
            _row(9, 0.0, 1),
            _row(3, 0.0, 9),
            _row(3, 0.0, 7),
        ]
    )

    summary, _, _ = _generate(tmp_path, tracks)

    assert summary.max_score_track_id == 7
    assert summary.max_score_frame_index == 3
    assert summary.max_score_timestamp_sec == 0.0


def test_summary_counts_unique_tracks_and_growth_valid_rows(
    tmp_path: Path,
) -> None:
    summary, output_csv, _ = _generate(tmp_path)
    result = pd.read_csv(output_csv)

    assert summary.unique_track_ids == 2
    assert summary.growth_valid_rows == int(result["growth_valid"].sum())
    assert summary.growth_valid_rows == 1


def test_row_count_change_raises_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv, input_video = _write_inputs(tmp_path)

    def filter_one_row(features: pd.DataFrame, **_: object) -> pd.DataFrame:
        result = features.iloc[:-1].copy()
        for column in SCORE_COLUMNS:
            result[column] = 0.0
        return result

    monkeypatch.setattr(
        "dashguard.analysis.risk_report.compute_video_risk_scores",
        filter_one_row,
    )

    with pytest.raises(RuntimeError, match="changed the row count"):
        generate_video_risk_report(
            input_csv=input_csv,
            input_video=input_video,
            output_csv=tmp_path / "risk.csv",
            output_plot=tmp_path / "risk.png",
            metadata_loader=lambda _: _metadata(),
        )

    assert not (tmp_path / "risk.csv").exists()
    assert not (tmp_path / "risk.png").exists()


def test_cli_success_prints_unicode_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generator = Mock(return_value=_summary())
    monkeypatch.setattr(day5_score_tracks, "generate_video_risk_report", generator)

    exit_code = day5_score_tracks.main(
        [
            "--input-csv",
            "tracks.csv",
            "--input-video",
            "video.mp4",
            "--output-csv",
            "risk.csv",
            "--output-plot",
            "risk.png",
            "--overwrite",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["input_csv_path"] == "입력.csv"
    assert payload["max_score_track_id"] == 7
    assert "입력.csv" in captured.out
    assert "  \"input_rows\": 1" in captured.out
    generator.assert_called_once()
    assert generator.call_args.kwargs["overwrite"] is True


def test_cli_missing_required_arguments_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as error:
        day5_score_tracks.main([])

    assert error.value.code != 0

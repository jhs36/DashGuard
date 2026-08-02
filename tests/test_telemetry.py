from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from dashguard.analysis.telemetry import (
    HARD_BRAKE_INTERVAL_SEC,
    MISSING_INTERVAL_SEC,
    SYNC_COLUMNS,
    TELEMETRY_COLUMNS,
    SyntheticTelemetrySummary,
    TelemetrySyncSummary,
    generate_synthetic_telemetry,
    generate_synthetic_telemetry_csv,
    sync_risk_telemetry_csv,
    sync_risk_with_telemetry,
    validate_telemetry,
)
from scripts import (
    day6_generate_synthetic_telemetry,
    day6_sync_telemetry,
)


def _telemetry(
    timestamps: tuple[float, ...] = (0.0, 1.0),
) -> pd.DataFrame:
    rows = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp_sec": timestamps,
            "speed_kmh": [40.0 + index for index in range(rows)],
            "accel_longitudinal": [0.0] * rows,
            "accel_lateral": [0.1] * rows,
            "brake": [0] * rows,
            "telemetry_source": ["synthetic"] * rows,
        },
        columns=TELEMETRY_COLUMNS,
    )


def _risk(timestamps: tuple[float, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": list(range(len(timestamps))),
            "timestamp_sec": timestamps,
            "track_id": [10 + index for index in range(len(timestamps))],
            "total_score": [0.1 * (index + 1) for index in range(len(timestamps))],
        }
    )


def test_synthetic_telemetry_is_deterministic_with_exact_default_grid() -> None:
    first = generate_synthetic_telemetry()
    second = generate_synthetic_telemetry()

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(TELEMETRY_COLUMNS)
    assert len(first) == 296
    assert first["timestamp_sec"].iloc[0] == 0.0
    assert first["timestamp_sec"].iloc[-1] == 30.0
    expected = np.arange(301, dtype=float) / 10.0
    expected = expected[(expected < 12.0) | (expected > 12.4)]
    np.testing.assert_array_equal(first["timestamp_sec"], expected)


def test_synthetic_missing_and_hard_brake_intervals_and_source() -> None:
    telemetry = generate_synthetic_telemetry()
    missing = telemetry["timestamp_sec"].between(
        *MISSING_INTERVAL_SEC, inclusive="both"
    )
    braking = telemetry["timestamp_sec"].between(
        *HARD_BRAKE_INTERVAL_SEC, inclusive="both"
    )

    assert not missing.any()
    assert braking.sum() == 10
    assert telemetry.loc[braking, "brake"].eq(1).all()
    assert telemetry.loc[braking, "accel_longitudinal"].le(-3.0).all()
    assert telemetry.loc[~braking, "brake"].eq(0).all()
    assert telemetry["telemetry_source"].eq("synthetic").all()
    assert np.isfinite(
        telemetry[
            ["speed_kmh", "accel_longitudinal", "accel_lateral"]
        ].to_numpy()
    ).all()
    assert telemetry["speed_kmh"].ge(0.0).all()


def test_validate_telemetry_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="accel_lateral"):
        validate_telemetry(_telemetry().drop(columns="accel_lateral"))


@pytest.mark.parametrize(
    "column,value",
    [
        ("timestamp_sec", np.inf),
        ("speed_kmh", -np.inf),
        ("accel_longitudinal", np.nan),
        ("accel_lateral", np.inf),
    ],
)
def test_validate_telemetry_rejects_non_finite_numeric_values(
    column: str,
    value: float,
) -> None:
    telemetry = _telemetry()
    telemetry.loc[1, column] = value

    with pytest.raises(ValueError, match=column):
        validate_telemetry(telemetry)


def test_validate_telemetry_rejects_negative_duplicate_and_decreasing_time() -> None:
    with pytest.raises(ValueError, match="timestamp_sec.*non-negative"):
        validate_telemetry(_telemetry((-0.1, 0.0)))
    with pytest.raises(ValueError, match="timestamp_sec.*duplicates"):
        validate_telemetry(_telemetry((0.0, 0.0)))
    with pytest.raises(ValueError, match="timestamp_sec.*strictly increasing"):
        validate_telemetry(_telemetry((1.0, 0.5)))


def test_validate_telemetry_rejects_negative_speed_invalid_brake_and_source() -> None:
    negative_speed = _telemetry()
    negative_speed.loc[0, "speed_kmh"] = -0.1
    with pytest.raises(ValueError, match="speed_kmh.*non-negative"):
        validate_telemetry(negative_speed)

    invalid_brake = _telemetry()
    invalid_brake.loc[0, "brake"] = 2
    with pytest.raises(ValueError, match="brake.*bool or 0/1"):
        validate_telemetry(invalid_brake)

    invalid_source = _telemetry()
    invalid_source.loc[0, "telemetry_source"] = " "
    with pytest.raises(ValueError, match="telemetry_source"):
        validate_telemetry(invalid_source)


def test_validate_telemetry_normalizes_supported_brake_encodings() -> None:
    telemetry = _telemetry((0.0, 1.0, 2.0, 3.0))
    telemetry["brake"] = [False, "true", "0", 1.0]

    validated = validate_telemetry(telemetry)

    assert validated["brake"].tolist() == [0, 1, 0, 1]
    assert validated["brake"].dtype == np.int64


@pytest.mark.parametrize(
    "kwargs,setting",
    [
        ({"duration_sec": np.inf}, "duration_sec"),
        ({"duration_sec": 0.0}, "duration_sec"),
        ({"frequency_hz": np.nan}, "frequency_hz"),
        ({"frequency_hz": -1.0}, "frequency_hz"),
    ],
)
def test_synthetic_settings_must_be_positive_and_finite(
    kwargs: dict[str, float],
    setting: str,
) -> None:
    with pytest.raises(ValueError, match=setting):
        generate_synthetic_telemetry(**kwargs)


@pytest.mark.parametrize("offset", [np.nan, np.inf, -np.inf, "invalid"])
def test_sync_rejects_invalid_offset(offset: object) -> None:
    with pytest.raises(ValueError, match="offset_sec"):
        sync_risk_with_telemetry(
            _risk((0.0,)),
            _telemetry(),
            offset_sec=offset,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -0.01, "invalid"])
def test_sync_rejects_invalid_tolerance(tolerance: object) -> None:
    with pytest.raises(ValueError, match="tolerance_sec"):
        sync_risk_with_telemetry(
            _risk((0.0,)),
            _telemetry(),
            tolerance_sec=tolerance,  # type: ignore[arg-type]
        )


def test_offset_moves_telemetry_forward_before_matching() -> None:
    result = sync_risk_with_telemetry(
        _risk((1.5,)),
        _telemetry((1.0,)),
        offset_sec=0.5,
        tolerance_sec=0.0,
    )

    assert bool(result.loc[0, "telemetry_matched"])
    assert result.loc[0, "telemetry_timestamp_sec"] == 1.0
    assert result.loc[0, "telemetry_aligned_timestamp_sec"] == 1.5
    assert result.loc[0, "telemetry_delta_sec"] == 0.0


def test_tolerance_boundary_is_inclusive_and_excess_is_unmatched() -> None:
    result = sync_risk_with_telemetry(
        _risk((0.05, 0.050001)),
        _telemetry((0.0,)),
        tolerance_sec=0.05,
    )

    assert result["telemetry_matched"].tolist() == [True, False]
    assert result.loc[0, "telemetry_delta_sec"] == pytest.approx(0.05)


def test_nearest_tie_explicitly_selects_earlier_aligned_timestamp() -> None:
    telemetry = _telemetry((0.0, 0.2))
    telemetry["speed_kmh"] = [10.0, 20.0]

    result = sync_risk_with_telemetry(
        _risk((0.1,)),
        telemetry,
        tolerance_sec=0.1,
    )

    assert result.loc[0, "telemetry_timestamp_sec"] == 0.0
    assert result.loc[0, "speed_kmh"] == 10.0
    assert result.loc[0, "telemetry_delta_sec"] == pytest.approx(0.1)


def test_unmatched_rows_are_not_forward_or_back_filled() -> None:
    result = sync_risk_with_telemetry(
        _risk((0.0, 1.0, 100.0)),
        _telemetry((0.0,)),
        tolerance_sec=0.1,
    )

    assert result["telemetry_matched"].tolist() == [True, False, False]
    unmatched = result.loc[~result["telemetry_matched"]]
    measurement_columns = [
        "telemetry_timestamp_sec",
        "telemetry_aligned_timestamp_sec",
        "speed_kmh",
        "accel_longitudinal",
        "accel_lateral",
        "brake",
        "telemetry_delta_sec",
    ]
    assert unmatched[measurement_columns].isna().all().all()
    assert unmatched["hard_brake"].isna().all()
    assert str(result["hard_brake"].dtype) == "boolean"
    assert unmatched["telemetry_source"].eq("synthetic").all()


def test_matched_hard_brake_has_true_and_false_evidence() -> None:
    telemetry = _telemetry((0.0, 1.0))
    telemetry["brake"] = [0, 1]
    telemetry["accel_longitudinal"] = [0.0, -3.0]

    result = sync_risk_with_telemetry(
        _risk((0.0, 1.0)),
        telemetry,
        tolerance_sec=0.0,
    )

    assert result["hard_brake"].tolist() == [False, True]


def test_risk_row_order_original_values_and_append_order_are_preserved() -> None:
    risk = pd.DataFrame(
        {
            "frame_index": [9, 1, 5],
            "timestamp_sec": [0.2, 0.0, 0.1],
            "track_id": [4, 9, 4],
            "label": ["last", "first", "middle"],
            "total_score": [0.9, 0.1, 0.5],
        },
        index=[7, 2, 11],
    )
    original = risk.copy(deep=True)

    result = sync_risk_with_telemetry(
        risk,
        _telemetry((0.0, 0.1, 0.2)),
        tolerance_sec=0.0,
    )

    pd.testing.assert_frame_equal(
        result.loc[:, original.columns],
        original,
    )
    pd.testing.assert_frame_equal(risk, original)
    assert list(result.columns) == [*original.columns, *SYNC_COLUMNS]
    assert result.index.tolist() == [7, 2, 11]


def test_output_default_refusal_and_overwrite_success(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "telemetry.csv"
    output.parent.mkdir()
    output.write_bytes(b"keep original")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_synthetic_telemetry_csv(output_csv=output)
    assert output.read_bytes() == b"keep original"

    summary = generate_synthetic_telemetry_csv(
        output_csv=output,
        overwrite=True,
    )
    written = pd.read_csv(output)

    assert output.read_bytes() != b"keep original"
    assert summary.rows == 296
    assert list(written.columns) == list(TELEMETRY_COLUMNS)


def test_output_generation_failure_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "telemetry.csv"
    output.write_bytes(b"must survive")
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        Mock(side_effect=OSError("disk write failed")),
    )

    with pytest.raises(OSError, match="disk write failed"):
        generate_synthetic_telemetry_csv(
            output_csv=output,
            overwrite=True,
        )

    assert output.read_bytes() == b"must survive"
    assert list(tmp_path.iterdir()) == [output]


def test_relative_temporary_paths_support_end_to_end_csv_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    risk = _risk((0.0, 0.03, 12.2, 28.2))
    risk.to_csv("risk.csv", index=False)

    generate_synthetic_telemetry_csv(output_csv="data/telemetry.csv")
    summary = sync_risk_telemetry_csv(
        risk_csv="risk.csv",
        telemetry_csv="data/telemetry.csv",
        output_csv="result/sync.csv",
    )
    result = pd.read_csv("result/sync.csv")

    assert summary.input_risk_rows == 4
    assert summary.output_rows == 4
    assert summary.output_columns == len(risk.columns) + len(SYNC_COLUMNS)
    assert summary.matched_rows == 3
    assert summary.unmatched_rows == 1
    assert summary.hard_brake_matched_rows == 1
    assert result.loc[2, "telemetry_matched"] == np.False_
    pd.testing.assert_series_equal(
        result["total_score"],
        risk["total_score"],
        check_names=True,
    )


def test_sync_csv_exactly_preserves_risk_float_values_and_dtypes(
    tmp_path: Path,
) -> None:
    risk_csv = tmp_path / "risk.csv"
    telemetry_csv = tmp_path / "telemetry.csv"
    output_csv = tmp_path / "sync.csv"
    risk = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "timestamp_sec": [0.0, 1.0],
            "track_id": [7, 7],
            "class_name": ["car", "car"],
            "bbox_area": [247037.72607227735, 198765.43210987654],
            "roi_signal": [0.0, 1.0],
            "total_score": [0.5483780123456789, 0.12345678901234568],
        }
    )
    telemetry = _telemetry((0.0, 1.0))
    telemetry["brake"] = [0, 1]
    telemetry["accel_longitudinal"] = [0.0, -3.5]
    risk.to_csv(risk_csv, index=False)
    telemetry.to_csv(telemetry_csv, index=False)

    original = pd.read_csv(risk_csv)
    sync_risk_telemetry_csv(
        risk_csv=risk_csv,
        telemetry_csv=telemetry_csv,
        output_csv=output_csv,
        tolerance_sec=0.0,
    )
    restored = pd.read_csv(output_csv)

    assert list(restored.columns[: len(original.columns)]) == list(
        original.columns
    )
    pd.testing.assert_frame_equal(
        restored.loc[:, original.columns],
        original,
        check_exact=True,
        check_dtype=True,
    )
    pd.testing.assert_series_equal(
        restored["total_score"],
        original["total_score"],
        check_exact=True,
        check_dtype=True,
    )
    assert list(restored.columns) == [*original.columns, *SYNC_COLUMNS]
    assert restored["telemetry_matched"].tolist() == [True, True]
    assert restored["telemetry_delta_sec"].eq(0.0).all()
    assert restored["hard_brake"].tolist() == [False, True]
    assert restored["telemetry_source"].eq("synthetic").all()


def test_generate_cli_passes_arguments_prints_summary_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = SyntheticTelemetrySummary(
        output_path="telemetry.csv",
        rows=5,
        timestamp_range_sec=(0.0, 1.0),
        frequency_hz=4.0,
        missing_interval_sec=MISSING_INTERVAL_SEC,
        hard_brake_rows=0,
        telemetry_source="synthetic",
    )
    generator = Mock(return_value=summary)
    monkeypatch.setattr(
        day6_generate_synthetic_telemetry,
        "generate_synthetic_telemetry_csv",
        generator,
    )

    exit_code = day6_generate_synthetic_telemetry.main(
        [
            "--output-csv",
            "telemetry.csv",
            "--duration-sec",
            "1",
            "--frequency-hz",
            "4",
            "--overwrite",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["rows"] == 5
    generator.assert_called_once_with(
        output_csv=Path("telemetry.csv"),
        duration_sec=1.0,
        frequency_hz=4.0,
        overwrite=True,
    )

    generator.side_effect = ValueError("bad duration_sec")
    assert day6_generate_synthetic_telemetry.main(
        ["--output-csv", "other.csv"]
    ) == 1
    assert "bad duration_sec" in capsys.readouterr().err


def test_sync_cli_passes_arguments_prints_summary_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = TelemetrySyncSummary(
        risk_csv_path="risk.csv",
        telemetry_csv_path="telemetry.csv",
        output_path="sync.csv",
        input_risk_rows=2,
        output_rows=2,
        output_columns=40,
        matched_rows=1,
        unmatched_rows=1,
        maximum_matched_absolute_delta_sec=0.01,
        hard_brake_matched_rows=1,
        telemetry_source="synthetic",
    )
    synchronizer = Mock(return_value=summary)
    monkeypatch.setattr(
        day6_sync_telemetry,
        "sync_risk_telemetry_csv",
        synchronizer,
    )

    exit_code = day6_sync_telemetry.main(
        [
            "--risk-csv",
            "risk.csv",
            "--telemetry-csv",
            "telemetry.csv",
            "--output-csv",
            "sync.csv",
            "--offset-sec",
            "0.2",
            "--tolerance-sec",
            "0.03",
            "--overwrite",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["matched_rows"] == 1
    synchronizer.assert_called_once_with(
        risk_csv=Path("risk.csv"),
        telemetry_csv=Path("telemetry.csv"),
        output_csv=Path("sync.csv"),
        offset_sec=0.2,
        tolerance_sec=0.03,
        overwrite=True,
    )

    synchronizer.side_effect = FileNotFoundError("missing risk")
    assert day6_sync_telemetry.main(
        [
            "--risk-csv",
            "missing.csv",
            "--telemetry-csv",
            "telemetry.csv",
            "--output-csv",
            "sync.csv",
        ]
    ) == 1
    assert "missing risk" in capsys.readouterr().err

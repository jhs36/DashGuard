from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dashguard.analysis.risk_features import (
    RiskFeatureConfig,
    compute_video_risk_features,
)


REQUIRED_COLUMNS = (
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

FEATURE_COLUMNS = (
    "bbox_valid",
    "bbox_clipped",
    "bbox_area",
    "area_ratio",
    "smoothed_area_ratio",
    "raw_growth_rate_per_sec",
    "growth_rate_per_sec",
    "history_length",
    "growth_valid",
    "growth_status",
    "center_proximity",
    "in_roi",
)


def _row(
    frame_index: int,
    timestamp_sec: float,
    *,
    track_id: int = 1,
    box: tuple[float, float, float, float] = (40.0, 40.0, 60.0, 60.0),
) -> dict[str, object]:
    x1, y1, x2, y2 = box
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "track_id": track_id,
        "class_name": "car",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": 0.9,
    }


def _tracks(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _config(**overrides: object) -> RiskFeatureConfig:
    values: dict[str, object] = {
        "frame_width": 100,
        "frame_height": 100,
    }
    values.update(overrides)
    return RiskFeatureConfig(**values)  # type: ignore[arg-type]


def _area_rows(
    areas: tuple[float, ...],
    *,
    timestamps: tuple[float, ...] | None = None,
    track_id: int = 1,
) -> list[dict[str, object]]:
    if timestamps is None:
        timestamps = tuple(index * 0.1 for index in range(len(areas)))
    return [
        _row(
            index,
            timestamps[index],
            track_id=track_id,
            box=(45.0, 0.0, 55.0, area / 10.0),
        )
        for index, area in enumerate(areas)
    ]


def test_config_is_frozen_dataclass_with_required_defaults() -> None:
    config = _config()

    assert is_dataclass(RiskFeatureConfig)
    assert [field.name for field in fields(RiskFeatureConfig)] == [
        "frame_width",
        "frame_height",
        "smoothing_window_observations",
        "max_gap_sec",
        "max_gap_frames",
        "roi_vertices_normalized",
    ]
    assert config.smoothing_window_observations == 3
    assert config.max_gap_sec == 0.20
    assert config.max_gap_frames == 6
    assert config.roi_vertices_normalized == (
        (0.35, 0.48),
        (0.65, 0.48),
        (0.95, 1.00),
        (0.05, 1.00),
    )
    with pytest.raises(FrozenInstanceError):
        config.frame_width = 200  # type: ignore[misc]


def test_bbox_area_and_area_ratio_use_clipped_pixel_area() -> None:
    result = compute_video_risk_features(
        _tracks([_row(0, 0.0, box=(10.0, 20.0, 30.0, 50.0))]),
        config=_config(),
    )

    assert result.loc[0, "bbox_area"] == pytest.approx(600.0)
    assert result.loc[0, "area_ratio"] == pytest.approx(0.06)


def test_input_dataframe_is_not_modified() -> None:
    tracks = _tracks([_row(0, 0.0), _row(1, 0.1)])
    before = tracks.copy(deep=True)

    compute_video_risk_features(tracks, config=_config())

    assert_frame_equal(tracks, before)


def test_output_preserves_row_count_order_index_and_original_values() -> None:
    tracks = _tracks(
        [
            _row(8, 0.8, track_id=2),
            _row(1, 0.1, track_id=1),
            _row(7, 0.7, track_id=2),
        ]
    )
    tracks.index = pd.Index([9, 3, 9], name="source_row")

    result = compute_video_risk_features(tracks, config=_config())

    assert len(result) == len(tracks)
    assert result.index.equals(tracks.index)
    assert list(result.columns) == [*tracks.columns, *FEATURE_COLUMNS]
    assert_frame_equal(result.loc[:, list(tracks.columns)], tracks)


def test_first_observation_has_no_growth() -> None:
    result = compute_video_risk_features(
        _tracks([_row(0, 0.0)]),
        config=_config(),
    )

    assert np.isnan(result.loc[0, "raw_growth_rate_per_sec"])
    assert np.isnan(result.loc[0, "growth_rate_per_sec"])
    assert not result.loc[0, "growth_valid"]
    assert result.loc[0, "history_length"] == 1
    assert result.loc[0, "growth_status"] == "first_observation"


def test_three_observation_smoothing_is_simple_moving_average() -> None:
    tracks = _tracks(_area_rows((100.0, 200.0, 300.0)))

    result = compute_video_risk_features(tracks, config=_config())

    assert result["smoothed_area_ratio"].iloc[:2].isna().all()
    assert result.loc[2, "smoothed_area_ratio"] == pytest.approx(0.02)
    assert not result.loc[2, "growth_valid"]
    assert result.loc[2, "growth_status"] == "insufficient_history"


def test_fourth_observation_has_valid_smoothed_growth() -> None:
    tracks = _tracks(_area_rows((100.0, 200.0, 300.0, 400.0)))

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[3, "smoothed_area_ratio"] == pytest.approx(0.03)
    assert result.loc[3, "growth_rate_per_sec"] == pytest.approx(5.0)
    assert result.loc[3, "growth_valid"]
    assert result.loc[3, "growth_status"] == "valid"
    assert result.loc[3, "history_length"] == 4


def test_growth_uses_elapsed_seconds_for_irregular_timestamps() -> None:
    tracks = _tracks(
        _area_rows(
            (100.0, 200.0, 300.0, 400.0),
            timestamps=(0.0, 0.05, 0.12, 0.20),
        )
    )

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[1, "raw_growth_rate_per_sec"] == pytest.approx(20.0)
    assert result.loc[3, "growth_rate_per_sec"] == pytest.approx(6.25)


def test_negative_growth_is_preserved() -> None:
    tracks = _tracks(_area_rows((400.0, 300.0, 200.0, 100.0)))

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[3, "raw_growth_rate_per_sec"] == pytest.approx(-5.0)
    assert result.loc[3, "growth_rate_per_sec"] == pytest.approx(-10.0 / 3.0)
    assert result.loc[3, "growth_valid"]


def test_duplicate_timestamp_starts_new_segment() -> None:
    tracks = _tracks([_row(0, 0.0), _row(1, 0.0)])

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[1, "growth_status"] == "nonpositive_delta"
    assert result.loc[1, "history_length"] == 1
    assert np.isnan(result.loc[1, "raw_growth_rate_per_sec"])


def test_timestamp_inversion_is_assigned_to_the_actual_later_frame() -> None:
    tracks = _tracks(
        [
            _row(0, 0.0),
            _row(1, 0.2),
            _row(2, 0.1),
        ]
    )

    result = compute_video_risk_features(tracks, config=_config())

    assert result["growth_status"].tolist() == [
        "first_observation",
        "insufficient_history",
        "nonpositive_delta",
    ]
    assert result["history_length"].tolist() == [1, 2, 1]


def test_shuffled_input_uses_frame_index_for_history_calculation() -> None:
    tracks = _tracks(
        [
            _row(2, 0.2),
            _row(0, 0.0),
            _row(1, 0.1),
        ]
    )
    tracks.index = pd.Index([20, 5, 11], name="input_row")

    result = compute_video_risk_features(tracks, config=_config())

    assert result.index.equals(tracks.index)
    assert result["frame_index"].tolist() == [2, 0, 1]
    assert result["history_length"].tolist() == [3, 1, 2]
    assert result["growth_status"].tolist() == [
        "insufficient_history",
        "first_observation",
        "insufficient_history",
    ]


def test_duplicate_frame_index_starts_new_segment() -> None:
    tracks = _tracks([_row(0, 0.0), _row(0, 0.1)])

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[1, "growth_status"] == "nonpositive_delta"
    assert result.loc[1, "history_length"] == 1
    assert np.isnan(result.loc[1, "raw_growth_rate_per_sec"])


def test_time_gap_over_limit_resets_history() -> None:
    tracks = _tracks([_row(0, 0.0), _row(1, 0.200001)])

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[1, "growth_status"] == "gap_reset"
    assert result.loc[1, "history_length"] == 1
    assert np.isnan(result.loc[1, "raw_growth_rate_per_sec"])


def test_frame_gap_over_limit_resets_history() -> None:
    tracks = _tracks([_row(0, 0.0), _row(7, 0.1)])

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[1, "growth_status"] == "gap_reset"
    assert result.loc[1, "history_length"] == 1


def test_tracks_keep_independent_histories_when_rows_are_interleaved() -> None:
    rows: list[dict[str, object]] = []
    for index, area in enumerate((100.0, 200.0, 300.0, 400.0)):
        rows.append(
            _row(
                index,
                index * 0.1,
                track_id=1,
                box=(45.0, 0.0, 55.0, area / 10.0),
            )
        )
        rows.append(
            _row(
                index,
                index * 0.1,
                track_id=2,
                box=(40.0, 0.0, 60.0, area / 20.0),
            )
        )

    result = compute_video_risk_features(_tracks(rows), config=_config())

    for track_id in (1, 2):
        track_result = result[result["track_id"] == track_id]
        assert track_result["history_length"].tolist() == [1, 2, 3, 4]
        assert track_result["growth_valid"].tolist() == [False, False, False, True]
        assert track_result["growth_rate_per_sec"].iloc[-1] == pytest.approx(5.0)


@pytest.mark.parametrize("observation_count", [1, 2])
def test_short_ghost_tracks_never_have_valid_growth(
    observation_count: int,
) -> None:
    rows = _area_rows(tuple(100.0 * (i + 1) for i in range(observation_count)))

    result = compute_video_risk_features(_tracks(rows), config=_config())

    assert not result["growth_valid"].any()
    assert result["growth_rate_per_sec"].isna().all()


def test_roi_inside_outside_and_boundary_are_classified() -> None:
    tracks = _tracks(
        [
            _row(0, 0.0, track_id=1, box=(45.0, 70.0, 55.0, 80.0)),
            _row(0, 0.0, track_id=2, box=(5.0, 30.0, 15.0, 40.0)),
            _row(0, 0.0, track_id=3, box=(45.0, 40.0, 55.0, 48.0)),
        ]
    )

    result = compute_video_risk_features(tracks, config=_config())

    assert result["in_roi"].tolist() == [True, False, True]


def test_bottom_center_has_maximum_center_proximity() -> None:
    result = compute_video_risk_features(
        _tracks([_row(0, 0.0, box=(45.0, 90.0, 55.0, 100.0))]),
        config=_config(),
    )

    assert result.loc[0, "center_proximity"] == pytest.approx(1.0)


def test_center_proximity_decreases_farther_from_bottom_center() -> None:
    tracks = _tracks(
        [
            _row(0, 0.0, track_id=1, box=(45.0, 80.0, 55.0, 90.0)),
            _row(0, 0.0, track_id=2, box=(0.0, 0.0, 10.0, 10.0)),
        ]
    )

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[0, "center_proximity"] > result.loc[1, "center_proximity"]
    assert result["center_proximity"].between(0.0, 1.0).all()


@pytest.mark.parametrize(
    "box",
    [
        (30.0, 20.0, 30.0, 40.0),
        (40.0, 20.0, 30.0, 40.0),
        (20.0, 30.0, 40.0, 30.0),
        (20.0, 40.0, 40.0, 30.0),
    ],
)
def test_nonpositive_original_bbox_dimensions_are_invalid(
    box: tuple[float, float, float, float],
) -> None:
    result = compute_video_risk_features(
        _tracks([_row(0, 0.0, box=box)]),
        config=_config(),
    )

    assert not result.loc[0, "bbox_valid"]
    assert result.loc[0, "growth_status"] == "invalid_bbox"
    assert result.loc[0, "history_length"] == 0
    assert np.isnan(result.loc[0, "bbox_area"])


def test_negative_coordinates_are_clipped_for_features_only() -> None:
    tracks = _tracks([_row(0, 0.0, box=(-10.0, -20.0, 30.0, 40.0))])

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[0, "bbox_valid"]
    assert result.loc[0, "bbox_clipped"]
    assert result.loc[0, "bbox_area"] == pytest.approx(1200.0)
    assert result.loc[0, "x1"] == -10.0
    assert result.loc[0, "y1"] == -20.0


def test_coordinates_beyond_frame_are_clipped() -> None:
    result = compute_video_risk_features(
        _tracks([_row(0, 0.0, box=(80.0, 70.0, 120.0, 130.0))]),
        config=_config(),
    )

    assert result.loc[0, "bbox_valid"]
    assert result.loc[0, "bbox_clipped"]
    assert result.loc[0, "bbox_area"] == pytest.approx(600.0)


def test_bbox_with_zero_area_after_clipping_is_invalid() -> None:
    result = compute_video_risk_features(
        _tracks([_row(0, 0.0, box=(-10.0, 10.0, 0.0, 20.0))]),
        config=_config(),
    )

    assert not result.loc[0, "bbox_valid"]
    assert result.loc[0, "bbox_clipped"]
    assert np.isnan(result.loc[0, "area_ratio"])
    assert np.isnan(result.loc[0, "center_proximity"])
    assert not result.loc[0, "in_roi"]


@pytest.mark.parametrize("bad_coordinate", [np.nan, np.inf, -np.inf])
def test_nonfinite_bbox_coordinates_are_invalid(bad_coordinate: float) -> None:
    result = compute_video_risk_features(
        _tracks([_row(0, 0.0, box=(bad_coordinate, 10.0, 30.0, 40.0))]),
        config=_config(),
    )

    assert not result.loc[0, "bbox_valid"]
    assert not result.loc[0, "bbox_clipped"]
    assert np.isnan(result.loc[0, "bbox_area"])
    assert result.loc[0, "growth_status"] == "invalid_bbox"


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_width": 0},
        {"frame_width": -1},
        {"frame_height": 0},
        {"frame_height": -1},
    ],
)
def test_frame_dimensions_must_be_positive(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="frame_(width|height)"):
        _config(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"smoothing_window_observations": 1}, "smoothing_window"),
        ({"max_gap_sec": 0.0}, "max_gap_sec"),
        ({"max_gap_sec": np.inf}, "max_gap_sec"),
        ({"max_gap_frames": 0}, "max_gap_frames"),
        ({"roi_vertices_normalized": ((0.0, 0.0),) * 3}, "four points"),
        (
            {
                "roi_vertices_normalized": (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                    (0.0, 1.0),
                )
            },
            "two coordinates",
        ),
        (
            {
                "roi_vertices_normalized": (
                    (-0.1, 0.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                    (0.0, 1.0),
                )
            },
            "within",
        ),
        (
            {
                "roi_vertices_normalized": (
                    (np.nan, 0.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                    (0.0, 1.0),
                )
            },
            "finite",
        ),
    ],
)
def test_invalid_smoothing_gap_and_roi_settings_are_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_missing_required_schema_lists_every_missing_column() -> None:
    tracks = _tracks([_row(0, 0.0)]).drop(columns=["x1", "confidence"])

    with pytest.raises(ValueError) as error:
        compute_video_risk_features(tracks, config=_config())

    assert "x1" in str(error.value)
    assert "confidence" in str(error.value)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1, 1.5])
def test_invalid_frame_index_is_rejected(bad_value: float) -> None:
    tracks = _tracks([_row(0, 0.0)])
    tracks["frame_index"] = pd.Series([bad_value], dtype=object)

    with pytest.raises(ValueError, match=r"frame_index has 1 invalid row"):
        compute_video_risk_features(tracks, config=_config())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -0.1])
def test_invalid_timestamp_is_rejected(bad_value: float) -> None:
    tracks = _tracks([_row(0, 0.0)])
    tracks.loc[0, "timestamp_sec"] = bad_value

    with pytest.raises(ValueError, match=r"timestamp_sec has 1 invalid row"):
        compute_video_risk_features(tracks, config=_config())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, 1.5])
def test_invalid_track_id_is_rejected(bad_value: float) -> None:
    tracks = _tracks([_row(0, 0.0)])
    tracks["track_id"] = pd.Series([bad_value], dtype=object)

    with pytest.raises(ValueError, match=r"track_id has 1 invalid row"):
        compute_video_risk_features(tracks, config=_config())


@pytest.mark.parametrize("column", ["frame_index", "track_id"])
def test_bool_is_not_an_integer_identity_value(column: str) -> None:
    tracks = _tracks([_row(0, 0.0)])
    tracks[column] = pd.Series([True], dtype=object)

    with pytest.raises(ValueError, match=rf"{column} has 1 invalid row"):
        compute_video_risk_features(tracks, config=_config())


def test_missing_track_ids_are_rejected_instead_of_linked() -> None:
    tracks = _tracks([_row(0, 0.0), _row(1, 0.1)])
    tracks["track_id"] = np.nan

    with pytest.raises(ValueError, match=r"track_id has 2 invalid row"):
        compute_video_risk_features(tracks, config=_config())


def test_validation_reports_all_bad_columns_and_invalid_row_counts() -> None:
    tracks = _tracks([_row(0, 0.0), _row(1, 0.1)])
    tracks["frame_index"] = [-1, 1]
    tracks["timestamp_sec"] = [np.nan, np.inf]
    tracks["track_id"] = [1.0, 1.5]

    with pytest.raises(ValueError) as error:
        compute_video_risk_features(tracks, config=_config())

    message = str(error.value)
    assert "frame_index has 1 invalid row" in message
    assert "timestamp_sec has 2 invalid row" in message
    assert "track_id has 1 invalid row" in message


def test_numeric_strings_are_allowed_without_changing_original_columns() -> None:
    tracks = _tracks([_row(0, 0.0), _row(1, 0.1)])
    tracks["frame_index"] = ["0", "1"]
    tracks["timestamp_sec"] = ["0.0", "0.1"]
    tracks["track_id"] = ["2", "2"]
    before = tracks.copy(deep=True)

    result = compute_video_risk_features(tracks, config=_config())

    assert_frame_equal(result.loc[:, list(tracks.columns)], before)
    assert result["history_length"].tolist() == [1, 2]


def test_valid_observation_after_invalid_bbox_starts_fresh_history() -> None:
    tracks = _tracks(
        [
            _row(0, 0.0),
            _row(1, 0.1, box=(30.0, 20.0, 30.0, 40.0)),
            _row(2, 0.2),
        ]
    )

    result = compute_video_risk_features(tracks, config=_config())

    assert result.loc[2, "growth_status"] == "first_observation"
    assert result.loc[2, "history_length"] == 1
    assert np.isnan(result.loc[2, "raw_growth_rate_per_sec"])
    assert np.isnan(result.loc[2, "growth_rate_per_sec"])
    assert not result.loc[2, "growth_valid"]


def test_empty_dataframe_returns_original_and_typed_feature_columns() -> None:
    tracks = _tracks([])

    result = compute_video_risk_features(tracks, config=_config())

    assert result.empty
    assert list(result.columns) == [*REQUIRED_COLUMNS, *FEATURE_COLUMNS]
    assert result["bbox_valid"].dtype == bool
    assert result["history_length"].dtype == "int64"


def test_feature_values_never_contain_infinity() -> None:
    tracks = _tracks(
        _area_rows(
            (100.0, 200.0, 300.0, 400.0),
            timestamps=(0.0, 1e-320, 2e-320, 3e-320),
        )
        + [_row(10, 1.0, track_id=2, box=(np.inf, 0.0, 10.0, 10.0))]
    )

    result = compute_video_risk_features(tracks, config=_config())
    numeric_features = result.loc[
        :,
        [
            "bbox_area",
            "area_ratio",
            "smoothed_area_ratio",
            "raw_growth_rate_per_sec",
            "growth_rate_per_sec",
            "center_proximity",
        ],
    ].to_numpy(dtype=float)

    assert not np.isinf(numeric_features).any()

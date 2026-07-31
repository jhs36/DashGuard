from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dashguard.analysis.risk_score import (
    RiskScoreConfig,
    compute_video_risk_scores,
)


REQUIRED_COLUMNS = (
    "bbox_valid",
    "area_ratio",
    "growth_rate_per_sec",
    "growth_valid",
    "center_proximity",
    "in_roi",
)

SCORE_COLUMNS = (
    "area_signal",
    "growth_signal",
    "center_signal",
    "roi_signal",
    "area_contribution",
    "growth_contribution",
    "center_contribution",
    "roi_contribution",
    "total_score",
)

SIGNAL_COLUMNS = (
    "area_signal",
    "growth_signal",
    "center_signal",
    "roi_signal",
)

CONTRIBUTION_COLUMNS = (
    "area_contribution",
    "growth_contribution",
    "center_contribution",
    "roi_contribution",
)


def _row(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "bbox_valid": True,
        "area_ratio": 0.05,
        "growth_rate_per_sec": np.nan,
        "growth_valid": False,
        "center_proximity": 0.5,
        "in_roi": False,
    }
    values.update(overrides)
    return values


def _features(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_config_is_frozen_dataclass_with_required_defaults() -> None:
    config = RiskScoreConfig()

    assert is_dataclass(RiskScoreConfig)
    assert [field.name for field in fields(RiskScoreConfig)] == [
        "area_full_scale",
        "growth_full_scale_per_sec",
        "area_weight",
        "growth_weight",
        "center_weight",
        "roi_weight",
    ]
    assert config.area_full_scale == 0.10
    assert config.growth_full_scale_per_sec == 3.0
    assert config.area_weight == 0.15
    assert config.growth_weight == 0.45
    assert config.center_weight == 0.15
    assert config.roi_weight == 0.25
    assert sum(
        (
            config.area_weight,
            config.growth_weight,
            config.center_weight,
            config.roi_weight,
        )
    ) == 1.0
    with pytest.raises(FrozenInstanceError):
        config.area_weight = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    ["area_full_scale", "growth_full_scale_per_sec"],
)
@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_full_scale_is_rejected(
    field_name: str,
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        RiskScoreConfig(**{field_name: bad_value})


@pytest.mark.parametrize(
    "field_name",
    ["area_weight", "growth_weight", "center_weight", "roi_weight"],
)
@pytest.mark.parametrize("bad_value", [-0.01, np.nan, np.inf, -np.inf])
def test_invalid_weight_is_rejected(
    field_name: str,
    bad_value: float,
) -> None:
    weights = {
        "area_weight": 0.0,
        "growth_weight": 0.0,
        "center_weight": 0.0,
        "roi_weight": 0.0,
    }
    weights[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        RiskScoreConfig(**weights)


def test_weight_sum_over_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight sum"):
        RiskScoreConfig(area_weight=0.150001)


def test_tiny_floating_point_weight_sum_artifact_is_tolerated() -> None:
    config = RiskScoreConfig(
        area_weight=0.1,
        growth_weight=0.2,
        center_weight=0.3,
        roi_weight=0.4000000000000001,
    )

    assert sum(
        (
            config.area_weight,
            config.growth_weight,
            config.center_weight,
            config.roi_weight,
        )
    ) > 1.0


def test_input_dataframe_is_not_modified() -> None:
    features = _features([_row(), _row(area_ratio=0.08)])
    before = features.copy(deep=True)

    compute_video_risk_scores(features)

    assert_frame_equal(features, before)


def test_output_preserves_rows_index_original_columns_values_and_order() -> None:
    features = _features(
        [
            _row(area_ratio=0.02),
            _row(
                growth_rate_per_sec=1.0,
                growth_valid=True,
                in_roi=True,
            ),
            _row(area_ratio=0.08),
        ]
    )
    features.insert(0, "track_id", [9, 3, 9])
    features["note"] = ["first", "second", "third"]
    features.index = pd.Index([12, 4, 12], name="source_row")

    result = compute_video_risk_scores(features)

    assert len(result) == len(features)
    assert result.index.equals(features.index)
    assert list(result.columns) == [*features.columns, *SCORE_COLUMNS]
    assert_frame_equal(result.loc[:, list(features.columns)], features)


@pytest.mark.parametrize(
    ("area_ratio", "expected_signal"),
    [(0.0, 0.0), (0.05, 0.5), (0.10, 1.0), (0.75, 1.0)],
)
def test_area_signal_has_zero_middle_and_saturation(
    area_ratio: float,
    expected_signal: float,
) -> None:
    result = compute_video_risk_scores(
        _features([_row(area_ratio=area_ratio)])
    )

    assert result.loc[0, "area_signal"] == pytest.approx(expected_signal)
    assert result.loc[0, "area_contribution"] == pytest.approx(
        0.15 * expected_signal
    )


@pytest.mark.parametrize(
    ("growth_rate", "expected_signal"),
    [(0.0, 0.0), (1.5, 0.5), (3.0, 1.0), (12.0, 1.0)],
)
def test_positive_growth_signal_has_zero_middle_and_saturation(
    growth_rate: float,
    expected_signal: float,
) -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(
                    growth_rate_per_sec=growth_rate,
                    growth_valid=True,
                )
            ]
        )
    )

    assert result.loc[0, "growth_signal"] == pytest.approx(expected_signal)
    assert result.loc[0, "growth_contribution"] == pytest.approx(
        0.45 * expected_signal
    )


def test_negative_growth_has_zero_signal_and_contribution() -> None:
    result = compute_video_risk_scores(
        _features(
            [_row(growth_rate_per_sec=-8.0, growth_valid=True)]
        )
    )

    assert result.loc[0, "growth_signal"] == 0.0
    assert result.loc[0, "growth_contribution"] == 0.0


def test_missing_growth_has_zero_signal_and_contribution() -> None:
    result = compute_video_risk_scores(_features([_row()]))

    assert result.loc[0, "growth_signal"] == 0.0
    assert result.loc[0, "growth_contribution"] == 0.0


@pytest.mark.parametrize(
    ("center_proximity", "expected"),
    [(0.0, 0.0), (0.4, 0.4), (1.0, 1.0)],
)
def test_center_signal_and_contribution(
    center_proximity: float,
    expected: float,
) -> None:
    result = compute_video_risk_scores(
        _features([_row(center_proximity=center_proximity)])
    )

    assert result.loc[0, "center_signal"] == pytest.approx(expected)
    assert result.loc[0, "center_contribution"] == pytest.approx(
        0.15 * expected
    )


def test_roi_inside_adds_exact_roi_weight() -> None:
    result = compute_video_risk_scores(
        _features([_row(in_roi=False), _row(in_roi=True)])
    )

    assert result.loc[0, "roi_signal"] == 0.0
    assert result.loc[1, "roi_signal"] == 1.0
    assert result.loc[1, "roi_contribution"] == pytest.approx(0.25)
    assert result.loc[1, "total_score"] - result.loc[
        0, "total_score"
    ] == pytest.approx(0.25)


def test_invalid_bbox_zeroes_every_signal_and_contribution() -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(
                    bbox_valid=False,
                    area_ratio=np.nan,
                    growth_rate_per_sec=np.nan,
                    growth_valid=False,
                    center_proximity=np.nan,
                    in_roi=False,
                )
            ]
        )
    )

    assert (result.loc[0, list(SIGNAL_COLUMNS)] == 0.0).all()
    assert (result.loc[0, list(CONTRIBUTION_COLUMNS)] == 0.0).all()
    assert result.loc[0, "total_score"] == 0.0


def test_growing_centered_roi_object_increases_in_score() -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(
                    growth_rate_per_sec=0.0,
                    growth_valid=True,
                    center_proximity=1.0,
                    in_roi=True,
                ),
                _row(
                    growth_rate_per_sec=2.0,
                    growth_valid=True,
                    center_proximity=1.0,
                    in_roi=True,
                ),
            ]
        )
    )

    assert result.loc[1, "total_score"] > result.loc[0, "total_score"]


def test_shrinking_object_scores_lower_than_otherwise_equal_growing_object() -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(growth_rate_per_sec=-1.0, growth_valid=True, in_roi=True),
                _row(growth_rate_per_sec=1.0, growth_valid=True, in_roi=True),
            ]
        )
    )

    assert result.loc[0, "total_score"] < result.loc[1, "total_score"]


def test_object_farther_from_center_scores_lower() -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(center_proximity=0.9),
                _row(center_proximity=0.2),
            ]
        )
    )

    assert result.loc[1, "total_score"] < result.loc[0, "total_score"]


def test_object_outside_roi_scores_lower() -> None:
    result = compute_video_risk_scores(
        _features([_row(in_roi=True), _row(in_roi=False)])
    )

    assert result.loc[1, "total_score"] < result.loc[0, "total_score"]


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        (
            _row(area_ratio=0.01),
            _row(area_ratio=0.09),
        ),
        (
            _row(growth_rate_per_sec=0.2, growth_valid=True),
            _row(growth_rate_per_sec=2.0, growth_valid=True),
        ),
        (
            _row(center_proximity=0.1),
            _row(center_proximity=0.9),
        ),
        (
            _row(in_roi=False),
            _row(in_roi=True),
        ),
    ],
)
def test_total_is_monotonic_when_one_feature_increases(
    lower: dict[str, object],
    higher: dict[str, object],
) -> None:
    result = compute_video_risk_scores(_features([lower, higher]))

    assert result.loc[1, "total_score"] >= result.loc[0, "total_score"]


def test_score_ranges_and_total_equals_direct_contribution_sum() -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(area_ratio=0.0, center_proximity=0.0),
                _row(
                    area_ratio=0.04,
                    growth_rate_per_sec=-1.0,
                    growth_valid=True,
                    center_proximity=0.2,
                ),
                _row(
                    area_ratio=0.1,
                    growth_rate_per_sec=3.0,
                    growth_valid=True,
                    center_proximity=1.0,
                    in_roi=True,
                ),
                _row(
                    area_ratio=1.0,
                    growth_rate_per_sec=100.0,
                    growth_valid=True,
                    center_proximity=0.7,
                ),
            ]
        )
    )

    assert result.loc[:, SIGNAL_COLUMNS].ge(0.0).all().all()
    assert result.loc[:, SIGNAL_COLUMNS].le(1.0).all().all()
    assert result.loc[:, CONTRIBUTION_COLUMNS].ge(0.0).all().all()
    assert result["total_score"].between(0.0, 1.0).all()
    expected_total = result.loc[:, CONTRIBUTION_COLUMNS].sum(axis=1)
    np.testing.assert_allclose(result["total_score"], expected_total, atol=1e-15)


def test_custom_weight_sum_below_one_is_not_post_clamped() -> None:
    config = RiskScoreConfig(
        area_weight=0.1,
        growth_weight=0.3,
        center_weight=0.1,
        roi_weight=0.3,
    )
    result = compute_video_risk_scores(
        _features(
            [
                _row(
                    area_ratio=1.0,
                    growth_rate_per_sec=10.0,
                    growth_valid=True,
                    center_proximity=1.0,
                    in_roi=True,
                )
            ]
        ),
        config=config,
    )

    assert result.loc[0, "total_score"] == pytest.approx(0.8)
    assert result.loc[0, "total_score"] != 1.0


def test_missing_growth_does_not_redistribute_growth_weight() -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(
                    area_ratio=0.1,
                    growth_rate_per_sec=np.nan,
                    growth_valid=False,
                    center_proximity=1.0,
                    in_roi=True,
                )
            ]
        )
    )

    assert result.loc[0, "growth_contribution"] == 0.0
    assert result.loc[0, "total_score"] == pytest.approx(0.55)


@pytest.mark.parametrize("column", ["bbox_valid", "growth_valid", "in_roi"])
@pytest.mark.parametrize("bad_value", ["True", 0, 1, np.nan])
def test_boolean_contract_rejects_non_boolean_values(
    column: str,
    bad_value: object,
) -> None:
    features = _features([_row()])
    features[column] = pd.Series([bad_value], dtype=object)

    with pytest.raises(ValueError) as error:
        compute_video_risk_scores(features)

    assert f"{column} has 1 invalid row" in str(error.value)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("area_ratio", np.nan),
        ("area_ratio", np.inf),
        ("area_ratio", -np.inf),
        ("area_ratio", -0.01),
        ("area_ratio", 1.01),
        ("center_proximity", np.nan),
        ("center_proximity", np.inf),
        ("center_proximity", -np.inf),
        ("center_proximity", -0.01),
        ("center_proximity", 1.01),
    ],
)
def test_valid_bbox_requires_finite_bounded_area_and_center(
    column: str,
    bad_value: float,
) -> None:
    features = _features([_row()])
    features.loc[0, column] = bad_value

    with pytest.raises(ValueError) as error:
        compute_video_risk_scores(features)

    assert f"{column} has 1 invalid row" in str(error.value)


@pytest.mark.parametrize("flag_column", ["in_roi", "growth_valid"])
def test_invalid_bbox_rejects_true_roi_or_growth_flag(
    flag_column: str,
) -> None:
    values = _row(
        bbox_valid=False,
        area_ratio=np.nan,
        growth_rate_per_sec=np.nan,
        growth_valid=False,
        center_proximity=np.nan,
        in_roi=False,
    )
    values[flag_column] = True

    with pytest.raises(ValueError) as error:
        compute_video_risk_scores(_features([values]))

    assert f"{flag_column} has 1 invalid row" in str(error.value)


@pytest.mark.parametrize("bad_growth", [np.nan, np.inf, -np.inf])
def test_valid_growth_requires_finite_growth_rate(bad_growth: float) -> None:
    features = _features(
        [_row(growth_rate_per_sec=bad_growth, growth_valid=True)]
    )

    with pytest.raises(ValueError, match=r"growth_rate_per_sec has 1 invalid row"):
        compute_video_risk_scores(features)


@pytest.mark.parametrize("finite_growth", [-1.0, 0.0, 2.0])
def test_missing_growth_rejects_finite_growth_rate(
    finite_growth: float,
) -> None:
    features = _features(
        [_row(growth_rate_per_sec=finite_growth, growth_valid=False)]
    )

    with pytest.raises(ValueError, match=r"growth_rate_per_sec has 1 invalid row"):
        compute_video_risk_scores(features)


def test_validation_reports_all_bad_columns_and_row_counts() -> None:
    features = _features(
        [
            _row(
                area_ratio=np.nan,
                growth_rate_per_sec=np.nan,
                growth_valid=True,
                center_proximity=np.inf,
            ),
            _row(
                bbox_valid=False,
                area_ratio=np.nan,
                growth_rate_per_sec=np.nan,
                growth_valid=True,
                center_proximity=np.nan,
                in_roi=True,
            ),
        ]
    )

    with pytest.raises(ValueError) as error:
        compute_video_risk_scores(features)

    message = str(error.value)
    assert "area_ratio has 1 invalid row" in message
    assert "center_proximity has 1 invalid row" in message
    assert "growth_rate_per_sec has 2 invalid row" in message
    assert "growth_valid has 1 invalid row" in message
    assert "in_roi has 1 invalid row" in message


def test_missing_required_columns_lists_every_missing_column() -> None:
    features = _features([_row()]).drop(
        columns=["area_ratio", "center_proximity"]
    )

    with pytest.raises(ValueError) as error:
        compute_video_risk_scores(features)

    assert "area_ratio" in str(error.value)
    assert "center_proximity" in str(error.value)


def test_empty_dataframe_preserves_columns_and_adds_float_scores() -> None:
    features = _features([])

    result = compute_video_risk_scores(features)

    assert result.empty
    assert list(result.columns) == [*REQUIRED_COLUMNS, *SCORE_COLUMNS]
    for column in SCORE_COLUMNS:
        assert result[column].dtype == float


def test_result_scores_never_contain_nan_or_infinity() -> None:
    result = compute_video_risk_scores(
        _features(
            [
                _row(),
                _row(
                    growth_rate_per_sec=-100.0,
                    growth_valid=True,
                    center_proximity=0.0,
                ),
                _row(
                    area_ratio=1.0,
                    growth_rate_per_sec=100.0,
                    growth_valid=True,
                    center_proximity=1.0,
                    in_roi=True,
                ),
                _row(
                    bbox_valid=False,
                    area_ratio=np.nan,
                    growth_rate_per_sec=np.nan,
                    growth_valid=False,
                    center_proximity=np.nan,
                    in_roi=False,
                ),
            ]
        )
    )

    score_values = result.loc[:, SCORE_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(score_values).all()

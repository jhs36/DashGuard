"""Pure rule-based risk scoring for per-object video features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "bbox_valid",
    "area_ratio",
    "growth_rate_per_sec",
    "growth_valid",
    "center_proximity",
    "in_roi",
)

SCORE_COLUMNS: Final[tuple[str, ...]] = (
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

_WEIGHT_SUM_REL_TOLERANCE: Final[float] = 1e-12
_WEIGHT_SUM_ABS_TOLERANCE: Final[float] = 1e-12


@dataclass(frozen=True)
class RiskScoreConfig:
    """Full-scale signal values and additive contribution weights."""

    area_full_scale: float = 0.10
    growth_full_scale_per_sec: float = 3.0
    area_weight: float = 0.15
    growth_weight: float = 0.45
    center_weight: float = 0.15
    roi_weight: float = 0.25

    def __post_init__(self) -> None:
        _validate_positive_finite("area_full_scale", self.area_full_scale)
        _validate_positive_finite(
            "growth_full_scale_per_sec",
            self.growth_full_scale_per_sec,
        )

        weights = (
            ("area_weight", self.area_weight),
            ("growth_weight", self.growth_weight),
            ("center_weight", self.center_weight),
            ("roi_weight", self.roi_weight),
        )
        for name, value in weights:
            _validate_nonnegative_finite(name, value)

        active_weight_sum = math.fsum(
            float(value) for _, value in weights if float(value) > 0.0
        )
        if active_weight_sum > 1.0 and not math.isclose(
            active_weight_sum,
            1.0,
            rel_tol=_WEIGHT_SUM_REL_TOLERANCE,
            abs_tol=_WEIGHT_SUM_ABS_TOLERANCE,
        ):
            raise ValueError(
                "active weight sum must be less than or equal to 1.0"
            )


def _validate_positive_finite(name: str, value: object) -> None:
    try:
        converted = float(value) if isinstance(value, Real) else math.nan
    except (OverflowError, TypeError, ValueError):
        converted = math.nan
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(converted) or (
        converted <= 0.0
    ):
        raise ValueError(f"{name} must be finite and greater than zero")


def _validate_nonnegative_finite(name: str, value: object) -> None:
    try:
        converted = float(value) if isinstance(value, Real) else math.nan
    except (OverflowError, TypeError, ValueError):
        converted = math.nan
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(converted) or (
        converted < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")


def _boolean_values(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return strict boolean values and a mask identifying boolean cells."""

    values = np.zeros(len(series), dtype=bool)
    valid = np.zeros(len(series), dtype=bool)
    for position, value in enumerate(series.array):
        if isinstance(value, (bool, np.bool_)):
            values[position] = bool(value)
            valid[position] = True
    return values, valid


def _numeric_values(
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return floats plus masks for real numeric and missing cells."""

    values = np.full(len(series), np.nan, dtype=float)
    numeric = np.zeros(len(series), dtype=bool)
    missing = np.zeros(len(series), dtype=bool)
    for position, value in enumerate(series.array):
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, Real):
            try:
                converted = float(value)
            except (OverflowError, TypeError, ValueError):
                continue
            numeric[position] = True
            values[position] = converted
            missing[position] = math.isnan(values[position])
            continue
        try:
            missing[position] = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing[position] = False
    return values, numeric, missing


def _raise_invalid_features(invalid_masks: dict[str, np.ndarray]) -> None:
    invalid_counts = {
        column: int(mask.sum())
        for column, mask in invalid_masks.items()
        if bool(mask.any())
    }
    if not invalid_counts:
        return

    details = "; ".join(
        f"{column} has {count} invalid row(s)"
        for column, count in invalid_counts.items()
    )
    raise ValueError(f"invalid risk feature values: {details}")


def compute_video_risk_scores(
    features: pd.DataFrame,
    *,
    config: RiskScoreConfig = RiskScoreConfig(),
) -> pd.DataFrame:
    """Append normalized risk signals, contributions, and their direct sum."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            "features is missing required columns: " + ", ".join(missing_columns)
        )

    bbox_valid, bbox_is_bool = _boolean_values(features["bbox_valid"])
    growth_valid, growth_is_bool = _boolean_values(features["growth_valid"])
    in_roi, roi_is_bool = _boolean_values(features["in_roi"])

    area_ratio, area_is_numeric, _ = _numeric_values(features["area_ratio"])
    growth_rate, growth_is_numeric, growth_is_missing = _numeric_values(
        features["growth_rate_per_sec"]
    )
    center_proximity, center_is_numeric, _ = _numeric_values(
        features["center_proximity"]
    )

    valid_bbox_rows = bbox_is_bool & bbox_valid
    invalid_bbox_rows = bbox_is_bool & ~bbox_valid
    valid_growth_rows = growth_is_bool & growth_valid
    missing_growth_rows = growth_is_bool & ~growth_valid

    invalid_masks = {
        "bbox_valid": ~bbox_is_bool,
        "growth_valid": (~growth_is_bool)
        | (invalid_bbox_rows & growth_valid),
        "in_roi": (~roi_is_bool) | (invalid_bbox_rows & in_roi),
        "area_ratio": valid_bbox_rows
        & (
            ~area_is_numeric
            | ~np.isfinite(area_ratio)
            | (area_ratio < 0.0)
            | (area_ratio > 1.0)
        ),
        "center_proximity": valid_bbox_rows
        & (
            ~center_is_numeric
            | ~np.isfinite(center_proximity)
            | (center_proximity < 0.0)
            | (center_proximity > 1.0)
        ),
        "growth_rate_per_sec": (
            valid_growth_rows
            & (~growth_is_numeric | ~np.isfinite(growth_rate))
        )
        | (
            missing_growth_rows
            & (~growth_is_missing | np.isfinite(growth_rate))
        ),
    }
    _raise_invalid_features(invalid_masks)

    row_count = len(features)
    area_signal = np.zeros(row_count, dtype=float)
    growth_signal = np.zeros(row_count, dtype=float)
    center_signal = np.zeros(row_count, dtype=float)
    roi_signal = np.zeros(row_count, dtype=float)

    area_signal[valid_bbox_rows] = np.clip(
        area_ratio[valid_bbox_rows] / float(config.area_full_scale),
        0.0,
        1.0,
    )
    scored_growth_rows = valid_bbox_rows & valid_growth_rows
    growth_signal[scored_growth_rows] = np.clip(
        np.maximum(growth_rate[scored_growth_rows], 0.0)
        / float(config.growth_full_scale_per_sec),
        0.0,
        1.0,
    )
    center_signal[valid_bbox_rows] = np.clip(
        center_proximity[valid_bbox_rows],
        0.0,
        1.0,
    )
    roi_signal[valid_bbox_rows & in_roi] = 1.0

    area_contribution = float(config.area_weight) * area_signal
    growth_contribution = float(config.growth_weight) * growth_signal
    center_contribution = float(config.center_weight) * center_signal
    roi_contribution = float(config.roi_weight) * roi_signal
    total_score = (
        area_contribution
        + growth_contribution
        + center_contribution
        + roi_contribution
    )

    result = features.copy(deep=True)
    result["area_signal"] = area_signal
    result["growth_signal"] = growth_signal
    result["center_signal"] = center_signal
    result["roi_signal"] = roi_signal
    result["area_contribution"] = area_contribution
    result["growth_contribution"] = growth_contribution
    result["center_contribution"] = center_contribution
    result["roi_contribution"] = roi_contribution
    result["total_score"] = total_score
    return result


__all__ = ["RiskScoreConfig", "compute_video_risk_scores"]

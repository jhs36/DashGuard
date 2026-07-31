"""Pure per-object video feature extraction from tracked-video rows."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
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

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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

DEFAULT_ROI_VERTICES: Final[tuple[tuple[float, float], ...]] = (
    (0.35, 0.48),
    (0.65, 0.48),
    (0.95, 1.00),
    (0.05, 1.00),
)

_BOUNDARY_TOLERANCE_SCALE: Final[float] = 1e-9


@dataclass(frozen=True)
class RiskFeatureConfig:
    """Configuration for geometry, continuity, and area smoothing."""

    frame_width: int
    frame_height: int
    smoothing_window_observations: int = 3
    max_gap_sec: float = 0.20
    max_gap_frames: int = 6
    roi_vertices_normalized: tuple[tuple[float, float], ...] = (
        DEFAULT_ROI_VERTICES
    )

    def __post_init__(self) -> None:
        _validate_positive_integer("frame_width", self.frame_width)
        _validate_positive_integer("frame_height", self.frame_height)
        if (
            isinstance(self.smoothing_window_observations, bool)
            or not isinstance(self.smoothing_window_observations, Integral)
            or self.smoothing_window_observations < 2
        ):
            raise ValueError(
                "smoothing_window_observations must be an integer of at least 2"
            )
        _validate_positive_finite("max_gap_sec", self.max_gap_sec)
        _validate_positive_integer("max_gap_frames", self.max_gap_frames)
        _validate_roi(self.roi_vertices_normalized)


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be an integer greater than zero")


def _validate_positive_finite(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and greater than zero")


def _validate_roi(vertices: object) -> None:
    try:
        vertex_count = len(vertices)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "roi_vertices_normalized must contain exactly four points"
        ) from exc
    if vertex_count != 4:
        raise ValueError(
            "roi_vertices_normalized must contain exactly four points"
        )

    for vertex in vertices:  # type: ignore[union-attr]
        try:
            coordinate_count = len(vertex)
        except TypeError as exc:
            raise ValueError(
                "each normalized ROI point must contain exactly two coordinates"
            ) from exc
        if coordinate_count != 2:
            raise ValueError(
                "each normalized ROI point must contain exactly two coordinates"
            )
        for coordinate in vertex:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, Real)
                or not math.isfinite(float(coordinate))
                or not 0.0 <= float(coordinate) <= 1.0
            ):
                raise ValueError(
                    "normalized ROI coordinates must be finite and within [0, 1]"
                )


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return one input column as float values, coercing bad values to NaN."""

    return pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )


def _validated_history_values(
    tracks: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return frame, timestamp, and track numeric arrays."""

    values_by_column: dict[str, np.ndarray] = {}
    invalid_counts: dict[str, int] = {}
    for column in ("frame_index", "timestamp_sec", "track_id"):
        original = tracks[column]
        converted = pd.to_numeric(original, errors="coerce")
        converted_array = converted.to_numpy()
        complex_values = np.fromiter(
            (
                isinstance(value, (complex, np.complexfloating))
                for value in original.array
            ),
            dtype=bool,
            count=len(original),
        )
        if np.iscomplexobj(converted_array):
            values = np.asarray(converted_array.real, dtype=float)
        else:
            values = converted.to_numpy(dtype=float, na_value=np.nan)

        invalid = ~np.isfinite(values) | complex_values
        if column in {"frame_index", "track_id"}:
            bool_values = np.fromiter(
                (
                    isinstance(value, (bool, np.bool_))
                    for value in original.array
                ),
                dtype=bool,
                count=len(original),
            )
            invalid |= bool_values | (values != np.trunc(values))
        if column in {"frame_index", "timestamp_sec"}:
            invalid |= values < 0.0

        values_by_column[column] = values
        invalid_count = int(invalid.sum())
        if invalid_count:
            invalid_counts[column] = invalid_count

    if invalid_counts:
        details = "; ".join(
            f"{column} has {count} invalid row(s)"
            for column, count in invalid_counts.items()
        )
        raise ValueError(f"invalid identity/time input values: {details}")

    return (
        values_by_column["frame_index"],
        values_by_column["timestamp_sec"],
        values_by_column["track_id"],
    )


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    tolerance: float,
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    segment_length = math.hypot(dx, dy)
    if segment_length <= tolerance:
        return math.hypot(px - ax, py - ay) <= tolerance

    cross = (px - ax) * dy - (py - ay) * dx
    if abs(cross) > tolerance * segment_length:
        return False
    return (
        min(ax, bx) - tolerance <= px <= max(ax, bx) + tolerance
        and min(ay, by) - tolerance <= py <= max(ay, by) + tolerance
    )


def _point_in_polygon(
    point: tuple[float, float],
    vertices: tuple[tuple[float, float], ...],
    tolerance: float,
) -> bool:
    """Return whether a point is inside or on an arbitrary polygon."""

    inside = False
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        if _point_on_segment(point, start, end, tolerance):
            return True

        _, py = point
        _, ay = start
        _, by = end
        if (ay > py) == (by > py):
            continue
        intersection_x = (
            start[0]
            + (py - ay) * (end[0] - start[0]) / (by - ay)
        )
        if point[0] < intersection_x:
            inside = not inside
    return inside


def _same_track_id(left: object, right: object) -> bool:
    left_missing = bool(pd.isna(left))
    right_missing = bool(pd.isna(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(left == right)


def _growth_rate(current: float, previous: float, delta_t: float) -> float:
    """Return finite relative growth per second, or NaN if undefined."""

    if (
        not math.isfinite(current)
        or not math.isfinite(previous)
        or previous <= 0.0
        or not math.isfinite(delta_t)
        or delta_t <= 0.0
    ):
        return math.nan
    try:
        candidate = ((current - previous) / previous) / delta_t
    except (OverflowError, ZeroDivisionError):
        return math.nan
    return candidate if math.isfinite(candidate) else math.nan


def _empty_feature_columns(result: pd.DataFrame) -> None:
    float_columns = (
        "bbox_area",
        "area_ratio",
        "smoothed_area_ratio",
        "raw_growth_rate_per_sec",
        "growth_rate_per_sec",
        "center_proximity",
    )
    for column in FEATURE_COLUMNS:
        if column in {"bbox_valid", "bbox_clipped", "growth_valid", "in_roi"}:
            result[column] = pd.Series(index=result.index, dtype=bool)
        elif column == "history_length":
            result[column] = pd.Series(index=result.index, dtype="int64")
        elif column == "growth_status":
            result[column] = pd.Series(index=result.index, dtype=object)
        elif column in float_columns:
            result[column] = pd.Series(index=result.index, dtype=float)


def compute_video_risk_features(
    tracks: pd.DataFrame,
    *,
    config: RiskFeatureConfig,
) -> pd.DataFrame:
    """Append deterministic per-track geometry and growth features."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in tracks.columns
    ]
    if missing_columns:
        raise ValueError(
            "tracks is missing required columns: " + ", ".join(missing_columns)
        )

    frame_index, timestamp, track_ids = _validated_history_values(tracks)
    result = tracks.copy(deep=True)
    row_count = len(result)
    if row_count == 0:
        _empty_feature_columns(result)
        return result

    x1 = _numeric_values(tracks, "x1")
    y1 = _numeric_values(tracks, "y1")
    x2 = _numeric_values(tracks, "x2")
    y2 = _numeric_values(tracks, "y2")
    coordinates_finite = (
        np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)
    )
    original_geometry_valid = coordinates_finite & (x2 > x1) & (y2 > y1)

    clipped_x1 = np.clip(x1, 0.0, float(config.frame_width))
    clipped_y1 = np.clip(y1, 0.0, float(config.frame_height))
    clipped_x2 = np.clip(x2, 0.0, float(config.frame_width))
    clipped_y2 = np.clip(y2, 0.0, float(config.frame_height))
    bbox_clipped = coordinates_finite & (
        (clipped_x1 != x1)
        | (clipped_y1 != y1)
        | (clipped_x2 != x2)
        | (clipped_y2 != y2)
    )

    clipped_width = clipped_x2 - clipped_x1
    clipped_height = clipped_y2 - clipped_y1
    bbox_valid = original_geometry_valid & (clipped_width > 0.0) & (
        clipped_height > 0.0
    )
    bbox_area = np.full(row_count, np.nan, dtype=float)
    bbox_area[bbox_valid] = clipped_width[bbox_valid] * clipped_height[bbox_valid]
    area_ratio = bbox_area / float(config.frame_width * config.frame_height)

    center_x = (clipped_x1 + clipped_x2) / 2.0
    center_y = clipped_y2
    center_proximity = np.full(row_count, np.nan, dtype=float)
    normalized_x = (
        center_x[bbox_valid] - config.frame_width / 2.0
    ) / (config.frame_width / 2.0)
    normalized_y = (
        config.frame_height - center_y[bbox_valid]
    ) / config.frame_height
    distance = np.hypot(normalized_x, normalized_y) / math.sqrt(2.0)
    center_proximity[bbox_valid] = 1.0 - np.clip(distance, 0.0, 1.0)

    roi_vertices = tuple(
        (x * config.frame_width, y * config.frame_height)
        for x, y in config.roi_vertices_normalized
    )
    boundary_tolerance = _BOUNDARY_TOLERANCE_SCALE * max(
        config.frame_width,
        config.frame_height,
        1,
    )
    in_roi = np.zeros(row_count, dtype=bool)
    for position in np.flatnonzero(bbox_valid):
        in_roi[position] = _point_in_polygon(
            (float(center_x[position]), float(center_y[position])),
            roi_vertices,
            boundary_tolerance,
        )

    ordering = pd.DataFrame(
        {
            "track_id": track_ids,
            "timestamp_sec": timestamp,
            "frame_index": frame_index,
            "original_position": np.arange(row_count, dtype=np.int64),
        }
    ).sort_values(
        ["track_id", "frame_index", "timestamp_sec", "original_position"],
        kind="mergesort",
        na_position="last",
    )

    smoothed_area_ratio = np.full(row_count, np.nan, dtype=float)
    raw_growth = np.full(row_count, np.nan, dtype=float)
    smoothed_growth = np.full(row_count, np.nan, dtype=float)
    history_length = np.zeros(row_count, dtype=np.int64)
    growth_valid = np.zeros(row_count, dtype=bool)
    growth_status = np.full(row_count, "", dtype=object)

    no_track = object()
    current_track: object = no_track
    seen_valid = False
    active_segment = False
    previous_timestamp = math.nan
    previous_frame_index = math.nan
    previous_area_ratio = math.nan
    previous_smoothed_area = math.nan
    segment_history = 0
    area_window: deque[float] = deque(
        maxlen=int(config.smoothing_window_observations)
    )

    for position_value in ordering["original_position"].to_numpy():
        position = int(position_value)
        track_id = track_ids[position]
        if current_track is no_track or not _same_track_id(
            current_track, track_id
        ):
            current_track = track_id
            seen_valid = False
            active_segment = False
            area_window.clear()

        if not bbox_valid[position]:
            growth_status[position] = "invalid_bbox"
            seen_valid = False
            active_segment = False
            segment_history = 0
            area_window.clear()
            previous_smoothed_area = math.nan
            continue

        row_timestamp = float(timestamp[position])
        row_frame_index = float(frame_index[position])
        status = "first_observation" if not seen_valid else "insufficient_history"
        reset_segment = not active_segment
        delta_t = math.nan

        if active_segment:
            delta_t = row_timestamp - previous_timestamp
            frame_gap = row_frame_index - previous_frame_index
            if (
                not math.isfinite(delta_t)
                or not math.isfinite(frame_gap)
                or delta_t <= 0.0
                or frame_gap <= 0.0
            ):
                status = "nonpositive_delta"
                reset_segment = True
            elif (
                delta_t > config.max_gap_sec
                or frame_gap > config.max_gap_frames
            ):
                status = "gap_reset"
                reset_segment = True
            else:
                reset_segment = False

        if reset_segment:
            segment_history = 1
            area_window.clear()
            previous_smoothed_area = math.nan
        else:
            segment_history += 1
            raw_growth[position] = _growth_rate(
                float(area_ratio[position]),
                previous_area_ratio,
                delta_t,
            )

        area_window.append(float(area_ratio[position]))
        if len(area_window) == config.smoothing_window_observations:
            current_smoothed_area = math.fsum(area_window) / len(area_window)
            smoothed_area_ratio[position] = current_smoothed_area
        else:
            current_smoothed_area = math.nan

        if (
            not reset_segment
            and math.isfinite(current_smoothed_area)
            and math.isfinite(previous_smoothed_area)
        ):
            smoothed_growth[position] = _growth_rate(
                current_smoothed_area,
                previous_smoothed_area,
                delta_t,
            )
            if math.isfinite(smoothed_growth[position]):
                growth_valid[position] = True
                status = "valid"

        history_length[position] = segment_history
        growth_status[position] = status
        seen_valid = True
        active_segment = True
        previous_timestamp = row_timestamp
        previous_frame_index = row_frame_index
        previous_area_ratio = float(area_ratio[position])
        previous_smoothed_area = current_smoothed_area

    result["bbox_valid"] = bbox_valid
    result["bbox_clipped"] = bbox_clipped
    result["bbox_area"] = bbox_area
    result["area_ratio"] = area_ratio
    result["smoothed_area_ratio"] = smoothed_area_ratio
    result["raw_growth_rate_per_sec"] = raw_growth
    result["growth_rate_per_sec"] = smoothed_growth
    result["history_length"] = history_length
    result["growth_valid"] = growth_valid
    result["growth_status"] = growth_status
    result["center_proximity"] = center_proximity
    result["in_roi"] = in_roi
    return result


__all__ = ["RiskFeatureConfig", "compute_video_risk_features"]
